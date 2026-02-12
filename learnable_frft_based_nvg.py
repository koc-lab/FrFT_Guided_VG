## Learning the best order for FrFT-based NVG (edge weights) 

import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import Linear
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import ChebConv, global_max_pool, GraphNorm
from torch_geometric.utils import to_undirected

from nvg   import compute_visibility_graph

# Differentiable FrFT layer (pip: torch-frft)
from torch_frft.layer import DFrFTLayer
from torch_frft.dfrft_module import dfrft


# ---------------------- Label utils ----------------------
def normalize_labels(raw: np.ndarray) -> np.ndarray:
    raw = raw.astype(np.int64, copy=False)
    uniq = np.unique(raw)

    # FordA/FordB case
    if len(uniq) == 2 and uniq[0] == -1 and uniq[1] == 1:
        return ((raw + 1) // 2).astype(np.int64)

    # Typical 1..K labels
    if uniq.min() == 1 and np.array_equal(uniq, np.arange(1, uniq.max() + 1)):
        return (raw - 1).astype(np.int64)

    # Fallback: sorted unique → 0..C-1
    lut = {lab: i for i, lab in enumerate(uniq.tolist())}
    return np.vectorize(lut.get, otypes=[np.int64])(raw)


# -------------------- Dataset loading --------------------
def load_arrays(dataset_format,
                train_path=None,
                test_path=None,
                csv_path=None,
                csv_label_col=-1):
    """
    dataset_format = 'ucr':
        - train_path, test_path are required
        - first column = label, remaining = time-series samples
    dataset_format = 'csv':
        - csv_path is required
        - csv_label_col: which column is label (default: -1 = last)
        - all other columns are treated as samples
    """
    dataset_format = dataset_format.lower()

    if dataset_format == "ucr":
        if train_path is None or test_path is None:
            raise ValueError("For dataset_format='ucr', train_path and test_path must be provided.")

        tr = pd.read_csv(train_path, header=None, sep=None, engine="python").astype(np.float32)
        te = pd.read_csv(test_path,  header=None, sep=None, engine="python").astype(np.float32)
        df = pd.concat([tr, te], axis=0, ignore_index=True)

        y_raw = df.iloc[:, 0].values
        X     = df.iloc[:, 1:].values
        y     = normalize_labels(y_raw)
        return X, y

    elif dataset_format == "csv":
        if csv_path is None:
            raise ValueError("For dataset_format='csv', csv_path must be provided.")

        df = pd.read_csv(csv_path)
        n_cols = df.shape[1]
        if csv_label_col < 0:
            csv_label_col = n_cols + csv_label_col
        y_raw = df.iloc[:, csv_label_col].values
        X     = df.drop(df.columns[csv_label_col], axis=1).astype(np.float32).values
        y     = normalize_labels(y_raw)
        return X, y

    else:
        raise ValueError(f"Unknown DATASET_FORMAT: {dataset_format}")


# -------------------- Filter definition --------------------
def make_filter_vector(N, kind):
    """
    Frequency-domain filter g[k] used inside the (learned) FrFT domain.

    - 'identity': g[k] = 1 for all k
    - 'lowpass' : keep ~15% of bins at the beginning and 15% at the end
                  of the spectrum, zero elsewhere.
    """
    kind = kind.lower()
    k = torch.arange(N, dtype=torch.float32)
    g = torch.zeros_like(k)

    if kind == "identity":
        g[:] = 1.0

    elif kind == "lowpass":
        cut = max(1, int(round(0.15 * N)))
        if 2 * cut >= N:
            # Degenerate small-N case: pass everything
            g[:] = 1.0
        else:
            g[:cut] = 1.0
            g[-cut:] = 1.0

    else:
        raise ValueError(f"Unknown filter kind: {kind}")

    return g


# -------------- Dataset: NVG topology on-the-fly --------------
class NVGOnTheFlyDataset(InMemoryDataset):
    """
    For each time series:
      - x: (N, 1) node features (raw signal)
      - edge_index: NVG edges computed on-the-fly from the series
    Topology can optionally be cached per index.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, cache_topology=True):
        super().__init__(".")
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.cache_topology = cache_topology
        self._edge_cache = {}  # idx -> edge_index (2,E)

    def len(self):
        return len(self.X)

    def get(self, idx):
        series = self.X[idx]
        N = series.shape[0]
        x = torch.tensor(series[:, None], dtype=torch.float32)  # (N,1)

        if self.cache_topology and idx in self._edge_cache:
            edge_index = self._edge_cache[idx]
        else:
            xs = np.arange(N, dtype=float)
            edges, _, _ = compute_visibility_graph(
                ts=series, xs=xs, directed=None, weighted=0, only_degrees=False
            )
            if edges:
                rows, cols = zip(*edges)
            else:
                rows, cols = [], []
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_index = to_undirected(edge_index, num_nodes=N)
            if self.cache_topology:
                self._edge_cache[idx] = edge_index

        y = torch.tensor([int(self.y[idx])], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)


# -------------- FrFT → filtered time series Y(a) --------------
def frft_time_filter_with_layer(x_1d,
                                frft_layer: DFrFTLayer,
                                kind: str,
                                padding_mode: str):
    """
    x_1d: (N,) float tensor (real)

    padding_mode:
      - 'none': no time-domain padding
      - 'zero': zero-pad in time to 2N before FrFT, then crop back to N

    Returns:
        Y = |FrFT^{-1}_a [ G ⊙ FrFT_a(x (padded/not)) ]|
        cropped back to length N_orig if padded.
    """
    padding_mode = padding_mode.lower()
    N_orig = x_1d.numel()

    if padding_mode == "zero":
        N_pad = 2 * N_orig
        x_pad = torch.zeros(N_pad, dtype=x_1d.dtype, device=x_1d.device)
        x_pad[:N_orig] = x_1d
        x_proc = x_pad
        N = N_pad
    elif padding_mode == "none":
        x_proc = x_1d
        N = N_orig
    else:
        raise ValueError(f"Unknown padding_mode: {padding_mode}")

    g = make_filter_vector(N, kind).to(x_proc.device)

    # Forward FrFT with learnable order
    Xa   = frft_layer(x_proc.to(torch.complex64))      # complex spectrum
    Xa_f = Xa * g.to(Xa.dtype)                         # filtered spectrum

    # Inverse FrFT with same (learned) order, but sign-flipped
    Ya   = dfrft(Xa_f, -frft_layer.order, dim=0)       # complex time-domain
    Y_full = torch.abs(Ya).float()

    if padding_mode == "zero":
        Y = Y_full[:N_orig]
    else:
        Y = Y_full

    return Y


def nvg_weights_from_Y(edge_index: torch.Tensor,
                       Y: torch.Tensor,
                       floor: float = 1e-6,
                       eps: float = 1e-8) -> torch.Tensor:
    """
    Visibility edges exist from NVG; we assign weights using Y.
    For an edge (u,v):
      d(u,v) = |Y[u] - Y[v]| / (|Y[u]| + |Y[v]| + eps) in [0,1]
      w(u,v) = max(1 - d(u,v), floor)
    """
    src, dst = edge_index
    yu, yv = Y[src], Y[dst]
    d = (yu - yv).abs() / (yu.abs() + yv.abs() + eps)
    w = 1.0 - d
    if floor > 0:
        w = torch.clamp(w, min=floor)
    return w


# -------------- Model: Learnable FrFT order GNN --------------
class LearnableOrderGNN(nn.Module):
    def __init__(self, in_channels, hidden, num_classes,
                 filter_kind="lowpass",
                 a_init=1.0,
                 dropout=0.5,
                 weight_floor=1e-6,
                 padding="none"):
        super().__init__()
        self.filter_kind  = filter_kind
        self.weight_floor = weight_floor
        self.padding      = padding

        # Learnable fractional order (initialized at a_init)
        self.frft = DFrFTLayer(order=float(a_init), dim=0, trainable=True)

        self.conv1 = ChebConv(in_channels, hidden, K=3)
        self.bn1   = GraphNorm(hidden)
        self.conv2 = ChebConv(hidden, hidden, K=3)
        self.bn2   = GraphNorm(hidden)
        self.conv3 = ChebConv(hidden, hidden, K=3)
        self.bn3   = GraphNorm(hidden)
        self.lin   = Linear(hidden, num_classes)
        self.dropout = dropout

    def alpha(self):
        """Return current learnable order (tensor scalar)."""
        return self.frft.order

    def _edge_weight_for_graph(self, x_1d, edge_index):
        Y = frft_time_filter_with_layer(
            x_1d.squeeze(-1), self.frft, self.filter_kind, self.padding
        )
        return nvg_weights_from_Y(edge_index, Y, floor=self.weight_floor)

    def forward(self, batch):
        outs = []
        for data in batch.to_data_list():
            x, ei = data.x, data.edge_index
            ew = self._edge_weight_for_graph(x.view(-1), ei)

            h = F.relu(self.bn1(self.conv1(x, ei, ew)))
            h = F.relu(self.bn2(self.conv2(h, ei, ew)))
            h = F.relu(self.bn3(self.conv3(h, ei, ew)))
            # Single graph per Data in this loop → batch index is all zeros
            g = global_max_pool(h, torch.zeros(h.size(0), dtype=torch.long, device=h.device))
            g = F.dropout(g, p=self.dropout, training=self.training)
            outs.append(self.lin(g))
        return torch.vstack(outs)


# ---------------- Train / Eval routines ----------------
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = 0
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = torch.cat([d.y for d in batch.to_data_list()]).to(device).view(-1)
        loss = criterion(logits, y)
        total_loss    += loss.item() * y.numel()
        total_correct += (logits.argmax(1) == y).sum().item()
        ys.append(y.cpu())
        ps.append(logits.argmax(1).cpu())
    y_true = torch.cat(ys).numpy() if ys else np.array([])
    y_pred = torch.cat(ps).numpy() if ps else np.array([])
    acc = (y_pred == y_true).mean() if len(y_true) else 0.0
    return total_loss / len(loader.dataset), acc, y_true, y_pred


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_correct = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        y = torch.cat([d.y for d in batch.to_data_list()]).to(device).view(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * y.numel()
        total_correct += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


# ------------------------- Argparse -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end learnable-order FrFT NVG classifier."
    )

    # Dataset
    p.add_argument(
        "--dataset_format",
        type=str,
        choices=["ucr", "csv"],
        default="ucr",
        help="Input format: 'ucr' (TRAIN/TEST files) or 'csv' (single CSV).",
    )
    p.add_argument(
        "--train_path",
        type=str,
        help="UCR TRAIN file path (used when dataset_format='ucr').",
    )
    p.add_argument(
        "--test_path",
        type=str,
        help="UCR TEST file path (used when dataset_format='ucr').",
    )
    p.add_argument(
        "--csv_path",
        type=str,
        help="CSV file path (used when dataset_format='csv').",
    )
    p.add_argument(
        "--csv_label_col",
        type=int,
        default=-1,
        help="Label column for CSV format (default: -1 = last column).",
    )

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for network weights.")
    p.add_argument("--lr_alpha", type=float, default=5e-3,
                   help="Learning rate for the FrFT order parameter.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)

    # Filter & weights
    p.add_argument(
        "--filter_kind",
        type=str,
        choices=["identity", "lowpass"],
        default="lowpass",
        help="Filter kind in FrFT domain: 'identity' or 'lowpass' (15% head & tail).",
    )
    p.add_argument(
        "--weight_floor",
        type=float,
        default=1e-6,
        help="Minimum edge weight.",
    )

    # Padding
    p.add_argument(
        "--padding",
        type=str,
        choices=["none", "zero"],
        default="none",
        help="Time-domain padding before FrFT: 'none' or 'zero' (pad to 2N, then crop).",
    )

    return p.parse_args()


# ----------------------------- Main -----------------------------
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load full array dataset (we will split into train/val/test)
    X, y = load_arrays(
        dataset_format=args.dataset_format,
        train_path=args.train_path,
        test_path=args.test_path,
        csv_path=args.csv_path,
        csv_label_col=args.csv_label_col,
    )

    # Split once; NVG topology fixed regardless of learned order
    idx_all = np.arange(len(X))
    train_val_idx, test_idx = train_test_split(
        idx_all, test_size=0.15, random_state=args.seed, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.1765, random_state=args.seed, stratify=y[train_val_idx]
    )

    ds_train = NVGOnTheFlyDataset(X[train_idx], y[train_idx], cache_topology=True)
    ds_val   = NVGOnTheFlyDataset(X[val_idx],   y[val_idx],   cache_topology=True)
    ds_test  = NVGOnTheFlyDataset(X[test_idx],  y[test_idx],  cache_topology=True)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False)

    num_classes = int(y.max() + 1)
    model = LearnableOrderGNN(
        in_channels=1,
        hidden=args.hidden_channels,
        num_classes=num_classes,
        filter_kind=args.filter_kind,
        a_init=1.0,
        dropout=args.dropout,
        weight_floor=args.weight_floor,
        padding=args.padding,
    ).to(device)

    # Two-branch optimizer: network weights vs. FrFT order
    params_order = [model.frft.order]
    params_others = [p for n, p in model.named_parameters() if n != "frft.order"]

    opt = torch.optim.Adam([
        {"params": params_others, "lr": args.lr},
        {"params": params_order,  "lr": args.lr_alpha},
    ])
    crit = nn.CrossEntropyLoss()

    # ---- Select best model by lowest validation loss ----
    best_val_loss = float("inf")
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, crit, device)
        va_loss, va_acc, *_ = eval_epoch(model, val_loader, crit, device)
        a_val = float(model.alpha().detach().cpu())

        print(f"Epoch {ep:03d} | alpha={a_val:.4f} | "
              f"Train: loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"Val: loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test with the best-val-loss checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc, y_true, y_pred = eval_epoch(model, test_loader, crit, device)
    a_final = float(model.alpha().detach().cpu())

    print("\n[TEST] Using best-val-loss checkpoint")
    print(f"alpha* = {a_final:.4f} | Test loss = {te_loss:.4f} | Test acc = {te_acc:.4f}")


if __name__ == "__main__":
    main()

