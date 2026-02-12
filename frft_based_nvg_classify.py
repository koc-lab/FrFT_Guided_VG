# Classification codes for FrFT-based NVGs

import os, sys, io, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import ChebConv, global_max_pool, GraphNorm
from datetime import datetime


# ===================== Helpers =====================
def graph_dir(base_out, transform, order, filter_kind, split_name):
    """
    Matches generator layout:
    base_out / transform / order_<order> / filter_<filter_kind> / (train|test|all)
    """
    filter_folder = f"filter_{filter_kind}"
    return os.path.join(base_out, transform, f"order_{order}", filter_folder, split_name)


def load_graphs_from_dir(graph_dir_path):
    data_list = []
    if not os.path.isdir(graph_dir_path):
        print(f"[WARN] Directory not found: {graph_dir_path} (skipping)")
        return data_list

    for fname in sorted(os.listdir(graph_dir_path)):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(graph_dir_path, fname)
        npz  = np.load(path)

        feats = npz["features"]
        if feats.ndim == 1:
            feats = feats[:, None]
        x = torch.tensor(feats, dtype=torch.float32)

        A = npz["adjacency"]
        rows, cols = np.nonzero(A)
        edge_index  = torch.tensor([rows, cols], dtype=torch.long)
        edge_weight = torch.tensor(A[rows, cols], dtype=torch.float32)

        y = int(npz["label"])

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([y], dtype=torch.long)
        )
        data_list.append(data)

    return data_list


class GraphConvNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = ChebConv(in_channels,  hidden_channels, K=3)
        self.bn1   = GraphNorm(hidden_channels)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=3)
        self.bn2   = GraphNorm(hidden_channels)
        self.conv3 = ChebConv(hidden_channels, hidden_channels, K=3)
        self.bn3   = GraphNorm(hidden_channels)
        self.lin   = Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight)))
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_weight)))
        x = global_max_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_correct = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * data.num_graphs
        total_correct += (out.argmax(1) == data.y).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = 0
    preds, labs = [], []
    for data in loader:
        data = data.to(device)
        out  = model(data.x, data.edge_index, data.edge_weight, data.batch)
        loss = criterion(out, data.y)
        total_loss    += loss.item() * data.num_graphs
        total_correct += (out.argmax(1) == data.y).sum().item()
        preds.append(out.argmax(1).cpu())
        labs.append(data.y.cpu())
    y_true = torch.cat(labs).numpy() if labs else np.array([])
    y_pred = torch.cat(preds).numpy() if preds else np.array([])
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset), y_true, y_pred


def compute_metrics(true, pred):
    acc = accuracy_score(true, pred)
    p, r, f1, _ = precision_recall_fscore_support(true, pred, average='macro', zero_division=0)
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1}


# --- Tee logger: write to both console and a file ---
class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)
    def flush(self):
        for st in self.streams:
            st.flush()


# ===================== Argparse =====================
def parse_args():
    parser = argparse.ArgumentParser(
        description="GNN classifier for NVG graphs generated by the FrFT-guided NVG generator."
    )

    parser.add_argument(
        "--dataset_format",
        type=str,
        choices=["ucr", "csv"],
        default="ucr",
        help="Graph split layout: 'ucr' expects (train,test), 'csv' expects (all).",
    )
    parser.add_argument(
        "--base_out",
        type=str,
        required=True,
        help="Base output directory used during graph generation.",
    )
    parser.add_argument(
        "--transform",
        type=str,
        choices=["frft", "dft"],
        default="frft",
        help="Transform name in the folder structure (must match generator).",
    )
    parser.add_argument(
        "--filter_kind",
        type=str,
        default="lowpass",
        help="Filter name in the folder structure (e.g., 'lowpass', 'identity').",
    )
    parser.add_argument(
        "--orders",
        type=float,
        nargs="+",
        default=[0.0],
        help="List of transform orders to train on (e.g., --orders 0.0 1.0 1.25).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[40, 41, 42],
        help="Random seeds for train/val/test splits.",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true",
                        help="Enable DataLoader pin_memory=True.")

    parser.add_argument(
        "--padding",
        type=str,
        choices=["none", "zero"],
        default="none",
        help="Padding mode used during graph generation (for bookkeeping & logs).",
    )

    return parser.parse_args()


# ========================= Main =========================
if __name__ == "__main__":
    args = parse_args()

    DATASET_FORMAT = args.dataset_format
    BASE_OUT       = args.base_out
    TRANSFORM      = args.transform
    FILTER_KIND    = args.filter_kind
    ORDERS         = args.orders
    SEEDS          = args.seeds

    BATCH_SIZE  = args.batch_size
    LR          = args.lr
    EPOCHS      = args.epochs
    DROPOUT     = args.dropout
    NUM_WORKERS = args.num_workers
    PIN_MEMORY  = args.pin_memory
    PADDING     = args.padding

    # Make sure output dir exists
    os.makedirs(BASE_OUT, exist_ok=True)

    # One results file per dataset (per BASE_OUT), shared across all orders & seeds
    dataset_tag = os.path.basename(os.path.normpath(BASE_OUT))
    results_txt = os.path.join(
        BASE_OUT,
        f"results_{dataset_tag}_{TRANSFORM}_{FILTER_KIND}_pad-{PADDING}.txt"
    )

    # Open file in append mode and tee stdout/stderr
    with open(results_txt, "a", buffering=1, encoding="utf-8") as log_f:
        sys.stdout = Tee(sys.stdout, log_f)
        sys.stderr = Tee(sys.stderr, log_f)

        print("\n" + "="*80)
        print(f"[{datetime.now().isoformat(timespec='seconds')}] START run")
        print(f"DATASET_FORMAT={DATASET_FORMAT} | BASE_OUT={BASE_OUT}")
        print(f"TRANSFORM={TRANSFORM} | FILTER_KIND={FILTER_KIND} | PADDING={PADDING}")
        print(f"ORDERS={ORDERS}")
        print(f"SEEDS={SEEDS}")
        print(f"BATCH_SIZE={BATCH_SIZE} | LR={LR} | EPOCHS={EPOCHS}")
        print("="*80)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # Checkpoints folder (kept per dataset)
        ckpt_dir = os.path.join(BASE_OUT, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Loop over each transform order
        for order in ORDERS:
            # Pick split names by format
            split_names = ["train", "test"] if DATASET_FORMAT == "ucr" else ["all"]

            # Load graphs for this order
            all_data = []
            for split in split_names:
                d = graph_dir(BASE_OUT, TRANSFORM, order, FILTER_KIND, split)
                all_data += load_graphs_from_dir(d)

            if len(all_data) == 0:
                print(f"[WARN] No graphs found for order={order}. Skipping.")
                continue

            # Basic info
            print("\n" + "-"*70)
            print(f"ORDER = {order} | splits: {split_names}")
            for split in split_names:
                print("  ", graph_dir(BASE_OUT, TRANSFORM, order, FILTER_KIND, split))
            print(f"Total graphs: {len(all_data)}")

            # Extract model dims / classes from data
            in_ch       = all_data[0].x.size(1)
            num_classes = int(max(d.y.item() for d in all_data) + 1)
            print(f"in_channels={in_ch}, num_classes={num_classes}")

            # Run each seed
            for SEED in SEEDS:
                # Repro per seed
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                train_val, test = train_test_split(all_data, test_size=0.15, random_state=SEED)
                train, val      = train_test_split(train_val, test_size=0.1765, random_state=SEED)
                print(f"\n[order={order} | seed={SEED}] → "
                      f"train: {len(train)}, val: {len(val)}, test: {len(test)}")

                # DataLoaders
                train_loader = DataLoader(
                    train, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
                )
                val_loader   = DataLoader(
                    val,   batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
                )
                test_loader  = DataLoader(
                    test,  batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
                )

                # Model / opt / loss
                model     = GraphConvNet(in_ch,
                                         hidden_channels=args.hidden_channels,
                                         num_classes=num_classes,
                                         dropout=DROPOUT).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                criterion = torch.nn.CrossEntropyLoss()

                tag = f"{TRANSFORM}_order{order}_{FILTER_KIND}_pad-{PADDING}"
                ckpt_path = os.path.join(
                    ckpt_dir, f"best_{DATASET_FORMAT}_{tag}_seed{SEED}.pt"
                )
                best_val_loss = float("inf")

                # Train (select best by lowest validation loss)
                for epoch in range(1, EPOCHS + 1):
                    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
                    va_loss, va_acc, _, _ = eval_epoch(model, val_loader, criterion, device)

                    if va_loss < best_val_loss:
                        best_val_loss = va_loss
                        torch.save(model.state_dict(), ckpt_path)

                    print(f"[order={order} | seed={SEED}] "
                          f"Epoch {epoch:03d} | "
                          f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
                          f"Val Loss:   {va_loss:.4f} | Val Acc:   {va_acc:.4f}")

                # Final test (load best)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                te_loss, te_acc, y_true, y_pred = eval_epoch(model, test_loader, criterion, device)
                metrics = compute_metrics(y_true, y_pred)

                print(f"\n[RESULT] order={order} | seed={SEED} | tag={tag}")
                print(f"Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.4f} | "
                      f"Precision: {metrics['precision']:.4f} | "
                      f"Recall: {metrics['recall']:.4f} | "
                      f"F1: {metrics['f1_score']:.4f}")
                print("-"*70)

        print(f"[{datetime.now().isoformat(timespec='seconds')}] END run")
        print("="*80)

