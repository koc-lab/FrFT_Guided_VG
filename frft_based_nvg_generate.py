# FrFT-based NVG generation and saving

import os
import math
import argparse
import numpy as np
import pandas as pd
import torch

from nvg   import compute_visibility_graph
from dfrft import dfrtmtrx2  # FrFT matrix


# ------------------------- Transform utils ------------------------
def dft_matrix(N: int) -> np.ndarray:
    n = np.arange(N, dtype=np.float32)
    k = n[:, None]
    W = np.exp(-2j * np.pi * k * n / N)
    return W.astype(np.complex64)


def get_transform_mats(N: int, order: float, transform: str, device: str | None = None):
    transform = transform.lower()
    if transform == "frft":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        F_t = dfrtmtrx2(N, order, device=device)
        F = F_t.detach().cpu().numpy().astype(np.complex64)
    elif transform == "dft":
        F = dft_matrix(N)
    else:
        raise ValueError(f"Unknown transform: {transform}")
    Finv = F.conj().T
    return F, Finv


# ---------------------- Label normalization -----------------------
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


# --------------------------- Filters ------------------------------
def make_filter_vector(N: int, kind: str) -> np.ndarray:
    """
    Frequency-domain filter g[k].

    - 'identity': g[k] = 1 for all k
    - 'lowpass' : keep ~15% of bins at the beginning and 15% at the end
                  of the spectrum, zero elsewhere.
    """
    kind = kind.lower()
    g = np.zeros(N, dtype=np.float32)

    if kind == "identity":
        g[:] = 1.0

    elif kind == "lowpass":
        # 15% from the beginning and 15% from the end
        cut = max(1, int(round(0.15 * N)))
        if 2 * cut >= N:
            # Degenerate small-N case: just pass everything
            g[:] = 1.0
        else:
            g[:cut] = 1.0
            g[-cut:] = 1.0
    else:
        raise ValueError(f"Unknown filter kind: {kind}")

    return g


# --------------------- Core: filter-then-compare ------------------
def filtered_time_series(series: np.ndarray,
                         order: float,
                         transform: str,
                         filter_kind: str,
                         padding: str = "none") -> np.ndarray:
    """
    series: 1D array (length N_orig)
    padding:
      - 'none': no time-domain padding (circular-type behaviour)
      - 'zero': zero-pad to 2*N_orig before transform, then crop back
    """
    x = series.astype(np.float32, copy=False)
    N_orig = x.size

    padding = padding.lower()
    if padding == "zero":
        N_pad = 2 * N_orig
        x_pad = np.zeros(N_pad, dtype=np.float32)
        x_pad[:N_orig] = x
        x_proc = x_pad
        N = N_pad
    elif padding == "none":
        x_proc = x
        N = N_orig
    else:
        raise ValueError(f"Unknown padding mode: {padding}")

    F, Finv = get_transform_mats(N, order, transform)
    x_c = x_proc.astype(np.complex64)

    S = (F @ x_c) / math.sqrt(N)
    g = make_filter_vector(N, filter_kind)
    Sf = g.astype(np.complex64) * S
    Yc = (Finv @ Sf) * math.sqrt(N)
    Y_full = np.abs(Yc).astype(np.float32)

    if padding == "zero":
        Y = Y_full[:N_orig]
    else:
        Y = Y_full

    return Y


def build_unweighted_nvg(series: np.ndarray, xs: np.ndarray) -> np.ndarray:
    N = series.size
    edges, _, _ = compute_visibility_graph(
        ts=series, xs=xs, directed=None, weighted=0, only_degrees=False
    )
    A = np.zeros((N, N), dtype=np.uint8)
    for u, v in edges:
        A[u, v] = A[v, u] = 1
    return A


def build_weighted_nvg_filter_then_compare(series: np.ndarray,
                                           xs: np.ndarray,
                                           order: float,
                                           transform: str,
                                           filter_kind: str,
                                           eps: float = 1e-8,
                                           weight_floor: float = 1e-6,
                                           padding: str = "none") -> np.ndarray:
    """
    NVG edges on ORIGINAL series; weights from filtered time features Y.
    d(u,v) = |Y[u]-Y[v]| / (|Y[u]| + |Y[v]| + eps) in [0,1]
    w(u,v) = max(1 - d(u,v), weight_floor)

    padding: passed to filtered_time_series ('none' or 'zero').
    """
    N = series.size

    edges, _, _ = compute_visibility_graph(
        ts=series, xs=xs, directed=None, weighted=0, only_degrees=False
    )

    Y = filtered_time_series(series, order, transform, filter_kind, padding=padding)

    A = np.zeros((N, N), dtype=np.float32)
    for (u, v) in edges:
        yu, yv = float(Y[u]), float(Y[v])
        num = abs(yu - yv)
        den = abs(yu) + abs(yv) + eps
        d = num / den
        w = 1.0 - d
        if w < weight_floor:
            w = weight_floor
        A[u, v] = A[v, u] = w

    return A


# --------------------------- Data loading -------------------------
def read_split(path: str,
               dataset_format: str,
               label_col: int = -1):
    """
    Generic data loader.

    dataset_format = 'ucr':
        - Plain text or CSV
        - First column: label
        - Remaining columns: time-series samples

    dataset_format = 'csv':
        - Standard CSV
        - label_col: index of label column (default: last)
        - Remaining columns (after dropping label_col): features
    """
    dataset_format = dataset_format.lower()
    if dataset_format == "ucr":
        df = pd.read_csv(path, header=None, sep=None, engine="python")
        y_raw = df.iloc[:, 0].values
        X = df.iloc[:, 1:].astype(np.float32).values
        y = normalize_labels(y_raw)
        return X, y

    elif dataset_format == "csv":
        df = pd.read_csv(path)
        n_cols = df.shape[1]
        if label_col < 0:
            label_col = n_cols + label_col
        y_raw = df.iloc[:, label_col].values
        X = df.drop(df.columns[label_col], axis=1).astype(np.float32).values
        y = normalize_labels(y_raw)
        return X, y

    else:
        raise ValueError(f"Unknown dataset_format: {dataset_format}")


# --------------------------- I/O pipelines ------------------------
def process_split_arrays(X: np.ndarray,
                         y: np.ndarray,
                         order: float,
                         transform: str,
                         filter_kind: str,
                         out_dir: str,
                         weight_floor: float = 1e-6,
                         padding: str = "none"):
    """
    Build NVG graphs from X, y for a GIVEN order.
    """
    N = X.shape[1]
    os.makedirs(out_dir, exist_ok=True)

    for idx, series in enumerate(X):
        xs = np.arange(N, dtype=float)
        A = build_weighted_nvg_filter_then_compare(
            series=series,
            xs=xs,
            order=order,
            transform=transform,
            filter_kind=filter_kind,
            weight_floor=weight_floor,
            padding=padding,
        )

        np.savez_compressed(
            os.path.join(out_dir, f"graph_{idx:06d}.npz"),
            adjacency=A,
            features=series.astype(np.float32, copy=False),
            label=y[idx]
        )

    print(f"[{transform} | order={order} | {filter_kind} | padding={padding}] "
          f"→ {len(X)} graphs at {out_dir}")


def process_split_unweighted(X: np.ndarray,
                             y: np.ndarray,
                             out_dir: str):
    """
    Pure unweighted NVGs (binary). For compatibility/baselines.
    """
    N = X.shape[1]
    os.makedirs(out_dir, exist_ok=True)

    for idx, series in enumerate(X):
        xs = np.arange(N, dtype=float)
        A = build_unweighted_nvg(series, xs)
        np.savez_compressed(
            os.path.join(out_dir, f"graph_{idx:06d}.npz"),
            adjacency=A,
            features=series.astype(np.float32, copy=False),
            label=y[idx]
        )

    print(f"[UNWEIGHTED NVG] → {len(X)} graphs at {out_dir}")


# ------------------------------- Main -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="NVG graph generator with FrFT/DFT-based filter-then-compare weights."
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        choices=["ucr", "csv"],
        default="ucr",
        help="Input format: 'ucr' (first col = label) or 'csv' (generic CSV, label_col controls label).",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        help="Path to UCR-style TRAIN file (used when dataset_format='ucr').",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        help="Path to UCR-style TEST file (used when dataset_format='ucr').",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to a CSV file (used when dataset_format='csv').",
    )
    parser.add_argument(
        "--csv_label_col",
        type=int,
        default=-1,
        help="Label column index for CSV format (default: -1 = last column).",
    )
    parser.add_argument(
        "--base_out",
        type=str,
        required=True,
        help="Base output directory for generated graph .npz files.",
    )
    parser.add_argument(
        "--transform",
        type=str,
        choices=["frft", "dft"],
        default="frft",
        help="Transform type: 'frft' or 'dft'.",
    )
    parser.add_argument(
        "--orders",
        type=float,
        nargs="+",
        default=[0.0],
        help="List of FrFT orders to use (e.g. --orders 0.0 0.25 0.5 1.0).",
    )
    parser.add_argument(
        "--filter_kind",
        type=str,
        choices=["identity", "lowpass"],
        default="lowpass",
        help="Filter type. 'lowpass' keeps ~15% of bins at both ends of the spectrum.",
    )
    parser.add_argument(
        "--weight_floor",
        type=float,
        default=1e-6,
        help="Minimum edge weight for weighted NVGs.",
    )
    parser.add_argument(
        "--build_unweighted_baseline",
        action="store_true",
        help="If set, also build pure unweighted NVGs (binary).",
    )
    parser.add_argument(
        "--padding",
        type=str,
        choices=["none", "zero"],
        default="none",
        help="Time-domain padding: 'none' or 'zero' (pad to 2N, then crop back).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate dataset args
    if args.dataset_format == "ucr":
        if not args.train_path or not args.test_path:
            raise SystemExit(
                "For dataset_format='ucr', --train_path and --test_path must be provided."
            )
        splits = {
            "train": args.train_path,
            "test": args.test_path,
        }
        csv_label_col = None  # unused
    elif args.dataset_format == "csv":
        if not args.csv_path:
            raise SystemExit(
                "For dataset_format='csv', --csv_path must be provided."
            )
        splits = {"all": args.csv_path}
        csv_label_col = args.csv_label_col
    else:
        raise SystemExit(f"Unknown dataset_format: {args.dataset_format}")

    # Optional: unweighted baseline
    if args.build_unweighted_baseline:
        for split_name, path in splits.items():
            if args.dataset_format == "ucr":
                X, y = read_split(path, "ucr")
            else:
                X, y = read_split(path, "csv", label_col=csv_label_col)

            out_dir = os.path.join(args.base_out, "unweighted", split_name)
            process_split_unweighted(X, y, out_dir)

    # Weighted NVGs for all orders
    for order in args.orders:
        for split_name, path in splits.items():
            if args.dataset_format == "ucr":
                X, y = read_split(path, "ucr")
            else:
                X, y = read_split(path, "csv", label_col=csv_label_col)

            out_dir = os.path.join(
                args.base_out,
                args.transform,
                f"order_{order}",
                f"filter_{args.filter_kind}",
                split_name,
            )

            process_split_arrays(
                X=X,
                y=y,
                order=order,
                transform=args.transform,
                filter_kind=args.filter_kind,
                out_dir=out_dir,
                weight_floor=args.weight_floor,
                padding=args.padding,
            )


if __name__ == "__main__":
    main()

