### Generic Natural Visibility Graph (NVG) implementation

import numpy as np
from collections import deque
from math import fabs, inf

# Visibility-graph direction options
_DIRECTED_OPTIONS = {
    None:             0,
    "left_to_right":  1,
    "top_to_bottom":  2,
}

# convenience constants
_DIRECTED_NONE           = _DIRECTED_OPTIONS[None]
_DIRECTED_LEFT_TO_RIGHT  = _DIRECTED_OPTIONS["left_to_right"]
_DIRECTED_TOP_TO_BOTTOM  = _DIRECTED_OPTIONS["top_to_bottom"]

# tolerances
ABS_TOL = 1e-14
REL_TOL = 1e-14

def _argmax(ts: np.ndarray, left: int, right: int) -> int:
    """Index of the maximum of ts[left:right]."""
    sub = ts[left:right]
    return left + int(np.argmax(sub))

def _greater(a: float, b: float, tol: float) -> bool:
    """Tolerance-aware comparison: a > b (with tol)."""
    return a > b + tol

def _get_weight_func(weighted: int):
    """
    Return a Python weight function:
      - if weighted==0: weight = 1.0
      - if weighted!=0: weight = slope
    """
    if weighted:
        return lambda x1, x2, y1, y2, slope: slope
    else:
        return lambda *args: 1.0

def compute_visibility_graph(
    ts: np.ndarray,
    xs: np.ndarray,
    directed: int    = _DIRECTED_LEFT_TO_RIGHT,
    weighted: int    = 0,
    only_degrees: bool = False,
    min_weight: float  = -inf,
    max_weight: float  = inf,
):
    """
    Natural Visibility Graph (divide & conquer).
    Returns (edges, deg_in, deg_out), where
      • edges is a list of (i, j) or (i, j, w)
      • deg_in/out are np.uint32 arrays of node degrees.
    """
    n = ts.size
    edges   = []
    deg_in  = np.zeros(n, dtype=np.uint32)
    deg_out = np.zeros(n, dtype=np.uint32)

    weight_func = _get_weight_func(weighted)
    queue = deque()
    queue.append((0, n))

    while queue:
        left, right = queue.popleft()
        if left + 1 < right:
            # 1) pivot: index of max in ts[left:right]
            i = _argmax(ts, left, right)
            x_a, y_a = xs[i], ts[i]

            # 2) sweep left from i
            max_slope = -inf
            for d in range(1, i - left + 1):
                x_b, y_b = xs[i - d], ts[i - d]
                slope = (y_b - y_a) / (-(x_b - x_a))
                tol = max(
                    ABS_TOL,
                    REL_TOL * max(fabs(x_a), fabs(x_b), fabs(y_a), fabs(y_b))
                )
                if _greater(slope, max_slope, tol):
                    w = weight_func(x_a, x_b, y_a, y_b, -slope)
                    if min_weight < w < max_weight:
                        # pick orientation
                        if directed == _DIRECTED_TOP_TO_BOTTOM:
                            u, v = i, i - d
                        else:
                            u, v = i - d, i
                        deg_out[u] += 1
                        deg_in[v]  += 1
                        if not only_degrees:
                            edges.append((u, v, w) if weighted else (u, v))
                    max_slope = slope

            # 3) sweep right from i
            max_slope = -inf
            for d in range(1, right - i):
                x_b, y_b = xs[i + d], ts[i + d]
                slope = (y_b - y_a) / (x_b - x_a)
                tol = max(
                    ABS_TOL,
                    REL_TOL * max(fabs(x_a), fabs(x_b), fabs(y_a), fabs(y_b))
                )
                if _greater(slope, max_slope, tol):
                    w = weight_func(x_a, x_b, y_a, y_b, slope)
                    if min_weight < w < max_weight:
                        u, v = i, i + d
                        deg_out[u] += 1
                        deg_in[v]  += 1
                        if not only_degrees:
                            edges.append((u, v, w) if weighted else (u, v))
                    max_slope = slope

            # 4) recurse on left/right intervals
            queue.append((left,   i))
            queue.append((i + 1, right))

    return edges, deg_in, deg_out
