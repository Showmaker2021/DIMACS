"""Parametric priority function and fast JIT greedy coloring.

Normalized Features:
phi_0 = sat * (n + 1)                     (scale ~ n^2, dominant saturation term)
phi_1 = uncolored_deg                     (scale ~ n, uncolored degree)
phi_2 = deg                               (scale ~ n, total degree)
phi_3 = (sat / max(deg, 1)) * (n + 1)     (scale ~ n, relative saturation)
phi_4 = (uncolored_deg / max(deg, 1)) * n (scale ~ n, relative uncolored degree)

DSATUR weight vector:
w = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
Reduces exactly to classic DSATUR: sat * (n + 1) + uncolored_deg.
"""

from typing import Callable, Tuple
import numpy as np
import numba as nb

DSATUR_WEIGHTS = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
NUM_FEATURES = 5


def make_parametric_priority(weights: np.ndarray) -> Callable:
    """Create a pure priority function matching the fixed contract API for given weights."""
    w = np.asarray(weights, dtype=np.float64)
    w0, w1, w2, w3, w4 = w[0], w[1], w[2], w[3], w[4]

    def priority_fn(
        adj: np.ndarray,
        colors: np.ndarray,
        sat: np.ndarray,
        deg: np.ndarray,
        uncolored_deg: np.ndarray,
    ) -> np.ndarray:
        n = adj.shape[0]
        np1 = float(n + 1)
        nf = float(n)
        d = deg.astype(np.float64)
        inv_deg = 1.0 / np.maximum(d, 1.0)
        s = sat.astype(np.float64)
        u = uncolored_deg.astype(np.float64)

        scores = (
            w0 * s * np1
            + w1 * u
            + w2 * d
            + w3 * s * inv_deg * np1
            + w4 * u * inv_deg * nf
        )
        return scores

    return priority_fn


@nb.njit
def _parametric_color_numba(adj: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    deg = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                deg[i] += 1

    max_k = 1
    for i in range(n):
        if deg[i] + 2 > max_k:
            max_k = deg[i] + 2

    nb_has_color = np.zeros((n, max_k), dtype=np.bool_)
    sat = np.zeros(n, dtype=np.int64)
    uncolored_deg = deg.copy()
    colored = np.zeros(n, dtype=np.bool_)
    colors = np.full(n, -1, dtype=np.int64)

    w0, w1, w2, w3, w4 = weights[0], weights[1], weights[2], weights[3], weights[4]
    np1 = float(n + 1)
    nf = float(n)

    for _ in range(n):
        best_v = -1
        best_score = -1e30
        for i in range(n):
            if not colored[i]:
                d = float(deg[i])
                inv_deg = 1.0 / (d if d > 1.0 else 1.0)
                s = float(sat[i])
                u = float(uncolored_deg[i])
                score = (
                    w0 * s * np1
                    + w1 * u
                    + w2 * d
                    + w3 * s * inv_deg * np1
                    + w4 * u * inv_deg * nf
                )
                if best_v == -1 or score > best_score:
                    best_score = score
                    best_v = i

        v = best_v
        c = 0
        while c < max_k and nb_has_color[v, c]:
            c += 1

        colors[v] = c
        colored[v] = True

        for u in range(n):
            if adj[v, u]:
                if not nb_has_color[u, c]:
                    nb_has_color[u, c] = True
                    sat[u] += 1
                uncolored_deg[u] -= 1

    return colors


def parametric_color(adj: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Fast JIT parametric greedy coloring."""
    w = np.asarray(weights, dtype=np.float64)
    return _parametric_color_numba(adj, w)


@nb.njit
def evaluate_weights_on_graphs(adjs: nb.types.List, weights: np.ndarray) -> Tuple[float, float]:
    """Evaluate weights across a list of graph adjacency matrices.

    Returns:
    (guided_loss, real_mean_colors)
    where guided_loss includes a tiny tie-breaker: + 1e-4 * (count_last_color / n).
    """
    total_loss = 0.0
    total_colors = 0.0
    num_graphs = len(adjs)

    for i in range(num_graphs):
        adj = adjs[i]
        n = adj.shape[0]
        col = _parametric_color_numba(adj, weights)

        # Count colors
        max_c = -1
        for j in range(n):
            if col[j] > max_c:
                max_c = col[j]
        k = max_c + 1
        total_colors += float(k)

        # Count vertices in last color class
        last_color_cnt = 0
        for j in range(n):
            if col[j] == max_c:
                last_color_cnt += 1

        tie_breaker = 1e-4 * (float(last_color_cnt) / float(n if n > 0 else 1))
        total_loss += float(k) + tie_breaker

    return total_loss / float(num_graphs), total_colors / float(num_graphs)
