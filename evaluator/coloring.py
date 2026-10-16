"""Graph coloring algorithms and greedy coloring framework.

Implements:
1. Priority function contract:
   priority(adj, colors, sat, deg, uncolored_deg) -> np.ndarray
2. greedy_color(adj, priority_fn) -> colors
3. dsatur(adj) -> standalone fast Numba DSATUR
4. rlf(adj) -> Recursive Largest First reference baseline
"""

from typing import Callable
import numpy as np
import numba as nb


def dsatur_priority(
    adj: np.ndarray,
    colors: np.ndarray,
    sat: np.ndarray,
    deg: np.ndarray,
    uncolored_deg: np.ndarray,
) -> np.ndarray:
    """Standard DSATUR priority formula.

    Score = sat * (n + 1) + uncolored_deg.
    Pure function: inputs are read-only.
    """
    n = adj.shape[0]
    return sat.astype(np.float64) * (n + 1) + uncolored_deg.astype(np.float64)


def rlf_priority(
    adj: np.ndarray,
    colors: np.ndarray,
    sat: np.ndarray,
    deg: np.ndarray,
    uncolored_deg: np.ndarray,
) -> np.ndarray:
    """RLF (Recursive Largest First) priority function matching the API contract.

    Mechanism:
    - Current color class k = max(colors) (or 0 if no vertex is colored).
    - Candidate set 'cand': uncolored vertices non-adjacent to any member of class k.
    - Excluded set 'excl': uncolored vertices adjacent to at least one member of class k.
    - If cand is not empty: score = ne * (n + 1) - nc, where ne is neighbors in excl
      and nc is neighbors in cand. Higher score = color earlier.
    - If cand is empty (class k complete): open class k + 1 by selecting uncolored
      vertex with maximum uncolored degree (udeg).
    - Ties broken by lowest index (handled by greedy framework).
    """
    n = adj.shape[0]
    scores = np.full(n, -1e18, dtype=np.float64)
    unc = (colors == -1)
    if not np.any(unc):
        return scores

    k = int(np.max(colors)) if np.any(colors >= 0) else -1
    if k == -1:
        scores[unc] = uncolored_deg[unc].astype(np.float64)
        return scores

    class_k = (colors == k)
    adj_to_k = adj[class_k].any(axis=0) if np.any(class_k) else np.zeros(n, dtype=bool)
    cand = unc & ~adj_to_k
    excl = unc & adj_to_k

    if np.any(cand):
        ne = adj.dot(excl.astype(np.int64))
        nc = adj.dot(cand.astype(np.int64))
        scores[cand] = (ne[cand] * (n + 1) - nc[cand]).astype(np.float64)
    else:
        scores[unc] = uncolored_deg[unc].astype(np.float64)

    return scores


def greedy_color(
    adj: np.ndarray,
    priority_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Greedy coloring framework with arbitrary priority function.

    Parameters:
    adj: (n, n) bool, symmetric, zero diagonal.
    priority_fn: Pure function returning (n,) float scores.

    Returns:
    colors: (n,) int, color assignment for each vertex.
    """
    n = adj.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    deg = adj.sum(axis=1).astype(np.int64)
    max_k = int(deg.max()) + 2
    colors = np.full(n, -1, dtype=np.int64)
    sat = np.zeros(n, dtype=np.int64)
    uncolored_deg = deg.copy()
    nb_has_color = np.zeros((n, max_k), dtype=bool)
    uncolored = np.ones(n, dtype=bool)

    # Make read-only views to enforce purity
    adj_ro = adj.view()
    adj_ro.flags.writeable = False

    for _ in range(n):
        colors_ro = colors.view()
        colors_ro.flags.writeable = False
        sat_ro = sat.view()
        sat_ro.flags.writeable = False
        deg_ro = deg.view()
        deg_ro.flags.writeable = False
        uncolored_deg_ro = uncolored_deg.view()
        uncolored_deg_ro.flags.writeable = False

        scores = priority_fn(adj_ro, colors_ro, sat_ro, deg_ro, uncolored_deg_ro)

        # Mask out already-colored vertices
        masked_scores = np.where(uncolored, scores, -np.inf)
        max_score = np.max(masked_scores)
        # Ties broken by lowest index
        candidates = np.flatnonzero(masked_scores == max_score)
        v = int(candidates[0])

        # Assign lowest feasible color >= 0
        c = int(np.argmin(nb_has_color[v]))
        colors[v] = c
        uncolored[v] = False

        # Incremental update of sat and uncolored_deg for neighbors
        neighbors = np.flatnonzero(adj[v])
        new_color_neighbors = neighbors[~nb_has_color[neighbors, c]]
        sat[new_color_neighbors] += 1
        nb_has_color[neighbors, c] = True
        uncolored_deg[neighbors] -= 1

    return colors


@nb.njit
def _dsatur_numba(adj: np.ndarray) -> np.ndarray:
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

    for _ in range(n):
        best_v = -1
        best_score = -1.0
        for i in range(n):
            if not colored[i]:
                score = float(sat[i] * (n + 1) + uncolored_deg[i])
                if best_v == -1 or score > best_score:
                    best_score = score
                    best_v = i

        v = best_v
        # Lowest available color
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


def dsatur(adj: np.ndarray) -> np.ndarray:
    """Fast Numba implementation of DSATUR.

    Guaranteed to produce identical colorings and tie-breaking as
    greedy_color(adj, dsatur_priority).
    """
    return _dsatur_numba(adj)


@nb.njit
def _rlf_numba(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    colors = np.full(n, -1, dtype=np.int64)
    uncolored = np.ones(n, dtype=np.bool_)
    c = 0

    has_uncolored = True
    while has_uncolored:
        # du = degree in uncolored subgraph
        best_v = -1
        max_du = -1
        for i in range(n):
            if uncolored[i]:
                du = 0
                for j in range(n):
                    if uncolored[j] and adj[i, j]:
                        du += 1
                if du > max_du:
                    max_du = du
                    best_v = i

        v = best_v
        colors[v] = c
        uncolored[v] = False

        # cand: uncolored vertices not adjacent to v
        # excl: uncolored vertices adjacent to v
        cand = np.zeros(n, dtype=np.bool_)
        excl = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if uncolored[i]:
                if adj[v, i]:
                    excl[i] = True
                else:
                    cand[i] = True

        has_cand = False
        for i in range(n):
            if cand[i]:
                has_cand = True
                break

        while has_cand:
            best_u = -1
            best_score = -1e18
            for i in range(n):
                if cand[i]:
                    ne = 0
                    nc = 0
                    for j in range(n):
                        if adj[i, j]:
                            if excl[j]:
                                ne += 1
                            elif cand[j]:
                                nc += 1
                    score = float(ne * (n + 1) - nc)
                    if score > best_score:
                        best_score = score
                        best_u = i

            u = best_u
            colors[u] = c
            uncolored[u] = False
            cand[u] = False

            # Update excl and cand
            for j in range(n):
                if cand[j] and adj[u, j]:
                    cand[j] = False
                    excl[j] = True

            has_cand = False
            for i in range(n):
                if cand[i]:
                    has_cand = True
                    break

        c += 1
        has_uncolored = False
        for i in range(n):
            if uncolored[i]:
                has_uncolored = True
                break

    return colors


def rlf(adj: np.ndarray) -> np.ndarray:
    """Recursive Largest First (RLF) graph coloring heuristic."""
    return _rlf_numba(adj)
