"""Diagnostics module for Phase 1 Oracle-Gap Gate.

Implements:
D1'. Graph ceiling: min(DSATUR, RLF, IG-2000-DS, IG-2000-RLF, TabuCol)
     With TabuCol iteratively descending K (while cur_target >= 1).
D4.  Richer non-LLM competitor: Parameterized RLF with CMA-ES per family,
     starting from exact RLF, val selection, and fallback to R(F).
D5.  Constructive ceiling: 200 randomized constructive variants per graph
     (randomized tie-breaking and class opening for DSATUR and RLF).
"""

from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import numba as nb
import cma

from gate.dataset import load_family, families
from evaluator.coloring import dsatur, rlf
from evaluator.verify import verify, count_colors
from gate.tune import compute_paired_bootstrap_ci


# ==============================================================================
# D1': ITERATED GREEDY (2000 ITERATIONS) + TABUCOL DESCENDING-K SOLVER
# ==============================================================================

@nb.njit
def _iterated_greedy_numba(
    adj: np.ndarray,
    init_colors: np.ndarray,
    max_iter: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    n = adj.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    best_colors = init_colors.copy()
    best_k = 0
    for i in range(n):
        if best_colors[i] + 1 > best_k:
            best_k = best_colors[i] + 1

    cur_colors = best_colors.copy()
    cur_k = best_k

    np.random.seed(seed)
    deg = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                deg[i] += 1
    max_k = int(np.max(deg)) + 2

    order = np.empty(n, dtype=np.int64)
    new_colors = np.full(n, -1, dtype=np.int64)

    for it in range(max_iter):
        perm = np.arange(cur_k)
        if it % 2 == 0:
            for i in range(cur_k - 1, 0, -1):
                j = np.random.randint(0, i + 1)
                tmp = perm[i]
                perm[i] = perm[j]
                perm[j] = tmp
        else:
            perm = perm[::-1]

        idx = 0
        for c in perm:
            for v in range(n):
                if cur_colors[v] == c:
                    order[idx] = v
                    idx += 1

        for i in range(n):
            new_colors[i] = -1

        for step in range(n):
            v = order[step]
            used = np.zeros(max_k, dtype=np.bool_)
            for u in range(n):
                if adj[v, u] and new_colors[u] >= 0:
                    used[new_colors[u]] = True
            c_val = 0
            while used[c_val]:
                c_val += 1
            new_colors[v] = c_val

        new_k = 0
        for i in range(n):
            if new_colors[i] + 1 > new_k:
                new_k = new_colors[i] + 1

        if new_k <= cur_k:
            cur_k = new_k
            cur_colors = new_colors.copy()
            if cur_k < best_k:
                best_k = cur_k
                best_colors = cur_colors.copy()

    return best_colors


@nb.njit
def _tabucol_check_k(
    adj: np.ndarray,
    init_colors: np.ndarray,
    K: int,
    max_iter: int = 2500,
    seed: int = 42,
) -> Tuple[np.ndarray, bool]:
    n = adj.shape[0]
    np.random.seed(seed)
    colors = np.empty(n, dtype=np.int64)
    conflicts = np.zeros((n, K), dtype=np.int64)

    for v in range(n):
        if init_colors[v] < K:
            colors[v] = init_colors[v]
        else:
            colors[v] = np.random.randint(0, K)

    for v in range(n):
        c = colors[v]
        for u in range(n):
            if adj[v, u]:
                conflicts[u, c] += 1

    total_conflicts = 0
    for v in range(n):
        total_conflicts += conflicts[v, colors[v]]
    total_conflicts //= 2

    if total_conflicts == 0:
        return colors, True

    tabu = np.zeros((n, K), dtype=np.int64)
    best_conflicts = total_conflicts

    for it in range(1, max_iter + 1):
        best_delta = 1000000
        best_v = -1
        best_c = -1

        for v in range(n):
            cur_c = colors[v]
            cur_conf = conflicts[v, cur_c]
            if cur_conf == 0:
                continue
            for c in range(K):
                if c == cur_c:
                    continue
                delta = conflicts[v, c] - cur_conf
                is_tabu = (tabu[v, c] >= it)
                aspiration = (total_conflicts + delta < best_conflicts)
                if aspiration or not is_tabu:
                    if delta < best_delta:
                        best_delta = delta
                        best_v = v
                        best_c = c

        if best_v == -1:
            break

        old_c = colors[best_v]
        new_c = best_c
        colors[best_v] = new_c
        for u in range(n):
            if adj[best_v, u]:
                conflicts[u, old_c] -= 1
                conflicts[u, new_c] += 1
        total_conflicts += best_delta
        if total_conflicts < best_conflicts:
            best_conflicts = total_conflicts

        if total_conflicts == 0:
            return colors, True

        tenure = np.random.randint(5, 15)
        tabu[best_v, old_c] = it + tenure

    return colors, (total_conflicts == 0)


def compute_graph_ceiling(adj: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, int]:
    """Compute strong near-optimal graph ceiling via min(DSATUR, RLF, IG-2000-DS, IG-2000-RLF, TabuCol).

    Note: TabuCol iteratively attempts cur_target = best_k - 1 and keeps decrementing
    K while it finds conflict-free verified colorings.
    """
    c_ds = dsatur(adj)
    c_rlf = rlf(adj)

    k_ds = count_colors(c_ds)
    k_rlf = count_colors(c_rlf)

    # 1. Run IG 2000 from DSATUR
    c_ig_ds = _iterated_greedy_numba(adj, c_ds, max_iter=2000, seed=seed)
    k_ig_ds = count_colors(c_ig_ds)

    # 2. Run IG 2000 from RLF
    c_ig_rlf = _iterated_greedy_numba(adj, c_rlf, max_iter=2000, seed=seed + 100)
    k_ig_rlf = count_colors(c_ig_rlf)

    # Best so far
    candidates = [(k_ds, c_ds), (k_rlf, c_rlf), (k_ig_ds, c_ig_ds), (k_ig_rlf, c_ig_rlf)]
    best_k, best_colors = min(candidates, key=lambda x: x[0])

    # 3. Try TabuCol with iterative K descent
    cur_target = best_k - 1
    tabu_seed = seed + 200
    while cur_target >= 1:
        tabu_sol, success = _tabucol_check_k(adj, best_colors, cur_target, max_iter=2500, seed=tabu_seed)
        if success and verify(adj, tabu_sol):
            best_colors = tabu_sol
            best_k = cur_target
            cur_target -= 1
            tabu_seed += 1
        else:
            break

    return best_colors, best_k


def run_d1_val_worker(args: Tuple[str, str]) -> Dict[str, Any]:
    """Worker to compute D1' graph ceiling on 20 VAL graphs per family."""
    family_id, r_ref_name = args

    val_graphs = load_family(family_id, "val")

    ds_counts = []
    rlf_counts = []
    r_counts = []
    ceiling_counts = []

    for seed, adj in val_graphs:
        c_ds = count_colors(dsatur(adj))
        c_rlf = count_colors(rlf(adj))

        if r_ref_name == "rlf":
            c_ref = c_rlf
        elif r_ref_name == "dsatur":
            c_ref = c_ds
        else:
            c_ref = min(c_ds, c_rlf)

        _, k_ceil = compute_graph_ceiling(adj, seed=seed)

        ds_counts.append(c_ds)
        rlf_counts.append(c_rlf)
        r_counts.append(c_ref)
        ceiling_counts.append(k_ceil)

    ds_arr = np.array(ds_counts, dtype=np.float64)
    rlf_arr = np.array(rlf_counts, dtype=np.float64)
    r_arr = np.array(r_counts, dtype=np.float64)
    ceil_arr = np.array(ceiling_counts, dtype=np.float64)

    # Headroom of ceiling vs R(F)
    headroom_colors, hr_c_lo, hr_c_hi, headroom_pct, hr_p_lo, hr_p_hi = compute_paired_bootstrap_ci(
        r_arr, ceil_arr
    )

    is_control = ("interval" in family_id)

    return {
        "family_id": family_id,
        "is_control": is_control,
        "dsatur_val": float(np.mean(ds_arr)),
        "rlf_val": float(np.mean(rlf_arr)),
        "r_ref_val": float(np.mean(r_arr)),
        "ceiling_val": float(np.mean(ceil_arr)),
        "headroom_vs_r_colors": headroom_colors,
        "headroom_vs_r_pct": headroom_pct,
        "headroom_ci_colors": (hr_c_lo, hr_c_hi),
        "headroom_ci_pct": (hr_p_lo, hr_p_hi),
    }


# ==============================================================================
# D4: PARAMETERIZED RLF WITH NUMBA KERNEL & CMA-ES TUNING
# ==============================================================================

@nb.njit
def _parametric_rlf_numba(adj: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Parameterized RLF kernel.

    weights: (6,) float array:
    w[0]: candidate ne weight (scaled by n+1)
    w[1]: candidate nc weight
    w[2]: candidate udeg weight
    w[3]: candidate deg weight
    w[4]: opening udeg weight
    w[5]: opening deg weight

    When weights = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0], strictly matches standard RLF.
    """
    n = adj.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    w1, w2, w3, w4, w5, w6 = weights[0], weights[1], weights[2], weights[3], weights[4], weights[5]

    deg = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                deg[i] += 1

    colors = np.full(n, -1, dtype=np.int64)
    uncolored = np.ones(n, dtype=np.bool_)
    c = 0

    has_uncolored = True
    while has_uncolored:
        best_v = -1
        best_open_score = -1e18
        for i in range(n):
            if uncolored[i]:
                du = 0
                for j in range(n):
                    if uncolored[j] and adj[i, j]:
                        du += 1
                open_score = float(w5 * du + w6 * deg[i])
                if open_score > best_open_score:
                    best_open_score = open_score
                    best_v = i

        v = best_v
        colors[v] = c
        uncolored[v] = False

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
                    du = 0
                    for j in range(n):
                        if adj[i, j]:
                            if excl[j]:
                                ne += 1
                            elif cand[j]:
                                nc += 1
                            if uncolored[j]:
                                du += 1
                    score = float(w1 * (n + 1) * ne - w2 * nc + w3 * du + w4 * deg[i])
                    if score > best_score:
                        best_score = score
                        best_u = i

            u = best_u
            colors[u] = c
            uncolored[u] = False
            cand[u] = False

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


RLF_DEFAULT_WEIGHTS = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)


def _eval_parametric_rlf(adjs: List[np.ndarray], weights: np.ndarray) -> float:
    total = 0.0
    for a in adjs:
        col = _parametric_rlf_numba(a, weights)
        k = int(np.max(col)) + 1
        # Tiny tie breaker: sum of color indices normalized
        total += float(k) + 1e-4 * float(np.sum(col)) / (len(col) * max(k, 1))
    return total / len(adjs)


def run_d4_worker(args: Tuple[str, str, int, Tuple[int, ...]]) -> Dict[str, Any]:
    """Worker to tune Parameterized RLF with CMA-ES per family, evaluate on VAL, fallback to R(F).

    args: (family_id, r_ref_name, budget_per_seed, seeds)
    """
    family_id, r_ref_name, budget, seeds = args

    train_graphs = load_family(family_id, "baseline_train")
    val_graphs = load_family(family_id, "val")

    train_adjs = [g[1] for g in train_graphs]
    val_adjs = [g[1] for g in val_graphs]

    # Calculate baseline R(F) performance on val
    ds_val_counts = np.array([count_colors(dsatur(a)) for a in val_adjs], dtype=np.float64)
    rlf_val_counts = np.array([count_colors(rlf(a)) for a in val_adjs], dtype=np.float64)

    if r_ref_name == "rlf":
        r_ref_counts = rlf_val_counts
    elif r_ref_name == "dsatur":
        r_ref_counts = ds_val_counts
    else:
        r_ref_counts = np.minimum(ds_val_counts, rlf_val_counts)
    r_ref_val_mean = float(np.mean(r_ref_counts))

    best_overall_val = float("inf")
    best_overall_w = RLF_DEFAULT_WEIGHTS.copy()
    best_overall_counts = None

    for s in seeds:
        es = cma.CMAEvolutionStrategy(
            RLF_DEFAULT_WEIGHTS.tolist(),
            0.3,
            {"seed": s, "maxfevals": budget, "verbose": -9},
        )
        while not es.stop():
            solutions = es.ask()
            fitnesses = [_eval_parametric_rlf(train_adjs, np.array(w, dtype=np.float64)) for w in solutions]
            es.tell(solutions, fitnesses)

        cand_w = np.array(es.result.xbest, dtype=np.float64)

        # Evaluate on val
        val_counts_s = np.array([count_colors(_parametric_rlf_numba(a, cand_w)) for a in val_adjs], dtype=np.float64)
        val_mean_s = float(np.mean(val_counts_s))

        if val_mean_s < best_overall_val:
            best_overall_val = val_mean_s
            best_overall_w = cand_w.copy()
            best_overall_counts = val_counts_s

    # Fallback to R(F) if tuned does not strictly improve on VAL
    fallback = False
    if best_overall_val >= r_ref_val_mean:
        fallback = True
        best_overall_val = r_ref_val_mean
        best_overall_counts = r_ref_counts.copy()

    # Gap vs R(F) on VAL
    gap_colors, g_c_lo, g_c_hi, gap_pct, g_p_lo, g_p_hi = compute_paired_bootstrap_ci(
        r_ref_counts, best_overall_counts
    )

    is_control = ("interval" in family_id)

    return {
        "family_id": family_id,
        "is_control": is_control,
        "r_ref_name": r_ref_name,
        "r_ref_val": r_ref_val_mean,
        "tuned_rlf_val": best_overall_val,
        "fallback": fallback,
        "weights": best_overall_w.tolist(),
        "gap_vs_r_colors": gap_colors,
        "gap_vs_r_pct": gap_pct,
        "gap_ci_colors": (g_c_lo, g_c_hi),
        "gap_ci_pct": (g_p_lo, g_p_hi),
    }


# ==============================================================================
# D5: CONSTRUCTIVE CEILING (200 RANDOMIZED CONSTRUCTIVE RUNS)
# ==============================================================================

@nb.njit
def _rand_dsatur(adj: np.ndarray, seed: int) -> int:
    np.random.seed(seed)
    n = adj.shape[0]
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
                score = float(sat[i] * (n + 1) + uncolored_deg[i]) + np.random.uniform(0.0, 0.99)
                if score > best_score:
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
    k = 0
    for i in range(n):
        if colors[i] + 1 > k:
            k = colors[i] + 1
    return k


@nb.njit
def _rand_rlf(adj: np.ndarray, seed: int, top_k_open: int = 1) -> int:
    np.random.seed(seed)
    n = adj.shape[0]
    colors = np.full(n, -1, dtype=np.int64)
    uncolored = np.ones(n, dtype=np.bool_)
    c = 0

    has_uncolored = True
    while has_uncolored:
        best_v = -1
        best_open_score = -1e18
        for i in range(n):
            if uncolored[i]:
                du = 0
                for j in range(n):
                    if uncolored[j] and adj[i, j]:
                        du += 1
                noise = np.random.uniform(0.0, 0.99) if top_k_open == 1 else np.random.uniform(0.0, float(top_k_open))
                open_score = float(du) + noise
                if open_score > best_open_score:
                    best_open_score = open_score
                    best_v = i

        v = best_v
        colors[v] = c
        uncolored[v] = False

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
                    score = float(ne * (n + 1) - nc) + np.random.uniform(0.0, 0.99)
                    if score > best_score:
                        best_score = score
                        best_u = i

            u = best_u
            colors[u] = c
            uncolored[u] = False
            cand[u] = False

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

    k = 0
    for i in range(n):
        if colors[i] + 1 > k:
            k = colors[i] + 1
    return k


def compute_constructive_ceiling(adj: np.ndarray, base_seed: int = 42) -> int:
    """Run 200 randomized constructive variants of DSATUR and RLF and return minimum colors."""
    best_k = 1000000

    # 100 randomized DSATUR
    for t in range(100):
        k = _rand_dsatur(adj, base_seed + t)
        if k < best_k:
            best_k = k

    # 50 randomized RLF (random tie-breaking on opening and candidates)
    for t in range(50):
        k = _rand_rlf(adj, base_seed + 1000 + t, top_k_open=1)
        if k < best_k:
            best_k = k

    # 50 randomized RLF with randomized opening among top uncolored degree vertices
    for t in range(50):
        k = _rand_rlf(adj, base_seed + 2000 + t, top_k_open=3)
        if k < best_k:
            best_k = k

    return best_k


def run_d5_worker(args: Tuple[str, str]) -> Dict[str, Any]:
    """Worker to compute constructive ceiling (200 trials) on 20 VAL graphs per family."""
    family_id, r_ref_name = args

    val_graphs = load_family(family_id, "val")

    ds_counts = []
    rlf_counts = []
    r_counts = []
    constr_counts = []

    for seed, adj in val_graphs:
        c_ds = count_colors(dsatur(adj))
        c_rlf = count_colors(rlf(adj))

        if r_ref_name == "rlf":
            c_ref = c_rlf
        elif r_ref_name == "dsatur":
            c_ref = c_ds
        else:
            c_ref = min(c_ds, c_rlf)

        k_c = compute_constructive_ceiling(adj, base_seed=seed)

        ds_counts.append(c_ds)
        rlf_counts.append(c_rlf)
        r_counts.append(c_ref)
        constr_counts.append(k_c)

    ds_arr = np.array(ds_counts, dtype=np.float64)
    rlf_arr = np.array(rlf_counts, dtype=np.float64)
    r_arr = np.array(r_counts, dtype=np.float64)
    c_arr = np.array(constr_counts, dtype=np.float64)

    # Headroom of constructive ceiling vs R(F)
    headroom_colors, hr_c_lo, hr_c_hi, headroom_pct, hr_p_lo, hr_p_hi = compute_paired_bootstrap_ci(
        r_arr, c_arr
    )

    is_control = ("interval" in family_id)

    return {
        "family_id": family_id,
        "is_control": is_control,
        "dsatur_val": float(np.mean(ds_arr)),
        "rlf_val": float(np.mean(rlf_arr)),
        "r_ref_val": float(np.mean(r_arr)),
        "constructive_val": float(np.mean(c_arr)),
        "headroom_vs_r_colors": headroom_colors,
        "headroom_vs_r_pct": headroom_pct,
        "headroom_ci_colors": (hr_c_lo, hr_c_hi),
        "headroom_ci_pct": (hr_p_lo, hr_p_hi),
    }
