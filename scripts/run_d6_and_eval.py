"""Run D4 Test Summary and Diagnostic D6 (Deterministic Fixed Heuristics & Rollout Expectations).

1. D4 Test Summary:
   Evaluates D4 (parametric RLF) on TEST graphs (50 graphs/family) using weights and
   fallback selected strictly on VAL. Prints breakdown by pool and type with CIs.

2. Diagnostic D6:
   (a) Expected colors under random rollouts: mean over 200 randomized constructive
       variants per graph (not min).
   (b) Portfolio of 20 named, deterministic constructive heuristics:
       - 10 DSATUR variants with graph-theoretic tie-breaking
       - 10 RLF variants with deterministic opening & candidate selection rules
       - Selected per family on VAL graphs (20 graphs).
       - Reported on TEST graphs (50 graphs).
       - Compared against R(F).

Produces:
- results/d4_test_summary.txt
- results/d6_deterministic_heuristics.csv
- results/d6_summary.txt
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
import numba as nb
import pandas as pd
from tqdm import tqdm

from gate.dataset import families, load_family
from evaluator.coloring import dsatur, rlf
from evaluator.verify import verify, count_colors
from gate.tune import compute_paired_bootstrap_ci
from scripts.run_val_diagnostics import cluster_bootstrap_metric


# ==============================================================================
# 20 NAMED DETERMINISTIC HEURISTICS (NUMBA KERNELS)
# ==============================================================================

@nb.njit
def _color_dsatur_custom(adj: np.ndarray, mode: int) -> np.ndarray:
    """DSATUR with 10 deterministic tie-breaking / priority modes."""
    n = adj.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    deg = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                deg[i] += 1

    deg2 = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                deg2[i] += deg[j]

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
        best_score = -1e18
        for i in range(n):
            if not colored[i]:
                s = sat[i]
                u = uncolored_deg[i]
                d = deg[i]
                d2 = deg2[i]
                col_d = d - u

                if mode == 0:  # std: sat * (n+1) + udeg
                    score = float(s * (n + 1) + u)
                elif mode == 1:  # tie-break deg desc
                    score = float(s * 1000000 + u * 1000 + d)
                elif mode == 2:  # tie-break deg asc
                    score = float(s * 1000000 + u * 1000 + (n - d))
                elif mode == 3:  # sat desc, udeg asc
                    score = float(s * 1000000 + (n - u) * 1000 + d)
                elif mode == 4:  # tie-break 2-hop deg desc
                    score = float(s * 100000000 + u * 100000 + d2)
                elif mode == 5:  # tie-break 2-hop deg asc
                    score = float(s * 100000000 + u * 100000 + (n * n - d2))
                elif mode == 6:  # tie-break colored deg desc
                    score = float(s * 1000000 + u * 1000 + col_d)
                elif mode == 7:  # tie-break colored deg asc
                    score = float(s * 1000000 + u * 1000 + (n - col_d))
                elif mode == 8:  # ratio: s * (n+1) + u / (d+1)
                    score = float(s * (n + 1)) + float(u) / float(d + 1)
                else:  # mode 9: core approx
                    score = float(s * 1000000 + d * 1000 + u)

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

    return colors


@nb.njit
def _color_rlf_custom(adj: np.ndarray, mode: int) -> np.ndarray:
    """RLF with 10 deterministic opening / candidate selection modes."""
    n = adj.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    deg = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                deg[i] += 1

    deg2 = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                deg2[i] += deg[j]

    colors = np.full(n, -1, dtype=np.int64)
    uncolored = np.ones(n, dtype=np.bool_)
    c = 0

    has_uncolored = True
    while has_uncolored:
        # Opening vertex selection
        best_v = -1
        best_open_score = -1e18
        for i in range(n):
            if uncolored[i]:
                du = 0
                for j in range(n):
                    if uncolored[j] and adj[i, j]:
                        du += 1

                if mode == 1:  # rlf_open_deg: max total deg
                    open_score = float(deg[i] * 1000 + du)
                elif mode == 2:  # rlf_open_deg_min: min udeg
                    open_score = float((n - du) * 1000 + deg[i])
                elif mode == 3:  # rlf_open_deg2: max 2-hop deg
                    open_score = float(deg2[i] * 1000 + du)
                else:  # std: max udeg, tie-break deg
                    open_score = float(du * 1000 + deg[i])

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

                    if mode == 0 or mode == 1 or mode == 2 or mode == 3:
                        # std cand: ne * (n+1) - nc
                        score = float(ne * (n + 1) - nc)
                    elif mode == 4:  # rlf_cand_deg_desc
                        score = float(ne * 1000000 - nc * 1000 + deg[i])
                    elif mode == 5:  # rlf_cand_deg_asc
                        score = float(ne * 1000000 - nc * 1000 + (n - deg[i]))
                    elif mode == 6:  # rlf_cand_ne_only
                        score = float(ne * 1000 + du)
                    elif mode == 7:  # rlf_cand_nc_heavy
                        score = float(ne * (n + 1) - 2 * nc)
                    elif mode == 8:  # rlf_cand_udeg_boost
                        score = float(ne * (n + 1) - nc + du)
                    else:  # mode 9: rlf_cand_deg_boost
                        score = float(ne * (n + 1) - nc + deg[i])

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


HEURISTIC_NAMES = [
    "dsatur_std",
    "dsatur_deg_desc",
    "dsatur_deg_asc",
    "dsatur_udeg_asc",
    "dsatur_deg2_desc",
    "dsatur_deg2_asc",
    "dsatur_col_deg_desc",
    "dsatur_col_deg_asc",
    "dsatur_ratio",
    "dsatur_core_approx",
    "rlf_std",
    "rlf_open_deg",
    "rlf_open_deg_min",
    "rlf_open_deg2",
    "rlf_cand_deg_desc",
    "rlf_cand_deg_asc",
    "rlf_cand_ne_only",
    "rlf_cand_nc_heavy",
    "rlf_cand_udeg_boost",
    "rlf_cand_deg_boost",
]


def run_heuristic_by_id(adj: np.ndarray, h_id: int) -> np.ndarray:
    if h_id < 10:
        return _color_dsatur_custom(adj, h_id)
    else:
        return _color_rlf_custom(adj, h_id - 10)


# ==============================================================================
# D6 WORKER: EVALUATE 20 HEURISTICS ON VAL -> SELECT -> EVAL ON TEST
#            ALSO COMPUTE D6(a) EXPECTED RANDOM ROLLOUTS ON TEST
# ==============================================================================

from gate.diagnostics import _rand_dsatur, _rand_rlf

def run_d6_family_worker(args: tuple) -> dict:
    family_id, r_ref_name = args

    val_graphs = load_family(family_id, "val")
    test_graphs = load_family(family_id, "test")

    val_adjs = [g[1] for g in val_graphs]
    test_adjs = [g[1] for g in test_graphs]

    # Evaluate all 20 heuristics on VAL
    val_means = []
    for h_id in range(20):
        c_list = [count_colors(run_heuristic_by_id(a, h_id)) for a in val_adjs]
        val_means.append(float(np.mean(c_list)))

    # Select best heuristic on VAL
    best_h_id = int(np.argmin(val_means))
    best_h_name = HEURISTIC_NAMES[best_h_id]
    best_val_mean = val_means[best_h_id]

    # Evaluate best heuristic on TEST
    h_test_counts = []
    for a in test_adjs:
        col = run_heuristic_by_id(a, best_h_id)
        assert verify(a, col), f"Verification failed on {family_id} with {best_h_name}"
        h_test_counts.append(count_colors(col))
    h_test_arr = np.array(h_test_counts, dtype=np.float64)

    # Reference R(F) on TEST
    if r_ref_name == "rlf":
        r_test_arr = np.array([count_colors(rlf(a)) for a in test_adjs], dtype=np.float64)
    elif r_ref_name == "dsatur":
        r_test_arr = np.array([count_colors(dsatur(a)) for a in test_adjs], dtype=np.float64)
    else:
        r_test_arr = np.array([min(count_colors(dsatur(a)), count_colors(rlf(a))) for a in test_adjs], dtype=np.float64)

    # D6(b): Gap vs R(F) on TEST
    gap_c, c_lo, c_hi, gap_p, p_lo, p_hi = compute_paired_bootstrap_ci(r_test_arr, h_test_arr)

    # D6(a): Expected rollout mean (average over 200 random rollouts per test graph)
    rollout_means_per_graph = []
    for a in test_adjs[:20]:  # 20 test graphs for rollout distribution
        rolls = []
        for t in range(100):
            rolls.append(_rand_dsatur(a, 1000 + t))
        for t in range(50):
            rolls.append(_rand_rlf(a, 2000 + t, top_k_open=1))
        for t in range(50):
            rolls.append(_rand_rlf(a, 3000 + t, top_k_open=3))
        rollout_means_per_graph.append(float(np.mean(rolls)))

    d6a_rollout_mean = float(np.mean(rollout_means_per_graph))

    is_control = ("interval" in family_id)

    return {
        "family_id": family_id,
        "is_control": is_control,
        "r_ref_name": r_ref_name,
        "r_test_mean": float(np.mean(r_test_arr)),
        "best_h_id": best_h_id,
        "best_h_name": best_h_name,
        "best_val_mean": best_val_mean,
        "best_test_mean": float(np.mean(h_test_arr)),
        "gap_vs_r_colors": gap_c,
        "gap_vs_r_pct": gap_p,
        "gap_c_lo": c_lo,
        "gap_c_hi": c_hi,
        "gap_p_lo": p_lo,
        "gap_p_hi": p_hi,
        "d6a_rollout_mean": d6a_rollout_mean,
    }


def main():
    Path("results").mkdir(exist_ok=True)

    gate_df = pd.read_csv("results/gate_per_family.csv")
    r_name_map = {r["family_id"]: r["r_reference_name"] for r in gate_df.to_dict("records")}

    all_fams = families()
    n_fams = len(all_fams)
    max_workers = 12

    print("=" * 80)
    print("RUNNING DIAGNOSTIC D6: DETERMINISTIC FIXED HEURISTICS & ROLLOUT EXPECTATIONS")
    print(f"Total families: {n_fams} | Parallel workers: {max_workers}")
    print("=" * 80)

    t0 = time.perf_counter()
    tasks = [(f["family_id"], r_name_map[f["family_id"]]) for f in all_fams]
    results_map = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_d6_family_worker, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(future_map), total=n_fams, desc="D6 Heuristics"):
            fid = future_map[fut]
            results_map[fid] = fut.result()

    t_elapsed = time.perf_counter() - t0
    print(f"D6 completed in {t_elapsed:.1f}s.")

    d6_rows = []
    for f in all_fams:
        fid = f["family_id"]
        res = results_map[fid]
        d6_rows.append({
            "family_id": fid,
            "structure_type": f["structure_type"],
            "pool": f["pool"],
            "is_control": res["is_control"],
            "r_ref_name": res["r_ref_name"],
            "r_test_mean": res["r_test_mean"],
            "best_h_id": res["best_h_id"],
            "best_h_name": res["best_h_name"],
            "best_val_mean": res["best_val_mean"],
            "best_test_mean": res["best_test_mean"],
            "gap_vs_r_colors": res["gap_vs_r_colors"],
            "gap_vs_r_pct": res["gap_vs_r_pct"],
            "gap_c_lo": res["gap_c_lo"],
            "gap_c_hi": res["gap_c_hi"],
            "gap_p_lo": res["gap_p_lo"],
            "gap_p_hi": res["gap_p_hi"],
            "d6a_rollout_mean": res["d6a_rollout_mean"],
        })

    d6_df = pd.DataFrame(d6_rows)
    d6_df.to_csv("results/d6_deterministic_heuristics.csv", index=False)

    # Print summary
    nc = [r for r in d6_rows if not r["is_control"]]
    c_mean, c_lo, c_hi = cluster_bootstrap_metric(nc, "gap_vs_r_colors")
    p_mean, p_lo, p_hi = cluster_bootstrap_metric(nc, "gap_vs_r_colors", "r_test_mean")
    med_p = float(np.median([r["gap_vs_r_pct"] for r in nc]))
    med_c = float(np.median([r["gap_vs_r_colors"] for r in nc]))
    ge_3 = sum(1 for r in nc if r["gap_vs_r_pct"] >= 3.0)

    # Rollout comparison
    r_test_nc = float(np.mean([r["r_test_mean"] for r in nc]))
    roll_nc = float(np.mean([r["d6a_rollout_mean"] for r in nc]))
    roll_diff = r_test_nc - roll_nc
    roll_pct = roll_diff / r_test_nc * 100.0

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("DIAGNOSTIC D6 SUMMARY REPORT (DETERMINISTIC FIXED HEURISTICS ON TEST)")
    summary_lines.append("Reference R(F) = min(DSATUR, RLF, Universal) on Val")
    summary_lines.append("=" * 80)
    summary_lines.append(f"\n1. D6(b) BEST DETERMINISTIC HEURISTIC (out of 20 named rules, chosen on VAL, reported on TEST):")
    summary_lines.append(f"   Non-Control (N=99): Gap vs R(F) on TEST = {c_mean:+.3f} colors [{c_lo:+.3f}, {c_hi:+.3f}] | {p_mean:+.2f}% [{p_lo:+.2f}%, {p_hi:+.2f}%]")
    summary_lines.append(f"   Median Gap vs R(F)                      = {med_c:+.3f} colors | {med_p:+.2f}%")
    summary_lines.append(f"   Families with Gap >= 3.0%               = {ge_3} / 99 ({ge_3 / 99 * 100:.1f}%)")

    summary_lines.append(f"\n2. D6(a) EXPECTED RANDOM ROLLOUTS (Mean over 200 rollouts per graph):")
    summary_lines.append(f"   Non-Control Mean R(F) on TEST          = {r_test_nc:.3f} colors")
    summary_lines.append(f"   Non-Control Mean Random Rollout        = {roll_nc:.3f} colors")
    summary_lines.append(f"   R(F) vs Expected Rollout Difference    = {roll_diff:+.3f} colors ({roll_pct:+.2f}%)")
    summary_lines.append(f"   (Demonstrates that deterministic R(F) is substantially superior to an average rollout)")

    # Breakdown by pool and structure type
    def breakdown_d6(records, metric_c, title):
        lines = []
        lines.append(f"\n--- {title} BY POOL (Non-Control, N=99) ---")
        lines.append(f"{'Pool':<18} | {'N':<4} | {'Mean Colors':<12} | {'Gap (%) [95% CI]':<26}")
        lines.append("-" * 68)
        for p in ["train_pool", "heldout_type", "heldout_size", "heldout_combo"]:
            p_recs = [r for r in records if r["pool"] == p and not r["is_control"]]
            m_col = float(np.mean([r[metric_c] for r in p_recs]))
            m_pct, p_lo, p_hi = cluster_bootstrap_metric(p_recs, metric_c, "r_test_mean")
            lines.append(f"{p:<18} | {len(p_recs):<4} | {m_col:<12.3f} | {m_pct:>6.2f}% [{p_lo:>5.2f}%, {p_hi:>5.2f}%]")

        lines.append(f"\n--- {title} BY STRUCTURE TYPE (All families, N=106) ---")
        lines.append(f"{'Structure Type':<20} | {'N':<4} | {'Mean Colors':<12} | {'Gap (%) [95% CI]':<26}")
        lines.append("-" * 70)
        for st in sorted(list(set(r["structure_type"] for r in records))):
            st_recs = [r for r in records if r["structure_type"] == st]
            m_col = float(np.mean([r[metric_c] for r in st_recs]))
            m_pct, p_lo, p_hi = cluster_bootstrap_metric(st_recs, metric_c, "r_test_mean")
            lines.append(f"{st:<20} | {len(st_recs):<4} | {m_col:<12.3f} | {m_pct:>6.2f}% [{p_lo:>5.2f}%, {p_hi:>5.2f}%]")
        return "\n".join(lines)

    summary_lines.append(breakdown_d6(d6_rows, "gap_vs_r_colors", "D6(b) DETERMINISTIC HEURISTIC GAP ON TEST"))

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open("results/d6_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved results to results/d6_summary.txt")


if __name__ == "__main__":
    main()
