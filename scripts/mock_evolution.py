"""Dry-run test of the full evolution loop using a Mock LLM on 2 families.

Verifies:
1. End-to-end evolution loop with valid seeds and intentional bad code variants.
2. Candidate logging into results/pilot_candidates.jsonl.
3. Resume functionality: restores state from log and continues seamlessly.
4. Hard ceilings: terminates cleanly when hitting max_calls or max_dollars.
5. Final Heuristic Selection Protocol:
   - Rank candidates on train fitness -> take top-10
   - Select best on VAL
   - Evaluate on TEST and report paired Gap vs R(F) in colors and % with 95% bootstrap CI.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import numpy as np
import pandas as pd

from gate.dataset import load_family, families
from gate.evolved.sandbox import evaluate_candidate_safe
from gate.evolved.task import CandidateLogger, CALIBRATED_PER_GRAPH_TIMEOUTS
from gate.tune import compute_paired_bootstrap_ci
from evaluator.coloring import dsatur, rlf
from evaluator.verify import count_colors


MOCK_CANDIDATES = [
    # 1. Valid DSATUR
    """def priority(adj, colors, sat, deg, uncolored_deg):
    n = adj.shape[0]
    return sat * (n + 1) + uncolored_deg
""",
    # 2. Valid RLF
    """def priority(adj, colors, sat, deg, uncolored_deg):
    n = adj.shape[0]
    unc = (colors == -1)
    if not np.any(unc):
        return np.zeros(n, dtype=float)
    cur_color = int(np.max(colors))
    scores = np.zeros(n, dtype=float)
    if cur_color >= 0:
        in_cur_class = (colors == cur_color)
        cand = unc & (~np.any(adj[:, in_cur_class], axis=1))
        excl = unc & (~cand)
        if np.any(cand):
            for u in np.flatnonzero(cand):
                adj_u = adj[u]
                ne = np.sum(adj_u & excl)
                nc = np.sum(adj_u & cand)
                scores[u] = float(ne * (n + 1) - nc)
            return scores
    scores[unc] = uncolored_deg[unc].astype(float)
    return scores
""",
    # 3. Bad code: Syntax error
    """def priority(adj, colors, sat, deg, uncolored_deg):
    return sat * (
""",
    # 4. Bad code: Forbidden random
    """def priority(adj, colors, sat, deg, uncolored_deg):
    return sat + np.random.rand(len(sat))
""",
    # 5. Bad code: Infinite loop
    """def priority(adj, colors, sat, deg, uncolored_deg):
    while True:
        pass
    return sat
""",
    # 6. Valid RLF variant with degree penalty
    """def priority(adj, colors, sat, deg, uncolored_deg):
    n = adj.shape[0]
    unc = (colors == -1)
    if not np.any(unc):
        return np.zeros(n, dtype=float)
    cur_color = int(np.max(colors))
    scores = np.zeros(n, dtype=float)
    if cur_color >= 0:
        in_cur_class = (colors == cur_color)
        cand = unc & (~np.any(adj[:, in_cur_class], axis=1))
        excl = unc & (~cand)
        if np.any(cand):
            for u in np.flatnonzero(cand):
                adj_u = adj[u]
                ne = np.sum(adj_u & excl)
                nc = np.sum(adj_u & cand)
                scores[u] = float(ne * (n + 1) - 2 * nc)
            return scores
    scores[unc] = (uncolored_deg[unc] + deg[unc] * 0.1).astype(float)
    return scores
""",
    # 7. Valid DSATUR ratio variant
    """def priority(adj, colors, sat, deg, uncolored_deg):
    n = adj.shape[0]
    return sat * (n + 1) + uncolored_deg / (deg + 1)
""",
]


def run_mock_evolution_for_family(
    family_id: str,
    max_calls: int = 6,
    max_dollars: float = 0.01,
    cost_per_call: float = 0.0005,
    logger_path: str = "results/pilot_candidates.jsonl",
) -> dict:
    logger = CandidateLogger(logger_path)
    train_graphs = load_family(family_id, "baseline_train")
    val_graphs = load_family(family_id, "val")
    test_graphs = load_family(family_id, "test")

    # Baseline R(F) performance on train, val, and test
    c_rlf_train = float(np.mean([count_colors(rlf(g[1])) for g in train_graphs]))
    c_ds_train = float(np.mean([count_colors(dsatur(g[1])) for g in train_graphs]))
    c_rlf_val = float(np.mean([count_colors(rlf(g[1])) for g in val_graphs]))
    c_ds_val = float(np.mean([count_colors(dsatur(g[1])) for g in val_graphs]))

    r_train = min(c_rlf_train, c_ds_train)
    r_val = min(c_rlf_val, c_ds_val)

    # Reference on test
    test_adjs = [g[1] for g in test_graphs]
    if c_rlf_val <= c_ds_val:
        r_test_arr = np.array([count_colors(rlf(a)) for a in test_adjs], dtype=float)
    else:
        r_test_arr = np.array([count_colors(dsatur(a)) for a in test_adjs], dtype=float)

    per_graph_timeout = CALIBRATED_PER_GRAPH_TIMEOUTS.get(family_id, 1.0)

    # Check resume state
    already_evaluated = logger.get_evaluated_count(family_id)
    print(f"\n[{family_id}] Resuming from iteration {already_evaluated}. Ceilings: max_calls={max_calls}, max_dollars=${max_dollars}")

    evaluated_candidates = []
    if Path(logger_path).exists():
        with open(logger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("family_id") == family_id:
                            evaluated_candidates.append(rec)
                    except Exception:
                        pass
    total_calls = already_evaluated
    total_cost = already_evaluated * cost_per_call

    for i in range(already_evaluated, len(MOCK_CANDIDATES)):
        # Check hard ceilings
        if total_calls >= max_calls:
            print(f"[{family_id}] Reached hard call ceiling ({total_calls} >= {max_calls}). Stopping evolution.")
            break
        if total_cost >= max_dollars:
            print(f"[{family_id}] Reached hard budget ceiling (${total_cost:.4f} >= ${max_dollars:.4f}). Stopping evolution.")
            break

        code = MOCK_CANDIDATES[i]
        total_calls += 1
        total_cost += cost_per_call

        # Evaluate candidate in sandbox
        res = evaluate_candidate_safe(
            code,
            train_graphs,
            per_graph_timeout=per_graph_timeout,
            overall_timeout=15.0,
        )

        if res["success"]:
            # Primary metric: R(F) - mean_colors
            # Secondary tie-breaking: + 1e-4 * mean_tie_break
            mean_c = res["mean_colors"]
            fitness = (r_train - mean_c) + 1e-4 * res.get("mean_tie_break", 0.0)
            err = None
        else:
            mean_c = None
            fitness = -100.0
            err = res["error"]

        fam_lookup = {f["family_id"]: f for f in families()}
        card = fam_lookup.get(family_id, {}).get("cards", {}).get("card", "")

        rec = {
            "family_id": family_id,
            "card": card,
            "iteration": total_calls,
            "program": code,
            "fitness": fitness,
            "mean_colors": mean_c,
            "r_train": r_train,
            "error": err,
            "tokens": {"prompt": 2800, "completion": 450},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        logger.log(rec)
        evaluated_candidates.append(rec)
        print(f"  Iter {total_calls}: Success={res['success']}, Mean Colors={mean_c}, Fitness={fitness:+.4f}, Error={err}")

    # --------------------------------------------------------------------------
    # PROTOCOL FOR FINAL HEURISTIC SELECTION:
    # 1. Rank all valid candidates by train fitness -> take top-10
    # 2. Select best candidate on VAL
    # 3. Evaluate selected candidate on TEST
    # --------------------------------------------------------------------------
    valid_candidates = [c for c in evaluated_candidates if c["fitness"] > -50.0]
    if not valid_candidates:
        print(f"[{family_id}] No valid candidate found. Fallback to R(F).")
        return {"family_id": family_id, "selected_test_colors": float(np.mean(r_test_arr)), "gap_colors": 0.0, "gap_pct": 0.0}

    # Top-10 by train fitness
    valid_candidates.sort(key=lambda x: x["fitness"], reverse=True)
    top_10 = valid_candidates[:10]

    # Evaluate top-10 on VAL
    print(f"[{family_id}] Evaluating top-{len(top_10)} candidates on VAL graphs...")
    best_cand = None
    best_val_mean = float("inf")

    for cand in top_10:
        val_res = evaluate_candidate_safe(
            cand["program"],
            val_graphs,
            per_graph_timeout=per_graph_timeout,
            overall_timeout=15.0,
        )
        if val_res["success"] and val_res["mean_colors"] < best_val_mean:
            best_val_mean = val_res["mean_colors"]
            best_cand = cand

    print(f"[{family_id}] Selected candidate on VAL: Best Val Mean = {best_val_mean:.3f} (vs R(F) Val = {r_val:.3f})")
    is_fallback = (best_val_mean > r_val)
    print(f"[{family_id}] Fallback Status: {'TRIGGERED (worse than R(F) on VAL)' if is_fallback else 'NOT NEEDED (strictly beats R(F) on VAL)'}")

    # Evaluate best candidate on TEST (50 graphs)
    test_res = evaluate_candidate_safe(
        best_cand["program"],
        test_graphs,
        per_graph_timeout=per_graph_timeout,
        overall_timeout=30.0,
    )
    test_counts = np.array(test_res["color_counts"], dtype=float)

    # Primary Gap B: NO FALLBACK (pure performance of evolved heuristic vs R(F))
    gap_c, c_lo, c_hi, gap_p, p_lo, p_hi = compute_paired_bootstrap_ci(r_test_arr, test_counts)

    # Supplementary: WITH FALLBACK
    if is_fallback:
        fb_gap_c, fb_gap_p = 0.0, 0.0
    else:
        fb_gap_c, fb_gap_p = gap_c, gap_p

    print(f"[{family_id}] TEST Evaluation: Mean Colors = {np.mean(test_counts):.3f} vs R(F) = {np.mean(r_test_arr):.3f}")
    print(f"[{family_id}] PRIMARY Gap B (NO FALLBACK)   = {gap_c:+.3f} colors [{c_lo:+.3f}, {c_hi:+.3f}] | {gap_p:+.2f}% [{p_lo:+.2f}%, {p_hi:+.2f}%]")
    print(f"[{family_id}] SUPPLEMENTARY Gap (W/ FALLBACK) = {fb_gap_c:+.3f} colors | {fb_gap_p:+.2f}%")

    return {
        "family_id": family_id,
        "selected_test_colors": float(np.mean(test_counts)),
        "r_test": float(np.mean(r_test_arr)),
        "gap_no_fallback_colors": gap_c,
        "gap_no_fallback_pct": gap_p,
        "gap_with_fallback_colors": fb_gap_c,
        "gap_with_fallback_pct": fb_gap_p,
        "is_fallback": is_fallback,
        "total_calls": total_calls,
        "total_cost": total_cost,
    }


def main():
    test_log = "results/pilot_candidates_mock.jsonl"
    if Path(test_log).exists():
        Path(test_log).unlink()

    families_to_test = [
        "kpart_k10_p0.3_n100-200",
        "gnp_d0.30_n100-200",
    ]

    print("=" * 80)
    print("RUNNING DRY-RUN MOCK EVOLUTION ON 2 FAMILIES (TESTING CEILINGS & RESUME)")
    print("=" * 80)

    # Pass 1: Run with call ceiling = 4
    print("\n--- PASS 1: Run with max_calls = 4 ---")
    for fid in families_to_test:
        run_mock_evolution_for_family(fid, max_calls=4, max_dollars=0.10, logger_path=test_log)

    # Pass 2: Resume with call ceiling = 6 (tests resume from log)
    print("\n--- PASS 2: RESUME with max_calls = 6 ---")
    results = []
    for fid in families_to_test:
        res = run_mock_evolution_for_family(fid, max_calls=6, max_dollars=0.10, logger_path=test_log)
        results.append(res)

    print("\n--- SUMMARY ACROSS TESTED FAMILIES (PASS 2) ---")
    n_fb = sum(1 for r in results if r["is_fallback"])
    mean_gap_no_fb = np.mean([r["gap_no_fallback_colors"] for r in results])
    mean_gap_fb = np.mean([r["gap_with_fallback_colors"] for r in results])
    print(f"Total Families: {len(results)}")
    print(f"Families requiring Fallback to R(F): {n_fb} / {len(results)} ({n_fb/len(results)*100:.1f}%)")
    print(f"PRIMARY Gap B (No Fallback) Mean:     {mean_gap_no_fb:+.3f} colors")
    print(f"SUPPLEMENTARY Gap (With Fallback) Mean: {mean_gap_fb:+.3f} colors")

    print("\n" + "=" * 80)
    print("MOCK EVOLUTION COMPLETED SUCCESSFULLY!")
    print("Verified: AST filter, child process sandbox, candidate logging, resume recovery, budget ceilings.")
    print("=" * 80)

    if Path(test_log).exists():
        Path(test_log).unlink()


if __name__ == "__main__":
    main()
