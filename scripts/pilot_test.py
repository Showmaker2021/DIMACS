"""Test Evaluator for Pilot Graph Coloring Heuristics.

Features:
- Reads frozen selected candidate from results/selections/{family_id}.json.
- STRICT LOCK ENFORCEMENT: Enforces results/pilot_test_locks/{family_id}.done to guarantee exactly ONE test evaluation per family and prevent test data snooping.
- Evaluates on all 50 TEST graphs.
- Reports:
  * Primary Gap B (NO FALLBACK): pure performance of evolved candidate vs R(F) on test.
  * Supplementary Gap B (WITH FALLBACK): fallback to R(F) if candidate was worse on VAL.
  * 95% bootstrap paired confidence intervals (both in colors and in %).
- Saves full test results to results/pilot_test_results.csv.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import time
import argparse
import numpy as np
import pandas as pd

from gate.dataset import load_family, families
from gate.evolved.sandbox import evaluate_candidate_safe
from gate.evolved.task import CALIBRATED_PER_GRAPH_TIMEOUTS
from gate.tune import compute_paired_bootstrap_ci
from evaluator.coloring import dsatur, rlf
from evaluator.verify import count_colors


def run_test_evaluation(
    family_id: str = "kpart_k10_p0.3_n100-200",
    split: str = "test_pilot",
    selection_dir: str = "results/selections",
    lock_dir: str = "results/pilot_test_locks",
    results_csv: str = "results/pilot_test_results.csv",
    mock: bool = False,
):
    Path(lock_dir).mkdir(parents=True, exist_ok=True)
    lock_path = Path(lock_dir) / f"{family_id}.done"

    # 1. STRICT LOCK CHECK: Prevents repeated test evaluations
    if lock_path.exists():
        print("=" * 80)
        print(f"[LOCK ACTIVE] Test evaluation for '{family_id}' has ALREADY been completed.")
        print(f"Lock file: {lock_path}")
        print("To maintain strict separation and zero test snooping, repeated test runs are BLOCKED.")
        print("=" * 80)
        return

    # 2. Read frozen selection JSON
    selection_path = Path(selection_dir) / f"{family_id}.json"
    if not selection_path.exists():
        if mock:
            # Generate dummy mock selection for testing the pipeline
            mock_selection = {
                "family_id": family_id,
                "card": "mock card",
                "selected_candidate": {
                    "program": """def priority(adj, colors, sat, deg, uncolored_deg):
    n = adj.shape[0]
    return sat.astype(float) * (n + 1) + uncolored_deg.astype(float)
"""
                },
                "val_mean_colors": 13.35,
                "r_val": 13.15,
                "is_fallback": True,
            }
            Path(selection_dir).mkdir(parents=True, exist_ok=True)
            with open(selection_path, "w", encoding="utf-8") as f:
                json.dump(mock_selection, f, indent=2)
            print(f"[MOCK] Created mock selection file at: {selection_path}")
        else:
            raise FileNotFoundError(
                f"Frozen selection file not found: {selection_path}. Run pilot_evolve.py first to freeze the selected candidate."
            )

    with open(selection_path, "r", encoding="utf-8") as f:
        selection_data = json.load(f)

    cand = selection_data.get("selected_candidate")
    if not cand or "program" not in cand:
        raise ValueError(f"Invalid selection file {selection_path}: missing candidate program.")

    prog = cand["program"]
    is_fallback = selection_data.get("is_fallback", False)
    val_mean = selection_data.get("val_mean_colors")
    r_val = selection_data.get("r_val")

    print("=" * 80)
    print(f"PILOT TEST EVALUATION: Family '{family_id}' (ONE-TIME LOCKED RUN)")
    print(f"Evaluation Split: '{split}' (seeds 4000-4199, disjoint test pilot set)")
    print(f"Selected Candidate Val Mean = {val_mean:.3f} vs R(F) Val = {r_val:.3f}")
    print(f"Fallback Status: {'TRIGGERED (worse than R(F) on VAL)' if is_fallback else 'NOT NEEDED (strictly beats R(F) on VAL)'}")
    print("=" * 80)

    # 3. Load TEST graphs
    test_graphs = load_family(family_id, split)
    test_adjs = [g[1] for g in test_graphs]

    # Precompute R(F) baseline on TEST
    # Choice of R(F) was determined by VAL
    val_graphs = load_family(family_id, "val")
    c_ds_val = float(np.mean([count_colors(dsatur(g[1])) for g in val_graphs]))
    c_rlf_val = float(np.mean([count_colors(rlf(g[1])) for g in val_graphs]))

    if c_rlf_val <= c_ds_val:
        r_test_arr = np.array([count_colors(rlf(a)) for a in test_adjs], dtype=float)
        rf_name = "RLF"
    else:
        r_test_arr = np.array([count_colors(dsatur(a)) for a in test_adjs], dtype=float)
        rf_name = "DSATUR"

    per_graph_timeout = CALIBRATED_PER_GRAPH_TIMEOUTS.get(family_id, 0.50)

    # 4. Evaluate candidate on 50 TEST graphs
    test_res = evaluate_candidate_safe(
        prog,
        test_graphs,
        per_graph_timeout=per_graph_timeout,
        overall_timeout=30.0,
    )

    if not test_res["success"]:
        raise RuntimeError(f"Candidate failed on TEST graphs: {test_res['error']}")

    cand_test_counts = np.array(test_res["color_counts"], dtype=float)
    mean_cand_test = float(np.mean(cand_test_counts))
    mean_rf_test = float(np.mean(r_test_arr))

    # 5. Compute Gaps with 95% paired bootstrap CI
    # Primary Gap B: NO FALLBACK
    gap_c, c_lo, c_hi, gap_p, p_lo, p_hi = compute_paired_bootstrap_ci(r_test_arr, cand_test_counts)

    # Supplementary Gap B: WITH FALLBACK
    if is_fallback:
        fb_gap_c, fb_c_lo, fb_c_hi = 0.0, 0.0, 0.0
        fb_gap_p, fb_p_lo, fb_p_hi = 0.0, 0.0, 0.0
        reported_mean = mean_rf_test
    else:
        fb_gap_c, fb_c_lo, fb_c_hi = gap_c, c_lo, c_hi
        fb_gap_p, fb_p_lo, fb_p_hi = gap_p, p_lo, p_hi
        reported_mean = mean_cand_test

    print(f"[{family_id}] TEST Evaluation Summary (N=50 graphs):")
    print(f"  * Evolved Candidate Mean Colors: {mean_cand_test:.3f}")
    print(f"  * Reference R(F) ({rf_name}) Mean:     {mean_rf_test:.3f}")
    print(f"  * PRIMARY Gap B (NO FALLBACK):    {gap_c:+.3f} colors [95% CI: {c_lo:+.3f}, {c_hi:+.3f}] | {gap_p:+.2f}% [{p_lo:+.2f}%, {p_hi:+.2f}%]")
    print(f"  * SUPPLEMENTARY Gap (W/ FALLBACK): {fb_gap_c:+.3f} colors [95% CI: {fb_c_lo:+.3f}, {fb_c_hi:+.3f}] | {fb_gap_p:+.2f}% [{fb_p_lo:+.2f}%, {fb_p_hi:+.2f}%]")

    # 6. Save result to CSV
    row = {
        "family_id": family_id,
        "rf_name": rf_name,
        "mean_rf_test": mean_rf_test,
        "mean_cand_test": mean_cand_test,
        "is_fallback": is_fallback,
        "primary_gap_colors": gap_c,
        "primary_ci_lo_colors": c_lo,
        "primary_ci_hi_colors": c_hi,
        "primary_gap_pct": gap_p,
        "primary_ci_lo_pct": p_lo,
        "primary_ci_hi_pct": p_hi,
        "supp_gap_colors": fb_gap_c,
        "supp_gap_pct": fb_gap_p,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    df = pd.DataFrame([row])
    if Path(results_csv).exists():
        df.to_csv(results_csv, mode="a", header=False, index=False)
    else:
        Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_csv, mode="w", header=True, index=False)

    # 7. Write strict completion lock file
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(f"Completed at {time.strftime('%Y-%m-%dT%H:%M:%S')}\nMean Colors: {mean_cand_test:.3f}\nPrimary Gap B: {gap_c:+.3f} colors\n")

    print("\n" + "=" * 80)
    print(f"TEST EVALUATION COMPLETE. LOCK FILE WRITTEN AT: {lock_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Pilot Test Evaluator (Reads Frozen Selection, Strict .done Lock)")
    parser.add_argument("--family", default="kpart_k10_p0.3_n100-200", help="Target family ID")
    parser.add_argument("--split", default="test_pilot", help="Test split name (default: test_pilot [seeds 4000-4199])")
    parser.add_argument("--selection-dir", default="results/selections", help="Directory containing frozen selections")
    parser.add_argument("--lock-dir", default="results/pilot_test_locks", help="Directory for strict test lock files")
    parser.add_argument("--results-csv", default="results/pilot_test_results.csv", help="CSV path for test results")
    parser.add_argument("--mock", action="store_true", help="Run with mock selection for testing pipeline")
    args = parser.parse_args()

    run_test_evaluation(
        family_id=args.family,
        split=args.split,
        selection_dir=args.selection_dir,
        lock_dir=args.lock_dir,
        results_csv=args.results_csv,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
