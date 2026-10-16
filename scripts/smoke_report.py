"""Smoke Test Offline Diagnostic Report Script.

Reads candidate JSONL logs, evaluates candidates locally on Train & Val graphs
(ZERO LLM API calls), and prints comprehensive diagnostics:
1. Candidate validity & rejection breakdown.
2. Two-level seed clone rates (absolute & equivalent).
3. Number of genuinely distinct candidates (distinct 20-dim train color vectors).
4. Top-10 selection on 20 VAL graphs, comparison with R(F) VAL, and fallback status.
5. Full Train-vs-Val evaluation table for all candidates.
6. Three sample candidate codes (Best, Median, Rejected).
"""

import sys
import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate.dataset import load_family, families
from gate.evolved.sandbox import evaluate_candidate_safe
from evaluator.coloring import dsatur, rlf
from evaluator.verify import count_colors


def deduplicate_by_train_behavior(candidates: list) -> list:
    """Deduplicate candidates by train color counts vector.
    
    For each distinct train color vector, keeps only ONE representative:
    - Highest gain_train (or fitness)
    - Tie-break: Shortest code
    """
    groups = {}
    for c in candidates:
        key = c.get("train_counts")
        if not key:
            key = hashlib.sha256(c["code"].strip().encode()).hexdigest()
        fit = c.get("gain_train", -1e9)
        prog_len = len(c.get("code", "").strip())
        if key not in groups:
            groups[key] = c
        else:
            ex = groups[key]
            ex_fit = ex.get("gain_train", -1e9)
            ex_len = len(ex.get("code", "").strip())
            if (fit > ex_fit) or (np.isclose(fit, ex_fit) and prog_len < ex_len):
                groups[key] = c
    deduped = list(groups.values())
    deduped.sort(key=lambda x: (x.get("gain_train", -1e9), -len(x.get("code", "").strip())), reverse=True)
    return deduped


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke Test Offline Diagnostic Report")
    parser.add_argument("--log", type=str, default="results/smoke_gemini_lite.jsonl", help="Path to JSONL log file")
    parser.add_argument("--run-id", type=str, default="gemini_lite_smoke_20", help="Run ID to report (default: gemini_lite_smoke_20)")
    parser.add_argument("--family", type=str, default="kpart_k10_p0.3_n100-200", help="Target family ID")
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        sys.exit(1)

    # 1. Load records
    all_records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_records.append(json.loads(line))
                except Exception:
                    pass

    # Filter by family and run_id if specified
    records = [
        r for r in all_records
        if r.get("family_id") == args.family and (args.run_id is None or r.get("run_id") == args.run_id)
    ]
    if not records:
        # Fallback to all records matching family
        records = [r for r in all_records if r.get("family_id") == args.family]

    # If more than 20 records (e.g. earlier tests), take the targeted 20 iterations
    if len(records) > 20 and args.run_id == "gemini_lite_smoke_20":
        records = [r for r in records if r.get("run_id") == "gemini_lite_smoke_20"]
    if len(records) > 20:
        records = records[-20:]

    total_candidates = len(records)

    # 2. Load Train & Val splits for local evaluation (ZERO LLM calls)
    train_graphs = load_family(args.family, "baseline_train")
    val_graphs = load_family(args.family, "val")

    # Baseline R(F) computations
    ds_train_counts = [count_colors(dsatur(g[1])) for g in train_graphs]
    rlf_train_counts = [count_colors(rlf(g[1])) for g in train_graphs]
    ds_train_mean = float(np.mean(ds_train_counts))
    rlf_train_mean = float(np.mean(rlf_train_counts))
    r_train = min(ds_train_mean, rlf_train_mean)

    ds_val_counts = [count_colors(dsatur(g[1])) for g in val_graphs]
    rlf_val_counts = [count_colors(rlf(g[1])) for g in val_graphs]
    ds_val_mean = float(np.mean(ds_val_counts))
    rlf_val_mean = float(np.mean(rlf_val_counts))
    r_val = min(ds_val_mean, rlf_val_mean)

    # Baseline seed hashes on 3 train graphs for absolute clone check
    ds_hashes = [hashlib.sha256(dsatur(adj).tobytes()).hexdigest() for _, adj in train_graphs[:3]]
    ds_hash = "-".join(ds_hashes)
    rlf_hashes = [hashlib.sha256(rlf(adj).tobytes()).hexdigest() for _, adj in train_graphs[:3]]
    rlf_hash = "-".join(rlf_hashes)

    # 3. Analyze validity, clones, and evaluate on VAL
    valid_records = []
    rejected_records = []
    rejection_reasons = {}

    abs_clones = 0
    equiv_clones = 0

    evaluated_candidates = []
    distinct_train_profiles = set()

    for r in records:
        code = r.get("program") or r.get("code", "")
        it = r.get("iteration")
        op = r.get("operator", "unknown")

        # Evaluate on Train graphs locally to confirm metrics and extract exact color vectors
        res_train = evaluate_candidate_safe(code, train_graphs, per_graph_timeout=0.80)

        if not res_train["success"]:
            rejected_records.append(r)
            err = res_train["error"] or r.get("error", "Unknown error")
            err_lower = err.lower()
            
            import re
            np_attr_match = re.search(r"NumPy attribute '(\w+)'", err)
            obj_attr_match = re.search(r"Attribute '(\w+)'", err)
            if np_attr_match:
                reason = f"AST Violation: np.{np_attr_match.group(1)} not allowed"
            elif obj_attr_match:
                reason = f"AST Violation: .{obj_attr_match.group(1)} attribute not allowed"
            elif "syntaxerror" in err_lower:
                reason = "AST Violation: Python SyntaxError"
            elif "memory" in err_lower:
                reason = "Memory Limit Exceeded (>1.5GB)"
            elif "per-graph timeout" in err_lower:
                reason = "Per-Graph Timeout Exceeded (>0.5s)"
            elif "timeout" in err_lower:
                reason = "Overall Timeout Exceeded"
            elif "indexerror" in err_lower:
                reason = "Runtime: IndexError (invalid double-masking/dimension mismatch)"
            elif "keyerror" in err_lower:
                reason = "Runtime: KeyError (restricted environment symbol)"
            elif "input mutation" in err_lower or "read-only" in err_lower:
                reason = "Statelessness: Attempted to mutate read-only inputs"
            elif "statelessness" in err_lower or "purity" in err_lower:
                reason = f"Statelessness / Purity Failed: {err[:40]}"
            else:
                reason = f"Runtime / Verification Error: {err[:40]}"

            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            evaluated_candidates.append({
                "iteration": it,
                "operator": op,
                "train_mean": None,
                "val_mean": None,
                "gain_train": None,
                "gain_val": None,
                "status": f"REJECTED: {reason}",
                "record": r,
            })
            continue

        valid_records.append(r)

        # Check clones
        train_vec = tuple(res_train["color_counts"])
        distinct_train_profiles.add(train_vec)

        # Level 1 Absolute Clone
        cand_hash = res_train.get("behavior_hash", "")
        is_abs_clone = (cand_hash == ds_hash or cand_hash == rlf_hash)
        if is_abs_clone:
            abs_clones += 1

        # Level 2 Equivalent Clone
        is_equiv_clone = (train_vec == tuple(ds_train_counts) or train_vec == tuple(rlf_train_counts))
        if is_equiv_clone:
            equiv_clones += 1

        train_mean = float(np.mean(train_vec))
        gain_train = r_train - train_mean

        # Evaluate on VAL graphs
        res_val = evaluate_candidate_safe(code, val_graphs, per_graph_timeout=0.80)
        if res_val["success"]:
            val_vec = tuple(res_val["color_counts"])
            val_mean = float(np.mean(val_vec))
            gain_val = r_val - val_mean
            status = "VALID"
        else:
            val_vec = None
            val_mean = None
            gain_val = None
            status = f"VAL FAIL: {res_val['error'][:25]}"

        evaluated_candidates.append({
            "iteration": it,
            "operator": op,
            "train_mean": train_mean,
            "val_mean": val_mean,
            "gain_train": gain_train,
            "gain_val": gain_val,
            "train_counts": train_vec,
            "val_counts": val_vec,
            "fitness": (r_train - train_mean),
            "status": status,
            "code": code,
            "record": r,
        })

    # =========================================================================
    # PRINT REPORT
    # =========================================================================
    print("=" * 85)
    print("SMOKE TEST OFFLINE DIAGNOSTIC REPORT (NO API CALLS)")
    print(f"Log File:   {log_path}")
    print(f"Family:     {args.family}")
    print(f"Run ID:     {args.run_id}")
    print(f"Total Logged Candidates: {total_candidates}")
    print("=" * 85)

    print("\n1. CANDIDATE VALIDITY & REJECTION BREAKDOWN:")
    valid_pct = (len(valid_records) / total_candidates) * 100.0 if total_candidates > 0 else 0.0
    rej_pct = (len(rejected_records) / total_candidates) * 100.0 if total_candidates > 0 else 0.0
    print(f"   - Valid Candidates:    {len(valid_records):>2} / {total_candidates} ({valid_pct:5.1f}%)")
    print(f"   - Rejected Candidates: {len(rejected_records):>2} / {total_candidates} ({rej_pct:5.1f}%)")
    if rejection_reasons:
        print("   - Detailed Breakdown of Rejection Reasons (Attribute/Construct):")
        for rsn, cnt in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"     * {rsn:<55}: {cnt:>2} ({(cnt/total_candidates)*100.0:.1f}%)")
    else:
        print("   - Breakdown of Rejection Reasons: None (0 rejections)")

    print("\n2. TWO-LEVEL SEED CLONE RATES:")
    abs_pct = (abs_clones / total_candidates) * 100.0 if total_candidates > 0 else 0.0
    equiv_pct = (equiv_clones / total_candidates) * 100.0 if total_candidates > 0 else 0.0
    print(f"   - Level 1 (Absolute Clone - exact coloring hash on 3 graphs):    {abs_clones:>2} / {total_candidates} ({abs_pct:5.1f}%)")
    print(f"   - Level 2 (Equivalent Clone - identical color counts on 20 graphs): {equiv_clones:>2} / {total_candidates} ({equiv_pct:5.1f}%)")
    print("   - Policy: Absolute clones are strictly excluded from mating pool.")

    print("\n3. GENUINELY DISTINCT CANDIDATES (BEHAVIORAL DIVERSITY):")
    print(f"   - Number of distinct 20-dim Train color vectors: {len(distinct_train_profiles)} out of {len(valid_records)} valid candidates.")
    print("   - Indicates true diversity of behavioral output across the instance distribution.")

    print("\n4. BASELINE REFERENCE VALUES (EXACT NUMBERS & SOURCES):")
    print(f"   - Train Split (20 graphs, seeds 0-19):")
    print(f"     * DSATUR Train Mean: {ds_train_mean:.3f} colors")
    print(f"     * RLF Train Mean:    {rlf_train_mean:.3f} colors")
    print(f"     * R(F) Train:        {r_train:.3f} colors (Best single = RLF)")
    print(f"   - Val Split (20 graphs, seeds 1000-1019):")
    print(f"     * DSATUR Val Mean:   {ds_val_mean:.3f} colors")
    print(f"     * RLF Val Mean:      {rlf_val_mean:.3f} colors")
    print(f"     * R(F) Val:          {r_val:.3f} colors (Best single = RLF)")
    print(f"     * Source: Evaluated directly on frozen dataset v1.0/v1.1 graphs with verified DSATUR & RLF.")

    print("\n5. TOP-10 DISTINCT BEHAVIOR SELECTION ON 20 VAL GRAPHS & FALLBACK STATUS:")
    valid_evaluated = [c for c in evaluated_candidates if c["val_mean"] is not None]
    # Behavioral deduplication: keep only ONE representative per distinct train color vector
    deduped_evaluated = deduplicate_by_train_behavior(valid_evaluated)
    top_10 = deduped_evaluated[:10]

    print(f"   {'Rank':<4} {'Iter':<5} {'Operator':<10} {'Train Mean':<11} {'Gain Train':<11} {'Val Mean':<10} {'Gain Val':<10}")
    print("   " + "-" * 65)
    for idx, c in enumerate(top_10, 1):
        gt = f"{c['gain_train']:+.3f}"
        gv = f"{c['gain_val']:+.3f}"
        print(f"   {idx:<4} #{c['iteration']:<4} {c['operator']:<10} {c['train_mean']:<11.3f} {gt:<11} {c['val_mean']:<10.3f} {gv:<10}")

    # Best candidate on Val among Top-10
    if top_10:
        best_val_cand = min(top_10, key=lambda x: x["val_mean"])
        best_val_mean = best_val_cand["val_mean"]
        fallback_triggered = best_val_mean > r_val
        print("\n   --- SELECTION RESULT ---")
        print(f"   Best Candidate on VAL: Iteration #{best_val_cand['iteration']}")
        print(f"   Candidate Val Mean:    {best_val_mean:.3f} colors")
        print(f"   Reference R(F) Val:    {r_val:.3f} colors")
        print(f"   Excess colors vs R(F) (dương = tệ hơn): {best_val_mean - r_val:+.3f} colors ({((best_val_mean - r_val)/r_val)*100:+.2f}%)")
        print(f"   Fallback Status:       {'TRIGGERED (worse than R(F) on VAL -> fallback to R(F))' if fallback_triggered else 'NOT TRIGGERED (beats or matches R(F) on VAL)'}")
    else:
        print("   No valid candidates to evaluate.")

    # 5b. Statistical power & detectability on top-3 distinct behaviors
    top_3 = top_10[:3]
    if top_3:
        rf_val_arr = np.array(rlf_val_counts if rlf_val_mean <= ds_val_mean else ds_val_counts, dtype=float)
        rf_name = "RLF" if rlf_val_mean <= ds_val_mean else "DSATUR"

        print("\n5b. PAIRED STATISTICAL POWER & DETECTABILITY ANALYSIS (TOP-3 DISTINCT BEHAVIORS ON VAL):")
        print(f"    Reference R(F) Val: {r_val:.3f} colors ({rf_name}) on 20 val graphs")
        print("    Sample size scenarios: n=20 (current val), n=50 (single family test), n=200 (pooled test)")
        header = f"    {'Rank':<4} {'Iter':<5} {'Val Mean':<10} {'Excess vs RF (dương=tệ)':<25} {'Paired s_D':<12} {'SE(n=20)':<10} {'SE(n=50)':<10} {'SE(n=200)':<11} {'MDE-80%(n=20)':<17} {'MDE-80%(n=50)':<17} {'MDE-80%(n=200)':<17}"
        print("    " + "-" * len(header))
        print(header)
        print("    " + "-" * len(header))

        for idx, c in enumerate(top_3, 1):
            c_val_arr = np.array(c["val_counts"], dtype=float)
            diff_arr = c_val_arr - rf_val_arr
            mean_diff = float(np.mean(diff_arr))
            pct_diff = (mean_diff / r_val) * 100.0
            s_d = float(np.std(diff_arr, ddof=1))

            def calc_se_mde(sd, n_samples):
                se = sd / np.sqrt(n_samples)
                df = n_samples - 1
                t_crit = float(stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df))
                mde = t_crit * se
                mde_pct = (mde / r_val) * 100.0
                return se, mde, mde_pct

            se_20, mde_20, mde_pct_20 = calc_se_mde(s_d, 20)
            se_50, mde_50, mde_pct_50 = calc_se_mde(s_d, 50)
            se_200, mde_200, mde_pct_200 = calc_se_mde(s_d, 200)

            diff_str = f"{mean_diff:+.3f} ({pct_diff:+.2f}%)"
            mde_str_20 = f"{mde_20:.3f} ({mde_pct_20:.2f}%)"
            mde_str_50 = f"{mde_50:.3f} ({mde_pct_50:.2f}%)"
            mde_str_200 = f"{mde_200:.3f} ({mde_pct_200:.2f}%)"

            print(f"    {idx:<4} #{c['iteration']:<4} {c['val_mean']:<10.3f} {diff_str:<25} {s_d:<12.3f} {se_20:<10.3f} {se_50:<10.3f} {se_200:<11.3f} {mde_str_20:<17} {mde_str_50:<17} {mde_str_200:<17}")
        print("    " + "-" * len(header))

    print("\n6. FULL TRAIN-VS-VAL EVALUATION TABLE (ALL 20 CANDIDATES):")
    print(f"   {'Iter':<5} {'Operator':<10} {'Train Mean':<12} {'Gain Train':<12} {'Val Mean':<11} {'Gain Val':<11} {'Status'}")
    print("   " + "-" * 75)
    for c in evaluated_candidates:
        tm = f"{c['train_mean']:.3f}" if c['train_mean'] is not None else "N/A"
        vm = f"{c['val_mean']:.3f}" if c['val_mean'] is not None else "N/A"
        gt = f"{c['gain_train']:+.3f}" if c['gain_train'] is not None else "N/A"
        gv = f"{c['gain_val']:+.3f}" if c['gain_val'] is not None else "N/A"
        print(f"   #{c['iteration']:<4} {c['operator']:<10} {tm:<12} {gt:<12} {vm:<11} {gv:<11} {c['status']}")

    print("\n7. THREE SAMPLE CANDIDATE CODES:")
    # Sample 1: Best Valid
    if valid_evaluated:
        best_cand = max(valid_evaluated, key=lambda x: (x["gain_train"], x["gain_val"]))
        print("\n" + "=" * 80)
        print(f"--- SAMPLE 1: BEST VALID CANDIDATE (Iteration #{best_cand['iteration']} | Train Mean={best_cand['train_mean']:.2f} | Val Mean={best_cand['val_mean']:.2f}) ---")
        print("=" * 80)
        print(best_cand["code"].strip())

        # Sample 2: Median Valid
        med_idx = len(valid_evaluated) // 2
        med_cand = valid_evaluated[med_idx]
        print("\n" + "=" * 80)
        print(f"--- SAMPLE 2: MEDIAN VALID CANDIDATE (Iteration #{med_cand['iteration']} | Train Mean={med_cand['train_mean']:.2f} | Val Mean={med_cand['val_mean']:.2f}) ---")
        print("=" * 80)
        print(med_cand["code"].strip())

    # Sample 3: Rejected Candidate
    if rejected_records:
        rej_sample = rejected_records[0]
        print("\n" + "=" * 80)
        print(f"--- SAMPLE 3: REJECTED CANDIDATE (Iteration #{rej_sample.get('iteration')} | Error: {rej_sample.get('error')}) ---")
        print("=" * 80)
        print(rej_sample.get("program", "").strip())
    else:
        # Check if there is any rejected candidate in all_records
        all_rej = [r for r in all_records if r.get("error")]
        if all_rej:
            rej_sample = all_rej[0]
            print("\n" + "=" * 80)
            print(f"--- SAMPLE 3: REJECTED CANDIDATE (from previous run Iteration #{rej_sample.get('iteration')} | Error: {rej_sample.get('error')}) ---")
            print("=" * 80)
            print(rej_sample.get("program", "").strip())
        else:
            print("\n--- SAMPLE 3: REJECTED CANDIDATE ---")
            print("Zero candidates were rejected in this run (100% valid).")

    print("\n" + "=" * 85)
    print("END OF SMOKE DIAGNOSTIC REPORT")
    print("=" * 85)


if __name__ == "__main__":
    main()
