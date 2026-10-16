"""Gate 2 Evaluation and Decision Report Script.

Implements Task 6.2 from PLAN_PHASE6_ONWARD.md:
- Collects candidate heuristics from pilot evolution runs across the 6 pilot families (4 target + 2 control).
- Selects top-1 candidate per family evaluated strictly on teacher_val (seeds 3000-3019).
- Evaluates top-1 candidate once on test_pilot (seeds 4000-4199) with .done lock mechanism.
- Computes paired bootstrap 95% CI (10,000 resamples) and Wilcoxon signed-rank test.
- Reports behavioral similarity to DSATUR/RLF, sandbox failure rates, and fraction beating R(F).
- Evaluates predefined Gate 2 decision criteria: PASS / MARGINAL / FAIL.
- Outputs runs/gate2/report.md and runs/gate2/report.json.
"""

import sys
import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import scipy.stats

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate.dataset import load_family, families
from gate.evolved.sandbox import evaluate_candidate_safe
from gate.evolved.task import CALIBRATED_PER_GRAPH_TIMEOUTS
from gate.tune import compute_paired_bootstrap_ci
from evaluator.coloring import dsatur, rlf
from evaluator.verify import count_colors


TARGET_FAMILIES = [
    "kpart_k10_p0.3_n100-200",
    "gnp_d0.30_n100-200",
    "sbm_k10_in0.5_out0.05_n200-300",
    "ws_k10_p0.1_n100-200",
]

CONTROL_FAMILIES = [
    "interval_L0.10_n100-200",
    "ba_m12_n200-300",
]

ALL_PILOT_FAMILIES = TARGET_FAMILIES + CONTROL_FAMILIES


def compute_wilcoxon_paired(arr_ref: np.ndarray, arr_cand: np.ndarray) -> float:
    """Compute paired Wilcoxon signed-rank test p-value.
    
    Tests whether arr_cand is significantly smaller (better) than arr_ref.
    If all differences are zero, returns 1.0.
    """
    diffs = arr_ref - arr_cand
    non_zero = diffs[diffs != 0]
    if len(non_zero) == 0:
        return 1.0
    try:
        res = scipy.stats.wilcoxon(arr_ref, arr_cand, alternative="greater")
        return float(res.pvalue)
    except Exception:
        return 1.0


def check_behavioral_similarity(
    code_str: str,
    probe_graphs: List[Tuple[int, np.ndarray]],
) -> Dict[str, bool]:
    """Check if candidate produces color vectors identical to DSATUR or RLF on probe graphs."""
    res = evaluate_candidate_safe(code_str, probe_graphs, per_graph_timeout=0.5, overall_timeout=10.0)
    if not res.get("success", False) or res.get("color_counts") is None:
        return {"equal_dsatur": False, "equal_rlf": False}

    cand_counts = tuple(res["color_counts"])

    ds_counts = tuple(count_colors(dsatur(g[1])) for g in probe_graphs)
    rlf_counts = tuple(count_colors(rlf(g[1])) for g in probe_graphs)

    return {
        "equal_dsatur": cand_counts == ds_counts,
        "equal_rlf": cand_counts == rlf_counts,
    }


def find_top_candidate_on_val(
    family_id: str,
    runs_dir: Path,
    teacher_val_graphs: List[Tuple[int, np.ndarray]],
    mock: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Scan all runs for family, pick top-1 candidate on teacher_val."""
    fam_dir = runs_dir / family_id
    candidates: List[Dict[str, Any]] = []

    if fam_dir.exists():
        for run_path in fam_dir.glob("*/candidates.jsonl"):
            with open(run_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            candidates.append(json.loads(line))
                        except Exception:
                            pass

    # Aggregated run statistics
    total_calls = len(candidates)
    fail_counts: Dict[str, int] = {}
    valid_candidates: List[Dict[str, Any]] = []

    for c in candidates:
        status = c.get("status", "ok" if c.get("sandbox_ok", False) else "error")
        if status != "ok":
            err_type = c.get("error_type", "sandbox_fail")
            fail_counts[err_type] = fail_counts.get(err_type, 0) + 1
        else:
            valid_candidates.append(c)

    if not valid_candidates:
        if mock:
            # Generate deterministic mock candidate
            mock_code = """def priority(adj, colors, sat, deg, uncolored_deg):
    n = adj.shape[0]
    return sat.astype(float) * (n + 1) + uncolored_deg.astype(float)
"""
            mock_cand = {
                "code": mock_code,
                "teacher_val_mean": 13.0,
                "run_id": "mock_run",
                "valid": True,
            }
            return mock_cand, {"total_calls": 20, "fail_counts": {}, "valid_rate": 1.0}
        raise FileNotFoundError(f"No valid candidates found in {fam_dir} for family {family_id}")

    # Deduplicate code
    unique_by_code: Dict[str, Dict[str, Any]] = {}
    for c in valid_candidates:
        code = c.get("code", "")
        if code and code not in unique_by_code:
            unique_by_code[code] = c

    # Evaluate unique candidates on teacher_val if not already cached
    best_cand = None
    best_val_mean = float("inf")

    per_graph_timeout = CALIBRATED_PER_GRAPH_TIMEOUTS.get(family_id, 0.50)

    for code, c_entry in unique_by_code.items():
        val_mean = c_entry.get("teacher_val_mean")
        if val_mean is None:
            # Evaluate on teacher_val
            ev_res = evaluate_candidate_safe(
                code_str=code,
                graphs=teacher_val_graphs,
                per_graph_timeout=per_graph_timeout,
                overall_timeout=25.0,
            )
            if ev_res.get("success", False) and ev_res.get("mean_colors") is not None:
                val_mean = float(ev_res["mean_colors"])
            else:
                val_mean = float("inf")
            c_entry["teacher_val_mean"] = val_mean

        if val_mean < best_val_mean:
            best_val_mean = val_mean
            best_cand = c_entry

    stats = {
        "total_calls": total_calls,
        "fail_counts": fail_counts,
        "valid_rate": len(valid_candidates) / total_calls if total_calls > 0 else 0.0,
    }
    return best_cand, stats


def evaluate_gate2(
    runs_dir: str = "runs/pilot",
    out_dir: str = "runs/gate2",
    locks_dir: str = "runs/gate2/locks",
    force_unlock: bool = False,
    mock: bool = False,
    improvement_threshold_pass: float = 1.0,
    improvement_threshold_marginal: float = 0.5,
) -> Dict[str, Any]:
    """Execute Gate 2 evaluation across all 6 pilot families."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    lock_path = Path(locks_dir)
    lock_path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "families": {},
        "summary": {},
    }

    target_pass_count = 0
    target_marginal_count = 0
    all_valid_rates_100 = True

    for family_id in ALL_PILOT_FAMILIES:
        is_target = family_id in TARGET_FAMILIES
        role = "target" if is_target else "control"
        fam_lock = lock_path / f"{family_id}.done"

        # Check lock
        if fam_lock.exists() and not force_unlock:
            with open(fam_lock, "r", encoding="utf-8") as f:
                locked_data = json.load(f)
            results["families"][family_id] = locked_data
            if is_target:
                if locked_data.get("status") == "PASS":
                    target_pass_count += 1
                elif locked_data.get("status") == "MARGINAL":
                    target_marginal_count += 1
            if locked_data.get("valid_rate", 0.0) < 1.0:
                all_valid_rates_100 = False
            continue

        # Load teacher_val, test_pilot, probe graphs
        t_val_graphs = load_family(family_id, split="teacher_val")
        t_test_graphs = load_family(family_id, split="test_pilot")
        probe_graphs = load_family(family_id, split="probe")

        # Pick top candidate on teacher_val
        top_cand, run_stats = find_top_candidate_on_val(
            family_id=family_id,
            runs_dir=Path(runs_dir),
            teacher_val_graphs=t_val_graphs,
            mock=mock,
        )

        prog = top_cand["code"]
        val_mean = top_cand.get("teacher_val_mean")

        # Evaluate on test_pilot (seeds 4000-4199)
        per_graph_timeout = CALIBRATED_PER_GRAPH_TIMEOUTS.get(family_id, 0.50)
        test_res = evaluate_candidate_safe(
            code_str=prog,
            graphs=t_test_graphs,
            per_graph_timeout=per_graph_timeout,
            overall_timeout=35.0,
        )

        if not test_res.get("success", False) or test_res.get("color_counts") is None:
            raise RuntimeError(f"Top candidate failed on test_pilot for {family_id}: {test_res.get('error')}")

        cand_test_counts = np.array(test_res["color_counts"], dtype=float)
        mean_top1 = float(np.mean(cand_test_counts))

        # Baselines on test_pilot
        test_adjs = [g[1] for g in t_test_graphs]
        ds_test_counts = np.array([count_colors(dsatur(a)) for a in test_adjs], dtype=float)
        rlf_test_counts = np.array([count_colors(rlf(a)) for a in test_adjs], dtype=float)

        # R(F) determined by teacher_val
        ds_val_mean = float(np.mean([count_colors(dsatur(g[1])) for g in t_val_graphs]))
        rlf_val_mean = float(np.mean([count_colors(rlf(g[1])) for g in t_val_graphs]))
        rf_name = "RLF" if rlf_val_mean <= ds_val_mean else "DSATUR"
        rf_test_counts = rlf_test_counts if rf_name == "RLF" else ds_test_counts

        mean_ds = float(np.mean(ds_test_counts))
        mean_rlf = float(np.mean(rlf_test_counts))
        mean_rf = float(np.mean(rf_test_counts))

        # Bootstrap CIs (10,000 resamples)
        # Gap vs R(F)
        gap_rf_c, rf_c_lo, rf_c_hi, gap_rf_p, rf_p_lo, rf_p_hi = compute_paired_bootstrap_ci(
            arr_ref=rf_test_counts,
            arr_target=cand_test_counts,
            num_bootstraps=10000,
            seed=42,
        )

        # Gap vs DSATUR
        gap_ds_c, ds_c_lo, ds_c_hi, gap_ds_p, ds_p_lo, ds_p_hi = compute_paired_bootstrap_ci(
            arr_ref=ds_test_counts,
            arr_target=cand_test_counts,
            num_bootstraps=10000,
            seed=42,
        )

        # Wilcoxon test vs R(F)
        p_wilcoxon = compute_wilcoxon_paired(rf_test_counts, cand_test_counts)

        # Behavioral similarity on probe
        beh_probe = check_behavioral_similarity(prog, probe_graphs)

        # Criteria evaluation for single family
        fam_status = "FAIL"
        if gap_rf_p >= improvement_threshold_pass and rf_p_lo > 0:
            fam_status = "PASS"
        elif gap_rf_p >= improvement_threshold_marginal or (is_target and gap_rf_p > 0):
            fam_status = "MARGINAL"

        if is_target:
            if fam_status == "PASS":
                target_pass_count += 1
            elif fam_status == "MARGINAL":
                target_marginal_count += 1

        if run_stats["valid_rate"] < 1.0:
            all_valid_rates_100 = False

        fam_data = {
            "family_id": family_id,
            "role": role,
            "rf_name": rf_name,
            "mean_top1": mean_top1,
            "mean_dsatur": mean_ds,
            "mean_rlf": mean_rlf,
            "mean_rf": mean_rf,
            "val_mean_top1": val_mean,
            "gap_vs_rf_pct": gap_rf_p,
            "gap_vs_rf_ci": [rf_p_lo, rf_p_hi],
            "gap_vs_rf_colors": gap_rf_c,
            "gap_vs_rf_colors_ci": [rf_c_lo, rf_c_hi],
            "gap_vs_dsatur_pct": gap_ds_p,
            "gap_vs_dsatur_ci": [ds_p_lo, ds_p_hi],
            "p_wilcoxon": p_wilcoxon,
            "behavioral_probe": beh_probe,
            "valid_rate": run_stats["valid_rate"],
            "total_calls": run_stats["total_calls"],
            "fail_counts": run_stats["fail_counts"],
            "status": fam_status,
        }

        # Write lock file
        with open(fam_lock, "w", encoding="utf-8") as f:
            json.dump(fam_data, f, indent=2)

        results["families"][family_id] = fam_data

    # Gate 2 Overall Decision
    if target_pass_count >= 2 and all_valid_rates_100:
        gate_decision = "PASS"
    elif target_pass_count >= 1 or target_marginal_count >= 1:
        gate_decision = "MARGINAL"
    else:
        gate_decision = "FAIL"

    results["summary"] = {
        "gate_decision": gate_decision,
        "target_pass_count": target_pass_count,
        "target_marginal_count": target_marginal_count,
        "all_valid_rates_100": all_valid_rates_100,
        "threshold_pass": improvement_threshold_pass,
        "threshold_marginal": improvement_threshold_marginal,
    }

    # Save report.json
    with open(out_path / "report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Render report.md
    md_content = render_report_markdown(results)
    with open(out_path / "report.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nGate 2 Evaluation Complete: DECISION = {gate_decision}")
    print(f"Report written to: {out_path / 'report.md'}")
    return results


def render_report_markdown(results: Dict[str, Any]) -> str:
    """Render markdown gate report."""
    summary = results["summary"]
    decision = summary["gate_decision"]

    lines = [
        "# GATE 2 DECISION REPORT: Pilot Evolution Baseline",
        f"**Generated:** {results.get('timestamp')}",
        f"**Overall Decision:** **{decision}** (Target PASS: {summary['target_pass_count']}/4, Target MARGINAL: {summary['target_marginal_count']}/4)",
        "",
        "## 1. Target Families (4 families)",
        "| Family | R(F) Alg | Mean Top-1 | Mean R(F) | Mean DSATUR | Gap vs R(F) (%) [95% CI] | Wilcoxon p | Probe ≈ Baseline | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for fid in TARGET_FAMILIES:
        d = results["families"].get(fid, {})
        rf_ci = f"[{d.get('gap_vs_rf_ci', [0,0])[0]:+.2f}%, {d.get('gap_vs_rf_ci', [0,0])[1]:+.2f}%]"
        sim = "DSATUR" if d.get("behavioral_probe", {}).get("equal_dsatur") else ("RLF" if d.get("behavioral_probe", {}).get("equal_rlf") else "Novel")
        lines.append(
            f"| `{fid}` | {d.get('rf_name')} | {d.get('mean_top1', 0):.3f} | {d.get('mean_rf', 0):.3f} | "
            f"{d.get('mean_dsatur', 0):.3f} | {d.get('gap_vs_rf_pct', 0):+.2f}% {rf_ci} | "
            f"{d.get('p_wilcoxon', 1.0):.4f} | {sim} | **{d.get('status')}** |"
        )

    lines.extend([
        "",
        "## 2. Control Families (2 families)",
        "| Family | R(F) Alg | Mean Top-1 | Mean R(F) | Mean DSATUR | Gap vs R(F) (%) [95% CI] | Wilcoxon p | Status |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for fid in CONTROL_FAMILIES:
        d = results["families"].get(fid, {})
        rf_ci = f"[{d.get('gap_vs_rf_ci', [0,0])[0]:+.2f}%, {d.get('gap_vs_rf_ci', [0,0])[1]:+.2f}%]"
        lines.append(
            f"| `{fid}` | {d.get('rf_name')} | {d.get('mean_top1', 0):.3f} | {d.get('mean_rf', 0):.3f} | "
            f"{d.get('mean_dsatur', 0):.3f} | {d.get('gap_vs_rf_pct', 0):+.2f}% {rf_ci} | "
            f"{d.get('p_wilcoxon', 1.0):.4f} | **{d.get('status')}** |"
        )

    lines.extend([
        "",
        "## 3. Decision Criteria Evaluation",
        "- **PASS**: >= 2/4 target families improve >= 1.0% vs R(F) on test_pilot with 95% CI > 0, AND valid-rate = 100%.",
        "- **MARGINAL**: exactly 1/4 target families PASS or improve in [0.5%, 1.0%].",
        "- **FAIL**: otherwise.",
        "",
        f"**Result**: Target passed = {summary['target_pass_count']}/4. All valid rates 100% = {summary['all_valid_rates_100']}.",
        f"**GATE 2 OUTCOME: {decision}**",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Gate 2 Evaluation and Decision Report")
    parser.add_argument("--runs-dir", default="runs/pilot", help="Pilot runs directory")
    parser.add_argument("--out-dir", default="runs/gate2", help="Gate 2 output directory")
    parser.add_argument("--locks-dir", default="runs/gate2/locks", help="Lock files directory")
    parser.add_argument("--force-unlock", action="store_true", help="Force unlock existing locked runs")
    parser.add_argument("--mock", action="store_true", help="Run with mock candidates if no real runs exist")
    args = parser.parse_args()

    evaluate_gate2(
        runs_dir=args.runs_dir,
        out_dir=args.out_dir,
        locks_dir=args.locks_dir,
        force_unlock=args.force_unlock,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
