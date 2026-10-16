"""Run Phase 1 Oracle-Gap Gate on frozen dataset v1.0.

Reference R(F) = best of {DSATUR, RLF, Universal} on VAL.

Produces:
- results/gate_per_family.csv
- results/gate_summary.txt
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from gate.dataset import families
from gate.tune import tune_universal, tune_family_worker


def cluster_bootstrap_gap(
    records: list,
    col_ref: str,
    col_target: str,
    num_bootstraps: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float, float, float, float]:
    """Cluster bootstrap over families (resample families, then compute mean).

    Returns:
    (mean_diff, diff_ci_lo, diff_ci_hi, mean_pct, pct_ci_lo, pct_ci_hi)
    """
    n_fams = len(records)
    if n_fams == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    diffs = np.array([r[col_ref] - r[col_target] for r in records])
    refs = np.array([r[col_ref] for r in records])

    mean_diff = float(np.mean(diffs))
    mean_ref = float(np.mean(refs))
    mean_pct = (mean_diff / mean_ref * 100.0) if mean_ref > 0 else 0.0

    rng = np.random.RandomState(seed)
    indices = rng.randint(0, n_fams, size=(num_bootstraps, n_fams))

    boot_diffs = np.mean(diffs[indices], axis=1)
    boot_refs = np.mean(refs[indices], axis=1)
    boot_pcts = np.where(boot_refs > 0, (boot_diffs / boot_refs) * 100.0, 0.0)

    diff_ci_lo = float(np.percentile(boot_diffs, 2.5))
    diff_ci_hi = float(np.percentile(boot_diffs, 97.5))
    pct_ci_lo = float(np.percentile(boot_pcts, 2.5))
    pct_ci_hi = float(np.percentile(boot_pcts, 97.5))

    return mean_diff, diff_ci_lo, diff_ci_hi, mean_pct, pct_ci_lo, pct_ci_hi


def main():
    Path("results").mkdir(exist_ok=True)

    all_fams = families()
    fam_metadata = {f["family_id"]: f for f in all_fams}
    print(f"Loaded {len(all_fams)} families from manifest.")

    # Step 1: Universal Heuristic Tuning
    print("\n" + "=" * 80)
    print("STEP 1: TUNING UNIVERSAL PARAMETRIC HEURISTIC")
    print("=" * 80)
    t0_univ = time.perf_counter()
    univ_weights, univ_info = tune_universal(budget=3000, seeds=(42, 43, 44), verbose=True)
    univ_time = time.perf_counter() - t0_univ
    print(f"Universal tuning completed in {univ_time:.1f}s.")

    # Step 2: Per-Family Tuning in Parallel
    print("\n" + "=" * 80)
    print(f"STEP 2: PER-FAMILY TUNING ACROSS {len(all_fams)} FAMILIES")
    print("Reference R(F) = min(DSATUR, RLF, Universal) chosen on Val")
    print("Budget: 3000 evaluations per seed | 3 seeds (42, 43, 44) | 12 Parallel Workers")
    print("=" * 80)

    tasks = [
        (f["family_id"], 3000, (42, 43, 44), univ_weights.tolist())
        for f in all_fams
    ]

    results_dict = {}
    t0_tune = time.perf_counter()
    max_workers = 12

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_fid = {
            executor.submit(tune_family_worker, task): task[0]
            for task in tasks
        }
        for future in tqdm(as_completed(future_to_fid), total=len(tasks), desc="Tuning families"):
            fid = future_to_fid[future]
            try:
                res = future.result()
                results_dict[fid] = res
            except Exception as e:
                print(f"Error tuning family {fid}: {e}")
                raise e

    tune_time = time.perf_counter() - t0_tune
    print(f"All 106 families tuned in {tune_time:.1f}s ({tune_time/len(all_fams):.2f}s/family across {max_workers} workers).")

    # Step 3: Build DataFrame & Save gate_per_family.csv
    rows = []
    for f in all_fams:
        fid = f["family_id"]
        res = results_dict[fid]
        meta = fam_metadata[fid]

        is_ctrl = res["is_control"]
        r_ref = res["r_reference"]
        tuned = res["tuned"]
        g_ds = res["gains_vs_dsatur"]
        g_a = res["gap_a"]

        rows.append({
            "family_id": fid,
            "structure_type": meta["structure_type"],
            "pool": meta["pool"],
            "is_control": is_ctrl,
            "dsatur_train": res["dsatur"]["train"],
            "dsatur_val": res["dsatur"]["val"],
            "dsatur_test": res["dsatur"]["test"],
            "rlf_train": res["rlf"]["train"],
            "rlf_val": res["rlf"]["val"],
            "rlf_test": res["rlf"]["test"],
            "universal_train": res["universal"]["train"],
            "universal_val": res["universal"]["val"],
            "universal_test": res["universal"]["test"],
            "r_reference_name": r_ref["name"],
            "r_reference_val": r_ref["val"],
            "r_reference_test": r_ref["test"],
            "tuned_weights": str([round(w, 4) for w in res["chosen_weights"]]),
            "tuned_fallback_dsatur": res["fallback_to_dsatur"],
            "tuned_train": tuned["train"],
            "tuned_val": tuned["val"],
            "tuned_test": tuned["test"],
            "gain_train_vs_ds": g_ds["train"],
            "gain_val_vs_ds": g_ds["val"],
            "gain_test_vs_ds": g_ds["test"],
            "gain_test_vs_ds_pct": g_ds["test_pct"],
            "gain_test_vs_ds_ci_lo": g_ds["test_ci"][0],
            "gain_test_vs_ds_ci_hi": g_ds["test_ci"][1],
            "gap_a_colors": g_a["colors"],
            "gap_a_pct": g_a["pct"],
            "gap_a_ci_lo": g_a["ci_colors"][0],
            "gap_a_ci_hi": g_a["ci_colors"][1],
            "gap_a_pct_ci_lo": g_a["ci_pct"][0],
            "gap_a_pct_ci_hi": g_a["ci_pct"][1],
        })

    df = pd.DataFrame(rows)
    csv_path = Path("results/gate_per_family.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved updated per-family gate results to {csv_path}")

    # Step 4: Statistical Summaries & gate_summary.txt
    summary_lines = []
    def log(msg=""):
        print(msg)
        summary_lines.append(msg)

    log("=" * 80)
    log("ORACLE-GAP GATE SUMMARY REPORT (PHASE 1 - FROZEN DATASET v1.0)")
    log("Reference R(F) = min(DSATUR, RLF, Universal) chosen on Val (all within API class)")
    log("Per-Family P(F) = CMA-ES tuned parametric heuristic (with DSATUR fallback)")
    log("=" * 80)

    # Subsets
    non_control_df = df[~df["is_control"]]
    control_df = df[df["is_control"]]

    # Overall & Non-Control cluster bootstrap for Gap A: R(F) - P(F)
    overall_gap, o_lo, o_hi, overall_pct, op_lo, op_hi = cluster_bootstrap_gap(
        df.to_dict("records"), "r_reference_test", "tuned_test"
    )
    non_ctrl_gap, nc_lo, nc_hi, non_ctrl_pct, ncp_lo, ncp_hi = cluster_bootstrap_gap(
        non_control_df.to_dict("records"), "r_reference_test", "tuned_test"
    )
    ctrl_gap, c_lo, c_hi, ctrl_pct, cp_lo, cp_hi = cluster_bootstrap_gap(
        control_df.to_dict("records"), "r_reference_test", "tuned_test"
    )

    # Vs DSATUR
    nc_ds_gap, ncds_lo, ncds_hi, nc_ds_pct, ncdsp_lo, ncdsp_hi = cluster_bootstrap_gap(
        non_control_df.to_dict("records"), "dsatur_test", "tuned_test"
    )
    all_ds_gap, allds_lo, allds_hi, all_ds_pct, alldsp_lo, alldsp_hi = cluster_bootstrap_gap(
        df.to_dict("records"), "dsatur_test", "tuned_test"
    )

    log(f"\n1. AGGREGATE GAP A: R(F) - P(F) (Cluster Bootstrap 95% CI over families)")
    log(f"   Non-Control Families (N={len(non_control_df)}):")
    log(f"     Gap A vs R(F) (colors) : {non_ctrl_gap:+.3f} colors  [95% CI: {nc_lo:+.3f}, {nc_hi:+.3f}]")
    log(f"     Gap A vs R(F) (%)      : {non_ctrl_pct:+.2f}%       [95% CI: {ncp_lo:+.2f}%, {ncp_hi:+.2f}%]")
    log(f"     Gain vs DSATUR (colors): {nc_ds_gap:+.3f} colors  [95% CI: {ncds_lo:+.3f}, {ncds_hi:+.3f}]")
    log(f"     Gain vs DSATUR (%)     : {nc_ds_pct:+.2f}%       [95% CI: {ncdsp_lo:+.2f}%, {ncdsp_hi:+.2f}%]")
    log(f"   Interval Control Families (N={len(control_df)}):")
    log(f"     Gap A vs R(F) (colors) : {ctrl_gap:+.3f} colors  [95% CI: {c_lo:+.3f}, {c_hi:+.3f}]")
    log(f"     Gap A vs R(F) (%)      : {ctrl_pct:+.2f}%       [95% CI: {cp_lo:+.2f}%, {cp_hi:+.2f}%]")
    log(f"   Overall (All N={len(df)}):")
    log(f"     Gap A vs R(F) (colors) : {overall_gap:+.3f} colors  [95% CI: {o_lo:+.3f}, {o_hi:+.3f}]")
    log(f"     Gap A vs R(F) (%)      : {overall_pct:+.2f}%       [95% CI: {op_lo:+.2f}%, {op_hi:+.2f}%]")
    log(f"     Gain vs DSATUR (colors): {all_ds_gap:+.3f} colors  [95% CI: {allds_lo:+.3f}, {allds_hi:+.3f}]")
    log(f"     Gain vs DSATUR (%)     : {all_ds_pct:+.2f}%       [95% CI: {alldsp_lo:+.2f}%, {alldsp_hi:+.2f}%]")

    # R(F) breakdown: how often was each heuristic chosen by VAL?
    r_counts = non_control_df["r_reference_name"].value_counts().to_dict()
    log(f"\n   Reference Portfolio Selection on Val (Non-Control, N={len(non_control_df)}):")
    for r_k, r_v in r_counts.items():
        log(f"     {r_k:<12}: {r_v:2d} / {len(non_control_df)} ({r_v/len(non_control_df)*100:.1f}%)")

    # By Pool
    log(f"\n2. BREAKDOWN BY POOL (Non-Control families):")
    log(f"{'Pool':<18} | {'N':<4} | {'Mean R(F)':<12} | {'Mean Tuned':<12} | {'Gap A vs R(F) [95% CI]':<26} | {'Gain vs DSATUR [95% CI]'}")
    log("-" * 105)
    for p in ["train_pool", "heldout_type", "heldout_size", "heldout_combo"]:
        sub = non_control_df[non_control_df["pool"] == p]
        if len(sub) == 0:
            continue
        g, glo, ghi, gp, gplo, gphi = cluster_bootstrap_gap(sub.to_dict("records"), "r_reference_test", "tuned_test")
        d_g, d_glo, d_ghi, d_gp, d_gplo, d_gphi = cluster_bootstrap_gap(sub.to_dict("records"), "dsatur_test", "tuned_test")
        log(f"{p:<18} | {len(sub):<4} | {sub['r_reference_test'].mean():<12.2f} | {sub['tuned_test'].mean():<12.2f} | {gp:+5.2f}% [{gplo:+5.2f}%, {gphi:+5.2f}%] | {d_gp:+5.2f}% [{d_gplo:+5.2f}%, {d_gphi:+5.2f}%]")

    # By Structure Type
    log(f"\n3. BREAKDOWN BY STRUCTURE TYPE (All families):")
    log(f"{'Structure Type':<20} | {'N':<4} | {'DSATUR':<8} | {'RLF':<8} | {'Mean R(F)':<10} | {'Tuned':<8} | {'Gap A vs R(F) (%)':<20} | {'Gain vs DS (%)'}")
    log("-" * 105)
    for st in sorted(df["structure_type"].unique()):
        sub = df[df["structure_type"] == st]
        g, glo, ghi, gp, gplo, gphi = cluster_bootstrap_gap(sub.to_dict("records"), "r_reference_test", "tuned_test")
        d_g, d_glo, d_ghi, d_gp, d_gplo, d_gphi = cluster_bootstrap_gap(sub.to_dict("records"), "dsatur_test", "tuned_test")
        log(f"{st:<20} | {len(sub):<4} | {sub['dsatur_test'].mean():<8.2f} | {sub['rlf_test'].mean():<8.2f} | {sub['r_reference_test'].mean():<10.2f} | {sub['tuned_test'].mean():<8.2f} | {gp:+6.2f}% [{gplo:+5.2f}%,{gphi:+5.2f}%] | {d_gp:+5.2f}%")

    # Final summary block
    log("\n" + "=" * 80)
    log(f"GATE SUMMARY: avg Gap A vs R(F) = {non_ctrl_pct:+.2f}% [95% CI: {ncp_lo:+.2f}%, {ncp_hi:+.2f}%] (Non-Control, N={len(non_control_df)})")
    log(f"              avg Gain vs DS    = {nc_ds_pct:+.2f}% [95% CI: {ncdsp_lo:+.2f}%, {ncdsp_hi:+.2f}%] (Non-Control, N={len(non_control_df)})")
    log(f"              Overall Gap A     = {overall_pct:+.2f}% [95% CI: {op_lo:+.2f}%, {op_hi:+.2f}%] (Overall, N={len(df)})")
    log("=" * 80)

    summary_path = Path("results/gate_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSaved updated summary report to {summary_path}")


if __name__ == "__main__":
    main()
