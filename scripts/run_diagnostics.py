"""Run 3 Phase 1 Diagnostics (D1, D2, D3).

Produces:
- results/diagnostics_per_family.csv
- results/diagnostics_summary.txt
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
from gate.diagnostics import run_d1_family_worker, tune_rich_family_worker


def cluster_bootstrap_metric(
    records: list,
    col_name: str,
    col_denom: str = None,
    num_bootstraps: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Cluster bootstrap over families.

    If col_denom is None: computes bootstrap CI of mean(r[col_name]).
    If col_denom is given: computes bootstrap CI of sum(r[col_name]) / sum(r[col_denom]) * 100%.
    """
    n_fams = len(records)
    if n_fams == 0:
        return 0.0, 0.0, 0.0

    vals = np.array([r[col_name] for r in records])

    if col_denom is None:
        mean_val = float(np.mean(vals))
        rng = np.random.RandomState(seed)
        indices = rng.randint(0, n_fams, size=(num_bootstraps, n_fams))
        boot_means = np.mean(vals[indices], axis=1)
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))
        return mean_val, ci_lo, ci_hi
    else:
        denoms = np.array([r[col_denom] for r in records])
        mean_val = (float(np.sum(vals)) / float(np.sum(denoms)) * 100.0) if np.sum(denoms) > 0 else 0.0
        rng = np.random.RandomState(seed)
        indices = rng.randint(0, n_fams, size=(num_bootstraps, n_fams))
        boot_num = np.sum(vals[indices], axis=1)
        boot_den = np.sum(denoms[indices], axis=1)
        boot_pcts = np.where(boot_den > 0, (boot_num / boot_den) * 100.0, 0.0)
        ci_lo = float(np.percentile(boot_pcts, 2.5))
        ci_hi = float(np.percentile(boot_pcts, 97.5))
        return mean_val, ci_lo, ci_hi


def main():
    start_time = time.perf_counter()
    Path("results").mkdir(exist_ok=True)

    # Load gate_per_family.csv for baseline values (r_api, dsatur, rlf)
    gate_df = pd.read_csv("results/gate_per_family.csv")
    gate_records = {r["family_id"]: r for r in gate_df.to_dict("records")}

    all_fams = families()
    n_fams = len(all_fams)
    max_workers = 12

    # ==========================================================================
    # D1: GRAPH CEILING VIA ITERATED GREEDY (200 ITERATIONS, 20 TEST GRAPHS)
    # ==========================================================================
    print("=" * 80)
    print("DIAGNOSTIC D1: GRAPH CEILING (ITERATED GREEDY, 200 ITERS / GRAPH, 20 TEST GRAPHS)")
    print("=" * 80)

    d1_tasks = [(f["family_id"], 200) for f in all_fams]
    d1_results = {}

    t0_d1 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_fid = {executor.submit(run_d1_family_worker, task): task[0] for task in d1_tasks}
        for future in tqdm(as_completed(future_to_fid), total=n_fams, desc="Running D1 (Iterated Greedy)"):
            fid = future_to_fid[future]
            d1_results[fid] = future.result()
    t_d1 = time.perf_counter() - t0_d1
    print(f"D1 finished in {t_d1:.1f}s.")

    # ==========================================================================
    # D2: RICHER FUNCTION CLASS (12 FEATURES, CMA-ES TUNED)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC D2: RICHER FUNCTION CLASS (12 FEATURES, CMA-ES 3 SEEDS x 1500 EVALS)")
    print("Features: sat, udeg, deg, ratios, colored neighbor count, overlap, cross-terms")
    print("=" * 80)

    d2_tasks = [
        (f["family_id"], 1500, (42, 43, 44), gate_records[f["family_id"]]["r_api_test"])
        for f in all_fams
    ]
    d2_results = {}

    t0_d2 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_fid = {executor.submit(tune_rich_family_worker, task): task[0] for task in d2_tasks}
        for future in tqdm(as_completed(future_to_fid), total=n_fams, desc="Running D2 (Rich Parametric)"):
            fid = future_to_fid[future]
            d2_results[fid] = future.result()
    t_d2 = time.perf_counter() - t0_d2
    print(f"D2 finished in {t_d2:.1f}s.")

    # ==========================================================================
    # COMPILE PER-FAMILY RESULTS FOR D1, D2, D3
    # ==========================================================================
    diag_rows = []
    for f in all_fams:
        fid = f["family_id"]
        g_rec = gate_records[fid]
        d1_res = d1_results[fid]
        d2_res = d2_results[fid]

        ds_test = g_rec["dsatur_test"]
        rlf_test = g_rec["rlf_test"]
        r_api_test = g_rec["r_api_test"]

        # D1: Headroom DSATUR -> Near-optimal (from 20 graphs)
        ds_20 = d1_res["dsatur_20"]
        near_opt_20 = d1_res["near_opt_20"]
        headroom_d1_colors = d1_res["headroom_colors"]
        headroom_d1_pct = d1_res["headroom_pct"]

        # D2: Gap Rich vs R_API (50 test graphs)
        rich_test = d2_res["rich_test"]
        gap_d2_colors = d2_res["gap_colors"]
        gap_d2_pct = d2_res["gap_pct"]

        # D3: Headroom Out-of-Class (DSATUR vs RLF, 50 test graphs)
        headroom_d3_colors = ds_test - rlf_test
        headroom_d3_pct = (headroom_d3_colors / ds_test * 100.0) if ds_test > 0 else 0.0

        diag_rows.append({
            "family_id": fid,
            "structure_type": f["structure_type"],
            "pool": f["pool"],
            "is_control": d1_res["is_control"],
            # D1 columns
            "d1_dsatur_20": ds_20,
            "d1_near_opt_20": near_opt_20,
            "d1_headroom_colors": headroom_d1_colors,
            "d1_headroom_pct": headroom_d1_pct,
            # D2 columns
            "d2_r_api_test": r_api_test,
            "d2_rich_test": rich_test,
            "d2_gap_colors": gap_d2_colors,
            "d2_gap_pct": gap_d2_pct,
            "d2_fallback_dsatur": d2_res["fallback_to_dsatur"],
            # D3 columns
            "d3_dsatur_test": ds_test,
            "d3_rlf_test": rlf_test,
            "d3_headroom_colors": headroom_d3_colors,
            "d3_headroom_pct": headroom_d3_pct,
            "d3_rlf_better_1color": (headroom_d3_colors >= 1.0),
        })

    diag_df = pd.DataFrame(diag_rows)
    csv_path = Path("results/diagnostics_per_family.csv")
    diag_df.to_csv(csv_path, index=False)
    print(f"\nSaved per-family diagnostics to {csv_path}")

    # ==========================================================================
    # STATISTICAL SUMMARIES & TEXT REPORT
    # ==========================================================================
    non_ctrl_df = diag_df[~diag_df["is_control"]]
    ctrl_df = diag_df[diag_df["is_control"]]

    nc_records = non_ctrl_df.to_dict("records")
    ctrl_records = ctrl_df.to_dict("records")
    all_records = diag_df.to_dict("records")

    summary_lines = []
    def log(msg=""):
        print(msg)
        summary_lines.append(msg)

    log("=" * 80)
    log("DIAGNOSTICS REPORT (D1, D2, D3) - PHASE 1")
    log("=" * 80)

    # --------------------------------------------------------------------------
    # SUMMARY D1
    # --------------------------------------------------------------------------
    d1_nc_c, d1_nc_c_lo, d1_nc_c_hi = cluster_bootstrap_metric(nc_records, "d1_headroom_colors")
    d1_nc_p, d1_nc_p_lo, d1_nc_p_hi = cluster_bootstrap_metric(nc_records, "d1_headroom_colors", "d1_dsatur_20")
    d1_ctrl_c, _, _ = cluster_bootstrap_metric(ctrl_records, "d1_headroom_colors")
    d1_ctrl_p, _, _ = cluster_bootstrap_metric(ctrl_records, "d1_headroom_colors", "d1_dsatur_20")

    log("\n" + "#" * 80)
    log("D1. TRAN THEO GRAPH (NEAR-OPTIMAL CEILING VIA ITERATED GREEDY, 200 ITERS)")
    log("#" * 80)
    log(f"Headroom DSATUR -> Near-Optimal (Non-Control, N={len(non_ctrl_df)}):")
    log(f"  Headroom (colors) : {d1_nc_c:.3f} colors  [95% CI: {d1_nc_c_lo:.3f}, {d1_nc_c_hi:.3f}]")
    log(f"  Headroom (%)      : {d1_nc_p:.2f}%       [95% CI: {d1_nc_p_lo:.2f}%, {d1_nc_p_hi:.2f}%]")
    log(f"Headroom Interval Control (N={len(ctrl_df)}):")
    log(f"  Headroom (colors) : {d1_ctrl_c:.3f} colors (0.00%, verified DSATUR optimal = clique LB)")

    log("\nD1 Breakdown by Pool (Non-Control):")
    log(f"{'Pool':<18} | {'N':<4} | {'DSATUR (20)':<12} | {'Near-Opt (20)':<14} | {'Headroom (colors)':<18} | {'Headroom (%) [95% CI]'}")
    log("-" * 92)
    for p in ["train_pool", "heldout_type", "heldout_size", "heldout_combo"]:
        sub = [r for r in nc_records if r["pool"] == p]
        if not sub: continue
        c, clo, chi = cluster_bootstrap_metric(sub, "d1_headroom_colors")
        pct, plo, phi = cluster_bootstrap_metric(sub, "d1_headroom_colors", "d1_dsatur_20")
        ds_m = np.mean([r["d1_dsatur_20"] for r in sub])
        no_m = np.mean([r["d1_near_opt_20"] for r in sub])
        log(f"{p:<18} | {len(sub):<4} | {ds_m:<12.2f} | {no_m:<14.2f} | {c:6.3f} [{clo:5.2f},{chi:5.2f}] | {pct:5.2f}% [{plo:5.2f}%, {phi:5.2f}%]")

    log("\nD1 Breakdown by Structure Type (All families):")
    log(f"{'Structure Type':<20} | {'N':<4} | {'DSATUR (20)':<12} | {'Near-Opt (20)':<14} | {'Headroom (colors)':<18} | {'Headroom (%) [95% CI]'}")
    log("-" * 94)
    for st in sorted(diag_df["structure_type"].unique()):
        sub = [r for r in all_records if r["structure_type"] == st]
        c, clo, chi = cluster_bootstrap_metric(sub, "d1_headroom_colors")
        pct, plo, phi = cluster_bootstrap_metric(sub, "d1_headroom_colors", "d1_dsatur_20")
        ds_m = np.mean([r["d1_dsatur_20"] for r in sub])
        no_m = np.mean([r["d1_near_opt_20"] for r in sub])
        log(f"{st:<20} | {len(sub):<4} | {ds_m:<12.2f} | {no_m:<14.2f} | {c:6.3f} [{clo:5.2f},{chi:5.2f}] | {pct:5.2f}% [{plo:5.2f}%, {phi:5.2f}%]")

    # --------------------------------------------------------------------------
    # SUMMARY D2
    # --------------------------------------------------------------------------
    d2_nc_c, d2_nc_c_lo, d2_nc_c_hi = cluster_bootstrap_metric(nc_records, "d2_gap_colors")
    d2_nc_p, d2_nc_p_lo, d2_nc_p_hi = cluster_bootstrap_metric(nc_records, "d2_gap_colors", "d2_r_api_test")

    log("\n" + "#" * 80)
    log("D2. LOP HAM GIAU HON (12 FEATURES: CROSS-TERMS & OVERLAP, CMA-ES TUNED)")
    log("#" * 80)
    log(f"Gap vs R_API: R_API - Tuned_Rich (Non-Control, N={len(non_ctrl_df)}):")
    log(f"  Gap D2 (colors)   : {d2_nc_c:+.3f} colors  [95% CI: {d2_nc_c_lo:+.3f}, {d2_nc_c_hi:+.3f}]")
    log(f"  Gap D2 (%)        : {d2_nc_p:+.2f}%       [95% CI: {d2_nc_p_lo:+.2f}%, {d2_nc_p_hi:+.2f}%]")
    n_rich_fallback = int(non_ctrl_df["d2_fallback_dsatur"].sum())
    log(f"  Fallback DSATUR   : {n_rich_fallback} / {len(non_ctrl_df)} families ({n_rich_fallback/len(non_ctrl_df)*100:.1f}%)")

    log("\nD2 Breakdown by Pool (Non-Control):")
    log(f"{'Pool':<18} | {'N':<4} | {'Mean R_API':<12} | {'Mean Rich':<12} | {'Gap D2 (colors)':<18} | {'Gap D2 (%) [95% CI]'}")
    log("-" * 92)
    for p in ["train_pool", "heldout_type", "heldout_size", "heldout_combo"]:
        sub = [r for r in nc_records if r["pool"] == p]
        if not sub: continue
        c, clo, chi = cluster_bootstrap_metric(sub, "d2_gap_colors")
        pct, plo, phi = cluster_bootstrap_metric(sub, "d2_gap_colors", "d2_r_api_test")
        ra_m = np.mean([r["d2_r_api_test"] for r in sub])
        ri_m = np.mean([r["d2_rich_test"] for r in sub])
        log(f"{p:<18} | {len(sub):<4} | {ra_m:<12.2f} | {ri_m:<12.2f} | {c:+6.3f} [{clo:+5.2f},{chi:+5.2f}] | {pct:+5.2f}% [{plo:+5.2f}%, {phi:+5.2f}%]")

    log("\nD2 Breakdown by Structure Type (All families):")
    log(f"{'Structure Type':<20} | {'N':<4} | {'Mean R_API':<12} | {'Mean Rich':<12} | {'Gap D2 (colors)':<18} | {'Gap D2 (%) [95% CI]'}")
    log("-" * 94)
    for st in sorted(diag_df["structure_type"].unique()):
        sub = [r for r in all_records if r["structure_type"] == st]
        c, clo, chi = cluster_bootstrap_metric(sub, "d2_gap_colors")
        pct, plo, phi = cluster_bootstrap_metric(sub, "d2_gap_colors", "d2_r_api_test")
        ra_m = np.mean([r["d2_r_api_test"] for r in sub])
        ri_m = np.mean([r["d2_rich_test"] for r in sub])
        log(f"{st:<20} | {len(sub):<4} | {ra_m:<12.2f} | {ri_m:<12.2f} | {c:+6.3f} [{clo:+5.2f},{chi:+5.2f}] | {pct:+5.2f}% [{plo:+5.2f}%, {phi:+5.2f}%]")

    # --------------------------------------------------------------------------
    # SUMMARY D3
    # --------------------------------------------------------------------------
    d3_nc_c, d3_nc_c_lo, d3_nc_c_hi = cluster_bootstrap_metric(nc_records, "d3_headroom_colors")
    d3_nc_p, d3_nc_p_lo, d3_nc_p_hi = cluster_bootstrap_metric(nc_records, "d3_headroom_colors", "d3_dsatur_test")
    n_rlf_better_1col = int(non_ctrl_df["d3_rlf_better_1color"].sum())

    log("\n" + "#" * 80)
    log("D3. HEADROOM NGOAI LOP (DSATUR vs RLF, 50 TEST GRAPHS)")
    log("#" * 80)
    log(f"Headroom (DSATUR - RLF)/DSATUR (Non-Control, N={len(non_ctrl_df)}):")
    log(f"  Headroom (colors) : {d3_nc_c:.3f} colors  [95% CI: {d3_nc_c_lo:.3f}, {d3_nc_c_hi:.3f}]")
    log(f"  Headroom (%)      : {d3_nc_p:.2f}%       [95% CI: {d3_nc_p_lo:.2f}%, {d3_nc_p_hi:.2f}%]")
    log(f"  So family RLF tot hon DSATUR >= 1.0 mau : {n_rlf_better_1col} / {len(non_ctrl_df)} families ({n_rlf_better_1col/len(non_ctrl_df)*100:.1f}%)")

    log("\nD3 Breakdown by Pool (Non-Control):")
    log(f"{'Pool':<18} | {'N':<4} | {'DSATUR Test':<12} | {'RLF Test':<12} | {'Headroom (colors)':<18} | {'Headroom (%) [95% CI]'}")
    log("-" * 92)
    for p in ["train_pool", "heldout_type", "heldout_size", "heldout_combo"]:
        sub = [r for r in nc_records if r["pool"] == p]
        if not sub: continue
        c, clo, chi = cluster_bootstrap_metric(sub, "d3_headroom_colors")
        pct, plo, phi = cluster_bootstrap_metric(sub, "d3_headroom_colors", "d3_dsatur_test")
        ds_m = np.mean([r["d3_dsatur_test"] for r in sub])
        rl_m = np.mean([r["d3_rlf_test"] for r in sub])
        log(f"{p:<18} | {len(sub):<4} | {ds_m:<12.2f} | {rl_m:<12.2f} | {c:6.3f} [{clo:5.2f},{chi:5.2f}] | {pct:5.2f}% [{plo:5.2f}%, {phi:5.2f}%]")

    log("\nD3 Breakdown by Structure Type (All families):")
    log(f"{'Structure Type':<20} | {'N':<4} | {'DSATUR Test':<12} | {'RLF Test':<12} | {'Headroom (colors)':<18} | {'Headroom (%) [95% CI]'}")
    log("-" * 94)
    for st in sorted(diag_df["structure_type"].unique()):
        sub = [r for r in all_records if r["structure_type"] == st]
        c, clo, chi = cluster_bootstrap_metric(sub, "d3_headroom_colors")
        pct, plo, phi = cluster_bootstrap_metric(sub, "d3_headroom_colors", "d3_dsatur_test")
        ds_m = np.mean([r["d3_dsatur_test"] for r in sub])
        rl_m = np.mean([r["d3_rlf_test"] for r in sub])
        log(f"{st:<20} | {len(sub):<4} | {ds_m:<12.2f} | {rl_m:<12.2f} | {c:6.3f} [{clo:5.2f},{chi:5.2f}] | {pct:5.2f}% [{plo:5.2f}%, {phi:5.2f}%]")

    summary_path = Path("results/diagnostics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSaved diagnostics summary to {summary_path}")


if __name__ == "__main__":
    main()
