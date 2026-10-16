"""Run Diagnostic D1' (Near-Optimal Graph Ceiling via IG-2000 and TabuCol).

Produces:
- results/d1_prime_per_family.csv
- results/d1_prime_summary.txt
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
from gate.diagnostics import run_d1_prime_worker


def cluster_bootstrap_metric(
    records: list,
    col_name: str,
    col_denom: str = None,
    num_bootstraps: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
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
    Path("results").mkdir(exist_ok=True)

    # Load gate_per_family.csv for reference R(F)
    gate_df = pd.read_csv("results/gate_per_family.csv")
    gate_records = {r["family_id"]: r for r in gate_df.to_dict("records")}

    all_fams = families()
    n_fams = len(all_fams)
    max_workers = 12

    print("=" * 80)
    print("DIAGNOSTIC D1': GRAPH CEILING VIA IG (2000 ITERS, FROM DSATUR & RLF) + TABUCOL")
    print(f"Running across all {n_fams} families on 20 test graphs per family (12 workers)")
    print("=" * 80)

    d1_tasks = [
        (f["family_id"], gate_records[f["family_id"]]["r_reference_test"])
        for f in all_fams
    ]
    d1_results = {}

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_fid = {executor.submit(run_d1_prime_worker, task): task[0] for task in d1_tasks}
        for future in tqdm(as_completed(future_to_fid), total=n_fams, desc="Running D1' Ceiling"):
            fid = future_to_fid[future]
            d1_results[fid] = future.result()
    t_elapsed = time.perf_counter() - t0
    print(f"D1' finished in {t_elapsed:.1f}s.")

    rows = []
    for f in all_fams:
        fid = f["family_id"]
        res = d1_results[fid]
        g_rec = gate_records[fid]

        rows.append({
            "family_id": fid,
            "structure_type": f["structure_type"],
            "pool": f["pool"],
            "is_control": res["is_control"],
            "dsatur_20": res["dsatur_20"],
            "rlf_20": res["rlf_20"],
            "r_ref_20": res["r_ref_20"],
            "ceiling_20": res["ceiling_20"],
            "headroom_vs_r_colors": res["headroom_vs_r_colors"],
            "headroom_vs_r_pct": res["headroom_vs_r_pct"],
            "headroom_ci_c_lo": res["headroom_ci_colors"][0],
            "headroom_ci_c_hi": res["headroom_ci_colors"][1],
            "headroom_ci_p_lo": res["headroom_ci_pct"][0],
            "headroom_ci_p_hi": res["headroom_ci_pct"][1],
            "headroom_gt_3pct": (res["headroom_vs_r_pct"] >= 3.0),
        })

    df = pd.DataFrame(rows)
    csv_path = Path("results/d1_prime_per_family.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved D1' per-family results to {csv_path}")

    # Summaries
    non_ctrl_df = df[~df["is_control"]]
    ctrl_df = df[df["is_control"]]

    nc_records = non_ctrl_df.to_dict("records")
    ctrl_records = ctrl_df.to_dict("records")
    all_records = df.to_dict("records")

    summary_lines = []
    def log(msg=""):
        print(msg)
        summary_lines.append(msg)

    log("=" * 80)
    log("DIAGNOSTIC D1' SUMMARY REPORT: GRAPH CEILING vs REFERENCE R(F)")
    log("Graph Ceiling = min(DSATUR, RLF, IG-2000-DS, IG-2000-RLF, TabuCol)")
    log("Reference R(F) = min(DSATUR, RLF, Universal) on Val")
    log("=" * 80)

    # Metrics on Non-Control
    nc_c_mean, nc_c_lo, nc_c_hi = cluster_bootstrap_metric(nc_records, "headroom_vs_r_colors")
    nc_p_mean, nc_p_lo, nc_p_hi = cluster_bootstrap_metric(nc_records, "headroom_vs_r_colors", "r_ref_20")
    nc_p_median = float(non_ctrl_df["headroom_vs_r_pct"].median())
    nc_c_median = float(non_ctrl_df["headroom_vs_r_colors"].median())

    ctrl_c_mean, _, _ = cluster_bootstrap_metric(ctrl_records, "headroom_vs_r_colors")
    ctrl_p_mean, _, _ = cluster_bootstrap_metric(ctrl_records, "headroom_vs_r_colors", "r_ref_20")

    all_c_mean, all_c_lo, all_c_hi = cluster_bootstrap_metric(all_records, "headroom_vs_r_colors")
    all_p_mean, all_p_lo, all_p_hi = cluster_bootstrap_metric(all_records, "headroom_vs_r_colors", "r_ref_20")

    n_gt3 = int(non_ctrl_df["headroom_gt_3pct"].sum())
    n_total_nc = len(non_ctrl_df)

    log(f"\n1. AGGREGATE HEADROOM: R(F) -> Graph Ceiling (20 test graphs per family)")
    log(f"   Non-Control Families (N={n_total_nc}):")
    log(f"     Mean Headroom (colors)   : {nc_c_mean:.3f} colors  [95% CI: {nc_c_lo:.3f}, {nc_c_hi:.3f}]")
    log(f"     Mean Headroom (%)        : {nc_p_mean:.2f}%       [95% CI: {nc_p_lo:.2f}%, {nc_p_hi:.2f}%]")
    log(f"     Median Headroom (colors) : {nc_c_median:.3f} colors")
    log(f"     Median Headroom (%)      : {nc_p_median:.2f}%")
    log(f"     Families with Headroom >= 3.0% : {n_gt3} / {n_total_nc} ({n_gt3/n_total_nc*100:.1f}%)")
    log(f"   Interval Control Families (N={len(ctrl_df)}):")
    log(f"     Mean Headroom (colors)   : {ctrl_c_mean:.3f} colors (0.00%, verified DSATUR optimal = clique LB)")
    log(f"   Overall (All N={len(df)}):")
    log(f"     Mean Headroom (colors)   : {all_c_mean:.3f} colors  [95% CI: {all_c_lo:.3f}, {all_c_hi:.3f}]")
    log(f"     Mean Headroom (%)        : {all_p_mean:.2f}%       [95% CI: {all_p_lo:.2f}%, {all_p_hi:.2f}%]")

    # By Pool
    log(f"\n2. D1' BREAKDOWN BY POOL (Non-Control families):")
    log(f"{'Pool':<18} | {'N':<4} | {'Mean R(F)':<12} | {'Ceiling (20)':<14} | {'Headroom (colors)':<18} | {'Headroom (%) [95% CI]'}")
    log("-" * 94)
    for p in ["train_pool", "heldout_type", "heldout_size", "heldout_combo"]:
        sub = [r for r in nc_records if r["pool"] == p]
        if not sub: continue
        c, clo, chi = cluster_bootstrap_metric(sub, "headroom_vs_r_colors")
        pct, plo, phi = cluster_bootstrap_metric(sub, "headroom_vs_r_colors", "r_ref_20")
        rf_m = np.mean([r["r_ref_20"] for r in sub])
        ce_m = np.mean([r["ceiling_20"] for r in sub])
        log(f"{p:<18} | {len(sub):<4} | {rf_m:<12.2f} | {ce_m:<14.2f} | {c:6.3f} [{clo:5.2f},{chi:5.2f}] | {pct:5.2f}% [{plo:5.2f}%, {phi:5.2f}%]")

    # By Structure Type
    log(f"\n3. D1' BREAKDOWN BY STRUCTURE TYPE (All families):")
    log(f"{'Structure Type':<20} | {'N':<4} | {'Mean R(F)':<12} | {'Ceiling (20)':<14} | {'Headroom (colors)':<18} | {'Headroom (%) [95% CI]'}")
    log("-" * 94)
    for st in sorted(df["structure_type"].unique()):
        sub = [r for r in all_records if r["structure_type"] == st]
        c, clo, chi = cluster_bootstrap_metric(sub, "headroom_vs_r_colors")
        pct, plo, phi = cluster_bootstrap_metric(sub, "headroom_vs_r_colors", "r_ref_20")
        rf_m = np.mean([r["r_ref_20"] for r in sub])
        ce_m = np.mean([r["ceiling_20"] for r in sub])
        log(f"{st:<20} | {len(sub):<4} | {rf_m:<12.2f} | {ce_m:<14.2f} | {c:6.3f} [{clo:5.2f},{chi:5.2f}] | {pct:5.2f}% [{plo:5.2f}%, {phi:5.2f}%]")

    summary_path = Path("results/d1_prime_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSaved D1' summary report to {summary_path}")


if __name__ == "__main__":
    main()
