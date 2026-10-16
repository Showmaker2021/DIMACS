"""Script to run Phase 1 Diagnostics (D1' on VAL, D4 on VAL, D5 on VAL).

All diagnostics are evaluated on the 20 VALIDATION graphs per family (seeds 1000..1019).
No test graphs are used for design decisions.

Produces:
- results/d1_prime_val.csv
- results/d4_parametric_rlf_val.csv
- results/d5_constructive_ceiling_val.csv
- results/val_diagnostics_summary.txt
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
from gate.diagnostics import run_d1_val_worker, run_d4_worker, run_d5_worker


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


def format_breakdown_table(
    records: list,
    metric_colors_col: str,
    metric_pct_col: str,
    title: str,
) -> str:
    lines = []
    lines.append(f"\n--- {title} BY POOL (Non-Control, N=99) ---")
    lines.append(f"{'Pool':<18} | {'N':<4} | {'Mean Colors':<12} | {'Headroom/Gap (%) [95% CI]':<26}")
    lines.append("-" * 68)

    pools = ["train_pool", "heldout_type", "heldout_size", "heldout_combo"]
    for p in pools:
        p_recs = [r for r in records if r["pool"] == p and not r["is_control"]]
        if not p_recs:
            continue
        m_col = float(np.mean([r[metric_colors_col] for r in p_recs]))
        m_pct, p_lo, p_hi = cluster_bootstrap_metric(p_recs, metric_colors_col, "r_ref_val")
        lines.append(f"{p:<18} | {len(p_recs):<4} | {m_col:<12.3f} | {m_pct:>6.2f}% [{p_lo:>5.2f}%, {p_hi:>5.2f}%]")

    lines.append(f"\n--- {title} BY STRUCTURE TYPE (All families, N=106) ---")
    lines.append(f"{'Structure Type':<20} | {'N':<4} | {'Mean Colors':<12} | {'Headroom/Gap (%) [95% CI]':<26}")
    lines.append("-" * 70)

    stypes = sorted(list(set(r["structure_type"] for r in records)))
    for st in stypes:
        st_recs = [r for r in records if r["structure_type"] == st]
        m_col = float(np.mean([r[metric_colors_col] for r in st_recs]))
        m_pct, p_lo, p_hi = cluster_bootstrap_metric(st_recs, metric_colors_col, "r_ref_val")
        lines.append(f"{st:<20} | {len(st_recs):<4} | {m_col:<12.3f} | {m_pct:>6.2f}% [{p_lo:>5.2f}%, {p_hi:>5.2f}%]")

    return "\n".join(lines)


def main():
    Path("results").mkdir(exist_ok=True)

    # Load gate_per_family.csv to get r_reference_name for each family
    gate_df = pd.read_csv("results/gate_per_family.csv")
    r_name_map = {r["family_id"]: r["r_reference_name"] for r in gate_df.to_dict("records")}

    all_fams = families()
    n_fams = len(all_fams)
    max_workers = 12

    print("=" * 80)
    print("PHASE 1 DIAGNOSTICS SUITE ON VALIDATION GRAPHS (VAL SPLIT: 20 GRAPHS/FAMILY)")
    print(f"Total families: {n_fams} | Parallel workers: {max_workers}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. DIAGNOSTIC D1': GRAPH CEILING ON VAL
    # --------------------------------------------------------------------------
    d1_csv_path = Path("results/d1_prime_val.csv")
    if d1_csv_path.exists() and len(pd.read_csv(d1_csv_path)) == n_fams:
        print("\n[Step 1] Loading existing D1' results from results/d1_prime_val.csv (106 families).")
        d1_df = pd.read_csv(d1_csv_path)
        d1_rows = d1_df.to_dict("records")
    else:
        print("\n" + "=" * 80)
        print("STEP 1: RUNNING D1' GRAPH CEILING (IG-2000-DS + IG-2000-RLF + TabuCol) ON VAL")
        print("=" * 80)
        t0 = time.perf_counter()
        d1_tasks = [(f["family_id"], r_name_map[f["family_id"]]) for f in all_fams]
        d1_results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(run_d1_val_worker, t): t[0] for t in d1_tasks}
            for fut in tqdm(as_completed(future_map), total=n_fams, desc="D1' on VAL"):
                fid = future_map[fut]
                d1_results[fid] = fut.result()
        print(f"D1' on VAL completed in {time.perf_counter() - t0:.1f}s.")

        d1_rows = []
        for f in all_fams:
            fid = f["family_id"]
            res = d1_results[fid]
            d1_rows.append({
                "family_id": fid,
                "structure_type": f["structure_type"],
                "pool": f["pool"],
                "is_control": res["is_control"],
                "dsatur_val": res["dsatur_val"],
                "rlf_val": res["rlf_val"],
                "r_ref_val": res["r_ref_val"],
                "ceiling_val": res["ceiling_val"],
                "headroom_vs_r_colors": res["headroom_vs_r_colors"],
                "headroom_vs_r_pct": res["headroom_vs_r_pct"],
                "headroom_ci_colors_lo": res["headroom_ci_colors"][0],
                "headroom_ci_colors_hi": res["headroom_ci_colors"][1],
                "headroom_ci_pct_lo": res["headroom_ci_pct"][0],
                "headroom_ci_pct_hi": res["headroom_ci_pct"][1],
            })
        d1_df = pd.DataFrame(d1_rows)
        d1_df.to_csv("results/d1_prime_val.csv", index=False)

    # --------------------------------------------------------------------------
    # 2. DIAGNOSTIC D4: PARAMETERIZED RLF WITH CMA-ES ON VAL
    # --------------------------------------------------------------------------
    d4_csv_path = Path("results/d4_parametric_rlf_val.csv")
    d4_existing = {}
    if d4_csv_path.exists():
        try:
            prev_df = pd.read_csv(d4_csv_path)
            for r in prev_df.to_dict("records"):
                d4_existing[r["family_id"]] = r
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("STEP 2: RUNNING D4 (PARAMETERIZED RLF TUNED VIA CMA-ES + VAL SELECTION & FALLBACK)")
    print(f"Already completed: {len(d4_existing)} / {n_fams}")
    print("Budget: 400 evaluations x 3 seeds (42, 43, 44)")
    print("=" * 80)

    remaining_fams = [f for f in all_fams if f["family_id"] not in d4_existing]
    if remaining_fams:
        t0 = time.perf_counter()
        d4_tasks = [(f["family_id"], r_name_map[f["family_id"]], 400, (42, 43, 44)) for f in remaining_fams]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(run_d4_worker, t): t[0] for t in d4_tasks}
            for fut in tqdm(as_completed(future_map), total=len(remaining_fams), desc="D4 Parametric RLF"):
                fid = future_map[fut]
                res = fut.result()
                f_meta = next(f for f in all_fams if f["family_id"] == fid)
                d4_existing[fid] = {
                    "family_id": fid,
                    "structure_type": f_meta["structure_type"],
                    "pool": f_meta["pool"],
                    "is_control": res["is_control"],
                    "r_ref_name": res["r_ref_name"],
                    "r_ref_val": res["r_ref_val"],
                    "tuned_rlf_val": res["tuned_rlf_val"],
                    "fallback": res["fallback"],
                    "gap_vs_r_colors": res["gap_vs_r_colors"],
                    "gap_vs_r_pct": res["gap_vs_r_pct"],
                    "gap_ci_colors_lo": res["gap_ci_colors"][0],
                    "gap_ci_colors_hi": res["gap_ci_colors"][1],
                    "gap_ci_pct_lo": res["gap_ci_pct"][0],
                    "gap_ci_pct_hi": res["gap_ci_pct"][1],
                    "weights": res["weights"],
                }
                # Checkpoint
                pd.DataFrame(list(d4_existing.values())).to_csv(d4_csv_path, index=False)
        print(f"D4 remaining completed in {time.perf_counter() - t0:.1f}s.")

    d4_rows = [d4_existing[f["family_id"]] for f in all_fams]
    d4_df = pd.DataFrame(d4_rows)
    d4_df.to_csv("results/d4_parametric_rlf_val.csv", index=False)

    # --------------------------------------------------------------------------
    # 3. DIAGNOSTIC D5: CONSTRUCTIVE CEILING (200 RUNS) ON VAL
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: RUNNING D5 CONSTRUCTIVE CEILING (200 RANDOMIZED RUNS) ON VAL")
    print("100 rand DSATUR + 50 rand RLF tie-break + 50 rand RLF top-open")
    print("=" * 80)
    t0 = time.perf_counter()
    d5_tasks = [(f["family_id"], r_name_map[f["family_id"]]) for f in all_fams]
    d5_results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_d5_worker, t): t[0] for t in d5_tasks}
        for fut in tqdm(as_completed(future_map), total=n_fams, desc="D5 Constructive Ceiling"):
            fid = future_map[fut]
            d5_results[fid] = fut.result()
    print(f"D5 completed in {time.perf_counter() - t0:.1f}s.")

    d5_rows = []
    for f in all_fams:
        fid = f["family_id"]
        res = d5_results[fid]
        d5_rows.append({
            "family_id": fid,
            "structure_type": f["structure_type"],
            "pool": f["pool"],
            "is_control": res["is_control"],
            "dsatur_val": res["dsatur_val"],
            "rlf_val": res["rlf_val"],
            "r_ref_val": res["r_ref_val"],
            "constructive_val": res["constructive_val"],
            "headroom_vs_r_colors": res["headroom_vs_r_colors"],
            "headroom_vs_r_pct": res["headroom_vs_r_pct"],
            "headroom_ci_colors_lo": res["headroom_ci_colors"][0],
            "headroom_ci_colors_hi": res["headroom_ci_colors"][1],
            "headroom_ci_pct_lo": res["headroom_ci_pct"][0],
            "headroom_ci_pct_hi": res["headroom_ci_pct"][1],
        })
    d5_df = pd.DataFrame(d5_rows)
    d5_df.to_csv("results/d5_constructive_ceiling_val.csv", index=False)

    # --------------------------------------------------------------------------
    # 4. GENERATE SUMMARY REPORT
    # --------------------------------------------------------------------------
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("PHASE 1 DIAGNOSTICS SUMMARY REPORT (EVALUATED ON VAL GRAPHS, N=20/FAMILY)")
    summary_lines.append("Reference R(F) = min(DSATUR, RLF, Universal) on Val")
    summary_lines.append("=" * 80)

    # Non-control records
    d1_nc = [r for r in d1_rows if not r["is_control"]]
    d4_nc = [r for r in d4_rows if not r["is_control"]]
    d5_nc = [r for r in d5_rows if not r["is_control"]]

    # D1' stats on VAL
    d1_c_mean, d1_c_lo, d1_c_hi = cluster_bootstrap_metric(d1_nc, "headroom_vs_r_colors")
    d1_p_mean, d1_p_lo, d1_p_hi = cluster_bootstrap_metric(d1_nc, "headroom_vs_r_colors", "r_ref_val")
    d1_p_med = float(np.median([r["headroom_vs_r_pct"] for r in d1_nc]))
    d1_c_med = float(np.median([r["headroom_vs_r_colors"] for r in d1_nc]))
    d1_ge_3 = sum(1 for r in d1_nc if r["headroom_vs_r_pct"] >= 3.0)

    # D4 stats on VAL
    d4_c_mean, d4_c_lo, d4_c_hi = cluster_bootstrap_metric(d4_nc, "gap_vs_r_colors")
    d4_p_mean, d4_p_lo, d4_p_hi = cluster_bootstrap_metric(d4_nc, "gap_vs_r_colors", "r_ref_val")
    d4_p_med = float(np.median([r["gap_vs_r_pct"] for r in d4_nc]))
    d4_c_med = float(np.median([r["gap_vs_r_colors"] for r in d4_nc]))
    d4_fallbacks = sum(1 for r in d4_nc if r["fallback"])
    d4_ge_3 = sum(1 for r in d4_nc if r["gap_vs_r_pct"] >= 3.0)

    # D5 stats on VAL
    d5_c_mean, d5_c_lo, d5_c_hi = cluster_bootstrap_metric(d5_nc, "headroom_vs_r_colors")
    d5_p_mean, d5_p_lo, d5_p_hi = cluster_bootstrap_metric(d5_nc, "headroom_vs_r_colors", "r_ref_val")
    d5_p_med = float(np.median([r["headroom_vs_r_pct"] for r in d5_nc]))
    d5_c_med = float(np.median([r["headroom_vs_r_colors"] for r in d5_nc]))
    d5_ge_3 = sum(1 for r in d5_nc if r["headroom_vs_r_pct"] >= 3.0)

    summary_lines.append("\n1. OVERVIEW COMPARISON ACROSS DIAGNOSTICS ON VAL (Non-Control, N=99 families):")
    summary_lines.append(f"   D1' Near-Optimal Ceiling vs R(F) : Headroom = {d1_c_mean:.3f} colors [{d1_c_lo:.3f}, {d1_c_hi:.3f}] | {d1_p_mean:.2f}% [{d1_p_lo:.2f}%, {d1_p_hi:.2f}%]")
    summary_lines.append(f"       Median Headroom              : {d1_c_med:.3f} colors | {d1_p_med:.2f}%")
    summary_lines.append(f"       Families with Headroom >= 3% : {d1_ge_3} / 99 ({d1_ge_3 / 99 * 100:.1f}%)")
    summary_lines.append(f"   D4  Tuned Parametric RLF vs R(F) : Gap = {d4_c_mean:+.3f} colors [{d4_c_lo:+.3f}, {d4_c_hi:+.3f}] | {d4_p_mean:+.2f}% [{d4_p_lo:+.2f}%, {d4_p_hi:+.2f}%]")
    summary_lines.append(f"       Median Gap                   : {d4_c_med:+.3f} colors | {d4_p_med:+.2f}%")
    summary_lines.append(f"       Fallback to R(F) triggered   : {d4_fallbacks} / 99 ({d4_fallbacks / 99 * 100:.1f}%)")
    summary_lines.append(f"       Families with Gap >= 3%      : {d4_ge_3} / 99 ({d4_ge_3 / 99 * 100:.1f}%)")
    summary_lines.append(f"   D5  Constructive Ceiling vs R(F) : Headroom = {d5_c_mean:.3f} colors [{d5_c_lo:.3f}, {d5_c_hi:.3f}] | {d5_p_mean:.2f}% [{d5_p_lo:.2f}%, {d5_p_hi:.2f}%]")
    summary_lines.append(f"       Median Headroom              : {d5_c_med:.3f} colors | {d5_p_med:.2f}%")
    summary_lines.append(f"       Families with Headroom >= 3% : {d5_ge_3} / 99 ({d5_ge_3 / 99 * 100:.1f}%)")

    # Control group check
    d1_ctrl = [r for r in d1_rows if r["is_control"]]
    d4_ctrl = [r for r in d4_rows if r["is_control"]]
    d5_ctrl = [r for r in d5_rows if r["is_control"]]
    summary_lines.append("\n2. INTERVAL CONTROL GROUP (N=7 families, verified DSATUR = clique LB):")
    summary_lines.append(f"   D1' Headroom : {float(np.mean([r['headroom_vs_r_colors'] for r in d1_ctrl])):.3f} colors (0.00%)")
    summary_lines.append(f"   D4  Gap      : {float(np.mean([r['gap_vs_r_colors'] for r in d4_ctrl])):.3f} colors (0.00%)")
    summary_lines.append(f"   D5  Headroom : {float(np.mean([r['headroom_vs_r_colors'] for r in d5_ctrl])):.3f} colors (0.00%)")

    # Breakdowns for D1', D4, D5
    summary_lines.append(format_breakdown_table(d1_rows, "headroom_vs_r_colors", "headroom_vs_r_pct", "D1' GRAPH CEILING ON VAL"))
    summary_lines.append(format_breakdown_table(d4_rows, "gap_vs_r_colors", "gap_vs_r_pct", "D4 TUNED PARAMETRIC RLF GAP ON VAL"))
    summary_lines.append(format_breakdown_table(d5_rows, "headroom_vs_r_colors", "headroom_vs_r_pct", "D5 CONSTRUCTIVE CEILING ON VAL"))

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open("results/val_diagnostics_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("\nSaved diagnostics summary to results/val_diagnostics_summary.txt")


if __name__ == "__main__":
    main()
