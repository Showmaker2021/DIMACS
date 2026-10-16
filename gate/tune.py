"""CMA-ES tuning for parametric DSATUR.

Provides:
- tune_family: tunes parametric weights on baseline_train graphs of a family.
- tune_universal: tunes a single parametric heuristic on union of baseline_train of train_pool families.
- tune_family_worker: top-level worker for ProcessPoolExecutor parallel execution.
"""

from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import cma

from gate.dataset import load_family, families
from gate.parametric import (
    parametric_color,
    DSATUR_WEIGHTS,
    NUM_FEATURES,
)
from evaluator.coloring import dsatur, rlf
from evaluator.verify import verify, count_colors


def _eval_graphs(adjs: List[np.ndarray], weights: np.ndarray) -> Tuple[float, float]:
    """Evaluate weights on a list of graphs.

    Returns:
    (guided_loss, real_mean_colors)
    """
    total_loss = 0.0
    total_colors = 0.0
    num_graphs = len(adjs)

    for adj in adjs:
        n = adj.shape[0]
        col = parametric_color(adj, weights)
        k = count_colors(col)
        total_colors += float(k)

        # Count vertices in last color class
        max_c = int(np.max(col)) if len(col) > 0 else 0
        last_color_cnt = int(np.sum(col == max_c)) if len(col) > 0 else 0
        tie_breaker = 1e-4 * (float(last_color_cnt) / float(n if n > 0 else 1))
        total_loss += float(k) + tie_breaker

    return total_loss / float(num_graphs), total_colors / float(num_graphs)


def evaluate_heuristic_on_split(
    graphs: List[Tuple[int, np.ndarray]],
    heuristic_fn: Any,
) -> float:
    """Evaluate a coloring heuristic on a list of (seed, adj) graphs, strictly verifying every coloring."""
    counts = []
    for seed, adj in graphs:
        col = heuristic_fn(adj)
        if not verify(adj, col):
            raise RuntimeError(f"Coloring verification failed for seed {seed}")
        counts.append(count_colors(col))
    return float(np.mean(counts))


def compute_paired_bootstrap_ci(
    arr_ref: np.ndarray,
    arr_target: np.ndarray,
    num_bootstraps: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float, float, float, float]:
    """Compute paired difference and percent difference with 95% bootstrap CI.

    Returns:
    (diff_mean, diff_ci_lo, diff_ci_hi, pct_mean, pct_ci_lo, pct_ci_hi)
    where diff = ref - target, and pct = diff / ref * 100.
    """
    n = len(arr_ref)
    diffs = arr_ref - arr_target
    diff_mean = float(np.mean(diffs))
    ref_mean = float(np.mean(arr_ref))
    pct_mean = (diff_mean / ref_mean * 100.0) if ref_mean > 0 else 0.0

    rng = np.random.RandomState(seed)
    indices = rng.randint(0, n, size=(num_bootstraps, n))

    boot_diffs = np.mean(diffs[indices], axis=1)
    boot_refs = np.mean(arr_ref[indices], axis=1)
    boot_pcts = np.where(boot_refs > 0, (boot_diffs / boot_refs) * 100.0, 0.0)

    diff_ci_lo = float(np.percentile(boot_diffs, 2.5))
    diff_ci_hi = float(np.percentile(boot_diffs, 97.5))
    pct_ci_lo = float(np.percentile(boot_pcts, 2.5))
    pct_ci_hi = float(np.percentile(boot_pcts, 97.5))

    return diff_mean, diff_ci_lo, diff_ci_hi, pct_mean, pct_ci_lo, pct_ci_hi


def tune_universal(
    budget: int = 3000,
    seeds: Tuple[int, ...] = (42, 43, 44),
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Tune a single universal heuristic on union of baseline_train of train_pool families.

    Uses a stratified sample of 4 graphs per family (seeds 0..3 of all 55 train_pool families = 220 graphs)
    for fast CMA-ES search, then evaluates and selects on all validation graphs of train_pool.
    """
    if verbose:
        print(">>> Tuning Universal Parametric Heuristic on train_pool families...")

    train_pool_fams = families(pool="train_pool")
    search_adjs = []
    val_graphs_all = []

    for f in train_pool_fams:
        fid = f["family_id"]
        bt = load_family(fid, "baseline_train")
        search_adjs.extend([g[1] for g in bt[:4]])
        val_graphs_all.extend(load_family(fid, "val"))

    val_adjs = [g[1] for g in val_graphs_all]
    ds_val = np.mean([count_colors(dsatur(a)) for a in val_adjs])

    best_overall_w = DSATUR_WEIGHTS.copy()
    best_overall_val = ds_val
    seed_details = []

    for s in seeds:
        x0 = DSATUR_WEIGHTS.copy()
        opts = {
            "CMA_stds": [0.2, 0.5, 0.5, 0.5, 0.5],
            "seed": s,
            "maxfevals": budget,
            "verbose": -9,
        }
        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=0.5, inopts=opts)

        best_train_loss = float("inf")
        best_w = x0.copy()

        while not es.stop():
            X = es.ask()
            fitnesses = []
            for candidate in X:
                loss, _ = _eval_graphs(search_adjs, candidate)
                fitnesses.append(loss)
                if loss < best_train_loss:
                    best_train_loss = loss
                    best_w = np.array(candidate, dtype=np.float64)
            es.tell(X, fitnesses)

        _, s_val_c = _eval_graphs(val_adjs, best_w)
        seed_details.append({
            "seed": s,
            "weights": best_w.tolist(),
            "val_colors": s_val_c,
        })
        if s_val_c < best_overall_val:
            best_overall_val = s_val_c
            best_overall_w = best_w.copy()

    fallback = False
    if best_overall_val >= ds_val:
        best_overall_w = DSATUR_WEIGHTS.copy()
        best_overall_val = ds_val
        fallback = True

    if verbose:
        print(f"Universal tuning complete. DSATUR Val: {ds_val:.3f}, Universal Val: {best_overall_val:.3f}, Fallback: {fallback}")
        print(f"Universal Weights: {[round(x, 4) for x in best_overall_w]}")

    return best_overall_w, {
        "weights": best_overall_w.tolist(),
        "val_colors": best_overall_val,
        "dsatur_val": ds_val,
        "fallback": fallback,
        "seed_details": seed_details,
    }


def tune_family_worker(args: Tuple[str, int, Tuple[int, ...], Optional[List[float]]]) -> Dict[str, Any]:
    """Worker function for parallel per-family tuning.

    Reference R(F) is selected as the best of {DSATUR, RLF, Universal} on VAL.
    """
    family_id, budget, seeds, univ_w_list = args

    univ_w = np.array(univ_w_list, dtype=np.float64) if univ_w_list is not None else DSATUR_WEIGHTS

    train_graphs = load_family(family_id, "baseline_train")
    val_graphs = load_family(family_id, "val")
    test_graphs = load_family(family_id, "test")

    train_adjs = [g[1] for g in train_graphs]
    val_adjs = [g[1] for g in val_graphs]
    test_adjs = [g[1] for g in test_graphs]

    # Baseline performances
    ds_train = evaluate_heuristic_on_split(train_graphs, dsatur)
    ds_val = evaluate_heuristic_on_split(val_graphs, dsatur)
    ds_test_counts = np.array([count_colors(dsatur(a)) for a in test_adjs], dtype=np.float64)
    ds_test = float(np.mean(ds_test_counts))

    # RLF performance on all splits
    rlf_train = evaluate_heuristic_on_split(train_graphs, rlf)
    rlf_val = evaluate_heuristic_on_split(val_graphs, rlf)
    rlf_test_counts = np.array([count_colors(rlf(a)) for a in test_adjs], dtype=np.float64)
    rlf_test = float(np.mean(rlf_test_counts))

    # Universal heuristic performance on all splits
    univ_fn = lambda a, w=univ_w: parametric_color(a, w)
    univ_train = evaluate_heuristic_on_split(train_graphs, univ_fn)
    univ_val = evaluate_heuristic_on_split(val_graphs, univ_fn)
    univ_test_counts = np.array([count_colors(univ_fn(a)) for a in test_adjs], dtype=np.float64)
    univ_test = float(np.mean(univ_test_counts))

    # Determine Reference R(F) = best of {DSATUR, RLF, Universal} on VAL
    candidates = [
        ("dsatur", ds_val, ds_test, ds_test_counts),
        ("rlf", rlf_val, rlf_test, rlf_test_counts),
        ("universal", univ_val, univ_test, univ_test_counts),
    ]
    r_name, r_val, r_test, r_test_counts = min(candidates, key=lambda c: c[1])

    # Run CMA-ES tuning across seeds on baseline_train
    seed_results = []
    for s in seeds:
        x0 = DSATUR_WEIGHTS.copy()
        opts = {
            "CMA_stds": [0.2, 0.5, 0.5, 0.5, 0.5],
            "seed": s,
            "maxfevals": budget,
            "verbose": -9,
        }
        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=0.5, inopts=opts)

        best_train_loss = float("inf")
        best_train_colors = float("inf")
        best_w = x0.copy()

        init_loss, init_c = _eval_graphs(train_adjs, x0)
        best_train_loss = init_loss
        best_train_colors = init_c

        while not es.stop():
            X = es.ask()
            fitnesses = []
            for candidate in X:
                loss, real_c = _eval_graphs(train_adjs, candidate)
                fitnesses.append(loss)
                if loss < best_train_loss:
                    best_train_loss = loss
                    best_train_colors = real_c
                    best_w = np.array(candidate, dtype=np.float64)
            es.tell(X, fitnesses)

        _, s_val_colors = _eval_graphs(val_adjs, best_w)
        seed_results.append({
            "seed": s,
            "weights": best_w.tolist(),
            "train_colors": best_train_colors,
            "val_colors": s_val_colors,
        })

    # Model selection on VAL
    best_seed_res = min(seed_results, key=lambda r: (r["val_colors"], r["train_colors"]))
    chosen_w = np.array(best_seed_res["weights"], dtype=np.float64)
    chosen_val = best_seed_res["val_colors"]

    # Fallback to DSATUR if tuned did not improve over DSATUR on VAL
    fallback_to_ds = False
    if chosen_val >= ds_val:
        chosen_w = DSATUR_WEIGHTS.copy()
        fallback_to_ds = True

    # Final evaluations for tuned heuristic
    tuned_fn = lambda a, w=chosen_w: parametric_color(a, w)
    tuned_train = evaluate_heuristic_on_split(train_graphs, tuned_fn)
    tuned_val = evaluate_heuristic_on_split(val_graphs, tuned_fn)
    tuned_test_counts = np.array([count_colors(tuned_fn(a)) for a in test_adjs], dtype=np.float64)
    tuned_test = float(np.mean(tuned_test_counts))

    # Gains over DSATUR
    gain_train = ds_train - tuned_train
    gain_val = ds_val - tuned_val
    gain_test = ds_test - tuned_test

    # Bootstrap CIs for test comparisons:
    # 1. Gain vs DSATUR (DSATUR - Tuned)
    gain_ds_mean, g_ds_lo, g_ds_hi, g_pct_mean, g_pct_lo, g_pct_hi = compute_paired_bootstrap_ci(
        ds_test_counts, tuned_test_counts
    )

    # 2. Gap A vs R(F) (R(F) - Tuned)
    gap_a_mean, gap_lo, gap_hi, gap_pct_mean, gap_pct_lo, gap_pct_hi = compute_paired_bootstrap_ci(
        r_test_counts, tuned_test_counts
    )

    is_control = ("interval" in family_id)

    return {
        "family_id": family_id,
        "is_control": is_control,
        "chosen_weights": chosen_w.tolist(),
        "fallback_to_dsatur": fallback_to_ds,
        "dsatur": {"train": ds_train, "val": ds_val, "test": ds_test},
        "rlf": {"train": rlf_train, "val": rlf_val, "test": rlf_test},
        "universal": {"train": univ_train, "val": univ_val, "test": univ_test},
        "r_reference": {"name": r_name, "val": r_val, "test": r_test},
        "tuned": {"train": tuned_train, "val": tuned_val, "test": tuned_test},
        "gains_vs_dsatur": {
            "train": gain_train,
            "val": gain_val,
            "test": gain_test,
            "test_pct": g_pct_mean,
            "test_ci": (g_ds_lo, g_ds_hi),
            "test_pct_ci": (g_pct_lo, g_pct_hi),
        },
        "gap_a": {
            "colors": gap_a_mean,
            "pct": gap_pct_mean,
            "ci_colors": (gap_lo, gap_hi),
            "ci_pct": (gap_pct_lo, gap_pct_hi),
        },
    }
