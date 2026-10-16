"""Step 2 Dry Run Script: Evaluates CMA-ES tuning on 3 representative families."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from gate.tune import tune_family


def print_convergence_table(history: list, seed: int):
    print(f"\n--- Convergence Curve (Seed {seed}) ---")
    print(f"{'Evals':<8} | {'Best Real Colors':<18} | {'Guided Loss':<15}")
    print("-" * 46)
    milestones = [1, 200, 500, 1000, 1500, 2000, 2500, 3000]
    m_idx = 0
    last_printed_eval = -1
    for pt in history:
        ev = pt["evals"]
        if m_idx < len(milestones) and ev >= milestones[m_idx]:
            print(f"{ev:<8} | {pt['colors']:<18.2f} | {pt['loss']:<15.6f}")
            last_printed_eval = ev
            while m_idx < len(milestones) and ev >= milestones[m_idx]:
                m_idx += 1
    # Ensure final is printed
    if history and history[-1]["evals"] != last_printed_eval:
        last = history[-1]
        print(f"{last['evals']:<8} | {last['colors']:<18.2f} | {last['loss']:<15.6f}")


def run():
    target_families = [
        ("kpart_k10_p0.6_n200-300", "RLF << DSATUR (RLF noticeably lower)"),
        ("interval_L0.10_n100-200", "All heuristics close (flat headroom)"),
        ("gnp_d0.30_n200-300", "Large headroom (gap to clique LB)"),
    ]

    all_results = []

    print("================================================================================")
    print("STEP 2 DRY RUN: CMA-ES TUNING ON 3 REPRESENTATIVE FAMILIES")
    print("Budget: 3000 evaluations per seed | 3 seeds (42, 43, 44)")
    print("================================================================================\n")

    for fid, category in target_families:
        print(f"\n>>>>>>>> Family: {fid} ({category}) <<<<<<<<")
        res = tune_family(fid, budget=3000, seeds=(42, 43, 44), verbose=True)
        all_results.append((fid, category, res))

        print(f"\nRuntime: {res['elapsed_seconds']:.2f} s ({res['elapsed_seconds']/9000*1000:.3f} ms / eval on 20 graphs)")
        print(f"Chosen Seed (by val): {res['chosen_seed']}")
        print(f"Chosen Weights: {[round(w, 4) for w in res['chosen_weights']]}")

        print("\nPerformance Comparison (Mean Number of Colors):")
        print(f"{'Method':<16} | {'Train (20)':<12} | {'Val (20)':<12} | {'Test (50)':<12}")
        print("-" * 56)
        print(f"{'DSATUR':<16} | {res['dsatur']['train']:<12.2f} | {res['dsatur']['val']:<12.2f} | {res['dsatur']['test']:<12.2f}")
        print(f"{'RLF':<16} | {res['rlf']['train']:<12.2f} | {res['rlf']['val']:<12.2f} | {res['rlf']['test']:<12.2f}")
        print(f"{'Parametric Tuned':<16} | {res['tuned']['train']:<12.2f} | {res['tuned']['val']:<12.2f} | {res['tuned']['test']:<12.2f}")

        print("\nOverfitting Gaps (Tuned):")
        print(f"  Val - Train  : {res['overfit']['val_minus_train']:+.2f} colors")
        print(f"  Test - Train : {res['overfit']['test_minus_train']:+.2f} colors")
        print(f"  Test - Val   : {res['overfit']['test_minus_val']:+.2f} colors")

        print("\nPer-seed Stability:")
        print(f"{'Seed':<6} | {'Train':<10} | {'Val':<10} | {'Test':<10} | {'Weights'}")
        print("-" * 65)
        for sr in res["seed_results"]:
            w_str = str([round(x, 3) for x in sr["weights"]])
            print(f"{sr['seed']:<6} | {sr['train_colors']:<10.2f} | {sr['val_colors']:<10.2f} | {sr['test_colors']:<10.2f} | {w_str}")

        # Print convergence for chosen seed
        print_convergence_table(res["convergence"][res["chosen_seed"]], res["chosen_seed"])

    print("\n================================================================================")
    print("DRY RUN SUMMARY TABLE")
    print("================================================================================")
    print(f"{'Family ID':<26} | {'DSATUR (test)':<14} | {'RLF (test)':<12} | {'Tuned (test)':<14} | {'Gain vs DS':<12} | {'Gain vs RLF':<12}")
    print("-" * 96)
    for fid, cat, r in all_results:
        ds_t = r['dsatur']['test']
        rlf_t = r['rlf']['test']
        tn_t = r['tuned']['test']
        gain_ds = ds_t - tn_t
        gain_rlf = rlf_t - tn_t
        print(f"{fid:<26} | {ds_t:<14.2f} | {rlf_t:<12.2f} | {tn_t:<14.2f} | {gain_ds:<+12.2f} | {gain_rlf:<+12.2f}")


if __name__ == "__main__":
    run()
