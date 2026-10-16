import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from gate.dataset import families, load_family
from evaluator.coloring import dsatur
from evaluator.verify import verify, count_colors


def main():
    fams = families()
    print(f"{'Family ID':<32} | {'Type':<18} | {'Pool':<14} | {'Eval DSATUR':<12} | {'Prov Label':<12} | {'Diff':<6}")
    print("-" * 104)

    results = []
    for f in fams:
        prov = f["labels"].get("dsatur_test_mean_provisional")
        if prov is None:
            continue
        graphs = load_family(f["family_id"], "test")
        cols = []
        for _, adj in graphs:
            col = dsatur(adj)
            assert verify(adj, col), f"Failed verification on {f['family_id']}"
            cols.append(count_colors(col))
        eval_m = float(np.mean(cols))
        diff = abs(eval_m - prov)
        results.append((f["family_id"], f["structure_type"], f["pool"], eval_m, prov, diff))

    for fid, stype, pool, ev, pr, df in results[:25]:
        print(f"{fid:<32} | {stype:<18} | {pool:<14} | {ev:<12.2f} | {pr:<12.2f} | {df:<6.2f}")

    diffs = [r[5] for r in results]
    print("-" * 104)
    print(f"Evaluated {len(results)} / {len(fams)} families on all 50 test graphs each ({len(results)*50} graphs total).")
    print(f"Mean absolute difference: {np.mean(diffs):.4f} colors")
    print(f"Median absolute difference: {np.median(diffs):.4f} colors")
    print(f"Max absolute difference: {np.max(diffs):.4f} colors")


if __name__ == "__main__":
    main()
