"""Inspect candidates for the 3 families needed for Step 2 dry run."""

import json
from pathlib import Path

fams = json.load(open("data/data_v1/families.json", encoding="utf-8"))

print("=== 1. FAMILIES WHERE RLF MEAN < DSATUR MEAN (RLF BEATS DSATUR) ===")
rlf_beats = sorted(
    [f for f in fams if f["labels"]["dsatur_mean"] - f["labels"]["rlf_mean"] >= 0.5],
    key=lambda f: f["labels"]["dsatur_mean"] - f["labels"]["rlf_mean"],
    reverse=True,
)
for f in rlf_beats[:12]:
    l = f["labels"]
    diff = l["dsatur_mean"] - l["rlf_mean"]
    print(f"{f['family_id']:<32} | {f['structure_type']:<18} | {f['pool']:<14} | ds:{l['dsatur_mean']:<5.1f} | rlf:{l['rlf_mean']:<5.1f} | diff:{diff:<5.1f}")

print("\n=== 2. FAMILIES WHERE ALL HEURISTICS ARE CLOSE ===")
flat = sorted(
    fams,
    key=lambda f: (
        abs(f["labels"]["dsatur_mean"] - f["labels"]["rlf_mean"])
        + abs(f["labels"]["dsatur_mean"] - f["labels"]["portfolio_best_mean"])
    ),
)
for f in flat[:10]:
    l = f["labels"]
    diff = abs(l["dsatur_mean"] - l["rlf_mean"])
    print(f"{f['family_id']:<32} | {f['structure_type']:<18} | {f['pool']:<14} | ds:{l['dsatur_mean']:<5.1f} | rlf:{l['rlf_mean']:<5.1f} | lb:{l['clique_lb_mean']:<5.1f} | headroom:{l['headroom_proxy_vs_dsatur']:<5.1f}")

print("\n=== 3. FAMILIES WITH LARGE HEADROOM (GAP TO CLIQUE LB) ===")
large_hr = sorted(
    [f for f in fams if f["pool"] == "train_pool"],
    key=lambda f: f["labels"]["gap_dsatur_to_lb"],
    reverse=True,
)
for f in large_hr[:10]:
    l = f["labels"]
    print(f"{f['family_id']:<32} | {f['structure_type']:<18} | {f['pool']:<14} | ds:{l['dsatur_mean']:<5.1f} | rlf:{l['rlf_mean']:<5.1f} | lb:{l['clique_lb_mean']:<5.1f} | gap_lb:{l['gap_dsatur_to_lb']:<5.1f} | headroom:{l['headroom_proxy_vs_dsatur']:<5.1f}")
