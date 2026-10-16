"""Benchmark pure Python RLF execution time for the 6 pilot families to calibrate sandbox limits."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd
from gate.dataset import load_family
from evaluator.coloring import greedy_color, rlf_priority

pilot_fids = [
    "kpart_k10_p0.3_n100-200",
    "gnp_d0.30_n100-200",
    "sbm_k10_in0.5_out0.05_n200-300",
    "ws_k10_p0.1_n100-200",
    "interval_L0.10_n100-200",
    "ba_m12_n200-300",
]

rows = []
for fid in pilot_fids:
    graphs = load_family(fid, "val")[:5]
    times = []
    n_list = []
    for s, adj in graphs:
        n_list.append(adj.shape[0])
        t0 = time.perf_counter()
        greedy_color(adj, rlf_priority)
        times.append(time.perf_counter() - t0)
    avg_t = float(np.mean(times))
    max_t = float(np.max(times))
    avg_n = float(np.mean(n_list))
    limit_4x = float(max_t * 4.0)

    rows.append({
        "family_id": fid,
        "n_avg": avg_n,
        "rlf_python_avg_ms": round(avg_t * 1000.0, 1),
        "rlf_python_max_ms": round(max_t * 1000.0, 1),
        "timeout_4x_ms": round(limit_4x * 1000.0, 1),
        "recommended_timeout_s": max(round(limit_4x * 1.5 / 1000.0, 2), 0.25),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
