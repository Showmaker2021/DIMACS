"""Compute and print R(F) on teacher_val split (seeds 3000-3019) for the 6 pilot families."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gate.dataset import families
from data.generate_dataset import make_graph, adj_from_edges
from evaluator.coloring import dsatur, rlf
from evaluator.verify import count_colors


PILOT_FAMILIES = [
    "kpart_k10_p0.3_n100-200",
    "gnp_d0.30_n100-200",
    "sbm_k10_in0.5_out0.05_n200-300",
    "ws_k10_p0.1_n100-200",
    "interval_L0.10_n100-200",   # Control 1
    "ba_m12_n200-300",           # Control 2
]


def compute_teacher_val_baselines():
    fam_map = {f["family_id"]: f for f in families()}
    
    results = {}
    print("=" * 90)
    print("TEACHER_VAL SPLIT (SEEDS 3000-3019, 20 GRAPHS) BASELINE REFERENCE R(F)")
    print("=" * 90)
    print(f"{'Family ID':<35} | {'DSATUR':<8} | {'RLF':<8} | {'R(F) teacher_val':<16} | {'Best Baseline':<12}")
    print("-" * 90)

    for fid in PILOT_FAMILIES:
        if fid not in fam_map:
            raise ValueError(f"Family {fid} not in families metadata!")
        f = fam_map[fid]
        fam_tuple = (
            fid,
            f["structure_type"],
            f["generator"],
            f.get("params", {}),
            tuple(f["n_range"]),
        )
        
        ds_counts = []
        rlf_counts = []
        for s in range(3000, 3020):
            n, edges = make_graph(fam_tuple, s)
            adj = adj_from_edges(n, edges)
            ds_counts.append(count_colors(dsatur(adj)))
            rlf_counts.append(count_colors(rlf(adj)))
            
        m_ds = float(np.mean(ds_counts))
        m_rlf = float(np.mean(rlf_counts))
        r_f = min(m_ds, m_rlf)
        best_name = "RLF" if m_rlf < m_ds else "DSATUR"
        
        results[fid] = {
            "dsatur_teacher_val": m_ds,
            "rlf_teacher_val": m_rlf,
            "r_f_teacher_val": r_f,
            "best_baseline": best_name,
        }
        print(f"{fid:<35} | {m_ds:<8.3f} | {m_rlf:<8.3f} | {r_f:<16.3f} | {best_name:<12}")

    print("=" * 90)
    return results


if __name__ == "__main__":
    compute_teacher_val_baselines()
