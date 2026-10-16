"""Diagnostic D8: Benchmark Vectorized RLF vs Vertex-level API.

Compares:
1. True Vectorized API: greedy_color(adj, rlf_priority) with matrix-vector dot products
   (no Python loops over candidates).
2. Vertex-level API: candidates = all uncolored vertices.
   priority_func(v, graph_dict, color_map, candidates) infers current color class from color_map.

Evaluated on 10 val graphs for 4 representative families:
- kpart_k10_p0.3_n100-200
- gnp_d0.30_n100-200
- sbm_k10_in0.5_out0.05_n200-300
- ws_k10_p0.1_n100-200
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from evaluator.coloring import greedy_color, rlf_priority, rlf
from evaluator.verify import verify, count_colors
from gate.dataset import load_family


def rlf_vertex_priority(
    v: int,
    graph_dict: Dict[int, List[int]],
    color_map: Dict[int, int],
    candidates: List[int],
) -> float:
    """RLF priority matching the vertex API specification:
    - candidates = all uncolored vertices.
    - Function infers current color class directly from color_map.
    """
    n = len(graph_dict)
    if not color_map:
        # Opening color class 0: max uncolored degree
        return float(sum(1 for w in graph_dict[v] if w not in color_map))

    k = max(color_map.values())
    v_has_k = any(color_map.get(w) == k for w in graph_dict[v])

    # Find candidates for color class k (uncolored vertices non-adjacent to class k)
    cand_k = set()
    for u in candidates:
        if not any(color_map.get(w) == k for w in graph_dict[u]):
            cand_k.add(u)

    if cand_k:
        if v_has_k or v not in cand_k:
            return -1e18
        # ne: neighbors in uncolored excluded set (candidates \ cand_k)
        # nc: neighbors in candidates for class k (cand_k)
        ne = sum(1 for w in graph_dict[v] if w in candidates and w not in cand_k)
        nc = sum(1 for w in graph_dict[v] if w in cand_k)
        return float(ne * (n + 1) - nc)
    else:
        # Class k is full, opening class k+1: score by uncolored degree
        return float(sum(1 for w in graph_dict[v] if w not in color_map))


def color_vertex_framework(adj: np.ndarray) -> np.ndarray:
    """Framework driving the vertex-level priority function.
    candidates = all uncolored vertices at each step.
    Framework assigns smallest feasible color to best_v.
    """
    n = adj.shape[0]
    graph_dict = {i: np.flatnonzero(adj[i]).tolist() for i in range(n)}
    color_map: Dict[int, int] = {}

    for _ in range(n):
        candidates = sorted([u for u in range(n) if u not in color_map])

        best_v = None
        best_score = -float("inf")
        for v in candidates:
            score = rlf_vertex_priority(v, graph_dict, color_map, candidates)
            if score > best_score:
                best_score = score
                best_v = v

        # Assign smallest feasible color
        used_colors = {color_map[w] for w in graph_dict[best_v] if w in color_map}
        c = 0
        while c in used_colors:
            c += 1
        color_map[best_v] = c

    return np.array([color_map[i] for i in range(n)], dtype=np.int64)


def run_d8_benchmark():
    target_families = [
        "kpart_k10_p0.3_n100-200",
        "gnp_d0.30_n100-200",
        "sbm_k10_in0.5_out0.05_n200-300",
        "ws_k10_p0.1_n100-200",
    ]

    records = []
    print("=" * 80)
    print("DIAGNOSTIC D8: TRUE VECTORIZED VS VERTEX-LEVEL API BENCHMARK")
    print("=" * 80)

    for fam_id in target_families:
        graphs = load_family(fam_id, "val")[:10]
        n_graphs = len(graphs)
        assert n_graphs == 10, f"Expected 10 graphs, got {n_graphs}"

        t_vec_total = 0.0
        t_vtx_total = 0.0
        exact_matches = 0

        for seed, adj in graphs:
            # 1. True Vectorized API
            t0 = time.perf_counter()
            c_vec = greedy_color(adj, rlf_priority)
            t1 = time.perf_counter()
            t_vec = t1 - t0
            t_vec_total += t_vec

            assert verify(adj, c_vec), f"Vectorized coloring invalid on {fam_id} seed {seed}"

            # 2. Vertex-level API
            t2 = time.perf_counter()
            c_vtx = color_vertex_framework(adj)
            t3 = time.perf_counter()
            t_vtx = t3 - t2
            t_vtx_total += t_vtx

            assert verify(adj, c_vtx), f"Vertex-level coloring invalid on {fam_id} seed {seed}"

            # Verify identical color assignment
            match = np.array_equal(c_vec, c_vtx)
            if match:
                exact_matches += 1
            else:
                raise AssertionError(f"Coloring mismatch on family {fam_id}, seed {seed}!")

            records.append({
                "family_id": fam_id,
                "seed": seed,
                "n_nodes": adj.shape[0],
                "n_edges": int(adj.sum() // 2),
                "colors": count_colors(c_vec),
                "exact_match": match,
                "t_vectorized_ms": t_vec * 1000.0,
                "t_vertex_ms": t_vtx * 1000.0,
                "slowdown": t_vtx / max(t_vec, 1e-9),
            })

        mean_vec_ms = (t_vec_total / n_graphs) * 1000.0
        mean_vtx_ms = (t_vtx_total / n_graphs) * 1000.0
        slowdown = t_vtx_total / max(t_vec_total, 1e-9)

        print(f"Family: {fam_id:<32} | 10/10 exact match: {exact_matches == 10}")
        print(f"  Vectorized: {mean_vec_ms:6.2f} ms/graph | Vertex-level: {mean_vtx_ms:8.2f} ms/graph | Slowdown: {slowdown:6.1f}x")

    df = pd.DataFrame(records)
    df.to_csv("results/d8_timing_results.csv", index=False)
    print("=" * 80)
    print("SUMMARY ACROSS ALL 40 GRAPHS:")
    total_vec_s = df["t_vectorized_ms"].sum() / 1000.0
    total_vtx_s = df["t_vertex_ms"].sum() / 1000.0
    overall_slowdown = total_vtx_s / max(total_vec_s, 1e-9)
    print(f"Total Vectorized time: {total_vec_s:.3f} s (mean {df['t_vectorized_ms'].mean():.2f} ms/graph)")
    print(f"Total Vertex time:     {total_vtx_s:.3f} s (mean {df['t_vertex_ms'].mean():.2f} ms/graph)")
    print(f"Overall Slowdown:      {overall_slowdown:.1f}x")
    print(f"All 40 graphs bitwise identical colorings: {df['exact_match'].all()}")
    print("Results saved to results/d8_timing_results.csv")
    print("=" * 80)


if __name__ == "__main__":
    run_d8_benchmark()
