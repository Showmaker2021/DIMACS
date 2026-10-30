"""Rollout and Parallel Group Evaluation Engine for GRPO.

Implements Task 11.2 from PLAN_PHASE6_ONWARD.md:
- Generates G candidates per card for a batch of cards.
- Pulls fresh, disjoint graphs from rl_train_pool (seeds 100000-199999).
- Evaluates group of G candidates on the EXACT SAME graph set in parallel.
- Produces CandidateResult objects directly consumable by train/reward.py.
"""

import sys
import os
import time
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate.dataset import load_family, families
from gate.evolved.sandbox import evaluate_candidate_safe
from train.reward import CandidateResult, compute_group_rewards, RewardConfig
from evaluator.coloring import dsatur, rlf
from evaluator.verify import count_colors


RL_SEED_START = 100000
RL_SEED_END = 199999


def sample_rl_graphs(
    family_id: str,
    n_graphs: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[int, np.ndarray]]:
    """Sample fresh graphs dynamically from rl_train_pool seeds."""
    if rng is None:
        rng = np.random.default_rng()

    from gate.dataset import _MANIFEST
    from data.generate_dataset import make_graph, adj_from_edges

    fam_rec = next((f for f in _MANIFEST.all_families if f["family_id"] == family_id), None)
    if fam_rec is None:
        raise ValueError(f"Family {family_id} not found in manifest")

    fam_tuple = (
        family_id,
        fam_rec["structure_type"],
        fam_rec["generator"],
        fam_rec.get("params", {}),
        tuple(fam_rec["n_range"]),
    )

    seeds = rng.integers(RL_SEED_START, RL_SEED_END + 1, size=n_graphs)
    graphs = []
    for s in seeds:
        n, edges = make_graph(fam_tuple, int(s))
        adj = adj_from_edges(n, edges)
        graphs.append((int(s), adj))

    return graphs


def evaluate_rollout_group(
    family_id: str,
    candidate_codes: List[str],
    graphs: List[Tuple[int, np.ndarray]],
    base_alg_name: str = "RLF",
    per_graph_timeout: float = 0.5,
) -> Tuple[List[CandidateResult], List[int], float]:
    """Evaluate a group of G candidates on the same graph set and return CandidateResult list."""
    t_start = time.perf_counter()

    # Precompute baseline c_base_per_graph and t_dsatur on the exact graphs
    adjs = [g[1] for g in graphs]
    t_dsatur: List[float] = []
    c_dsatur: List[int] = []
    for a in adjs:
        t0_d = time.perf_counter()
        col = dsatur(a)
        t_dsatur.append(time.perf_counter() - t0_d)
        c_dsatur.append(count_colors(col))

    if base_alg_name == "RLF":
        c_base = [count_colors(rlf(a)) for a in adjs]
    else:
        c_base = c_dsatur

    results: List[CandidateResult] = []
    code_cache: Dict[str, Dict[str, Any]] = {}

    for code in candidate_codes:
        chash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if chash in code_cache:
            res = code_cache[chash]
        else:
            res = evaluate_candidate_safe(
                code_str=code,
                graphs=graphs,
                per_graph_timeout=per_graph_timeout,
                overall_timeout=15.0,
            )
            code_cache[chash] = res

        if not res.get("success", False) or res.get("color_counts") is None:
            err = res.get("error", "unknown")
            status = "ast_fail" if "AST" in err else ("timeout" if "timeout" in err.lower() else "exception")
            cand_res = CandidateResult(
                status=status,
                colors_per_graph=[],
                runtime_per_graph=[],
                c_base_per_graph=c_base,
                t_dsatur_per_graph=t_dsatur,
                code=code,
            )
        else:
            counts = res.get("color_counts", [])
            runtimes = res.get("runtimes", [0.001] * len(counts))
            orders = res.get("vertex_orderings", None)
            cand_res = CandidateResult(
                status="ok",
                colors_per_graph=counts,
                runtime_per_graph=runtimes,
                probe_orderings=orders,
                c_base_per_graph=c_base,
                t_dsatur_per_graph=t_dsatur,
                code=code,
            )

        results.append(cand_res)

    elapsed = time.perf_counter() - t_start
    return results, c_base, elapsed
