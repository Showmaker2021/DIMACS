"""GRPO Reward Engine for Graph Coloring heuristics (Task 11.1 / Section 14.3).

This module is strictly decoupled from the execution sandbox and does NOT
invoke the sandbox, run graphs, or pick seeds. It purely consumes structured
candidate evaluation results (CandidateResult) and computes scalar rewards,
component breakdowns, and population-standardized group advantages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats


# Status categories
VALID_STATUS = "ok"
KNOWN_FAILURE_STATUSES = frozenset(
    {"ast_fail", "purity_fail", "timeout", "memory", "invalid_coloring", "exception"}
)


@dataclass
class CandidateResult:
    """Evaluation result for one candidate on a group's graph set S.
    
    Attributes:
        status: "ok" or failure reason ("ast_fail", "purity_fail", "timeout",
                "memory", "invalid_coloring", "exception", etc.)
        colors_per_graph: Number of colors used by candidate on each graph g in S.
        runtime_per_graph: Runtime in seconds on each graph g in S.
        probe_orderings: Vertex orderings on probe graphs (typically 2), or None.
        c_base_per_graph: Baseline color counts on S (optional per-candidate override).
        c_ref_per_graph: Reference/optimal color counts on S (optional per-candidate override).
        t_dsatur_per_graph: Baseline vectorized DSATUR runtimes on S (optional).
        color_classes_per_graph: Optional color class sizes per graph for r_tb.
        code: Optional Python source code string.
    """
    status: str
    colors_per_graph: List[int] = field(default_factory=list)
    runtime_per_graph: List[float] = field(default_factory=list)
    probe_orderings: Optional[List[Any]] = None
    c_base_per_graph: Optional[List[int]] = None
    c_ref_per_graph: Optional[List[int]] = None
    t_dsatur_per_graph: Optional[List[float]] = None
    color_classes_per_graph: Optional[List[List[int]]] = None
    code: Optional[str] = None


@dataclass
class RewardConfig:
    """Hyperparameters and coefficients for GRPO reward calculation.
    
    Mathematical Bounds:
        Assuming a >= 0, b >= 0, d >= 0:
        For failed candidate:
            r = failure_reward = -1.0
        For valid candidate:
            r = r_sav + a * r_tb - b * r_time + d * r_div
            Since:
                r_sav in [clip_sav_lo, clip_sav_hi] = [-1.0, 1.0]
                r_tb in [0.0, 1.0]
                r_time in [0.0, 1.0]
                r_div in [0.0, 1.0]
            Minimum possible valid reward:
                r_min = clip_sav_lo + a * 0 - b * 1 + d * 0 = -1.0 - b
            Maximum possible valid reward:
                r_max = clip_sav_hi + a * 1 - b * 0 + d * 1 = 1.0 + a + d
            Total lower bound: min(failure_reward, -1.0 - b) = -1.0 - b
            Total upper bound: max(failure_reward, 1.0 + a + d) = 1.0 + a + d
            
        Under defaults (a=0.0, b=0.1, d=0.0):
            r in [-1.1, 1.0]
        When diversity is active (e.g. d=0.1):
            r in [-1.1, 1.1]
    """
    a: float = 0.0                 # Weight for r_tb (tie-breaking, default 0.0 OFF)
    b: float = 0.1                 # Weight for r_time (runtime penalty, default 0.1)
    d: float = 0.0                 # Weight for r_div (behavioral diversity, default 0.0)
    time_factor: float = 3.0       # Multiplier: t_budget = time_factor * time_dsatur
    failure_reward: float = -1.0   # Exact reward assigned to any non-ok candidate
    eps_std: float = 1e-8          # Numerical stability for advantage normalization
    clip_sav_lo: float = -1.0      # Lower bound for r_sav clipping
    clip_sav_hi: float = 1.0       # Upper bound for r_sav clipping
    fallback_ref_offset: int = 1   # Fallback when c_ref is None: c_base(g) - offset

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardConfig:
        """Create RewardConfig from dictionary with type casting."""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> RewardConfig:
        """Load RewardConfig from a YAML file."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"YAML file {path} did not parse into a dictionary")
        return cls.from_dict(data)


@dataclass
class RewardBreakdown:
    """Detailed decomposition of reward components for one candidate."""
    status: str
    r_sav: float
    r_time: float
    r_tb: float
    r_pre: float
    r_div: float
    reward: float
    is_eligible_diversity: bool


def get_reward_bounds(config: Optional[RewardConfig] = None) -> Tuple[float, float]:
    """Compute exact mathematical lower and upper bounds for candidate rewards.
    
    Proof:
        r_sav in [clip_sav_lo, clip_sav_hi]
        r_tb in [0.0, 1.0]
        r_time in [0.0, 1.0]
        r_div in [0.0, 1.0]
        Failed candidates receive failure_reward.
        
        Lower bound:
            min(failure_reward, clip_sav_lo - b * 1.0)
        Upper bound:
            max(failure_reward, clip_sav_hi + a * 1.0 + d * 1.0)
    """
    if config is None:
        config = RewardConfig()
    lo = min(config.failure_reward, config.clip_sav_lo - max(0.0, config.b))
    hi = max(config.failure_reward, config.clip_sav_hi + max(0.0, config.a) + max(0.0, config.d))
    return (float(lo), float(hi))


def kendall_tau_distance(ord1: Any, ord2: Any) -> float:
    """Compute normalized Kendall-tau distance in [0, 1] between two orderings.
    
    Properties:
        - Identical orderings -> 0.0 exactly.
        - Exactly reversed orderings -> 1.0 exactly.
        - Distance equals the fraction of discordant pairs out of total pairs N*(N-1)/2.
        - Returns 0.0 for None, empty, or single-element inputs.
    """
    if ord1 is None or ord2 is None:
        return 0.0
    a1 = np.asarray(ord1)
    a2 = np.asarray(ord2)
    if a1.shape != a2.shape or a1.size < 2:
        return 0.0
    if np.array_equal(a1, a2):
        return 0.0
    if np.array_equal(a1, a2[::-1]):
        return 1.0

    # If both are permutations of the same unique items (e.g. vertex visitation orders)
    try:
        if a1.ndim == 1:
            items1 = set(a1.tolist())
            if len(items1) == len(a1) and items1 == set(a2.tolist()):
                pos1 = {val: i for i, val in enumerate(a1.tolist())}
                pos2 = {val: i for i, val in enumerate(a2.tolist())}
                # Track ranks of identical elements
                keys = list(pos1.keys())
                r1 = [pos1[k] for k in keys]
                r2 = [pos2[k] for k in keys]
                res = scipy.stats.kendalltau(r1, r2)
                tau = res.statistic
                if math.isnan(tau):
                    return 0.0
                dist = (1.0 - float(tau)) / 2.0
                return float(np.clip(dist, 0.0, 1.0))
    except Exception:
        pass

    res = scipy.stats.kendalltau(a1, a2)
    tau = res.statistic
    if math.isnan(tau):
        return 0.0
    dist = (1.0 - float(tau)) / 2.0
    return float(np.clip(dist, 0.0, 1.0))


def pairwise_probe_distance(
    probe_list_a: Optional[Sequence[Any]],
    probe_list_b: Optional[Sequence[Any]],
) -> float:
    """Compute mean normalized Kendall-tau distance across corresponding probe graphs."""
    if probe_list_a is None or probe_list_b is None:
        return 0.0
    n_a = len(probe_list_a)
    n_b = len(probe_list_b)
    if n_a == 0 or n_b == 0:
        return 0.0
    
    n_compare = min(n_a, n_b)
    distances: List[float] = []
    for k in range(n_compare):
        d_k = kendall_tau_distance(probe_list_a[k], probe_list_b[k])
        distances.append(d_k)
        
    if not distances:
        return 0.0
    return float(np.mean(distances))


def compute_r_sav(
    colors: Sequence[int],
    c_base: Sequence[int],
    c_ref: Optional[Sequence[int]] = None,
    config: Optional[RewardConfig] = None,
) -> float:
    """Compute mean color savings reward r_sav.
    
    Formula:
        r_sav = mean_g clip((c_base(g) - c_h(g)) / max(1, c_base(g) - c_ref(g)), -1, 1)
        Fallback if c_ref is None: c_base(g) - fallback_ref_offset (denominator = 1).
    """
    if config is None:
        config = RewardConfig()
    if len(colors) == 0 or len(c_base) == 0:
        return 0.0
    if len(colors) != len(c_base):
        raise ValueError(
            f"Length mismatch: colors ({len(colors)}) vs c_base ({len(c_base)})"
        )
    if c_ref is not None and len(c_ref) != len(colors):
        raise ValueError(
            f"Length mismatch: c_ref ({len(c_ref)}) vs colors ({len(colors)})"
        )

    savings_list: List[float] = []
    for i in range(len(colors)):
        cb = float(c_base[i])
        ch = float(colors[i])
        if c_ref is None or c_ref[i] is None:
            cr = cb - float(config.fallback_ref_offset)
        else:
            cr = float(c_ref[i])
            
        denom = max(1.0, cb - cr)
        score = (cb - ch) / denom
        clipped = float(np.clip(score, config.clip_sav_lo, config.clip_sav_hi))
        savings_list.append(clipped)

    return float(np.mean(savings_list))


def compute_r_time(
    runtimes: Sequence[float],
    t_dsatur: Optional[Union[float, Sequence[float]]] = None,
    config: Optional[RewardConfig] = None,
) -> float:
    """Compute runtime penalty r_time.
    
    Formula:
        r_time = clip(t_h / t_budget - 1, 0, 1)
        t_budget = time_factor * time_of_vectorized_DSATUR
    """
    if config is None:
        config = RewardConfig()
    if not runtimes:
        return 0.0
    if t_dsatur is None:
        return 0.0

    if isinstance(t_dsatur, (int, float)):
        base_time = float(t_dsatur)
    else:
        if len(t_dsatur) == 0:
            return 0.0
        base_time = float(np.mean(t_dsatur))

    if base_time <= 0.0:
        return 0.0

    t_h = float(np.mean(runtimes))
    t_budget = config.time_factor * base_time
    if t_budget <= 0.0:
        return 0.0

    ratio_penalty = (t_h / t_budget) - 1.0
    return float(np.clip(ratio_penalty, 0.0, 1.0))


def compute_r_tb(
    color_classes_per_graph: Optional[Sequence[Sequence[int]]],
) -> float:
    """Compute optional tie-breaking score based on color class concentration in [0, 1]."""
    if not color_classes_per_graph:
        return 0.0

    graph_scores: List[float] = []
    for classes in color_classes_per_graph:
        if not classes:
            continue
        n = sum(classes)
        if n <= 0:
            continue
        # Herfindahl index of color class fractions
        h = sum((size / n) ** 2 for size in classes)
        graph_scores.append(h)

    if not graph_scores:
        return 0.0
    return float(np.clip(np.mean(graph_scores), 0.0, 1.0))


def compute_group_rewards_detailed(
    candidates: Sequence[CandidateResult],
    config: Optional[RewardConfig] = None,
    c_base_per_graph: Optional[Sequence[int]] = None,
    c_ref_per_graph: Optional[Sequence[int]] = None,
    t_dsatur_per_graph: Optional[Union[float, Sequence[float]]] = None,
) -> List[RewardBreakdown]:
    """Compute detailed reward breakdown for each candidate in a group.
    
    Execution semantics:
      1. Pre-reward r_pre = r_sav + a * r_tb - b * r_time.
         For any candidate with status != 'ok', r_pre = failure_reward (-1.0).
      2. Median r_pre is calculated across the ENTIRE candidate group.
      3. Diversity r_div is applied ONLY to candidates strictly above group median r_pre.
         Failed candidates are excluded from diversity peer comparisons.
      4. Final reward r = r_pre + d * r_div.
         Any failure status yields r = failure_reward (-1.0) exactly.
    """
    if config is None:
        config = RewardConfig()

    num_candidates = len(candidates)
    if num_candidates == 0:
        return []

    # Step 1: Compute r_pre, r_sav, r_time, r_tb for each candidate
    pre_results: List[Tuple[float, float, float, float]] = []
    for cand in candidates:
        if cand.status != VALID_STATUS:
            # Failed candidate: fixed penalty
            pre_results.append((config.failure_reward, 0.0, 0.0, 0.0))
            continue

        # Resolve c_base
        cb = cand.c_base_per_graph if cand.c_base_per_graph is not None else c_base_per_graph
        if cb is None:
            raise ValueError(
                "c_base_per_graph must be provided either on CandidateResult or at group level"
            )

        # Resolve c_ref
        cr = cand.c_ref_per_graph if cand.c_ref_per_graph is not None else c_ref_per_graph

        # Resolve t_dsatur
        td = cand.t_dsatur_per_graph if cand.t_dsatur_per_graph is not None else t_dsatur_per_graph

        # Calculate component rewards
        r_sav = compute_r_sav(cand.colors_per_graph, cb, cr, config)
        r_time = compute_r_time(cand.runtime_per_graph, td, config)
        r_tb = compute_r_tb(cand.color_classes_per_graph) if config.a > 0.0 else 0.0
        r_pre = r_sav + (config.a * r_tb) - (config.b * r_time)

        pre_results.append((r_pre, r_sav, r_time, r_tb))

    # Step 2: Compute group median of r_pre across all candidates
    all_r_pre = [res[0] for res in pre_results]
    median_r_pre = float(np.median(all_r_pre))

    # Step 3: Identify diversity peers (valid 'ok' candidates only)
    valid_peer_indices = [
        idx for idx, cand in enumerate(candidates) if cand.status == VALID_STATUS
    ]

    # Step 4: Compute r_div and final reward
    breakdowns: List[RewardBreakdown] = []
    for idx, cand in enumerate(candidates):
        r_pre, r_sav, r_time, r_tb = pre_results[idx]

        if cand.status != VALID_STATUS:
            # Failure status gets exact failure_reward (-1.0)
            breakdowns.append(
                RewardBreakdown(
                    status=cand.status,
                    r_sav=0.0,
                    r_time=0.0,
                    r_tb=0.0,
                    r_pre=config.failure_reward,
                    r_div=0.0,
                    reward=config.failure_reward,
                    is_eligible_diversity=False,
                )
            )
            continue

        # Diversity is applied ONLY if r_pre is strictly above group median
        is_above_median = r_pre > median_r_pre
        r_div = 0.0

        if is_above_median:
            # Peers: other valid candidates in group
            peers = [p for p in valid_peer_indices if p != idx]
            if peers:
                peer_dists: List[float] = []
                for p in peers:
                    d_peer = pairwise_probe_distance(
                        cand.probe_orderings, candidates[p].probe_orderings
                    )
                    peer_dists.append(d_peer)
                r_div = float(np.mean(peer_dists)) if peer_dists else 0.0

        final_reward = r_pre + (config.d * r_div)

        breakdowns.append(
            RewardBreakdown(
                status=cand.status,
                r_sav=r_sav,
                r_time=r_time,
                r_tb=r_tb,
                r_pre=r_pre,
                r_div=r_div,
                reward=final_reward,
                is_eligible_diversity=is_above_median,
            )
        )

    return breakdowns


def compute_group_rewards(
    candidates: Sequence[CandidateResult],
    config: Optional[RewardConfig] = None,
    c_base_per_graph: Optional[Sequence[int]] = None,
    c_ref_per_graph: Optional[Sequence[int]] = None,
    t_dsatur_per_graph: Optional[Union[float, Sequence[float]]] = None,
) -> List[float]:
    """Compute final scalar rewards for all candidates in a group."""
    breakdowns = compute_group_rewards_detailed(
        candidates,
        config=config,
        c_base_per_graph=c_base_per_graph,
        c_ref_per_graph=c_ref_per_graph,
        t_dsatur_per_graph=t_dsatur_per_graph,
    )
    return [b.reward for b in breakdowns]


def compute_advantages(
    rewards: Sequence[float],
    eps_std: float = 1e-8,
) -> np.ndarray:
    """Compute normalized GRPO advantages for a group of rewards.
    
    Formula:
        A_i = (r_i - mean(r)) / (std(r) + eps_std)
        
    Invariant:
        Standard deviation is computed as the POPULATION standard deviation (ddof=0).
        If std(r) <= eps_std or np.isclose(std(r), 0.0) (e.g. all candidates share identical rewards),
        all advantages are set to exactly 0.0 with NO NaN or Inf.
    """
    if len(rewards) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(rewards) == 1:
        return np.zeros((1,), dtype=np.float64)

    r_arr = np.asarray(rewards, dtype=np.float64)
    mean = float(np.mean(r_arr))
    # Population standard deviation
    std = float(np.std(r_arr, ddof=0))

    if std <= eps_std or math.isclose(std, 0.0, abs_tol=1e-12):
        return np.zeros_like(r_arr, dtype=np.float64)

    advantages = (r_arr - mean) / (std + eps_std)
    advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    return advantages
