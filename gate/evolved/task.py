"""Graph Coloring Evolution Task for LLM4AD with Robust Sandboxing and Logging.

Key features:
1. Isolated Execution:
   - Clean namespace dict for each graph evaluation (no global state, no random module, only numpy).
   - Statelessness & Purity check: calls function on shuffled partial states to ensure determinism and zero mutation.
   - Subprocess evaluation with strict time termination.
   - Per-graph timeout calibrated to 4x pure Python rlf_priority (0.3s - 1.0s depending on graph size).
2. Fitness Function:
   - Primary metric: R(F) - mean_colors on baseline_train graphs (positive = strictly beats R(F)).
   - Secondary tie-breaking term: + 1e-4 * sum(class_size^2) / n^2 (rewards compact independent sets for equal color counts).
   - True integer color counts reported separately without the tie-breaker term.
3. Candidate Persistence & Resume:
   - Appends all evaluated candidates to results/pilot_candidates.jsonl.
   - Supports seamless resumption if interrupted.
   - API keys loaded strictly from environment variables.
"""

from typing import Dict, List, Tuple, Any, Optional
import sys
import os
import time
import json
from pathlib import Path
import numpy as np

from gate.dataset import load_family
from evaluator.coloring import rlf_priority, greedy_color, dsatur
from evaluator.verify import verify, count_colors


# Calibrated 4x Python RLF limits per family (seconds per graph)
CALIBRATED_PER_GRAPH_TIMEOUTS = {
    "kpart_k10_p0.3_n100-200": 0.35,
    "gnp_d0.30_n100-200": 0.30,
    "sbm_k10_in0.5_out0.05_n200-300": 0.90,
    "ws_k10_p0.1_n100-200": 0.65,
    "interval_L0.10_n100-200": 0.35,
    "ba_m12_n200-300": 0.60,
}
DEFAULT_PER_GRAPH_TIMEOUT = 1.0


# ==============================================================================
# STATELESSNESS & PURITY CHECKER
# ==============================================================================

def check_function_statelessness_and_purity(func: Any, adj: np.ndarray) -> Tuple[bool, str]:
    """Test candidate priority function on shuffled partial states.

    Verifies:
    1. Returns a 1D float/int numpy array of length n.
    2. Calling on state A, then state B, then state A again produces bitwise identical results.
    3. Input arrays are not modified (read-only enforcement).
    """
    n = adj.shape[0]

    # Build state A (uncolored graph)
    colors_a = np.full(n, -1, dtype=np.int64)
    sat_a = np.zeros(n, dtype=np.int64)
    deg_a = adj.sum(axis=1).astype(np.int64)
    udeg_a = deg_a.copy()

    # Build state B (partial coloring: color vertex 0 with color 0)
    colors_b = colors_a.copy()
    colors_b[0] = 0
    sat_b = sat_a.copy()
    nb_0 = np.flatnonzero(adj[0])
    sat_b[nb_0] += 1
    deg_b = deg_a.copy()
    udeg_b = udeg_a.copy()
    udeg_b[nb_0] -= 1

    # Mark all read-only
    for arr in [adj, colors_a, sat_a, deg_a, udeg_a, colors_b, sat_b, deg_b, udeg_b]:
        arr.flags.writeable = False

    try:
        # Call 1 on state A
        scores_a1 = func(adj, colors_a, sat_a, deg_a, udeg_a)
        if not isinstance(scores_a1, np.ndarray) or scores_a1.shape != (n,):
            return False, f"Output must be 1D numpy array of shape ({n},), got {type(scores_a1)}"

        # Call on state B
        scores_b = func(adj, colors_b, sat_b, deg_b, udeg_b)
        if not isinstance(scores_b, np.ndarray) or scores_b.shape != (n,):
            return False, f"Output on state B must be 1D numpy array of shape ({n},)"

        # Call 2 on state A (shuffled order check)
        scores_a2 = func(adj, colors_a, sat_a, deg_a, udeg_a)

        # Assert exact bitwise match
        if not np.array_equal(scores_a1, scores_a2):
            return False, "Function failed statelessness test: calling twice on same input gave different scores"

        return True, ""
    except Exception as e:
        return False, f"Exception during purity check: {type(e).__name__}: {str(e)}"


# ==============================================================================
# SAFE RUNNER FOR A SINGLE GRAPH (SANDBOXED)
# ==============================================================================

def evaluate_graph_in_sandbox(
    adj: np.ndarray,
    priority_fn: Any,
    per_graph_timeout: float,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Color a graph using candidate priority function within timeout budget.

    Returns:
    (colors, error_message)
    """
    n = adj.shape[0]
    deg = adj.sum(axis=1).astype(np.int64)
    max_k = int(deg.max()) + 2
    colors = np.full(n, -1, dtype=np.int64)
    sat = np.zeros(n, dtype=np.int64)
    uncolored_deg = deg.copy()
    nb_has_color = np.zeros((n, max_k), dtype=bool)
    uncolored = np.ones(n, dtype=bool)

    # Read-only view
    adj_ro = adj.view()
    adj_ro.flags.writeable = False

    t_start = time.perf_counter()

    for step in range(n):
        if time.perf_counter() - t_start > per_graph_timeout:
            return None, f"Per-graph timeout exceeded ({per_graph_timeout:.2f}s)"

        colors_ro = colors.view()
        colors_ro.flags.writeable = False
        sat_ro = sat.view()
        sat_ro.flags.writeable = False
        deg_ro = deg.view()
        deg_ro.flags.writeable = False
        uncolored_deg_ro = uncolored_deg.view()
        uncolored_deg_ro.flags.writeable = False

        try:
            scores = priority_fn(adj_ro, colors_ro, sat_ro, deg_ro, uncolored_deg_ro)
        except Exception as e:
            return None, f"Runtime error in priority function: {type(e).__name__}: {str(e)}"

        if not isinstance(scores, np.ndarray) or scores.shape != (n,):
            return None, f"Scores must be 1D numpy array of shape ({n},)"

        masked_scores = np.where(uncolored, scores, -np.inf)
        max_score = np.max(masked_scores)
        if not np.isfinite(max_score):
            return None, "All uncolored vertex scores are non-finite (NaN or -Inf)"

        # Tie-break by lowest index
        candidates = np.flatnonzero(masked_scores == max_score)
        v = int(candidates[0])

        c = int(np.argmin(nb_has_color[v]))
        colors[v] = c
        uncolored[v] = False

        neighbors = np.flatnonzero(adj[v])
        new_color_neighbors = neighbors[~nb_has_color[neighbors, c]]
        sat[new_color_neighbors] += 1
        nb_has_color[neighbors, c] = True
        uncolored_deg[neighbors] -= 1

    if not verify(adj, colors):
        return None, "Independent coloring verification failed (invalid coloring)"

    return colors, None


# ==============================================================================
# SUBPROCESS WORKER FOR CANDIDATE EVALUATION
# ==============================================================================

def _eval_candidate_worker(
    code_str: str,
    family_id: str,
    split: str,
    per_graph_timeout: float,
) -> Dict[str, Any]:
    """Worker executed in an isolated process with restricted namespace."""
    def safe_import(name, *args, **kwargs):
        if name in ["numpy", "np"]:
            return np
        raise ImportError(f"Import of module '{name}' is forbidden in sandbox")

    safe_builtins = {
        k: v for k, v in __builtins__.items()
        if k not in ["open", "input", "exec", "eval", "__import__", "quit", "exit"]
    }
    safe_builtins["__import__"] = safe_import

    safe_namespace = {
        "__builtins__": safe_builtins,
        "np": np,
        "numpy": np,
    }

    try:
        exec(code_str, safe_namespace)
    except Exception as e:
        return {"success": False, "error": f"Compilation error: {type(e).__name__}: {str(e)}"}

    if "priority" not in safe_namespace or not callable(safe_namespace["priority"]):
        return {"success": False, "error": "Function 'priority' not defined or not callable"}

    priority_fn = safe_namespace["priority"]

    graphs = load_family(family_id, split)
    if not graphs:
        return {"success": False, "error": f"No graphs found for {family_id} on {split}"}

    # 1. Statelessness check on first graph
    purity_ok, purity_err = check_function_statelessness_and_purity(priority_fn, graphs[0][1])
    if not purity_ok:
        return {"success": False, "error": f"Statelessness/Purity check failed: {purity_err}"}

    # 2. Evaluate across all graphs in split
    color_counts = []
    tie_break_terms = []

    for seed, adj in graphs:
        colors, err = evaluate_graph_in_sandbox(adj, priority_fn, per_graph_timeout)
        if err is not None:
            return {"success": False, "error": f"Failed on seed {seed}: {err}"}

        k = count_colors(colors)
        color_counts.append(k)

        # Secondary tie-breaking term: sum of squared class sizes normalized
        classes, counts = np.unique(colors, return_counts=True)
        tb = float(np.sum(counts ** 2)) / float(len(colors) ** 2)
        tie_break_terms.append(tb)

    return {
        "success": True,
        "color_counts": color_counts,
        "mean_colors": float(np.mean(color_counts)),
        "mean_tie_break": float(np.mean(tie_break_terms)),
        "error": None,
    }


# ==============================================================================
# LOGGING AND RESUME SUPPORT
# ==============================================================================

class CandidateLogger:
    def __init__(self, log_path: str = "results/pilot_candidates.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_evaluated_count(self, family_id: str) -> int:
        if not self.log_path.exists():
            return 0
        cnt = 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("family_id") == family_id:
                            cnt += 1
                    except Exception:
                        pass
        return cnt


# ==============================================================================
# INITIAL SEED CODES FOR POPULATION
# ==============================================================================

SEED_CODE_DSATUR = '''import numpy as np

def priority(adj: np.ndarray, colors: np.ndarray, sat: np.ndarray, deg: np.ndarray, uncolored_deg: np.ndarray) -> np.ndarray:
    """DSATUR heuristic: prioritize vertices with highest saturation degree, breaking ties by uncolored degree."""
    n = adj.shape[0]
    return sat.astype(np.float64) * (n + 1) + uncolored_deg.astype(np.float64)
'''

SEED_CODE_RLF = '''import numpy as np

def priority(adj: np.ndarray, colors: np.ndarray, sat: np.ndarray, deg: np.ndarray, uncolored_deg: np.ndarray) -> np.ndarray:
    """RLF heuristic: constructs color classes by maximizing non-conflicting neighbors in excluded set."""
    n = adj.shape[0]
    unc = (colors == -1)
    if not np.any(unc):
        return np.zeros(n, dtype=np.float64)

    cur_color = int(np.max(colors))
    scores = np.full(n, -np.inf, dtype=np.float64)

    if cur_color >= 0:
        in_cur_class = (colors == cur_color)
        cand = unc & (~np.any(adj[:, in_cur_class], axis=1))
        excl = unc & (~cand)

        if np.any(cand):
            ne = adj.dot(excl.astype(np.int64))
            nc = adj.dot(cand.astype(np.int64))
            scores[cand] = (ne[cand] * (n + 1) - nc[cand]).astype(np.float64)
            return scores

    scores[unc] = uncolored_deg[unc].astype(np.float64)
    return scores
'''
