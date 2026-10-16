"""LLM4AD Task Adapter for Graph Coloring Heuristics.

Implements Task 7.1 from PLAN_PHASE6_ONWARD.md:
- Inherits from llm4ad Evaluation base class (with graceful fallback if llm4ad not installed).
- Evaluates candidate priority functions using the project's strict 4-layer sandbox and verify.py.
- Returns -mean_colors (since LLM4AD/EoH/ReEvo maximize fitness).
- Uses exact vectorized API contract and loop.py system description.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gate.dataset import load_family, families
from gate.evolved.sandbox import evaluate_candidate_safe
from gate.evolved.prompts import SYSTEM_PROMPT


try:
    from llm4ad.base.evaluate import Evaluation as BaseEvaluation
except ImportError:
    class BaseEvaluation:
        """Fallback base Evaluation class compatible with LLM4AD interface."""
        def __init__(self, template_program: str = "", task_description: str = ""):
            self.template_program = template_program
            self.task_description = task_description

        def evaluate_program(self, program: str) -> Any:
            raise NotImplementedError("evaluate_program must be implemented by subclass.")


TEMPLATE_PROGRAM = '''def priority(adj, colors, sat, deg, uncolored_deg):
    """
    Parameters:
      adj:           (n, n) bool ndarray, symmetric, no self-loops (read-only)
      colors:        (n,) int64 ndarray, -1 = uncolored, >=0 color id (read-only)
      sat:           (n,) int64 ndarray, number of distinct colors among colored neighbors (read-only)
      deg:           (n,) int64 ndarray, total degree in full graph (read-only)
      uncolored_deg: (n,) int64 ndarray, number of uncolored neighbors (read-only)

    Returns:
      (n,) float ndarray of scores. Higher score = color earlier.
    """
    # Example (DSATUR baseline):
    n = adj.shape[0]
    return sat * (n + 1) + uncolored_deg
'''


class GraphColoringEvaluation(BaseEvaluation):
    """LLM4AD Evaluation adapter evaluating graph coloring heuristics on a specific family."""

    def __init__(
        self,
        family_name: str,
        split: str = "train",
        seed_range: Optional[Tuple[int, int]] = None,
        per_graph_timeout: float = 1.0,
        overall_timeout: float = 20.0,
        return_penalty_on_fail: bool = False,
    ):
        self.family_name = family_name
        self.split = split
        self.seed_range = seed_range
        self.per_graph_timeout = per_graph_timeout
        self.overall_timeout = overall_timeout
        self.return_penalty_on_fail = return_penalty_on_fail

        # Load graphs
        raw_graphs = load_family(family_name, split=split)
        if seed_range is not None:
            s_min, s_max = seed_range
            self.graphs = [g for g in raw_graphs if s_min <= g[0] <= s_max]
        else:
            self.graphs = raw_graphs

        # Retrieve card description if available
        card = ""
        for f in families():
            fid = f.get("family_id") if isinstance(f, dict) else getattr(f, "name", None)
            if fid == family_name:
                if isinstance(f, dict):
                    cards_dict = f.get("cards", {})
                    card = cards_dict.get("card", "") if isinstance(cards_dict, dict) else ""
                else:
                    card = getattr(f, "card", "")
                break

        task_desc = f"{SYSTEM_PROMPT}\n\n### TARGET GRAPH FAMILY: {family_name}\nDescription: {card}\n"

        super().__init__(
            template_program=TEMPLATE_PROGRAM,
            task_description=task_desc,
        )
        self.last_result: Optional[Dict[str, Any]] = None

    def evaluate_program(self, program: str) -> Optional[float]:
        """Evaluate program through the project's 4-layer sandbox.
        
        Returns:
            -mean_colors on success (since LLM4AD maximizes fitness),
            None (or -1e6 if return_penalty_on_fail=True) on failure.
        """
        res = evaluate_candidate_safe(
            code_str=program,
            graphs=self.graphs,
            per_graph_timeout=self.per_graph_timeout,
            overall_timeout=self.overall_timeout,
        )
        self.last_result = res

        if not res.get("success", False) or res.get("mean_colors") is None:
            return -1e6 if self.return_penalty_on_fail else None

        # LLM4AD maximizes fitness -> return negative mean colors
        return -float(res["mean_colors"])
