"""Prompt generator for Graph Coloring Evolution with Card conditioning."""

from typing import List, Dict, Any, Optional


SYSTEM_PROMPT = """You are an expert algorithm designer specializing in combinatorial optimization and graph coloring heuristics.
Your task is to design an innovative vertex priority function for a greedy graph coloring framework.

The greedy framework repeatedly:
1. Calls your `priority` function to score all vertices.
2. Selects the uncolored vertex with the highest score (ties broken by lowest index).
3. Assigns it the smallest feasible non-conflicting color (0, 1, 2, ...).
4. Updates incremental graph coloring states (colors, sat, uncolored_deg) and repeats until all vertices are colored.

Your goal is to minimize the total number of colors used on instances of a specific graph family.

### FIXED API CONTRACT:
```python
def priority(adj, colors, sat, deg, uncolored_deg):
    \"\"\"
    Parameters:
      adj:           (n, n) bool ndarray, symmetric, no self-loops (read-only)
      colors:        (n,) int64 ndarray, -1 = uncolored, >=0 color id (read-only)
      sat:           (n,) int64 ndarray, number of distinct colors among colored neighbors (read-only)
      deg:           (n,) int64 ndarray, total degree in full graph (read-only)
      uncolored_deg: (n,) int64 ndarray, number of uncolored neighbors (read-only)

    Returns:
      (n,) float ndarray of scores. Higher score = color earlier.
    \"\"\"
```

### STRICT SANDBOX EXECUTION RULES:
1. Positional arguments only: exactly 5 positional parameters (adj, colors, sat, deg, uncolored_deg).
2. Absolutely NO default parameter values anywhere (not in priority, helper functions, nested functions, or lambdas).
3. NO identifiers or attributes starting with '_' (no dunders, no __class__, no __dict__, no _name).
4. NO `import` or `from ... import` statements. `np` is pre-imported. Only standard whitelisted numpy functions and array methods are permitted:
   - Whitelisted np functions: np.zeros, np.ones, np.full, np.where, np.flatnonzero, np.nonzero, np.dot, np.matmul, np.diag, np.diagonal, np.triu, np.tril, np.outer, np.bincount, np.unique, np.sign, np.sum, np.mean, np.min, np.max, np.maximum, np.minimum, np.argmax, np.clip, np.count_nonzero, np.cumsum, np.arange, np.isin, np.any, np.all.
   - Whitelisted array methods & attributes: .dot, .sum, .max, .min, .argmax, .argmin, .any, .all, .astype, .copy, .reshape, .ravel, .nonzero, .cumsum, .diagonal, .clip, .round, .T, .shape, .size, .dtype.
   - Strictly forbidden: NO np.linalg, NO np.random, NO I/O, NO .view, NO .data, NO .base, NO non-whitelisted modules/methods.
5. NO mutable module-level variables (only immutable numbers and def allowed at top-level).
6. NO random generation (`np.random` is forbidden) and NO I/O (`open`, `eval`, `exec`, `np.savetxt`, `np.loadtxt` are forbidden).
7. Must be a pure, deterministic function. All inputs are read-only; do NOT mutate them.
8. Must be efficient: vectorized numpy operations preferred, execution must complete within < 0.5s per graph.
"""


def build_user_prompt(
    family_id: str,
    card: str,
    r_train: float,
    parent_candidates: List[Dict[str, Any]],
    operator: str = "crossover",
    past_behaviors: Optional[List[str]] = None,
) -> str:
    """Build user prompt conditioning on the family card, parent candidate programs, and operator."""
    parents_str = ""
    for idx, p in enumerate(parent_candidates, 1):
        fit_str = f"Mean colors: {p.get('mean_colors', 'N/A')}"
        if "fitness" in p:
            fit_str += f" (Fitness vs R(F): {p['fitness']:+.4f})"
        parents_str += f"\n### Parent Program {idx} ({fit_str}):\n```python\n{p['program'].strip()}\n```\n"

    behaviors_str = ""
    if past_behaviors:
        behaviors_str = "### Previously Explored Behaviors (DO NOT REPEAT THESE APPROACHES):\n"
        for idx, desc in enumerate(past_behaviors[:8], 1):
            behaviors_str += f"- Mechanism {idx}: {desc}\n"
        behaviors_str += "\n"

    if operator == "explore":
        task_instruction = (
            "You must propose a structurally NOVEL priority heuristic that explores a completely different "
            "algorithmic hypothesis from the parent programs and previously explored behaviors. "
            "Do NOT make minor tweaks or adjust coefficients on DSATUR or RLF. Instead, invent and implement "
            "a genuinely distinct mechanism tailored to the instance family card above. "
            "Promising ideas to explore:\n"
            "  * Common neighbor counts / triangle density around uncolored vertices (e.g. using adj.dot(adj)).\n"
            "  * 2-hop neighborhood expansion or second-order saturation / reachability effects.\n"
            "  * Independent set / cluster / community cues (e.g. uncolored degree ratios, local clustering proxies).\n"
            "  * Dynamic non-standard tie-breaking (e.g. neighbor degree variance or degree-ratio proxies).\n"
            "Remember all strict sandbox constraints (pure function, exactly 5 positional args, no default parameters, "
            "no forbidden attributes/imports, pure vectorized numpy, execution < 0.5s per graph)."
        )
    elif operator == "crossover":
        task_instruction = (
            "Analyze the structural characteristics described in the card above and the mechanisms used in the parent "
            "programs. Create a NEW priority function that synergizes the strengths of both parents or introduces a "
            "novel combination of graph features (e.g. saturation dynamics, candidate/exclusion sets, degree ratios) "
            "specifically tailored to beat the baseline on this family."
        )
    else:  # refine / mutate
        task_instruction = (
            "Analyze the best-performing parent program above and the specific structural properties from the card. "
            "Identify bottlenecks, poor tie-breaking, or sub-optimal weighting on this family. Propose a refined, "
            "improved priority function that achieves fewer colors."
        )

    prompt = f"""Target Graph Family: `{family_id}`
Instance Family Card:
"{card}"

Current Baseline Performance on Training Graphs: R(F) = {r_train:.3f} colors.
{parents_str}
{behaviors_str}### Objective:
{task_instruction}

Respond ONLY with the complete Python code implementing `priority(adj, colors, sat, deg, uncolored_deg)` inside a single ```python ... ``` block. Do not include markdown commentary outside the code block."""
    return prompt
