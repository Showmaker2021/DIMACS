# AGENTS.md: One-Shot Heuristic Generation for Graph Coloring

## 0. Context

Research project: fine-tune a 7B code model with RL (GRPO) to write a graph-coloring
priority function (DSATUR-style) directly from a short text "card" describing an instance
family, with no per-family evolutionary search at deployment.

**You are working on PHASE 1 ONLY: the go/no-go gate (oracle gap).**
Do NOT write any RL / TRL / GRPO / LoRA / regressor / selector / DIMACS-loader code.
Do not add dependencies or features not listed here. If something is ambiguous, write
your assumption in `NOTES.md` and continue; do not stop to ask unless blocked.

### Gate question
For each family F, how much better is the best heuristic for F than the best single
heuristic? Report three gaps (in number of colors, averaged over test graphs):

- **Gap A** = best_single (DSATUR, RLF, universal) minus parametric_tuned_per_family
- **Gap B** = best_single minus evolved_per_family (EoH/ReEvo, small subset of families)
- **Gap C** = parametric_tuned minus evolved (headroom left for an LLM beyond CMA-ES)

If average Gap B is under about 3-5% in colors, the project needs rethinking. Gap C
shows whether an LLM can add anything beyond tuned weights. All gaps need confidence
intervals (bootstrap over graphs and seeds), because 3-5% on a 200-node graph is about
1 color and noise can look like signal.

## 1. Environment

- Python 3.11+, `uv` or `venv`. Pin all versions in `requirements.txt`.
- Allowed libs: numpy, scipy, numba, networkx, pandas, pycma (`cma`), pytest, tqdm, matplotlib.
- Phase 1 needs no GPU and no LLM calls, except Step 7 (optional, evolved baseline).
- Everything seeded and deterministic. One `seed` argument everywhere; no global RNG state.

## 2. Repo layout

```
project/
  AGENTS.md
  NOTES.md                 # your assumptions, decisions, open questions
  requirements.txt
  graphs/
    families.py            # family specs + card text + generators
    splits.py              # train/test graph splits per family
  evaluator/
    coloring.py            # greedy framework, DSATUR, RLF
    verify.py              # independent verifier
    parametric.py          # parametric DSATUR
  baselines/
    tune_parametric.py     # CMA-ES per family
    universal.py           # one parametric heuristic tuned on all train families
    evolved/               # Step 7 only: wrapper around LLM4AD / ReEvo output
  scripts/
    run_gate.py            # produces the gate table
  tests/
  third_party/             # READ-ONLY reference repos (see section 9)
  results/                 # CSV outputs (gitignored except small tables)
```

## 3. Priority function API (fixed contract)

All heuristics, handwritten, parametric, or later LLM-generated, share this API:

```python
def priority(adj, colors, sat, deg, uncolored_deg) -> np.ndarray:
    """
    adj:           (n, n) bool, symmetric, no self loops (read-only)
    colors:        (n,) int, -1 = uncolored, else color id >= 0 (read-only)
    sat:           (n,) int, number of distinct colors among colored neighbors
    deg:           (n,) int, degree in the full graph
    uncolored_deg: (n,) int, number of uncolored neighbors
    returns:       (n,) float scores; higher = color earlier
    """
```

Rules:
- Pure function: no globals, no state between calls, no I/O, no randomness, no recursion
  into a solver. Inputs must not be mutated.
- The **framework** (not the function) does the rest: selects the uncolored vertex with
  max score (ties: lowest index), assigns the smallest feasible color, repeats.
- The framework ignores scores of already-colored vertices.
- DSATUR expressed in this API: `sat * (n + 1) + uncolored_deg` (document tie handling).
- RLF is NOT expressible in this API (it builds color classes). Implement it separately
  as a reference baseline.

## 4. Evaluator (most important part; write it first and test it hard)

`evaluator/coloring.py`
- `greedy_color(adj, priority_fn) -> colors` implementing the framework above.
- Maintain `sat` and `uncolored_deg` incrementally (do not recompute from scratch each step).
- `dsatur(adj)` as a standalone Numba implementation (fast path), plus DSATUR through
  `greedy_color` with the formula above. Both must give identical color counts.
- `rlf(adj)`: Recursive Largest First.
- Target: DSATUR on a 200-node graph in a few ms.

`evaluator/verify.py`
- `verify(adj, colors) -> bool`: independent code path (no import from `coloring.py`).
  Checks length, all colors >= 0, and no edge with equal endpoint colors.
- `count_colors(colors)`.
- Every coloring produced anywhere in the project must pass `verify` before its color
  count is recorded.

`evaluator/parametric.py`
- Score = `w1*sat + w2*deg + w3*uncolored_deg + w4*(sat/ max(1, max sat)) + w5*(uncolored_deg / max(1, max uncolored_deg))`
  (5 weights, vectorized numpy; propose a better small feature set in NOTES.md if you
  have a reason). Weights normalized so the search space is bounded.
- Must reduce to DSATUR for a specific weight setting (test this).

## 5. Graph families and cards

`graphs/families.py`
- Define 16-24 families, each with: `name`, generator `gen(seed) -> adj`, parameters, and
  a short **card string** like `"nodes 100-300, density 0.3, high degree variance, clustered"`.
- Cover at least these structure types: G(n,p) random (several densities), random
  geometric, scale-free (Barabasi-Albert), planted partition / stochastic block model,
  Watts-Strogatz small world, interval graphs, queen-like grids, Mycielski-like,
  hybrids (e.g. geometric + random edges). Vary n in 50-300.
- Each family tagged with `structure_type` so splits can hold out whole types later.
- Card text is generated programmatically from measured/declared properties, not
  handwritten per family, so it is reproducible. Also expose the numeric features
  (n range, density, degree variance, clustering) for later use.

`graphs/splits.py`
- Per family: 20 train graphs, 20 validation graphs, 50 test graphs, disjoint seeds.
- Never tune on test graphs. Document seed ranges.

## 6. Baselines for the gate

`baselines/tune_parametric.py`
- For each family, CMA-ES (`cma`) over the weights, objective = mean colors on that
  family's train graphs, evaluated on validation graphs for model selection, reported
  on test graphs. Fixed evaluation budget per family (e.g. 500-2000 evaluations); record it.
- Run with several seeds (>=5) and report mean and spread.

`baselines/universal.py`
- Same CMA-ES but one weight vector tuned on the union of all train graphs.

Best single heuristic = the best of {DSATUR, RLF, universal} on the family's test
graphs, **chosen using validation graphs, not test**.

## 7. Evolved baseline (optional, only after Steps 1-6 are done and green)

- Pick 3-5 representative families. Run per-family evolution (EoH or ReEvo, via LLM4AD
  if it supports it) with a small budget (a few hundred LLM calls), using the same
  priority API and same train/val/test graphs.
- Put wrapper code in `baselines/evolved/`. Evaluate evolved heuristics with the same
  `verify` and evaluator. Log number of LLM calls per family.
- Ask the user for API keys/budget before running anything that costs money. Never
  hardcode keys.

## 8. Gate script

`scripts/run_gate.py` writes `results/gate_table.csv` and a printed summary with, per
family: mean colors (test) for DSATUR, RLF, universal, parametric_tuned, evolved (if
available), then Gap A, Gap B, Gap C in colors and in %, each with 95% bootstrap CI.
Also print the average across families and a final line:
`GATE: PASS/FAIL (avg Gap B = x.x% [lo, hi])`, with threshold 4% as a default
(configurable). Do not make the decision for the user; just print the numbers.

## 9. Reference repos (read-only)

If told to, shallow-clone into `third_party/` with `git clone --depth 1`, never modify
them, and record URL, commit hash and license in `third_party/REFS.md`. Candidates:
LLM4AD (Optima-CityU), ReEvo (ai4co/reevo), EoH (FeiLiu36/EoH). Do not read whole
repos; answer specific questions (how is a task defined, how is the evaluator called,
how to add a task) and write a short summary into `NOTES.md`. Do not copy code from them
into the core evaluator.

## 10. Tests (required, `pytest`)

- `dsatur` (Numba) == `greedy_color` with the DSATUR formula, on 200+ random graphs.
- DSATUR color count within expectation versus `networkx.greedy_color(strategy="saturation_largest_first")` (tie-breaking may differ; compare distributions, and assert both are proper).
- `verify` accepts proper colorings and rejects: wrong length, uncolored vertex, a single bad edge, symmetric/asymmetric inputs.
- RLF always returns proper colorings.
- Parametric with the DSATUR weights reproduces DSATUR.
- Family generators: deterministic given seed, symmetric adjacency, no self loops,
  train/val/test seeds disjoint.
- Priority function purity: calling twice with the same inputs gives the same output and
  does not mutate inputs.

## 11. Order of work and done criteria

1. Skeleton, `requirements.txt`, test setup. (Done: `pytest` runs.)
2. `verify.py`, `coloring.py` (greedy framework, DSATUR, RLF) + tests. (Done: all green.)
3. `families.py`, `splits.py` + tests. (Done: table of families with cards printed.)
4. `parametric.py` + tests.
5. `tune_parametric.py`, `universal.py`; run on all families. (Done: results CSV.)
6. `run_gate.py` producing Gap A (and the DSATUR/RLF/universal/parametric columns).
7. Optional: evolved baseline and Gap B/C.

Stop after each numbered step with a short summary of what was done and test status.

## 12. Rules

- Correctness over speed: every reported color count comes from a verified coloring.
- No test-set tuning, ever. No silent changes of splits or seeds once results exist.
- Keep code small and readable; type hints and short docstrings; no premature abstraction.
- Do not modify `third_party/`. Do not commit large data or secrets.
- Record every non-obvious decision in `NOTES.md`.
- If a result looks too good (e.g. parametric far better than DSATUR everywhere), first
  suspect a bug, check verification and the split, and report it.
