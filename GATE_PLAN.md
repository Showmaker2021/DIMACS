# GATE_PLAN.md: oracle-gap gate on dataset v1

Read this together with `AGENTS.md`. **This file wins where they conflict.** The dataset
now exists (`data_v1/`, frozen as v1.0): do NOT regenerate graphs, do NOT write graph
generators or `families.py`/`splits.py` from AGENTS.md sections 2, 5. Keep from AGENTS.md:
the priority-function API (section 3), the evaluator and independent verifier (section 4),
tests (section 10) and rules (section 12).

Scope: phase 1 only. No RL / TRL / GRPO / LoRA / regressor / DIMACS loader.
If something is ambiguous, write your assumption in `NOTES.md` and continue.

## 1. The question the gate answers

For each family F, how much better is the best heuristic *for F* than a fixed reference?

Reference **R(F)** = the best of {DSATUR, RLF, universal parametric} on F, chosen by mean
colors on F's **val** graphs (this is the strongest non-per-family baseline).
Per-family heuristics:
- **P(F)** = parametric DSATUR with weights tuned on F (CMA-ES).
- **E(F)** = best heuristic found by per-family evolution (EoH/ReEvo), on a subset of families.

Gaps, in colors and in %, on **test** graphs (50 per family), mean over graphs:
- **Gap A(F)** = R(F) - P(F): what non-LLM per-family tuning gains.
- **Gap B(F)** = R(F) - E(F): what per-family evolution gains (the headline gate number).
- **Gap C(F)** = P(F) - E(F): what an LLM adds beyond tuned weights (can be negative).
Percent gap = gap / R(F). Also report the same gaps against plain DSATUR (R=DSATUR), since
the proposal's reward is relative to DSATUR.

Do not decide go/no-go yourself: print the numbers; the decision rule (about 3-5% average
Gap B) is the user's call.

## 2. Data contract (read-only)

`data_v1/` contains `config.json`, `families.json`, `manifest.jsonl`,
`graphs/<family_id>/<split>.npz` (key `s<seed>` -> int32 edge list (m,2), i<j; n from the
manifest). Provide `gate/dataset.py`:
- `load_family(family_id, split) -> list[(seed, adj: np.ndarray bool (n,n))]`
- `families(pool=None)`: records from `families.json` (fields: `pool`, `structure_type`,
  `labels`, `cards`, `n_range`).
- optional sha256 check on load (`n`, edges) against the manifest.
Splits: use `baseline_train` graphs only (train seeds 0-19) for tuning/evolution, **val**
for choosing among candidates (CMA-ES restarts, R(F), which evolved heuristic), **test**
only for the final numbers. Never tune on test. Never touch probe seeds.
Labels in `families.json` (DSATUR, LB, headroom) are provisional; do not use them as results.

## 3. Evaluator (write first, test hard)

Follow AGENTS.md section 3-4. Requirements for the gate:
- `numba`-jitted greedy coloring with a parametric priority (see section 4) and jitted
  DSATUR. Target: under ~2 ms for n=200, so 1000 CMA-ES evaluations x 20 graphs take
  seconds. (numba is allowed for this project; pin the version.)
- A plain-Python path `greedy_color(adj, priority_fn)` for arbitrary priority functions
  (needed for evolved code); vectorised numpy, calls priority once per step.
- RLF implemented separately (it is not expressible in the priority API).
- Independent `verify(adj, colors)`; every color count recorded anywhere must come from a
  verified coloring.
- Tests: jitted DSATUR == plain `greedy_color` with the DSATUR formula (colors count, many
  graphs); verifier rejects bad colorings; RLF proper; priority purity (no mutation);
  results identical across runs (determinism).
- `provisional dsatur` numbers in the dataset labels should roughly match your DSATUR
  (small differences from tie-breaking are fine; large ones indicate a bug).

## 4. Parametric DSATUR and CMA-ES (no LLM)

`gate/parametric.py`: score = w . phi, features phi from the API inputs, e.g.
`sat, udeg, deg, sat/max(deg,1), udeg/max(deg,1)` (propose a better small set in NOTES.md
if you have a reason). Requirements:
- DSATUR must be exactly representable (including its tie logic) and CMA-ES must start from
  it, so tuned >= DSATUR on the tuning graphs by construction. Document the encoding.
- Ties: lowest index. No randomness inside the heuristic.
`gate/tune.py`:
- Per family, CMA-ES (`cma`) on `baseline_train` graphs, objective = mean colors (add a
  tiny tie-breaking term, e.g. mean of normalized color-class imbalance or the number of
  vertices colored with the last color, only to guide search; report real colors).
- Budget fixed and logged (start with ~600 evaluations, 3 seeds; scale later). Pick the
  best seed/solution by val colors, report on test. Save weights, per-seed val/test, runtime.
- `universal`: same procedure on the union of baseline_train graphs of `train_pool` families
  only (so the universal heuristic never sees held-out types/sizes/combos).
- Parallelise over families (`ProcessPoolExecutor`; the user's machine is Windows, so keep
  workers top-level picklable and guard `if __name__ == "__main__"`).
- Rough cost estimate (unmeasured): ~tens of seconds per family-seed with numba. If it is
  much slower, report it instead of lowering quality silently.

## 5. Gate script and statistics

`scripts/run_gate.py` -> `results/gate_per_family.csv` and `results/gate_summary.txt`.
Per family: pool, type, mean test colors for DSATUR, RLF, universal, P(F), E(F) if
available, R(F) and which one it is, Gap A/B/C (colors, %), 95% CI.
- Paired differences per test graph; bootstrap over graphs for family CIs.
- Aggregate over families with a **cluster bootstrap over families** (resample families, then
  graphs). Report averages overall, per pool (train_pool, heldout_type, heldout_size,
  heldout_combo) and per structure type, plus the number of families with Gap > 3%.
- Show how many families have Gap A near zero (nothing to learn) vs large.
- Final printed block, no verdict: `avg Gap A = x.x% [lo, hi]`, `avg Gap B = ...` (on the
  evolved subset), `avg Gap C = ...`, and the same for the subset of families without E.

## 6. Evolved baseline (costs money: ask first)

Do this only after sections 3-5 are done and the free results are reported.
1. Shallow-clone LLM4AD into `third_party/` (read-only; record URL, commit, license in
   `third_party/REFS.md`). Answer in NOTES.md: how a task is defined, how the evaluator is
   called, how to add a task. Do not copy code into the core evaluator.
2. Define a task `graph_coloring_priority` using our API and evaluator (fitness = mean
   colors on the family's `baseline_train` graphs). Run LLM-generated code in a separate
   process with a timeout and memory cap; log crashes. (Full anti-hacking sandbox is a later
   phase; here just enforce the API, the time cap and independent verification.)
3. **Propose** a family subset (6-8): about one per structure type in `train_pool`, chosen
   for non-trivial headroom, plus 2 low-headroom controls. Show the list and the cost
   estimate (LLM calls x model price) and **wait for the user's approval and API key**.
   Never hardcode keys; read them from environment variables.
4. Pilot on 2-3 families with a small budget (~100-200 LLM calls each) before scaling.
   Log calls, tokens, cost, best fitness curve. Support resume and caching.
5. Pick the final heuristic per family by val, report on test. Save the code of every
   selected heuristic (it becomes SFT data later).

## 7. Order of work and stopping points

1. `dataset.py` + evaluator + tests. **Stop:** tests green; show DSATUR vs dataset labels.
2. `parametric.py` + `tune.py` on 3 families as a dry run. **Stop:** show runtime, weights, gains.
3. Run all kept families (universal + per-family). **Stop:** `gate_per_family.csv` with Gap A, DSATUR/RLF/universal columns.
4. `run_gate.py` statistics. **Stop:** summary text (the user reviews before any spending).
5. LLM4AD reading and task definition. **Stop:** cost estimate + family list, wait for approval.
6. Pilot, then full evolution on the approved subset. **Stop:** final Gap B/C table.

## 8. Report format back to the user (after each stop)

What was done, test status, files produced, key numbers with CIs, anything suspicious.
If a result looks too good (for example tuned far better than DSATUR on every family),
suspect a bug first: check verification, the split used, and test leakage.

## 9. Rules

- Test graphs are used only for final numbers. Tuning and selection use baseline_train and val.
- Everything seeded and deterministic; log all seeds, budgets and versions.
- Every reported color count comes from a verified coloring.
- No silent changes to dataset, splits, budgets or thresholds once results exist.
- Do not modify `data_v1/` or `third_party/`. No secrets in the repo.
- Keep code small and readable; type hints; record non-obvious decisions in `NOTES.md`.
