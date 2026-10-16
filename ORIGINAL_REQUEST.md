# Original User Request

## 2026-10-16T02:52:04Z

Implement the foundational Phase 6 / Batch A modules from `PLAN_PHASE6_ONWARD.md` for the Graph Coloring project: SEEDS registry and invariant tests (Task 6.0), Dataset v1.2 splits definition on EXISTING families (Task 8.1, partial), Evaluation harness with coverage@k (Task 9.2), and GRPO reward logic (Task 11.1).

Working directory: repo root (use pathlib; no hardcoded drive letters)
Integrity mode: development

## Step 0 (before writing any code)
Read `PLAN_PHASE6_ONWARD.md` (Sections 2, 3, 5, 6.0, 8.1, 9.2, 11.1, 14), `NOTES.md`, and the existing modules you will touch or reuse (`gate/evolved/loop.py`, the 4-layer sandbox, `evaluator/verify.py`, DSATUR/RLF implementations, `data/data_v1_1/` metadata). Write a summary of at most 10 lines: the exact vectorized API signature `priority(adj, colors, sat, deg, uncolored_deg)`, how the sandbox is invoked, how DSATUR/RLF colorings are computed, and your implementation plan. If anything in this prompt contradicts the code or the plan, STOP and list the questions instead of guessing.

## General Rules (strict)
1. Do NOT modify: `data/data_v1/`, `data/data_v1_1/`, `evaluator/verify.py`, existing `.done` lock files, sandbox whitelist. Add a test (`tests/test_frozen_files.py`) that hashes these paths against a recorded manifest (or uses `git diff --exit-code` when git is available) and fails on any change.
2. All 52 existing pytest tests must stay green. Every new component needs dedicated unit tests.
3. Zero real LLM API calls; zero API keys written, printed, or committed. Config references env-var NAMES only.
4. Only APPEND to `NOTES.md`; never edit existing sections.
5. Heavy/optional dependencies (torch, transformers, vLLM) must be imported lazily so tests run on a CPU-only Windows machine. Multiprocessing must be Windows-safe (`spawn`, `if __name__ == "__main__"` guards, picklable workers).
6. Scope: only what is listed here. Anything else you notice goes into NOTES.md under "Additional findings".

## Execution order
R1 -> R2 -> R3 and R4 (R2 depends on SEEDS.md; R3/R4 depend on both).

---

## R1. SEEDS registry (Task 6.0)

- Create `SEEDS.md` with a machine-readable markdown table with columns:
  `name | start | end | status | purpose`, ranges CLOSED on both ends, status in {frozen, proposed}.
  Rows:
  * frozen: train 0-99, val 1000-1019, test 2000-2049, teacher_val 3000-3019, test_pilot 4000-4199, probe 9000-9009
  * proposed: teacher_rescore 6000-6049, rl_train_pool 100000-199999, sft_val_probe 5000-5019, test_interp 7000-7049, test_extrap 7100-7149, test_comp 7200-7249, test_heldout_type 7300-7349
  If any proposed range collides with a frozen/used range in the repo, choose another and document why.
- `tests/test_seeds.py`: parse SEEDS.md programmatically; assert (a) every row parses, (b) start <= end, (c) all ranges pairwise disjoint (including rl_train_pool vs everything), (d) all frozen rows match the values above, (e) all required names are present.
- Create `configs/teacher_llm.yaml`: provider, model name placeholder, base temperature, max_tokens, rate limit (requests/min, pacing seconds), retry/backoff settings, and `api_key_env: <ENV_VAR_NAME>` only. Add a test that the file parses and contains no string resembling a key.
- Cost logging in `gate/evolved/loop.py`: verify that EVERY LLM call logs `prompt_tokens`, `completion_tokens`, `latency_s`, `operator`, `temperature`, `status`, plus optional `cost_dollars` (null if unknown). You ARE allowed to modify `loop.py`, but ONLY by adding log fields (no behavior change). Add a test with a mock LLM verifying the fields appear in the log, and confirm existing loop tests still pass.

## R2. Dataset v1.2 splits on existing families (Task 8.1, partial)

Scope decision: this batch does NOT write a new parametric-variant generator. It partitions the families that already exist (106 families, v1.1 metadata) into splits. Some splits (test_interp, test_extrap, test_comp) may therefore be small; record the number of families per split in NOTES.md and flag any split with fewer than 3 families.

- Create `data/data_v1_2/` (do not touch v1 / v1_1). Build `data/data_v1_2/splits.json` with, for every family: `family_id`, `structure_type`, and attributes `n_range`, `density_bin`, `degree_variance_bin`, `clustering_bin`, derived deterministically from the existing family parameters or measured on sampled graphs (document the binning thresholds in the file itself under a `"binning"` key).
- Groups: `train_pool` (must include `scale_free` per v1.1), `val`, `test_interp`, `test_extrap`, `test_comp`, `heldout_type` (= `small_world`). In `test_comp`, list explicitly the held-out attribute COMBINATIONS under a `"heldout_combinations"` key; those combinations must be absent from `train_pool`.
- Each group references its seed range name from SEEDS.md (e.g. test_interp -> "test_interp").
- Provide `scripts/build_splits.py` that regenerates splits.json deterministically (no randomness without a fixed seed recorded in the file).
- `tests/test_splits.py` verifies:
  * every family appears in exactly one group (or in an explicitly allowed overlap documented in the file; default is none);
  * no held-out compositional combination appears in `train_pool`;
  * `small_world` appears neither in `train_pool` nor in `val`;
  * extrapolation families really exceed the train range in n or density (assert against train_pool maxima);
  * every referenced seed-range name exists in SEEDS.md and ranges are consistent;
  * `scale_free` is in `train_pool`.

## R3. Offline evaluation harness (Task 9.2)

### `eval/coverage.py` (pure functions, no I/O)
- Define and DOCUMENT one canonical notion of "beats R(F)": a candidate beats R(F) on a card iff its mean colors on the split graphs is smaller than R(F) by at least `eps` colors (default `eps = 0.05`, configurable); ties do NOT count as wins. R(F) = min(mean DSATUR, mean RLF) on the same graphs.
- Implement `coverage_at_k` (fraction of cards with >= 1 winner among k candidates), `best_of_k` (mean over cards of the best candidate quality among k), and unbiased `pass_at_k(n, c, k)` (Chen et al. estimator, numerically stable) for n >= k generated samples.
- Handle edge cases: k > n (error), zero cards, all candidates failing.

### `eval/eval_model.py`
- CLI: `--backend {mock,hf,vllm,api} --split NAME --k K --n-samples N --temperature T --n-graphs M --run-id ID --seed S`.
- Backends behind one interface (`generate(prompt, n, temperature, seed) -> list[str]`). `mock` is a deterministic fake (e.g. emits DSATUR-like, RLF-like, broken, and slow code). `hf`/`vllm`/`api` are thin stubs that import lazily and raise a clear error if dependencies/keys are missing; they must not be exercised in tests.
- Evaluation: run each candidate through the project's 4-layer sandbox and verified color counting on the graphs of the requested split (seeds from SEEDS.md), in a pool of worker processes.
- Metrics per card and in summary: valid-rate and sandbox failure counts by type; mean colors; savings vs R(F) and vs DSATUR (colors and %); coverage@k, best-of-k, pass@k; runtime ratio vs vectorized DSATUR; code length; fraction of candidates behaviorally equal to DSATUR/RLF (fingerprint on probe seeds 9000-9009); within-card diversity (number of distinct fingerprints).
- Cache: key = `(code_hash, graph_set_id, config_version)`; store on disk under `runs/eval/_cache/`; identical code on identical graph set is never re-evaluated; report cache hit rate.
- Test-lock: any split whose name starts with `test` is measured ONCE per (model_id, config hash). Create a `.done` file with the config hash on completion; a second run must refuse unless `--force-unlock` is given, which must be logged in the output. `val` and `probe`-based runs are not locked.
- Output: `runs/eval/{run_id}/per_card.jsonl`, `summary.json`, and `config.json` (full config, seed, git commit hash, library versions). Never overwrite an existing run_id.
- Prompts for generation must be built from a single function using the vectorized API signature, with a `prompt_version` field recorded in the output.

### `tests/test_eval_harness.py`
- Metrics on synthetic data (coverage, best-of-k, pass@k against known values, e.g. n=k gives 0/1; k=1 equals c/n).
- Mock backend end-to-end on a tiny split (2-3 families, few graphs): produces valid `per_card.jsonl` and `summary.json`; DSATUR-like/RLF-like candidates give gap 0 and coverage 0; broken candidates counted as failures and never crash the run; slow candidate hits timeout.
- Cache: second identical run performs zero new sandbox evaluations.
- Test-lock: second run on a `test*` split is refused; `--force-unlock` is allowed and logged.
- No network access and no torch import during tests.

## R4. GRPO reward (Task 11.1)

### Interface
`train/reward.py` must NOT call the sandbox itself. It consumes already-computed evaluation results so tests are fast and deterministic. Define a dataclass for one candidate's result: `status` (ok / ast_fail / purity_fail / timeout / memory / invalid_coloring / exception), `colors_per_graph` (list), `runtime_per_graph` (list), `probe_orderings` (2 orderings or None), plus group-level inputs: `c_base_per_graph`, `c_ref_per_graph` (or None).

### Semantics (follow Section 14.3 of the plan exactly)
- `base_alg(card)` is determined beforehand on `teacher_val` (a separate helper may compute and cache it). The reward function itself is computed on whatever graph set is passed in (in Task 11.2 these will be FRESH graphs from `rl_train_pool`). The reward module must NOT choose seeds or touch `teacher_val` graphs.
- `r_sav = mean_g clip((c_base(g) - c_h(g)) / max(1, c_base(g) - c_ref(g)), -1, 1)`. If `c_ref` is None use `c_base(g) - 1` (configurable fallback).
- `r_time = clip(t_h / t_budget - 1, 0, 1)`, with `t_budget = time_factor * time_of_vectorized_DSATUR` (time_factor configurable).
- `r_tb` optional, default OFF (weight `a = 0`).
- `r_pre = r_sav + a*r_tb - b*r_time`.
- `r_div`: mean pairwise Kendall-tau DISTANCE (normalized to [0,1]) of probe orderings between this candidate and the other group members; applied ONLY if `r_pre` is strictly above the group median of `r_pre` (use `r_pre`, NOT the final reward, for the median); otherwise 0.
- `r = r_pre + d * r_div`. Any failure status -> `r = -1.0` exactly, and failed candidates are excluded from the diversity pairs.
- Coefficients `a, b, d, time_factor, eps` come from a config dataclass / YAML with defaults `a=0.0, b=0.1, d=0.0`; do not hardcode them in formulas.
- Advantages: `A_i = (r_i - mean(r)) / (std(r) + eps_std)`; if std is ~0, all advantages are exactly 0 (no NaN/inf). Document whether std is population or sample and keep it consistent.
- Bounded output: assert reward lies in the documented range given the config (compute and document bounds).

### `tests/test_reward.py`
- Determinism (same input -> identical output, run twice).
- Each failure status yields exactly -1.0.
- A candidate identical to the baseline gets `r_sav ~ 0`.
- Better-than-baseline gets positive `r_sav`; worse gets negative; clipping at +/-1 works.
- Whole group with equal rewards -> advantages all 0, no NaN.
- Diversity applied only above median `r_pre`; zero when `d = 0`; failed candidates excluded from pairs.
- Group where only one candidate succeeds, and group where all fail, are handled without exceptions.
- Kendall-tau distance: identical orderings -> 0, reversed -> 1.
- Reward values stay within documented bounds under random inputs (property-style test with a fixed seed).

---

## Acceptance Criteria
- [ ] Step 0 summary written (in NOTES.md or the final report).
- [ ] All 52 existing tests + new tests green: `pytest` from repo root.
- [ ] New test files: `tests/test_seeds.py`, `tests/test_splits.py`, `tests/test_eval_harness.py`, `tests/test_reward.py`, `tests/test_frozen_files.py`.
- [ ] SEEDS ranges provably disjoint (test asserts it); frozen values unchanged.
- [ ] `data/data_v1/`, `data/data_v1_1/`, `evaluator/verify.py` unmodified (test asserts it).
- [ ] Mock run of `eval/eval_model.py` produces valid `per_card.jsonl`, `summary.json`, `config.json`; test-lock behavior verified.
- [ ] `loop.py` logs all cost fields (only additive changes).
- [ ] New section appended to NOTES.md with: modules added, number of families per split (flag splits with < 3), canonical definition of "beats R(F)" and eps, reward default coefficients and bounds, exact commands to reproduce (pytest and a mock eval run), and "Additional findings".
- [ ] Final report lists files created/modified and the pytest result.
