# Phase 6 / Batch A - Step 0 Architecture & Codebase Investigation Report

## 1. Executive Summary & Step 0 10-Line Brief
```
1. API: priority(adj: (n,n) bool, colors: (n,) int64, sat: (n,) int64, deg: (n,) int64, uncolored_deg: (n,) int64) -> (n,) float scores.
2. Greedy framework masks colored vertices, picks max score (tie: lowest index), assigns min feasible color >= 0, updates sat/udeg.
3. Sandbox: 4 layers in gate/evolved/sandbox.py (AST whitelist, isolated builtins, statelessness/purity check, subprocess with 1.5GB/timeout).
4. Sandbox invocation: evaluate_candidate_safe(code_str, graphs, per_graph_timeout, overall_timeout) returning success, color_counts, error.
5. DSATUR: standalone Numba fast path dsatur(adj) == greedy_color(adj, dsatur_priority) with sat*(n+1) + uncolored_deg.
6. RLF: standalone Numba rlf(adj) == greedy_color(adj, rlf_priority) using vectorized candidate/exclusion sets ne*(n+1) - nc.
7. Verifier: independent evaluator/verify.py checks dimensions, symmetry, no self-loops, all colors >= 0, and proper coloring.
8. R1-R2 Plan: SEEDS.md (pairwise disjoint ranges), teacher_llm.yaml, loop.py latency/status cost logging; build data_v1_2/splits.json on 106 families.
9. R3-R4 Plan: eval/coverage.py (coverage@k, best-of-k, pass@k), eval/eval_model.py (cached/test-locked eval CLI), train/reward.py (pure GRPO reward dataclass).
10. Integrity: frozen files test (data_v1, data_v1_1, verify.py, whitelist), lazy imports, 52/52 existing pytest green + new test suites.
```

---

## 2. Priority Function Contract & Greedy Coloring Engine

### 2.1 Function Signature & Types (`evaluator/coloring.py`, `gate/evolved/prompts.py`)
```python
def priority(adj, colors, sat, deg, uncolored_deg) -> np.ndarray:
```
- **`adj`**: `(n, n)` NumPy array, `dtype=bool` (or `np.bool_`), symmetric ($A = A^T$), zero diagonal ($A_{ii} = \text{False}$). Passed as a read-only view (`flags.writeable = False`).
- **`colors`**: `(n,)` NumPy array, `dtype=np.int64`, $-1$ represents an uncolored vertex, $\ge 0$ represents an assigned color ID. Passed as a read-only view.
- **`sat`**: `(n,)` NumPy array, `dtype=np.int64`, number of distinct colors currently assigned to neighbors of vertex $v$. Passed as a read-only view.
- **`deg`**: `(n,)` NumPy array, `dtype=np.int64`, total static degree of vertex $v$ in the full graph. Passed as a read-only view.
- **`uncolored_deg`**: `(n,)` NumPy array, `dtype=np.int64`, number of uncolored neighbors of vertex $v$. Passed as a read-only view.
- **Output**: `(n,)` NumPy array, `dtype=float` (e.g. `np.float64`), finite scores for each vertex. Higher score indicates earlier coloring.

### 2.2 Framework Execution Rules (`evaluator/coloring.py:greedy_color`)
1. **Filtering**: Scores of already-colored vertices are ignored (`masked_scores = np.where(uncolored, scores, -np.inf)`).
2. **Vertex Selection**: Uncolored vertex with maximum score is selected. Ties are broken deterministically by the lowest vertex index (`np.flatnonzero(masked_scores == max_score)[0]`).
3. **Color Assignment**: Selected vertex $v$ receives the smallest non-negative integer color $c \ge 0$ not present among its neighbors (`np.argmin(nb_has_color[v])`).
4. **Incremental Update**: Saturation `sat` and `uncolored_deg` are updated incrementally in $O(\text{deg}(v))$ without full graph recomputation.
5. **Purity**: Functions must be deterministic, pure, and free of side-effects or persistent state.

---

## 3. Four-Layer Hardened Sandbox (`gate/evolved/sandbox.py`)

The sandbox provides rigorous defense-in-depth across 4 distinct layers:

### Layer 1: AST Static Analysis (`validate_ast`, `SecurityASTVisitor`)
- Parses AST with `ast.parse`.
- Restricts module-level statements: only `def`, `pass`, or immutable literal constants (`is_immutable_literal`) allowed.
- Enforces exact function name `priority` with exactly 5 positional parameters (`adj, colors, sat, deg, uncolored_deg`).
- Forbids `*args`, `**kwargs`, and default argument values in any function, helper, or lambda.
- Forbids identifiers and attributes starting with `_` (blocks `__class__`, `__bases__`, etc.).
- Forbids `global` and `nonlocal` declarations.
- Forbids `import` and `from ... import` statements.
- Forbids dangerous builtin calls (`open`, `eval`, `exec`, `__import__`, `compile`, `getattr`, `setattr`, `delattr`).
- Attribute whitelist enforcement:
  - NumPy attributes: restricted to `ALLOWED_NP_ATTRS` (math, indexing, linear algebra, e.g., `dot`, `matmul`, `zeros`, `where`, `flatnonzero`, `diag`, etc.).
  - Object attributes: restricted to `ALLOWED_OBJECT_ATTRS` (`sum`, `max`, `min`, `astype`, `copy`, `dot`, `shape`, `diagonal`, `clip`, etc.).
  - Explicitly blocks `FORBIDDEN_ATTRS` (`setflags`, `ctypes`, `data`, `base`, `view`, etc.).

### Layer 2: Restricted Execution Environment (`SAFE_BUILTINS`)
- Instantiates a clean namespace dictionary: `{"__builtins__": SAFE_BUILTINS, "np": np, "numpy": np}`.
- `SAFE_BUILTINS` contains only safe math and basic utilities (`range`, `len`, `int`, `float`, `bool`, `min`, `max`, `abs`, `sum`, `round`, `enumerate`, `zip`, `isinstance`, `True`, `False`, `None`).
- Custom `_safe_import`: only allows `numpy` and `numpy.*` internal C-extension imports (e.g. `numpy._core._methods` required by Python 3.13 NumPy reductions), strictly blocking all other module imports.

### Layer 3: Statelessness, Purity Check & Incremental Fresh Namespaces
- `check_statelessness_and_purity(compiled_code, adj)`:
  - Invokes priority on State A (uncolored), then State B (partially colored), then State A again with writeable flags set to `False`.
  - Verifies output shape `(n,)`, finiteness, and exact score equivalence `scores_a1 == scores_a2`.
  - Catches mutation attempts via read-only array exceptions.
- `color_graph_with_priority`:
  - A completely **fresh execution namespace** is created and `exec`'d before **every single step** of the coloring loop ($n$ times per graph), ensuring no internal state or cache survives across steps.
- Every coloring produced is verified by `evaluator.verify.verify(adj, colors)` and color counts extracted via `evaluator.verify.count_colors(colors)`.

### Layer 4: Isolated Subprocess Worker with Hard Resource Guards
- Entry point: `evaluate_candidate_safe(code_str, graphs, per_graph_timeout=1.0, overall_timeout=20.0) -> Dict[str, Any]`.
- Worker process is spawned using `multiprocessing.get_context("spawn")`.
- Child process environment is scrubbed of all `*_API_KEY*` variables.
- Thread concurrency is locked to 1 (`OMP_NUM_THREADS=1`, `threadpoolctl.threadpool_limits(1)`).
- Memory monitoring: parent process samples worker via `psutil`; kills process if RSS, private, or VMS memory exceeds 1.5 GB.
- Dual timeout: per-graph timer inside the loop, and hard wall-clock timeout on the child process with `proc.terminate()` / `proc.kill()`.

---

## 4. DSATUR and RLF Implementation Architectures

### 4.1 DSATUR
1. **Numba Standalone Fast Path**: `evaluator.coloring.dsatur(adj)` calls `_dsatur_numba(adj)`. Maintains incremental neighbor saturation and uncolored degrees, selects vertex maximizing `sat * (n + 1) + uncolored_deg`, assigns lowest feasible color. Runs in single-digit milliseconds.
2. **Greedy Framework Path**: `greedy_color(adj, dsatur_priority)` with `dsatur_priority(adj, colors, sat, deg, uncolored_deg)` computing `sat * (n + 1) + uncolored_deg`.
3. **Parametric Path**: `gate.parametric.parametric_color(adj, DSATUR_WEIGHTS)` where `DSATUR_WEIGHTS = [1.0, 1.0, 0.0, 0.0, 0.0]`.
4. **Equivalence Guarantee**: Verified by `test_dsatur_numba_vs_greedy_color_200_graphs` (210 random graphs) to produce identical colorings.

### 4.2 RLF (Recursive Largest First)
1. **Numba Standalone Fast Path**: `evaluator.coloring.rlf(adj)` calls `_rlf_numba(adj)`. Iteratively builds independent color classes $c=0, 1, \dots$. Picks initial vertex of class maximizing uncolored degree; repeatedly selects candidate vertex $u$ maximizing $ne \cdot (n+1) - nc$ (neighbors in excluded set vs candidate set).
2. **Vectorized Priority Path**: `evaluator.coloring.rlf_priority(adj, colors, sat, deg, uncolored_deg)`. Implements full RLF within the priority API contract! Evaluates active class $k = \max(colors)$, partitions uncolored vertices into `cand` and `excl`, computes $ne$ and $nc$ via vectorized NumPy matrix-vector dot products (`adj.dot(excl)` and `adj.dot(cand)`), and scores candidates. When `cand` is empty, opens class $k+1$ using `uncolored_deg`.
3. **Equivalence Guarantee**: Verified by `test_rlf_priority_matches_standalone_rlf` (25 random graphs) to match standalone RLF colorings bitwise.

---

## 5. Verifier Contract (`evaluator/verify.py`)

- Independent code path: strictly forbidden from importing from `evaluator.coloring` or any heuristic module.
- `verify(adj: np.ndarray, colors: np.ndarray) -> bool`:
  1. Checks `colors`: 1D NumPy integer array, shape $(n,)$.
  2. Checks `adj`: 2D NumPy boolean array, shape $(n, n)$.
  3. Checks empty graph: $n=0$ returns True.
  4. Checks self-loops: `np.any(np.diag(adj))` must be False.
  5. Checks symmetry: `np.array_equal(adj, adj.T)` must be True.
  6. Checks valid non-negative colors: `not np.any(colors < 0)`.
  7. Checks proper coloring: for all edges $(u, v)$ where `adj[u, v] == True`, `colors[u] != colors[v]`.
- `count_colors(colors: np.ndarray) -> int`:
  - Returns `len(np.unique(colors))` (or 0 for empty array).
- Integrity: `evaluator/verify.py` is frozen and must be validated by `tests/test_frozen_files.py`.

---

## 6. Discrepancy & Consistency Analysis

| Item | Reference / Prompt | Codebase Reality | Analysis & Resolution |
| :--- | :--- | :--- | :--- |
| **Path of `parametric.py`** | `ORIGINAL_REQUEST.md` line 11 mentions `evaluator/parametric.py`. | File is located at `gate/parametric.py`. | No file exists at `evaluator/parametric.py`. New code and tests must reference `gate.parametric`. |
| **Scope of Dataset v1.2** | `PLAN_PHASE6_ONWARD.md` Task 8.1 mentions generating parametric variants to reach ~300 cards. | `ORIGINAL_REQUEST.md` R2 scopes Batch A to partitioning existing 106 families only. | Follow `ORIGINAL_REQUEST.md` strictly: partition existing 106 families into splits; record counts and flag any split with < 3 families. |
| **SEEDS.md Ranges** | `ORIGINAL_REQUEST.md` R1 specifies closed ranges for frozen & proposed. | Matches `PLAN_PHASE6_ONWARD.md` Section 5 & `data/data_v1_1/config.json`. | All 13 ranges (0-99, 1000-1019, 2000-2049, 3000-3019, 4000-4199, 5000-5019, 6000-6049, 7000-7049, 7100-7149, 7200-7249, 7300-7349, 9000-9009, 100000-199999) are pairwise disjoint. |
| **`loop.py` Cost Logging** | `ORIGINAL_REQUEST.md` R1 requires logging `prompt_tokens`, `completion_tokens`, `latency_s`, `operator`, `temperature`, `status`, `cost_dollars`. | `loop.py` logs `tokens` dict, `operator`, `temperature`, `cost_dollars`, but lacks `latency_s` and `status`. | Update `loop.py` additively: record call latency and status without modifying loop behavior. |
| **Kendall-tau Diversity Distance** | `ORIGINAL_REQUEST.md` R4: normalized Kendall-tau distance in $[0, 1]$, identical $\to 0$, reversed $\to 1$. | `scipy.stats.kendalltau` returns correlation $\tau \in [-1, 1]$. | Distance is computed as $d_\tau = (1 - \tau) / 2$. Normalized range $[0, 1]$ satisfies all constraints. |
| **Reward Architecture** | `ORIGINAL_REQUEST.md` R4: `train/reward.py` must NOT call sandbox. | Consumes pre-computed dataclass `CandidateEvalResult`. | Fully decoupled reward evaluation; fast, deterministic unit testing. |
| **Frozen Files Enforcement** | `ORIGINAL_REQUEST.md` Rule 1: frozen files must not be edited. | Must add `tests/test_frozen_files.py`. | Hash-manifest test asserting zero changes to `data/data_v1/`, `data/data_v1_1/`, `evaluator/verify.py`, lock files, and sandbox whitelists. |

---

## 7. Implementation Plan for Phase 6 / Batch A Modules

1. **R1: SEEDS Registry & Cost Logging**:
   - `SEEDS.md`: Markdown table with columns `name | start | end | status | purpose`.
   - `tests/test_seeds.py`: Parse table, verify disjointness, check frozen ranges.
   - `configs/teacher_llm.yaml`: Provider, model, rate limits, `api_key_env: GEMINI_API_KEY`.
   - `gate/evolved/loop.py`: Add `latency_s` and `status` to candidate logs (additive only).
2. **R2: Dataset v1.2 Splits (`data/data_v1_2/`)**:
   - `scripts/build_splits.py`: Partition 106 families into `train_pool` (includes `scale_free`), `val`, `test_interp`, `test_extrap`, `test_comp`, and `heldout_type` (`small_world`).
   - `data/data_v1_2/splits.json`: Formatted with binning thresholds and `"heldout_combinations"`.
   - `tests/test_splits.py`: Assert single assignment, holdout exclusions, seed consistency.
3. **R3: Offline Evaluation Harness (`eval/`)**:
   - `eval/coverage.py`: Pure functions for "beats R(F)" ($\text{mean} < R(F) - \text{eps}$), `coverage_at_k`, `best_of_k`, and numerically stable Chen et al. `pass_at_k`.
   - `eval/eval_model.py`: CLI with `--backend {mock,hf,vllm,api}`, disk caching under `runs/eval/_cache/`, test-lock mechanism with `.done` files, output to `runs/eval/{run_id}/`.
   - `tests/test_eval_harness.py`: Unit and end-to-end mock tests.
4. **R4: GRPO Reward Module (`train/reward.py`)**:
   - Dataclass for candidate evaluation results.
   - Computes $r_{\text{sav}}$, $r_{\text{time}}$, $r_{\text{tb}}$, $r_{\text{pre}}$, $r_{\text{div}}$ (pairwise Kendall-tau distance applied above group median $r_{\text{pre}}$).
   - Group advantage computation with zero-std handling.
   - `tests/test_reward.py`: Determinism, bounds, failure penalty ($-1.0$), baseline symmetry.
5. **Frozen Files Guard (`tests/test_frozen_files.py`)**:
   - Hashes `data/data_v1/`, `data/data_v1_1/`, `evaluator/verify.py`, and sandbox whitelists to ensure strict immutability.
