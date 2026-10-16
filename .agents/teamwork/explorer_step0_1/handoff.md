# Step 0 Handoff Report - explorer_step0_1

## 1. Observation
- **Priority API Contract**:
  - `evaluator/coloring.py:16-30`: `def dsatur_priority(adj: np.ndarray, colors: np.ndarray, sat: np.ndarray, deg: np.ndarray, uncolored_deg: np.ndarray) -> np.ndarray:`.
  - `evaluator/coloring.py:32-75`: `def rlf_priority(adj: np.ndarray, colors: np.ndarray, sat: np.ndarray, deg: np.ndarray, uncolored_deg: np.ndarray) -> np.ndarray:`.
  - `gate/evolved/prompts.py:19-31`: Verbatim contract:
    - `adj`: `(n, n)` bool ndarray, symmetric, no self-loops (read-only).
    - `colors`: `(n,)` int64 ndarray, -1 = uncolored, >=0 color id (read-only).
    - `sat`: `(n,)` int64 ndarray, number of distinct colors among colored neighbors (read-only).
    - `deg`: `(n,)` int64 ndarray, total degree in full graph (read-only).
    - `uncolored_deg`: `(n,)` int64 ndarray, number of uncolored neighbors (read-only).
    - Returns: `(n,)` float ndarray of scores. Higher score = color earlier.
- **Greedy Framework**:
  - `evaluator/coloring.py:77-137`: `greedy_color(adj, priority_fn)`.
  - Masking: `masked_scores = np.where(uncolored, scores, -np.inf)` (line 119).
  - Tie-breaking: `candidates = np.flatnonzero(masked_scores == max_score); v = int(candidates[0])` (lines 122-123).
  - Smallest feasible color assignment: `c = int(np.argmin(nb_has_color[v]))` (line 126).
- **Fast Paths & Numba**:
  - `evaluator/coloring.py:140-198`: `_dsatur_numba(adj)` and `dsatur(adj)`.
  - `evaluator/coloring.py:201-294`: `_rlf_numba(adj)` and `rlf(adj)`.
  - `tests/test_evaluator.py:20-36`: Confirms `dsatur(adj) == greedy_color(adj, dsatur_priority)` on 210 graphs.
  - `tests/test_evaluator.py:136-150`: Confirms `rlf(adj) == greedy_color(adj, rlf_priority)` on 25 graphs.
- **Parametric File Location**:
  - `gate/parametric.py:1-161` exists; `evaluator/parametric.py` does not exist.
- **4-Layer Sandbox Architecture**:
  - Defined in `gate/evolved/sandbox.py`.
  - Layer 1 (AST): `validate_ast(code_str)` lines 195-229, `SecurityASTVisitor` lines 104-183 enforcing exact 5 args, no dunders/underscores, no defaults, no imports, whitelist `ALLOWED_NP_ATTRS` and `ALLOWED_OBJECT_ATTRS`.
  - Layer 2 (Restricted Builtins): `SAFE_BUILTINS` lines 79-98, `_safe_import` lines 72-77 allowing only `numpy` and `numpy.*`.
  - Layer 3 (Purity & Statelessness): `check_statelessness_and_purity(code_obj, adj)` lines 235-294; `color_graph_with_priority` lines 300-376 instantiating fresh namespace for every step. Calls `evaluator.verify.verify(adj, colors)` line 373.
  - Layer 4 (Subprocess & Hard Limits): `_sandbox_subprocess_target` and `evaluate_candidate_safe` lines 380-535. Uses `mp.get_context("spawn")`, scrubs `*_API_KEY`, limits threads to 1, monitors 1.5 GB memory limit via `psutil`, enforces `per_graph_timeout` and `overall_timeout`.
- **Verifier Contract**:
  - `evaluator/verify.py:13-64`: `verify(adj, colors)` independent of `coloring.py`. Validates types, 1D integer colors, 2D boolean symmetric adj, no self-loops, all colors >= 0, and no adjacent vertices share color.
  - `evaluator/verify.py:66-77`: `count_colors(colors)` returns `len(np.unique(colors))`.
- **Test Suite Status**:
  - Tool execution: `pytest` -> `52 passed in 8.46s` on Python 3.13.11.

## 2. Logic Chain
1. *Observation 1 & 2* establish that the greedy framework and priority function have a strict, unchanging contract: (n,) float output, ties to lowest index, non-conflicting color assignment.
2. *Observation 3* establishes that DSATUR and RLF have two validated paths: standalone Numba fast paths and vectorized priority representations that produce identical color counts and assignments.
3. *Observation 4* reveals that while older documentation referenced `evaluator/parametric.py`, the active module is `gate/parametric.py`.
4. *Observation 5* shows that candidate execution is fully encapsulated within `gate/evolved/sandbox.py` with `evaluate_candidate_safe` as the sole entry point, fulfilling all security and resource guarantees.
5. *Observation 6* confirms that the evaluation and verifier pipelines are decoupled, ensuring `verify()` can act as an incorruptible arbiter for all candidate colorings.
6. *Observation 7* confirms that all 52 existing regression tests pass cleanly on the current environment.

## 3. Caveats
- `data/data_v1_2/` does not yet exist and will be constructed during Task R2 using existing 106 families metadata from `data/data_v1_1/`.
- In `gate/evolved/loop.py`, `call_llm_with_retry` and candidate logging do not currently record `latency_s` or `status` explicitly; these will be added additively in Task R1.
- No heavy GPU packages (torch, vllm) are currently installed; all modules for Batch A must use lazy imports so pytest remains functional on CPU.

## 4. Conclusion
The codebase is clean, well-tested (52/52 passing), and structurally aligned with the requirements for Phase 6 / Batch A. Step 0 investigation is complete. The exact vectorized priority API, 4-layer sandbox architecture, DSATUR/RLF evaluation pipelines, and verification contracts are fully mapped out. There are no blocking contradictions, and implementation can proceed with Tasks R1 through R4.

## 5. Verification Method
- Independent verification command:
  ```powershell
  pytest
  ```
  Expected: 52 passed.
- Inspect detailed report:
  `d:\Graph Coloring\.agents\teamwork\explorer_step0_1\report.md`
- Invalidation conditions: Any modification to `data/data_v1/`, `data/data_v1_1/`, `evaluator/verify.py`, or failure of existing 52 tests.
