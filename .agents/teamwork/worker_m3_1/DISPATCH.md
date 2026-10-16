## 2026-10-16T03:55:59Z

You are Worker subagent worker_m3_1.
Your working directory is: d:\Graph Coloring\.agents\teamwork\worker_m3_1
You MUST read:
1. d:\Graph Coloring\.agents\teamwork\ORIGINAL_REQUEST.md (specifically Section R3)
2. d:\Graph Coloring\.agents\teamwork\orchestrator_1\PROJECT.md
3. d:\Graph Coloring\.agents\teamwork\explorer_r3r4_1\report.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Scope & Write Ownership:
You exclusively own:
- eval/coverage.py
- eval/eval_model.py
- tests/test_eval_harness.py

Tasks:
1. Implement eval/coverage.py:
   - Pure functions, no I/O.
   - Canonical definition of "beats R(F)": candidate mean colors on split graphs < R(F) by at least eps (default eps=0.05, configurable). Ties do NOT count as wins. R(F) = min(mean DSATUR, mean RLF) on same graphs.
   - Implement coverage_at_k, best_of_k, and pass_at_k(n, c, k) (Chen et al. estimator, log/comb numerical stability).
   - Handle edge cases: k > n (ValueError), zero cards, all candidates failing.
2. Implement eval/eval_model.py:
   - CLI flags: --backend {mock,hf,vllm,api} --split NAME --k K --n-samples N --temperature T --n-graphs M --run-id ID --seed S --force-unlock.
   - Backend interface: generate(prompt, n, temperature, seed) -> list[str].
   - MockBackend emits deterministic fake code: DSATUR-like, RLF-like, broken, slow code.
   - Thin stubs for hf/vllm/api with lazy imports and clear error messages when dependencies/keys are missing.
   - Subprocess candidate evaluation using gate/evolved/sandbox.py evaluate_candidate_safe (Windows-safe spawn / ThreadPoolExecutor).
   - Metrics per card & summary: valid-rate, failure counts, mean colors, savings vs R(F) and vs DSATUR (colors and %), coverage@k, best-of-k, pass@k, runtime ratio vs vectorized DSATUR, code length, fingerprint equality to DSATUR/RLF on probe seeds 9000-9009, within-card diversity.
   - Disk cache under runs/eval/_cache/ key=(code_hash, graph_set_id, config_version).
   - Test-lock: any split starting with "test" is measured ONCE per (model_id, config hash), creates .done file on completion; second run refuses unless --force-unlock.
   - Output: runs/eval/{run_id}/per_card.jsonl, summary.json, config.json. Never overwrite an existing run_id.
   - Single prompt builder function using vectorized API signature.
3. Implement tests/test_eval_harness.py:
   - Test metrics on synthetic data.
   - Mock backend end-to-end on tiny split (2-3 families, few graphs).
   - Test cache hits (second identical run performs 0 new sandbox evaluations).
   - Test test-lock refusal and --force-unlock.
   - Assert zero network access and zero torch import during tests.
4. Run pytest tests/test_eval_harness.py and full pytest to confirm all tests pass green.
5. Write handoff to d:\Graph Coloring\.agents\teamwork\worker_m3_1\handoff.md.
6. Send a message to orchestrator (553217b8-f1c7-4ae8-b2fd-86f4b1a6dc9b) with your test results and summary.
