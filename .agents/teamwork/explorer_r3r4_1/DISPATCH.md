## 2026-10-16T02:55:39Z
You are an Explorer subagent (explorer_r3r4_1).
Your working directory is: d:\Graph Coloring\.agents\teamwork\explorer_r3r4_1
You MUST read:
1. d:\Graph Coloring\.agents\teamwork\ORIGINAL_REQUEST.md
2. d:\Graph Coloring\PLAN_PHASE6_ONWARD.md (focus on Sections 9.2, 11.1, 14, 14.3)
3. d:\Graph Coloring\NOTES.md
4. Codebase inspection: existing eval/ (if any), evaluator/coloring.py, evaluator/verify.py, gate/evolved/sandbox.py, gate/evolved/loop.py.

Tasks:
1. Design eval/coverage.py:
   - Canonical definition of "beats R(F)": candidate mean colors on split graphs < R(F) by at least eps (default 0.05). Ties do not count. R(F) = min(mean DSATUR, mean RLF) on same graphs.
   - Exact implementation formulas for coverage_at_k, best_of_k, pass_at_k (Chen et al. estimator, log/comb numerical stability).
   - Edge cases: k > n, zero cards, all candidates failing.
2. Design eval/eval_model.py:
   - CLI flags: --backend {mock,hf,vllm,api} --split NAME --k K --n-samples N --temperature T --n-graphs M --run-id ID --seed S.
   - Backend interface: generate(prompt, n, temperature, seed) -> list[str]. Mock backend logic. Lazy import stubs for hf/vllm/api.
   - Evaluation execution: sandbox invocation with multiprocessing pool (Windows-safe spawn), timeout, verify.
   - Metrics computation per card & summary.
   - Caching mechanism: disk cache under runs/eval/_cache/ key=(code_hash, graph_set_id, config_version).
   - Test-lock mechanism: .done file with config hash for test* splits. Refuse on repeat unless --force-unlock.
   - Output format: per_card.jsonl, summary.json, config.json.
   - Single prompt builder function using vectorized API signature.
3. Design train/reward.py (GRPO reward):
   - Dataclass for CandidateResult.
   - base_alg(card) caching and fallback.
   - r_sav formula with clipping and denominator protection.
   - r_time with budget factor.
   - r_tb (optional).
   - r_div: Kendall-tau distance on probe orderings, median filtering using r_pre.
   - Failure status -> r = -1.0.
   - Advantage calculation: (r_i - mean(r)) / (std(r) + eps_std), handle zero std.
   - Config dataclass with default coefficients (a=0.0, b=0.1, d=0.0). Document bounds.
4. Write your complete analysis and recommendations to d:\Graph Coloring\.agents\teamwork\explorer_r3r4_1\report.md.
5. Update progress.md in your working directory.
6. Send a message to the orchestrator (conversation ID: 553217b8-f1c7-4ae8-b2fd-86f4b1a6dc9b) with your findings.
