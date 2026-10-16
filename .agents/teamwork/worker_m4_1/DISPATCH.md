## 2026-10-16T03:55:59Z
You are Worker subagent worker_m4_1.
Your working directory is: d:\Graph Coloring\.agents\teamwork\worker_m4_1
You MUST read:
1. d:\Graph Coloring\.agents\teamwork\ORIGINAL_REQUEST.md (specifically Section R4)
2. d:\Graph Coloring\.agents\teamwork\orchestrator_1\PROJECT.md
3. d:\Graph Coloring\.agents\teamwork\explorer_r3r4_1\report.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Scope & Write Ownership:
You exclusively own:
- train/reward.py
- tests/test_reward.py

Tasks:
1. Implement train/reward.py:
   - Must NOT call the sandbox itself. Consumes CandidateResult dataclass:
     status (ok, ast_fail, purity_fail, timeout, memory, invalid_coloring, exception),
     colors_per_graph (list), runtime_per_graph (list), probe_orderings (2 orderings or None),
     c_base_per_graph (list), c_ref_per_graph (list or None).
   - r_sav = mean_g clip((c_base(g) - c_h(g)) / max(1, c_base(g) - c_ref(g)), -1, 1). Fallback if c_ref is None: c_base(g) - 1.
   - r_time = clip(t_h / t_budget - 1, 0, 1), t_budget = time_factor * time_of_vectorized_DSATUR.
   - r_tb optional, default OFF (a=0.0).
   - r_pre = r_sav + a*r_tb - b*r_time.
   - r_div: mean pairwise Kendall-tau distance (normalized to [0,1]) of probe orderings between this candidate and other group members; applied ONLY if r_pre is strictly above group median of r_pre (use r_pre, NOT final reward); otherwise 0.
   - r = r_pre + d * r_div. Any failure status -> r = -1.0 exactly; failed candidates excluded from diversity pairs.
   - Config dataclass with defaults a=0.0, b=0.1, d=0.0, time_factor=3.0. Compute and document mathematical bounds.
   - Advantages: A_i = (r_i - mean(r)) / (std(r) + eps_std). If std ~ 0, all advantages are exactly 0 (no NaN/inf). Document whether std is population or sample (e.g. population ddof=0).
2. Implement tests/test_reward.py:
   - Determinism (same input -> identical output).
   - Each failure status yields exactly -1.0.
   - Baseline-identical candidate gets r_sav ~ 0.
   - Better/worse candidates get positive/negative r_sav with clipping at +/-1.
   - Equal rewards group -> advantages all 0.
   - Diversity applied only above median r_pre; zero when d=0; failed candidates excluded.
   - Single candidate success and all fail groups handled without exceptions.
   - Kendall-tau distance: identical -> 0, reversed -> 1.
   - Bounded output property test with fixed seed.
3. Run pytest tests/test_reward.py and full pytest to confirm all tests pass green.
4. Write handoff to d:\Graph Coloring\.agents\teamwork\worker_m4_1\handoff.md.
5. Send a message to orchestrator (553217b8-f1c7-4ae8-b2fd-86f4b1a6dc9b) with your test results and summary.
