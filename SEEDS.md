# SEEDS Registry

| name | start | end | status | purpose |
|---|---|---|---|---|
| train | 0 | 99 | frozen | Training and in-loop evolution graphs (baseline uses seeds 0-19) |
| val | 1000 | 1019 | frozen | Validation graphs for model selection and early stopping |
| test | 2000 | 2049 | frozen | Primary test graphs for Phase 1 gate evaluations |
| teacher_val | 3000 | 3019 | frozen | Teacher candidate selection and reference R(F) calculation |
| test_pilot | 4000 | 4199 | frozen | Pilot evolution test evaluation (strict .done lock) |
| sft_val_probe | 5000 | 5019 | proposed | Validation split for SFT early stopping heuristic score callback |
| teacher_rescore | 6000 | 6049 | proposed | Teacher heuristic re-scoring on unseen graphs to eliminate selection bias |
| test_interp | 7000 | 7049 | proposed | Interpolation test evaluation (unseen parameters within training range) |
| test_extrap | 7100 | 7149 | proposed | Extrapolation test evaluation (n or density exceeding training range) |
| test_comp | 7200 | 7249 | proposed | Compositional generalization test (held-out attribute combinations) |
| test_heldout_type | 7300 | 7349 | proposed | Out-of-distribution structure type evaluation (small_world) |
| probe | 9000 | 9009 | frozen | Behavioral fingerprinting and Kendall-tau ranking diversity distance |
| rl_train_pool | 100000 | 199999 | proposed | Fresh graph seed pool for GRPO rollouts and reinforcement learning |
