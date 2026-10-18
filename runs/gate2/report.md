# GATE 2 DECISION REPORT: Pilot Evolution Baseline
**Generated:** 2026-10-18T10:35:37
**Overall Decision:** **FAIL** (Target PASS: 0/4, Target MARGINAL: 0/4)

## 1. Target Families (4 families)
| Family | R(F) Alg | Mean Top-1 | Mean R(F) | Mean DSATUR | Gap vs R(F) (%) [95% CI] | Wilcoxon p | Probe ≈ Baseline | Status |
|---|---|---|---|---|---|---|---|---|
| `kpart_k10_p0.3_n100-200` | RLF | 14.270 | 13.395 | 14.270 | -6.53% [-7.28%, -5.80%] | 1.0000 | DSATUR | **FAIL** |
| `gnp_d0.30_n100-200` | RLF | 15.560 | 14.545 | 15.560 | -6.98% [-7.66%, -6.31%] | 1.0000 | DSATUR | **FAIL** |
| `sbm_k10_in0.5_out0.05_n200-300` | RLF | 10.005 | 9.375 | 10.005 | -6.72% [-7.67%, -5.75%] | 1.0000 | DSATUR | **FAIL** |
| `ws_k10_p0.1_n100-200` | DSATUR | 7.125 | 7.125 | 7.125 | +0.00% [+0.00%, +0.00%] | 1.0000 | DSATUR | **FAIL** |

## 2. Control Families (2 families)
| Family | R(F) Alg | Mean Top-1 | Mean R(F) | Mean DSATUR | Gap vs R(F) (%) [95% CI] | Wilcoxon p | Status |
|---|---|---|---|---|---|---|---|
| `interval_L0.10_n100-200` | RLF | 23.950 | 23.950 | 23.950 | +0.00% [+0.00%, +0.00%] | 1.0000 | **FAIL** |
| `ba_m12_n200-300` | DSATUR | 9.130 | 9.130 | 9.130 | +0.00% [+0.00%, +0.00%] | 1.0000 | **FAIL** |

## 3. Decision Criteria Evaluation
- **PASS**: >= 2/4 target families improve >= 1.0% vs R(F) on test_pilot with 95% CI > 0, AND valid-rate = 100%.
- **MARGINAL**: exactly 1/4 target families PASS or improve in [0.5%, 1.0%].
- **FAIL**: otherwise.

**Result**: Target passed = 0/4. All valid rates 100% = True.
**GATE 2 OUTCOME: FAIL**