# D7 Card-Based Algorithm Selector Baseline ({DSATUR, RLF})

Trained strictly on `train_pool` (41 families).

## Decision Tree Evaluation across Splits
| Split | N Families | Accuracy | Mean Regret (colors) | Regret (%) | Always RLF Regret | Always DSATUR Regret |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_pool` | 41 | 90.2% | 0.060 | 0.44% | 0.062 | 0.637 |
| `val` | 7 | 71.4% | 0.086 | 1.08% | 0.057 | 0.286 |
| `test_interp` | 7 | 71.4% | 0.054 | 0.43% | 0.054 | 0.557 |
| `test_extrap` | 15 | 100.0% | 0.000 | 0.00% | 0.000 | 1.656 |
| `test_comp` | 10 | 100.0% | 0.000 | 0.00% | 0.000 | 0.782 |
| `heldout_type` | 26 | 88.5% | 0.006 | 0.05% | 0.002 | 0.293 |

## Logistic Regression Evaluation across Splits
| Split | N Families | Accuracy | Mean Regret (colors) | Regret (%) | Always RLF Regret | Always DSATUR Regret |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_pool` | 41 | 85.4% | 0.062 | 0.46% | 0.062 | 0.637 |
| `val` | 7 | 85.7% | 0.057 | 0.72% | 0.057 | 0.286 |
| `test_interp` | 7 | 85.7% | 0.051 | 0.40% | 0.054 | 0.557 |
| `test_extrap` | 15 | 93.3% | 0.001 | 0.01% | 0.000 | 1.656 |
| `test_comp` | 10 | 100.0% | 0.000 | 0.00% | 0.000 | 0.782 |
| `heldout_type` | 26 | 96.2% | 0.002 | 0.01% | 0.002 | 0.293 |
