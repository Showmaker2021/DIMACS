# Cost Model and Amortization Analysis

- **Per-Family Evolutionary Search Cost**: $0.0196
- **Amortized Generator Deployment Inference Cost**: $0.000099 per card
- **Total Amortized Training Investment**: $33.96
  * Teacher Data Collection: $1.96
  * SFT LoRA Training: $8.00
  * GRPO Training: $24.00
- **Break-Even Threshold**: **1741.5 families**

## Sensitivity Analysis
| API Multiplier | GPU Rate ($/h) | Evo Cost / Fam | Amortized Cost | Break-Even Families |
|---|---|---|---|---|
| 0.5x | $1.00 | $0.0098 | $16.98 | **1741.5** |
| 0.5x | $2.00 | $0.0098 | $32.98 | **3382.5** |
| 0.5x | $4.00 | $0.0098 | $64.98 | **6664.4** |
| 1.0x | $1.00 | $0.0196 | $17.96 | **921.0** |
| 1.0x | $2.00 | $0.0196 | $33.96 | **1741.5** |
| 1.0x | $4.00 | $0.0196 | $65.96 | **3382.5** |
| 2.0x | $1.00 | $0.0392 | $19.92 | **510.8** |
| 2.0x | $2.00 | $0.0392 | $35.92 | **921.0** |
| 2.0x | $4.00 | $0.0392 | $67.92 | **1741.5** |
| 5.0x | $1.00 | $0.0980 | $25.80 | **264.6** |
| 5.0x | $2.00 | $0.0980 | $41.80 | **428.7** |
| 5.0x | $4.00 | $0.0980 | $73.80 | **756.9** |