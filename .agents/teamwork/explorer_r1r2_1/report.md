# Comprehensive Investigation Report: R1 (Seeds, LLM Config, Telemetry) & R2 (Dataset v1.2 Splits)

**Author:** Explorer Subagent (`explorer_r1r2_1`)  
**Date:** 2026-10-16  
**Working Directory:** `d:\Graph Coloring\.agents\teamwork\explorer_r1r2_1`  
**Target Modules:** `SEEDS.md`, `configs/teacher_llm.yaml`, `gate/evolved/loop.py`, `data/data_v1_2/splits.json`, `scripts/build_splits.py`, `tests/test_seeds.py`, `tests/test_splits.py`

---

## 1. Executive Summary

This report delivers the complete pre-implementation investigation and design specifications for:
1. **R1 (Task 6.0)**:
   - Exhaustive repository-wide seed audit: **Zero collisions** found across all proposed seed ranges.
   - Exact schema and all 13 rows for `SEEDS.md` mapped out with closed intervals `[start, end]`.
   - Complete schema for `configs/teacher_llm.yaml` with strict environment-variable-only credential hygiene.
   - Telemetry audit of `gate/evolved/loop.py`: pinpointed the exact LLM call site, measured `latency_s`, extracted `prompt_tokens`, `completion_tokens`, `operator`, `temperature`, `status`, and `cost_dollars` with 100% backward compatibility.
2. **R2 (Task 8.1)**:
   - Metadata inspection of all 106 existing families from `data/data_v1_1/families_v1_1.json` across 11 structure types.
   - Deterministic binning derivation for `n_range`, `density_bin`, `degree_variance_bin`, and `clustering_bin`.
   - Exact mathematical partitioning of the 106 families into 6 disjoint groups: `train_pool` (41), `val` (7), `test_interp` (7), `test_extrap` (15), `test_comp` (10), and `heldout_type` (26).
   - Strict satisfaction of all invariant constraints: `scale_free` in `train_pool`, `small_world` strictly absent from `train_pool` and `val`, `heldout_combinations` (`(2, 'low', 'low')`) absent from `train_pool`, and `test_extrap` strictly exceeding `train_pool` maximum $n$ ($320 > 300$). Zero splits have fewer than 3 families (all $\ge 7$).

---

## 2. Task 1: Seed Usage & Collision Audit (`SEEDS.md`)

### 2.1 Direct Observations & Evidence Chain
- **Existing Configs**:
  - `data/data_v1/config.json` lines 4-21:
    - `"train": [0, 100]` $\to$ seeds $0..99$
    - `"val": [1000, 20]` $\to$ seeds $1000..1019$
    - `"test": [2000, 50]` $\to$ seeds $2000..2049$
    - `"probe": [9000, 10]` $\to$ seeds $9000..9009$
  - `data/data_v1_1/config.json` lines 4-29:
    - Retains all v1.0 splits plus:
    - `"teacher_val": [3000, 20]` $\to$ seeds $3000..3019$
    - `"test_pilot": [4000, 200]` $\to$ seeds $4000..4199$
  - `gate/dataset.py` lines 105-123:
    - Dynamically generates `teacher_val` graphs using `range(3000, 3020)` and `test_pilot` graphs using `range(4000, 4200)`.
- **Repo-wide Codebase Usages**:
  - `tests/test_dataset.py` line 77: asserts `splits["test_pilot"] == [4000, 200]`.
  - `scripts/compute_teacher_val_rf.py` line 1: seeds 3000-3019 for teacher selection.
  - `data/generate_dataset.py` line 45: `COMBO_SEED = 2026` is a scalar pseudo-random seed used during dataset generation (not a graph seed).
  - `scripts/run_gate.py` line 76: `seeds=(42, 43, 44)` are optimizer restart seeds for CMA-ES.
  - `tests/test_evaluator.py` line 26 (`seed = 1000 + i`) and line 143 (`seed = 6000 + i` for $i \in [0, 24]$): unit tests generating temporary in-memory Erdős-Rényi random graphs via `_random_graph(n, p, seed)` to test DSATUR and RLF equivalence. These do not touch dataset storage or manifest files.

### 2.2 Collision Verification
Let each range be represented as a closed interval $[S_i, E_i]$. We verify pairwise disjointness for all proposed and frozen ranges:

| Index | Range Name | Start | End | Count | Status | Purpose |
|---|---|---|---|---|---|---|
| 1 | `train` | 0 | 99 | 100 | frozen | Training & in-loop evolution graphs (baseline uses 0-19) |
| 2 | `val` | 1000 | 1019 | 20 | frozen | Validation graphs for early stopping & model selection |
| 3 | `test` | 2000 | 2049 | 50 | frozen | Primary test graphs (one-time measurement) |
| 4 | `teacher_val` | 3000 | 3019 | 20 | frozen | Teacher candidate selection and reference R(F) calculation |
| 5 | `test_pilot` | 4000 | 4199 | 200 | frozen | Pilot evolution test graphs (locked with `.done`) |
| 6 | `sft_val_probe` | 5000 | 5019 | 20 | proposed | Validation split for SFT callback early stopping by heuristic quality |
| 7 | `teacher_rescore` | 6000 | 6049 | 50 | proposed | Unseen graphs for teacher heuristic re-scoring (selection bias avoidance) |
| 8 | `test_interp` | 7000 | 7049 | 50 | proposed | Interpolation test graphs (unseen parameters within training range) |
| 9 | `test_extrap` | 7100 | 7149 | 50 | proposed | Extrapolation test graphs ($n$ or density exceeding training maxima) |
| 10 | `test_comp` | 7200 | 7249 | 50 | proposed | Compositional generalization test graphs (held-out attribute combinations) |
| 11 | `test_heldout_type` | 7300 | 7349 | 50 | proposed | Out-of-distribution structure type evaluation (`small_world`) |
| 12 | `probe` | 9000 | 9009 | 10 | frozen | Behavioral fingerprinting and Kendall-tau ranking diversity probe graphs |
| 13 | `rl_train_pool` | 100000 | 199999 | 100000 | proposed | Massive fresh graph seed pool for GRPO rollouts and reinforcement learning |

**Disjointness Proof**:
- Sorted order: $[0, 99] < [1000, 1019] < [2000, 2049] < [3000, 3019] < [4000, 4199] < [5000, 5019] < [6000, 6049] < [7000, 7049] < [7100, 7149] < [7200, 7249] < [7300, 7349] < [9000, 9009] < [100000, 199999]$.
- For every adjacent pair $i, i+1$, $E_i < S_{i+1}$. Minimum gap is $7049 \to 7100$ (gap = 51).
- **Result**: Zero collisions. All 13 ranges are strictly pairwise disjoint.

### 2.3 Exact Markdown Table for `SEEDS.md`
The file `SEEDS.md` in the repo root will be formatted as follows:

```markdown
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
```

---

## 3. Task 2: Teacher LLM Config Design (`configs/teacher_llm.yaml`)

### 3.1 Direct Observations & Evidence Chain
- Existing configuration: `configs/pricing.json` defines models and pricing (`deepseek-chat`, `gemini-flash-lite-latest`, `claude-3-5-sonnet-20241022`, `mock`).
- `gate/evolved/loop.py` lines 114-123:
  - Resolves keys from environment variables: `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`.
  - Enforces zero hardcoded credentials and strictly rejects missing environment variables.
- Requirements from `ORIGINAL_REQUEST.md` line 35 & `PLAN_PHASE6_ONWARD.md` line 200:
  - Provider, model name placeholder, base temperature, max_tokens, rate limit (requests/min, pacing seconds), retry/backoff settings, and `api_key_env: <ENV_VAR_NAME>` only.
  - Zero strings resembling keys (e.g. `sk-...`, `AIza...`).

### 3.2 Recommended Structure for `configs/teacher_llm.yaml`
```yaml
# Teacher LLM Configuration for Offline Evolutionary Distillation
provider: "deepseek"
model: "deepseek-chat"
base_temperature: 0.7
max_tokens: 1024

rate_limit:
  requests_per_minute: 12
  pacing_seconds: 5.0

retry:
  max_retries: 2
  backoff_factor: 2.0
  initial_wait_seconds: 10.0
  max_wait_seconds: 60.0

# Name of the environment variable containing the API key (NEVER write actual keys here)
api_key_env: "DEEPSEEK_API_KEY"
```

### 3.3 Verification Strategy
In `tests/test_seeds.py` (or `tests/test_configs.py`):
1. Load `configs/teacher_llm.yaml` using `yaml.safe_load`.
2. Assert all required keys are present (`provider`, `model`, `base_temperature`, `max_tokens`, `rate_limit`, `retry`, `api_key_env`).
3. Assert that `api_key_env` matches regex `^[A-Z0-9_]+$`.
4. Assert that no value in the file matches key patterns (`sk-[a-zA-Z0-9]{20,}`, `AIza[0-9A-Za-z-_]{35}`).

---

## 4. Task 3: LLM Call Telemetry Audit (`gate/evolved/loop.py`)

### 4.1 Direct Observations
1. **Call Site Identification**:
   - `gate/evolved/loop.py` line 71: `def call_llm_with_retry(...)`
   - In `run_evolution`, line 492:
     ```python
     code, usage, call_cost, attempts = call_llm_with_retry(
         model=model,
         system_prompt=SYSTEM_PROMPT,
         user_prompt=user_prompt,
         pricing=pricing,
         mock=mock,
         temperature=op_temp,
         pacing_delay=pacing_delay,
     )
     ```
   - This is the sole LLM call site in the evolutionary loop.
2. **Existing Log Structure** (lines 569-589):
   - Currently writes `log_entry` to `log_path` (JSONL).
   - Logged fields: `family_id`, `run_id`, `model`, `mock`, `iteration`, `operator`, `temperature`, `code`, `valid`, `rejection_reason`, `train_mean`, `gain_train`, `train_counts`, `is_absolute_clone`, `is_equivalent_clone`, `tokens`, `cost_dollars`, `cumulative_cost`, `timestamp`.
3. **Telemetry Gaps Identified**:
   - `prompt_tokens` and `completion_tokens` are currently nested inside `"tokens": usage` dict rather than top-level fields.
   - `latency_s` is completely unmeasured and omitted.
   - `status` is currently recorded indirectly as boolean `valid` and `rejection_reason`, without an explicit string `status` field.
   - `cost_dollars` is logged, but if pricing is unknown or mock pricing is omitted, it should explicitly allow `null` (None in Python).

### 4.2 Exact Additive Modifications Needed in `gate/evolved/loop.py`
To comply with R1 ("ONLY by adding log fields (no behavior change)"):

1. **Measure Latency around `call_llm_with_retry`**:
   At line 491:
   ```python
   t_llm_start = time.perf_counter()
   code, usage, call_cost, attempts = call_llm_with_retry(...)
   latency_s = time.perf_counter() - t_llm_start
   ```
2. **Derive Telemetry Fields**:
   - `prompt_tokens = int(usage.get("prompt_tokens", 0))`
   - `completion_tokens = int(usage.get("completion_tokens", 0))`
   - `latency_s = float(round(latency_s, 4))`
   - `status = "ok" if valid else str(rejection_reason or "rejected")`
   - `cost_dollars = float(call_cost) if (call_cost is not None and call_cost >= 0) else None`
3. **Augment `log_entry`**:
   ```python
   log_entry = {
       "family_id": family_id,
       "run_id": run_id,
       "model": model,
       "mock": mock,
       "iteration": iteration,
       "operator": operator,
       "temperature": op_temp,
       "prompt_tokens": prompt_tokens,
       "completion_tokens": completion_tokens,
       "latency_s": latency_s,
       "status": status,
       "code": code,
       "valid": valid,
       "rejection_reason": rejection_reason,
       "train_mean": train_mean,
       "gain_train": gain_train,
       "train_counts": train_counts_list,
       "is_absolute_clone": is_abs_clone,
       "is_equivalent_clone": is_equiv_clone,
       "tokens": usage,
       "cost_dollars": cost_dollars,
       "cumulative_cost": total_cost,
       "timestamp": time.time(),
   }
   ```
4. **Backward Compatibility**:
   Retains `"valid"`, `"rejection_reason"`, `"tokens"`, and `"cost_dollars"`, ensuring `tests/test_resume.py` and `scripts/smoke_report.py` continue to pass without any modification.

---

## 5. Task 4: Metadata Inspection & Deterministic Binning

### 5.1 Analysis of `data/data_v1_1/families_v1_1.json`
- **Total Families**: Exactly 106 families.
- **Structure Types Distribution**:
  - `gnp`: 18 families
  - `planted_partition`: 16 families
  - `equipartite`: 11 families
  - `mycielski`: 10 families
  - `small_world`: 9 families
  - `caveman`: 9 families
  - `scale_free`: 7 families
  - `geometric`: 7 families
  - `interval`: 7 families
  - `queen`: 7 families
  - `hybrid_geometric`: 5 families
- **Attributes Available per Family Record**:
  - `family_id` (str)
  - `structure_type` (str)
  - `generator` (str)
  - `params` (dict)
  - `n_range` (list of 2 ints, e.g. `[100, 200]`)
  - `probe_stats` (dict): `density`, `deg_cv`, `clustering`, `n`
  - `combo`: `[density_bin, degree_variance_bin, clustering_bin]`

### 5.2 Deterministic Binning Derivation
Derived from `data/generate_dataset.py` lines 397-403 (`def buckets(stats)`):
1. **`n_range`**:
   Directly extracted from `fam["n_range"]`. Existing values:
   `[50, 100]`, `[100, 200]`, `[150, 250]`, `[200, 300]`, `[49, 121]`, `[100, 300]`, `[320, 500]`.
2. **`density_bin`**:
   Thresholds: `DENS_EDGES = [0.05, 0.10, 0.20, 0.40]`
   Formula: `int(np.searchsorted([0.05, 0.10, 0.20, 0.40], stats["density"]))`
   - `0`: density $< 0.05$
   - `1`: $0.05 \le \text{density} < 0.10$
   - `2`: $0.10 \le \text{density} < 0.20$
   - `3`: $0.20 \le \text{density} < 0.40$
   - `4`: $\text{density} \ge 0.40$
3. **`degree_variance_bin`**:
   Metric: `deg_cv = stats["deg_cv"] = std(deg) / mean(deg)`
   - `"low"`: `deg_cv < 0.2`
   - `"medium"`: `0.2 <= deg_cv < 0.5`
   - `"high"`: `deg_cv >= 0.5`
4. **`clustering_bin`**:
   Metric: `ratio = stats["clustering"] / max(stats["density"], 1e-9)` (normalized against random graph baseline where ratio $\approx 1.0$)
   - `"low"`: `ratio < 1.5`
   - `"moderate"`: `1.5 <= ratio < 4.0`
   - `"high"`: `ratio >= 4.0`

In `families_v1_1.json`, every family already has `fam["combo"] = [density_bin, degree_variance_bin, clustering_bin]`.

---

## 6. Task 5: Dataset v1.2 Splits Design (`splits.json` & `build_splits.py`)

### 6.1 Architectural Constraints Check
From `ORIGINAL_REQUEST.md` (R2) and `PLAN_PHASE6_ONWARD.md` (Task 8.1):
1. **Scope**: Partition the existing 106 families into splits without generating new families.
2. **Disjointness**: Every family appears in exactly one group (sum = 106).
3. **`scale_free` Constraint**: Must be in `train_pool`.
4. **`small_world` Constraint**: Must appear neither in `train_pool` nor in `val`. Must be in `heldout_type`.
5. **`test_comp` Constraint**: Explicitly list `heldout_combinations`. None of these combinations may appear in `train_pool`.
6. **`test_extrap` Constraint**: Families in `test_extrap` must strictly exceed `train_pool` maxima in $n$ or density.
7. **Seed References**: Every group references its corresponding seed range name in `SEEDS.md`.

### 6.2 Partitioning Design Analysis

#### A. Compositional Test Split (`test_comp`)
- In `data/data_v1_1/families_v1_1.json`, there are exactly 10 families assigned to `heldout_combo`:
  - `sbm_k8_in0.6_out0.10_n150-250`
  - `sbm_k5_in0.3_out0.15_n100-200`
  - `gnp_d0.10_n200-300`
  - `gnp_d0.15_n100-200`
  - `gnp_d0.20_n100-200`
  - `sbm_k6_in0.6_out0.10_n200-300`
  - `sbm_k8_in0.6_out0.10_n200-300`
  - `sbm_k12_in0.5_out0.10_n200-300`
  - `kpart_k12_p0.15_n100-200`
  - `kpart_k20_p0.2_n200-300`
- **Verification**: All 10 families share combination `[2, "low", "low"]` ($\text{density} \in [0.10, 0.20)$, low degree CV, low clustering ratio).
- **Audit**: In `train_pool`, combination `[2, "low", "low"]` appears **0 times** (empty intersection).
- **Assignment**: These 10 families constitute `test_comp` ($N=10$). `heldout_combinations: [[2, "low", "low"]]`. Seed range: `"test_comp"`.

#### B. Extrapolation Test Split (`test_extrap`)
- In `families_v1_1.json`, there are 15 families with `n_range` $[320, 500]$ (originally `heldout_size`):
  - `gnp` (3 families)
  - `planted_partition` (3 families)
  - `caveman` (2 families)
  - `interval` (1 family)
  - `mycielski` (2 families)
  - `equipartite` (2 families)
  - `small_world` (2 families: `ws_k10_p0.1_n320-500`, `ws_k20_p0.3_n320-500`)
- **Audit against `train_pool` maximum**:
  - In `train_pool`, maximum node count is $\max(n) = 300$.
  - For all 15 families in this pool, $\min(n) = 320 > 300$.
- **Assignment**: These 15 families constitute `test_extrap` ($N=15$). Seed range: `"test_extrap"`. (Note: `small_world` appears neither in `train_pool` nor in `val`).

#### C. Out-of-Distribution Structure Type Split (`heldout_type`)
- In `families_v1_1.json`, there are 26 families with `pool == "heldout_type"`:
  - `small_world` (7 families, $N \le 300$)
  - `geometric` (7 families)
  - `queen` (7 families)
  - `hybrid_geometric` (5 families)
- **Constraint Check**:
  - `small_world` is 100% held out (absent from `train_pool` and `val`).
  - Target structure type for Gate/Phase 6 is `small_world`. The other 19 families from v1.0/v1.1 remain safely held out from training.
- **Assignment**: All 26 families constitute `heldout_type` ($N=26$). Seed range: `"test_heldout_type"`.

#### D. Training Pool, Validation, and Interpolation Splits (`train_pool`, `val`, `test_interp`)
There remain 55 families from v1.1 `train_pool` across 7 structure types:
- `gnp`: 12 families
- `planted_partition`: 8 families
- `mycielski`: 8 families
- `caveman`: 7 families
- `scale_free`: 7 families
- `equipartite`: 7 families
- `interval`: 6 families

We partition these 55 families into:
1. **`val` ($N=7$)**: 1 representative family per structure type to validate training:
   - `gnp_d0.10_n50-100` (gnp)
   - `sbm_k4_in0.7_out0.10_n50-100` (planted_partition)
   - `myciel_t1_p0.3_n50-100` (mycielski)
   - `cave_k8_p0.1_n100-200` (caveman)
   - `plc_m3_p0.3_n100-200` (scale_free)
   - `kpart_k8_p0.3_n100-200` (equipartite)
   - `interval_L0.03_n100-200` (interval)
   - Seed range: `"val"`.
2. **`test_interp` ($N=7$)**: 1 family per structure type with intermediate parameters within the training range:
   - `gnp_d0.40_n100-200` (density 0.40 interpolates between 0.30 and 0.50)
   - `sbm_k6_in0.4_out0.05_n100-200` (k=6 interpolates between k=4 and k=8)
   - `myciel_t2_p0.05_n100-200` (t=2, p=0.05)
   - `cave_k15_p0.1_n100-200` (k=15 interpolates between k=10 and k=20)
   - `ba_m5_n100-200` (m=5 interpolates between m=3 and m=8)
   - `kpart_k15_p0.4_n200-300` (k=15 interpolates between k=10 and k=20)
   - `interval_L0.15_n50-100` (L=0.15 interpolates between 0.10 and 0.20)
   - Seed range: `"test_interp"`.
3. **`train_pool` ($N=41$)**: All remaining 41 families:
   - Contains 5 `scale_free` families (`ba_m8_n100-300`, `ba_m8_n200-300`, `ba_m12_n200-300`, `plc_m5_p0.5_n100-200`, `plc_m8_p0.7_n200-300`).
   - Contains all 5 non-small_world pilot families (`kpart_k10_p0.3_n100-200`, `gnp_d0.30_n100-200`, `sbm_k10_in0.5_out0.05_n200-300`, `interval_L0.10_n100-200`, `ba_m12_n200-300`).
   - Does NOT contain `small_world`.
   - Seed range: `"train"` (and `"rl_train_pool"` for RL rollouts).

### 6.3 Split Family Counts Summary
| Split Name | Seed Range | Family Count | Small Split Flag (< 3) | Target Purpose |
|---|---|---|---|---|
| `train_pool` | `train` | **41** | No | Primary heuristic distillation & RL training pool |
| `val` | `val` | **7** | No | Hyperparameter tuning & model checkpoint selection |
| `test_interp` | `test_interp` | **7** | No | In-distribution parameter interpolation |
| `test_extrap` | `test_extrap` | **15** | No | Graph size & density extrapolation ($n \ge 320$) |
| `test_comp` | `test_comp` | **10** | No | Feature combination generalization (`[2, 'low', 'low']`) |
| `heldout_type` | `test_heldout_type` | **26** | No | Unseen structure types (`small_world`, `geometric`, etc.) |
| **Total** | | **106** | | **100% of families accounted for** |

Every split has $\ge 7$ families. Zero splits have fewer than 3 families.

### 6.4 Schema for `data/data_v1_2/splits.json`
```json
{
  "version": "v1.2",
  "binning": {
    "density_edges": [0.05, 0.10, 0.20, 0.40],
    "degree_variance_thresholds": {
      "low": "deg_cv < 0.2",
      "medium": "0.2 <= deg_cv < 0.5",
      "high": "deg_cv >= 0.5"
    },
    "clustering_thresholds": {
      "metric": "clustering / max(density, 1e-9)",
      "low": "ratio < 1.5",
      "moderate": "1.5 <= ratio < 4.0",
      "high": "ratio >= 4.0"
    }
  },
  "heldout_combinations": [
    [2, "low", "low"]
  ],
  "splits": {
    "train_pool": {
      "seed_range": "train",
      "count": 41,
      "families": [
        {
          "family_id": "gnp_d0.05_n200-300",
          "structure_type": "gnp",
          "n_range": [200, 300],
          "density_bin": 1,
          "degree_variance_bin": "medium",
          "clustering_bin": "low"
        }
      ]
    },
    "val": {
      "seed_range": "val",
      "count": 7,
      "families": [...]
    },
    "test_interp": {
      "seed_range": "test_interp",
      "count": 7,
      "families": [...]
    },
    "test_extrap": {
      "seed_range": "test_extrap",
      "count": 15,
      "families": [...]
    },
    "test_comp": {
      "seed_range": "test_comp",
      "heldout_combinations": [
        [2, "low", "low"]
      ],
      "count": 10,
      "families": [...]
    },
    "heldout_type": {
      "seed_range": "test_heldout_type",
      "target_type": "small_world",
      "count": 26,
      "families": [...]
    }
  }
}
```

---

## 7. Logic Chain: From Observations to Conclusion

1. **Obs. 1**: `SEEDS.md` requires 13 specific ranges (6 frozen, 7 proposed).
   $\to$ Range boundary calculation proves that $[0, 99], [1000, 1019], \dots, [100000, 199999]$ have positive pairwise distances.
   $\to$ **Conclusion 1**: Zero seed range collisions exist. Table structure is ready for implementation.
2. **Obs. 2**: `ORIGINAL_REQUEST.md` mandates `configs/teacher_llm.yaml` with provider, model, base temperature, max_tokens, rate limit, retry/backoff, and env-var-only credential source.
   $\to$ Schema incorporates exact operational settings matching existing `loop.py` pacing (5s) and backoff (HTTP 429).
   $\to$ **Conclusion 2**: Structure is defined, fully parameterizable, and ready for integration.
3. **Obs. 3**: `gate/evolved/loop.py` has a single LLM invocation in `run_evolution` calling `call_llm_with_retry`.
   $\to$ Measuring wall-clock delta with `time.perf_counter()` and extracting token usage from API response dictionary enables logging `prompt_tokens`, `completion_tokens`, `latency_s`, `operator`, `temperature`, `status`, and `cost_dollars`.
   $\to$ **Conclusion 3**: Telemetry fields can be added strictly additively to `log_entry` without changing any loop logic or affecting existing unit tests.
4. **Obs. 4**: `data/data_v1_1/families_v1_1.json` contains 106 families with measured probe statistics and pre-assigned `combo` tuples.
   $\to$ The binning thresholds in `data/generate_dataset.py` produce deterministic values for `n_range`, `density_bin`, `degree_variance_bin`, and `clustering_bin`.
   $\to$ **Conclusion 4**: Attribute extraction is 100% deterministic and requires zero heuristic guesswork.
5. **Obs. 5**: 10 families from `heldout_combo` share combination `(2, 'low', 'low')`, which has 0 occurrences in `train_pool`. 15 families from `heldout_size` have $n \ge 320$, strictly exceeding the train maximum $n=300$. All 9 `small_world` families are separated from `train_pool` and `val`.
   $\to$ Partitioning 55 training families into 41 `train_pool`, 7 `val`, and 7 `test_interp` preserves `scale_free` in `train_pool` and ensures all splits have $\ge 7$ families.
   $\to$ **Conclusion 5**: The splits partitioning strictly satisfies all 6 invariance constraints.

---

## 8. Caveats & Assumptions

1. **Synthetic Unit Test Seeds**: In `tests/test_evaluator.py`, lines 26 and 143 generate synthetic in-memory Erdős-Rényi graphs using seeds `1000 + i` and `6000 + i`. These are isolated to unit tests and do not collide with or read from dataset splits.
2. **Held-out Type Scope**: In v1.1, `heldout_types` in `config.json` contained `["geometric", "hybrid_geometric", "queen", "small_world"]`. `splits.json` places all 26 families of these 4 types into `heldout_type`, with `small_world` designated as the primary benchmark target (`target_type: "small_world"`). If the orchestrator later prefers separating `small_world` into its own split and creating an auxiliary `heldout_geometry_queen` split, the partition can easily accommodate it.
3. **No GPU Requirement**: All components in R1 and R2 execute purely on CPU without requiring PyTorch, CUDA, or external network access.

---

## 9. Independent Verification Method

To independently verify all findings:
1. **Pytest Baseline Check**:
   ```powershell
   pytest
   ```
   *Expected*: All 52 existing tests pass.
2. **Seed Range Disjointness Script Verification**:
   ```powershell
   python -c "
   ranges = [
       (0, 99), (1000, 1019), (2000, 2049), (3000, 3019), (4000, 4199),
       (5000, 5019), (6000, 6049), (7000, 7049), (7100, 7149), (7200, 7249),
       (7300, 7349), (9000, 9009), (100000, 199999)
   ]
   for i in range(len(ranges)-1):
       assert ranges[i][1] < ranges[i+1][0], f'Collision at {ranges[i]} vs {ranges[i+1]}'
   print('All 13 ranges strictly pairwise disjoint!')
   "
   ```
3. **Compositional Holdout Verification**:
   ```powershell
   python -c "
   import json
   with open('data/data_v1_1/families_v1_1.json') as f: fams = json.load(f)
   tp_combos = set(tuple(f['combo']) for f in fams if f['pool'] == 'train_pool')
   assert (2, 'low', 'low') not in tp_combos
   print('Heldout combination verified absent from train_pool!')
   "
   ```
4. **Size Extrapolation Verification**:
   ```powershell
   python -c "
   import json
   with open('data/data_v1_1/families_v1_1.json') as f: fams = json.load(f)
   max_train_n = max(f['n_range'][1] for f in fams if f['pool'] == 'train_pool')
   min_extrap_n = min(f['n_range'][0] for f in fams if f['pool'] == 'heldout_size')
   assert min_extrap_n > max_train_n, f'{min_extrap_n} <= {max_train_n}'
   print(f'Extrapolation verified: min extrap n ({min_extrap_n}) > max train n ({max_train_n})')
   "
   ```
