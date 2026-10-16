# Handoff Report: Adversarial Challenge of Milestone 1 (M1)

**Agent ID**: `challenger_m1_2`  
**Milestone**: M1 (R1 Seeds, Config & Telemetry)  
**Date**: 2026-10-16  
**Final Verdict**: **REQUEST_CHANGES**  

---

## 1. Observation

### Empirical Test Execution & Failure Reproduction
An empirical test harness was executed at `tests/test_adversarial_m1.py` targeting edge cases in `gate/evolved/loop.py`. Running `pytest tests/test_adversarial_m1.py` yielded 4 failures:

```
=========================== short test summary info ===========================
FAILED tests/test_adversarial_m1.py::test_adversarial_loop_telemetry_none_cost
FAILED tests/test_adversarial_m1.py::test_adversarial_loop_telemetry_none_usage
FAILED tests/test_adversarial_m1.py::test_adversarial_loop_telemetry_llm_exception_logged
FAILED tests/test_adversarial_m1.py::test_adversarial_loop_resume_with_null_cost_dollars
========================= 4 failed, 1 passed in 3.88s =========================
```

### Observation 1: Unhandled `TypeError` on `call_cost is None`
In `gate/evolved/loop.py`:
- **Line 509**:
  ```python
  total_cost += call_cost
  ```
  When `call_cost` is `None` (as explicitly allowed by specification: `"plus optional cost_dollars (null if unknown)"`), `loop.py` crashes with:
  ```
  TypeError: unsupported operand type(s) for +=: 'float' and 'NoneType'
  ```
- **Line 568**:
  ```python
  print(f"[{family_id} #{iteration:03d}] Op={operator:<9} (T={op_temp:.1f}) | {status_str} | Cost=${call_cost:.4f} (Total=${total_cost:.4f})")
  ```
  When `call_cost` is `None`, formatting `${call_cost:.4f}` raises:
  ```
  TypeError: unsupported format string passed to NoneType.__format__
  ```
Both unhandled exceptions abort execution before line 573 where `cost_dollars` is constructed.

### Observation 2: Silently Discarding Resumed Records with `cost_dollars: null`
In `gate/evolved/loop.py`:
- **Line 386**:
  ```python
  total_cost += rec.get("cost_dollars", 0.0)
  ```
- When a previous execution logs `"cost_dollars": null`, Python's `rec.get("cost_dollars", 0.0)` returns `None` (not `0.0`, because the key exists in the dict with value `None`).
- `total_cost += None` raises `TypeError: unsupported operand type(s) for +=: 'float' and 'NoneType'`.
- This exception is caught by line 410:
  ```python
  except Exception:
      continue
  ```
  This silently discards and skips previously evaluated candidates with null cost during resume, resetting the population and repeating work. In `test_adversarial_loop_resume_with_null_cost_dollars`, `len(lines)` was 3 instead of 2 because candidate 1 was dropped and run again from scratch.

### Observation 3: Unhandled `AttributeError` on `usage is None`
In `gate/evolved/loop.py`:
- **Line 507**:
  ```python
  total_prompt_tokens += usage.get("prompt_tokens", 0)
  total_completion_tokens += usage.get("completion_tokens", 0)
  ```
  When `usage` is `None`, this crashes with:
  ```
  AttributeError: 'NoneType' object has no attribute 'get'
  ```
  (Note: Defensive handling `if isinstance(usage, dict)` was added to line 570, but omitted from lines 507-508).

### Observation 4: Omission of Telemetry Logging on LLM Call Failure / Exception
In `gate/evolved/loop.py`:
- **Lines 491-505**:
  ```python
  t_llm_start = time.perf_counter()
  try:
      code, usage, call_cost, attempts = call_llm_with_retry(...)
      latency_s = float(round(time.perf_counter() - t_llm_start, 4))
  except Exception as e:
      print(f"[{family_id} #{iteration}] API Call Failed: {e}")
      break
  ```
  When `call_llm_with_retry` raises an exception (e.g. timeout, rate limit exhaustion, mock failure), the loop breaks immediately.
- Zero telemetry is logged: `log_path` does not receive any entry for this failed LLM attempt. Latency, status (`"error"` / exception message), token counts (0), and cost (None) are completely lost.
- This directly violates the task requirement: *"verify that EVERY LLM call logs prompt_tokens, completion_tokens, latency_s, operator, temperature, status, plus optional cost_dollars (null if unknown)"* and Dispatch Task 1: *"Empirically verify that loop.py telemetry correctly logs latency, status, tokens, and cost under edge conditions (e.g. mock LLM failure/exception, zero tokens, None cost)"*.

### Observation 5: Config & Secret Parsing
- In `configs/teacher_llm.yaml`: File exists and contains no embedded secrets.
- In `tests/test_seeds.py` (lines 149-197): The checks for required fields and regex patterns (`sk-`, `AIza`, `ghp_`, uppercase env var) are present.
- However, `tests/test_seeds.py` only tests the static file and provides no negative unit tests to verify that malformed configs or embedded secrets are rejected. Moreover, the validation logic is not exported as a reusable parsing helper.

---

## 2. Logic Chain

1. **Premise 1**: The project specification (`ORIGINAL_REQUEST.md` line 36) states:
   *"verify that EVERY LLM call logs prompt_tokens, completion_tokens, latency_s, operator, temperature, status, plus optional cost_dollars (null if unknown)."*
2. **Premise 2**: Milestone 1 acceptance criteria requires loop robustness against unhandled exceptions and proper telemetry logging under edge conditions (`mock LLM failure/exception`, `zero tokens`, `None cost`).
3. **Inference from Observations 1, 2, 3**:
   - `loop.py` line 509 directly adds `call_cost` to `total_cost`. If `call_cost is None`, Python throws a `TypeError`.
   - `loop.py` line 568 attempts to format `call_cost` as `${call_cost:.4f}` without a `None` check, throwing a `TypeError`.
   - `loop.py` line 386 uses `rec.get("cost_dollars", 0.0)`. When JSON contains `"cost_dollars": null`, `get()` returns `None`, raising `TypeError` which silently corrupts resume behavior.
   - `loop.py` lines 507-508 calls `.get()` on `usage` without verifying `usage is not None`.
4. **Inference from Observation 4**:
   - When an LLM call fails, `loop.py` catches `Exception` and executes `break` before any log writing occurs. No telemetry entry is written for the failed call.
5. **Conclusion**:
   Because `loop.py` crashes on `cost is None`, corrupts resume on `cost_dollars is null`, and fails to log telemetry when an LLM call fails, Milestone 1 cannot be approved in its current state.

---

## 3. Caveats

- The base test suite (`52` original tests + `15` worker tests) passes when given ideal inputs where `call_cost` is a positive float and no LLM call throws an exception.
- The issues identified only manifest under edge conditions (API failures, missing/null cost, non-dict usage) and resume with null cost, which are standard operational hazards during long-running evolutionary loops.

---

## 4. Conclusion & Required Changes

**Verdict**: **REQUEST_CHANGES**

The following changes must be implemented by the worker to resolve all issues:

1. **In `gate/evolved/loop.py` (line 386)**:
   Change:
   ```python
   total_cost += (rec.get("cost_dollars") or 0.0)
   ```
   (ensuring `None` values default to `0.0` and do not cause `TypeError`).

2. **In `gate/evolved/loop.py` (lines 507-510)**:
   Change:
   ```python
   if isinstance(usage, dict):
       total_prompt_tokens += usage.get("prompt_tokens", 0)
       total_completion_tokens += usage.get("completion_tokens", 0)
   if call_cost is not None:
       total_cost += call_cost
   ```

3. **In `gate/evolved/loop.py` (line 568)**:
   Change:
   ```python
   cost_str = f"${call_cost:.4f}" if call_cost is not None else "None"
   print(f"[{family_id} #{iteration:03d}] Op={operator:<9} (T={op_temp:.1f}) | {status_str} | Cost={cost_str} (Total=${total_cost:.4f})")
   ```

4. **In `gate/evolved/loop.py` (lines 491-506)**:
   Record latency and log a failed telemetry entry before breaking or continuing when an LLM call raises an exception:
   ```python
   t_llm_start = time.perf_counter()
   try:
       code, usage, call_cost, attempts = call_llm_with_retry(...)
       latency_s = float(round(time.perf_counter() - t_llm_start, 4))
   except Exception as e:
       latency_s = float(round(time.perf_counter() - t_llm_start, 4))
       print(f"[{family_id} #{iteration}] API Call Failed: {e}")
       # Log failed attempt to telemetry
       log_entry = {
           "family_id": family_id,
           "run_id": run_id,
           "model": model,
           "mock": mock,
           "iteration": iteration,
           "operator": operator,
           "temperature": op_temp,
           "prompt_tokens": 0,
           "completion_tokens": 0,
           "latency_s": latency_s,
           "status": f"llm_error: {e}",
           "code": None,
           "valid": False,
           "rejection_reason": f"llm_error: {e}",
           "train_mean": None,
           "gain_train": None,
           "train_counts": None,
           "is_absolute_clone": False,
           "is_equivalent_clone": False,
           "tokens": {},
           "cost_dollars": None,
           "cumulative_cost": total_cost,
           "timestamp": time.time(),
       }
       with open(log_path, "a", encoding="utf-8") as f:
           f.write(json.dumps(log_entry) + "\n")
       break
   ```

5. **In `tests/test_seeds.py`**:
   Add unit tests verifying that invalid `teacher_llm.yaml` configurations (missing required keys, invalid ranges, embedded secrets like `sk-...` or lowercase env names) are rejected.

---

## 5. Verification Method

To independently verify the fixes:

1. **Run the adversarial challenge test suite**:
   ```powershell
   pytest tests/test_adversarial_m1.py -v
   ```
   *Current state*: 4 failed, 1 passed.
   *Target state after fix*: 5 passed.

2. **Run the full test suite**:
   ```powershell
   pytest
   ```
   *Target state*: All tests green.
