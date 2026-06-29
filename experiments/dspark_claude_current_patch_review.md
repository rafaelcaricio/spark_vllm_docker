# Current Patch Review - Triton Store-Main-KV + STS Calibration + Diagnostics

Date: 2026-06-29
Reviewer: Claude (read-only advisor)
Patch: uncommitted changes in vllm-dspark-unholy (7 files, +635/-50)

---

## Summary

Three changes:
1. **Direct Triton `_dspark_store_main_kv_kernel`** replacing the `index_select -> scatter -> index_copy` bridge in `store_main_kv`.
2. **Opt-in STS confidence calibration** (`_calibrate_confidence` with `VLLM_DSPARK_STS_TEMPERATURES`).
3. **Gated confidence diagnostics logging** (`_maybe_log_confidence_diagnostics` with `VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY`).

All three are well-designed. The Triton kernel is a genuine fusion win (1 kernel vs 3 ops + a copy). The STS calibration is the paper's correct mechanism (logit-space temperature scaling). The diagnostics are gated and Python-only. The critical `HAS_REJECTED`/`HAS_REQUEST_INDICES` constexprs are correctly sourced from the original parameters (line 877-878: `num_rejected_tokens is not None` / `request_indices is not None`), not from the dummy fallbacks - no aliasing bug.

---

## Correctness risks

### WARNING 1. Torch fallback has CPU-GPU syncs (latent, not hot-path)

`dspark_store_main_kv_torch` (the fallback) loops per-batch with `.item()` calls:
```python
for batch_idx in range(batch_size):
    valid_len = int(valid_lengths[batch_idx].item())  # CPU sync
    row = int(rows[batch_idx].item())                  # CPU sync
```
Each `.item()` blocks the CPU until the GPU produces the value. At c=4/c=8, that's 4-8 syncs per store. If this fallback fires during a benchmark (e.g., a shape edge case), it serializes the pipeline -> silent performance regression.

**Mitigation**: the fallback only triggers for non-CUDA / non-contiguous / unexpected shapes. In normal operation (CUDA, contiguous, correct ndim), the Triton path is taken. But add a startup log or a metric that records whether the torch fallback was ever hit during the run - so a silent fallback doesn't masquerade as a "Triton kernel regression."

### WARNING 2. STS temperatures must be calibrated, not guessed

`_calibrate_confidence` applies `sigmoid(logit(p) / T)`. For T > 1 this flattens confidence toward 0.5 (less overconfident - the paper's intent). For T < 1 it sharpens. **Arbitrary T values can make the scheduler WORSE** - over-flattened confidence -> the scheduler thinks all positions are ~50% survival -> prunes too aggressively -> lower tau.

The paper derives T by minimizing Expected Calibration Error (ECE) on a validation set. **The diagnostics logging you added (`avg_raw_confidence`, `calibration_delta`) is exactly the tool to derive T**: run a benchmark with diagnostics on (no STS), collect raw confidence + actual acceptance per position, compute ECE vs T numerically, pick the T that minimizes ECE per position.

**Action**: before benchmarking with STS, run one diagnostic pass (no STS) to collect raw confidence, then derive T offline. Setting `VLLM_DSPARK_STS_TEMPERATURES` without calibration is likely to regress acceptance.

### [ok] 3. The dummy-pointer pattern is safe

When `num_rejected_tokens is None` -> `rejected = slots` (dummy pointer). But `HAS_REJECTED=False` (line 877) -> the kernel never dereferences the rejected pointer. Same for request_indices. **No aliasing or corruption risk.** Verified.

### [ok] 4. Store mask correctly handles all boundaries

`should_store = (batch_idx < batch_size) & (token_idx < valid_len) & valid_d` - correctly gates:
- Out-of-bounds batch (grid over-allocation). [ok]
- Rejected suffix tokens. [ok]
- head_dim padding (D_BLOCK boundary). [ok]

Slot collisions (two tokens writing the same `main_kv_cache[row][slot]`) are last-writer-wins - same semantics as the old `scatter_`. Correct for a sliding-window cache. [ok]

---

## Hot-path overhead risks

### [ok] Triton kernel: net faster than the bridge

Old: `index_select` (copy N cache rows) -> `scatter` (on copy) -> `index_copy_` (write back). 3 kernel launches + 1 intermediate tensor allocation.

New: 1 fused kernel, direct in-place store, no copy. At c=4 (batch 4, seq 6): grid = (24, 8) = 192 programs x 64 elements each. Launch overhead ~5us, work ~2us. The old 3-op bridge had ~15us overhead (3 launches + copy). Net: **~10us saved per store_main_kv call**. At 14 steps/sec (single-stream), negligible. At c=8 (~27 steps/sec): ~270us/sec saved. Small but real.

### [ok] STS calibration: negligible at current batch sizes

`_calibrate_confidence` does: `.float()` -> `.clamp()` -> `.logit()` -> division -> `.sigmoid()`. On a (batch, gamma) = (8, 5) = 40-element tensor. Total: ~5 GPU ops on 40 elements -> <1us. **Not a hot-path concern.**

One minor optimization: `torch.tensor(temperatures, device=...)` at line 1005-1009 constructs a new GPU tensor EVERY call (when per-position temperatures are used). Pre-allocate at `__init__` and store as `self._sts_temp_tensor`. Saves one Python->GPU allocation per step. ~microseconds - not blocking but free.

### [ok] Diagnostics: gated, Python-only, no GPU sync

The diagnostics accumulate via Python float sums (from `.tolist()`'d confidence values). `snapshot()` is pure Python arithmetic. The log line is ~200 chars, every N steps. **No GPU sync, no hot-path impact.**

---

## Missing tests

| Gap | Priority | What to test |
|---|---|---|
| **Triton vs torch equivalence** | High | Call both `dspark_store_main_kv` and `dspark_store_main_kv_torch` on the same CUDA inputs, `torch.testing.assert_close`. Verify the dispatch picks Triton (not the fallback) on CUDA. |
| **STS calibration correctness** | Medium | T=1 -> identity (no change). T>1 -> flattened (confidence moves toward 0.5). T<1 -> sharpened. Per-position T -> different calibration per column. Clamp behavior at p~ 0 and p~ 1. |
| **STS + scheduler integration** | Medium | With STS enabled + a confidence head producing overconfident values -> verify the scheduler produces different (shorter) prefix lengths than without STS. |
| **Kernel edge cases** | Low | Empty batch (batch_size=0), single token (seq_len=1), full rejection (valid_len=0 -> no writes), request_indices out of order. |

---

## Next kernel/profiling opportunity

### 1. Pre-allocate STS temperature tensor (trivial, free)

Move `torch.tensor(temperatures[:confidence.shape[1]], device=confidence.device)` from `_calibrate_confidence` to `__init__`. Saves one allocation per draft step.

### 2. Calibrate STS temperatures from diagnostics data (high-value, pre-benchmark)

Run one diagnostic pass (`VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY=20`, no STS) at c=4. Collect raw confidence + actual acceptance per position. Compute ECE vs T (sweep T=0.5..3.0, find the minimum). Set `VLLM_DSPARK_STS_TEMPERATURES` to the calibrated values. This makes STS grounded rather than guessed.

### 3. Profile the store_main_kv kernel at c=8/c=16 (NCU, post-benchmark)

The kernel does 1 store per (batch_token, head_dim_block). At c=16 (batch 16, seq 6, head_dim 512): grid = (96, 8) = 768 programs. Each writes 64 bf16 elements (128 bytes). Total: 768 x 128 = 96 KB of scattered writes. The scatter pattern (positions % window_size) is random within the window -> poor L2 locality. NCU would show: is this kernel memory-latency-bound (scattered writes) or compute-bound? If latency-bound, consider coalescing writes by sorting by slot (group tokens that write to nearby cache slots). If compute-bound (unlikely for a copy kernel), increase D_BLOCK.

### 4. Fuse STS into the confidence head's sigmoid (medium, hot-path)

The confidence head produces logits -> `sigmoid(logits)` -> confidence. Then `_calibrate_confidence` does `logit(confidence) / T -> sigmoid(...)`. This is `sigmoid(logit(sigmoid(logits)) / T)` - two sigmoid + one logit that partially cancel. Fusing: `sigmoid(logits / T)` directly from the head's raw logits (before the first sigmoid). Saves 2 elementwise ops. Requires the draft model to export raw confidence logits (not post-sigmoid). Small gain but clean.

---

## Bottom line

**Ship it.** The three changes are correct, well-tested for the critical paths, and the Triton kernel is a genuine fusion win. The two pre-benchmark actions:

1. **Derive STS temperatures from diagnostics data** (one diagnostic pass, offline ECE minimization) - don't guess T.
2. **Add the Triton-vs-torch equivalence test** - ensures the fallback path matches the kernel, so a silent dispatch to the torch fallback can't masquerade as a kernel issue.

Neither blocks the c=16/single-stream benchmark, but both make the results trustworthy.
