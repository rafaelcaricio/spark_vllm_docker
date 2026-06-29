# DSpark Direct Main-KV Store Benchmark - 2026-06-29

## Change Under Test

- Rebuilt `vllm-dspark-runtime:clean` from
  `/home/pieter/Code/vllm-dspark-unholy/docker/Dockerfile.dspark-runtime-overlay`.
- Image on both nodes:
  `sha256:b74eee540e779c223d06f61c5bcfda5c6a2ff65d2e5554b609db67d16f2f7d81`.
- Replaced the selected-row `index_select` / `index_copy_` main-KV update bridge
  with a direct Triton DSpark main-KV store kernel.
- Scheduler off, confidence diagnostics off, `MAX_NUM_SEQS=16`.

## Validation

- Runtime image focused tests:
  `12 passed, 56 deselected`.
- Follow-up clean-image test after adding direct-store warmup and Triton JIT
  compiler dependencies (`gcc`, `libc6-dev`) passed:
  `12 passed, 56 deselected`.
- Final clean-image test after STS temperature tensor preallocation passed:
  `13 passed, 56 deselected`. Final image before fresh server validation:
  `sha256:dd5d0877318f32a9004f9bd9f1c23d51f4c154a841afc8a055ab070036b41a06`.
- Server bound to `0.0.0.0:8000`; `/health` returned 200 after startup.
- Startup confirmed DSpark fast draft-output mode:
  confidence head and returned draft logits skipped on the speed path.

## c16 Throughput Gate

Pattern:

`experiments/dspark-benchmarks/concurrent_interactive_262k_window_c16_directstore_kernel_native_off_mseq16_c16_20260629_081549_run*.json`

| run | per-user tok/s | aggregate tok/s | acceptance | accepted/draft |
| --- | ---: | ---: | ---: | ---: |
| 1 | 19.624 | 313.988 | 0.6122 | 3.061 |
| 2 | 19.841 | 317.462 | 0.6254 | 3.127 |
| 3 | 20.391 | 326.257 | 0.6306 | 3.153 |
| mean | 19.952 | 319.236 | 0.6227 | 3.114 |

Prior scheduler-off c16 baseline:

`experiments/dspark-benchmarks/concurrent_interactive_262k_window_c16_screenshot_scaling_verify_off_mseq16_c16_20260629_064107_run*.json`

| metric | baseline mean | direct-store mean |
| --- | ---: | ---: |
| aggregate tok/s | 319.366 | 319.236 |
| per-user tok/s | 19.960 | 19.952 |
| acceptance | 0.6233 | 0.6227 |

Result: no measurable steady-state speedup. The selected-row copy bridge was not
the c16 throughput bottleneck for this workload.

## Single-Stream Repeat

Pattern:

`experiments/dspark-benchmarks/single_stream_interactive_262k_window_directstore_kernel_native_single_20260629_081549_run*.json`

| run | tok/s | acceptance | accepted/draft | cycle ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 56.394 | 0.6633 | 3.317 | 76.545 |
| 2 | 54.053 | 0.6222 | 3.111 | 76.056 |
| 3 | 54.390 | 0.6355 | 3.177 | 76.805 |
| mean | 54.946 | 0.6403 | 3.202 | 76.469 |

This is a warmed, current-prompt sanity check. Do not compare it directly to
older single-stream sets with different prompt/metric conditions.

## Evidence and Next Step

- First real inference JIT-compiled `_dspark_store_main_kv_kernel`.
  Follow-up code adds warmup coverage for the direct-store kernel to remove the
  one-time latency spike; validate on a fresh server start by checking logs for
  absence of `_dspark_store_main_kv_kernel` JIT during the first live request.
- Fresh validation with image
  `sha256:dd5d0877318f32a9004f9bd9f1c23d51f4c154a841afc8a055ab070036b41a06`
  started successfully on `0.0.0.0:8000` with `MAX_NUM_SEQS=16`.
  `/health` passed and a first chat request returned 200 at
  `2026-06-29T08:51:06+02:00`. Head and worker logs showed only JIT monitor
  activation, with no `_dspark_store_main_kv_kernel` or inference-time Triton
  JIT warning after the first live request.
- Since steady c16 throughput did not move, the next large decode-speed work
  should target fused DSpark kernels:
  projection + RoPE + main-KV store, sparse attention output + inverse RoPE +
  `wo` projection, and fused Markov logits/argmax/confidence.
- Keep `VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY=0` for throughput gates.
  Enable it only with `VLLM_DSPARK_COLLECT_CONFIDENCE_DIAGNOSTICS=1` for
  calibration/profiling runs.
