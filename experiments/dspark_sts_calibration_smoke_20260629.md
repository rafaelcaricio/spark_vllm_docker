# DSpark STS Calibration Diagnostics Smoke — 2026-06-29

Purpose: validate the opt-in STS calibration-label path on the real
DeepSeek-V4-Flash-DSpark server while keeping diagnostics memory bounded.

This was a short plumbing smoke with `MAX_TOKENS=128`. Treat it as evidence
that the bounded calibration-label path works and logs real acceptance labels,
not as throughput evidence. Future DSpark evidence smoke/benchmark runs should
use `MAX_TOKENS=1024` or higher unless the run is explicitly marked as warmup or
plumbing-only.

## Runtime

- Image: `vllm-dspark-runtime:clean`
- Image id on both nodes:
  `sha256:29ca07cf390aa5772a03ef01140210ac4d65a6434952bb9f9055c4dddbe78fcf`
- `MAX_NUM_SEQS=4`
- `MAX_MODEL_LEN=262144`
- `VLLM_DSPARK_STS_CALIBRATION_DIAGNOSTICS=1`
- `VLLM_DSPARK_STS_TEMPERATURES=` empty
- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`
- `VLLM_LOG_STATS_INTERVAL=1`

## Validation

- Host `py_compile` passed for touched vLLM Python files.
- Ruff passed for touched vLLM Python/test files.
- Runtime image build py-compiled copied DSpark runtime files.
- In-image direct assertions passed for:
  - no calibration matrices allocated when no DSpark confidence row is present;
  - confidence bin counts/accepted/sums;
  - interval logger reset to `None` after logging.
- Advisor review:
  `/home/pieter/Code/vllm-dspark-unholy/experiments/dspark_sts_memory_review.md`
  reported no unbounded object/tensor accumulation. After that review, the
  calibration matrices were made lazy when diagnostics are absent.

## Smoke Result

Benchmark JSON:

`experiments/dspark-benchmarks/single_stream_interactive_262k_window_sts_calibration_smoke_20260629_093033_run1.json`

Key values:

- Prompt tokens: `411` local / `415` server metric delta
- Output tokens: `128`
- Decode elapsed after first content: `2.6316 s`
- Decode speed after first content: `48.26 tok/s`
- End-to-end elapsed: `5.6297 s`
- End-to-end output rate: `22.74 tok/s`
- Drafts: `34`
- Draft tokens: `170`
- Accepted draft tokens: `94`
- Acceptance rate: `55.29%`
- Accepted tokens per draft: `2.76`
- First-token acceptance: `85.29%`
- Mean scheduled draft length: `5.0`

The server emitted repeated `DSpark STS calibration bins` JSON log lines with
per-position `counts`, `accepted`, and `confidence_sums`, confirming the
acceptance labels are reaching the logger. Containers were stopped after the
smoke run.
