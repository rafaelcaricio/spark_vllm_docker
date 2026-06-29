# DSpark 1M NVFP4 Derived Repo Review

Date: 2026-06-29

Reviewed repo:
`/home/pieter/Code/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark`

Purpose: identify what differs from our current DSpark/vLLM experiment and what
is worth pulling back here.

## Verdict

The derived repo is valuable as a 1M-context operating recipe, not as a source
tree to merge wholesale.

It packages a two-node DGX Spark launch for `DeepSeek-V4-Flash-DSpark` with:

- `--kv-cache-dtype nvfp4_ds_mla`
- `--max-model-len 1048576`
- `--max-num-seqs 1`
- `--max-num-batched-tokens 8192`
- `--gpu-memory-utilization 0.80`
- `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`

The checkpoint evidence reports:

- API-advertised `max_model_len=1048576`
- KV pool `2,044,166` tokens
- maximum concurrency for a 1,048,576-token request of `1.95x`
- short single-stream probes above `50` tok/s

This is useful for our 1M context-window verification work. It does not prove a
full 1M-token retrieval/correctness benchmark, and it does not solve the true
compact NVFP4 sparse-MLA layout.

## Key Difference

The repo uses a Stage C padded `nvfp4_ds_mla` path:

- Stage A adds `nvfp4_ds_mla` dtype plumbing to vLLM cache config, torch dtype
  lookup, and KV quant-mode selection.
- Stage B tries a true 416-byte NVFP4 sparse-MLA layout for DeepSeek V4.
- Stage C backs DeepSeek V4 down to its known-good 584-byte sparse-MLA cache
  envelope while still routing through `nvfp4_ds_mla`.

Stage C is therefore a pragmatic long-context boot/profile path. It is not the
final 416-byte compact NVFP4 kernel path.

## Why Not Merge It Wholesale

The derived overlay is older than our current DSpark fork for the decode work we
care about:

- It does not contain our compact ragged selected-row main-KV update path.
- It does not contain our mixed prefill+decode stability fixes.
- It does not contain the hardware scheduler instrumentation and draft-length
  histogram metrics added during this pass.
- Its patching model is Dockerfile text replacement against installed site
  packages; for our branch, those changes should become source patches with
  tests.

Pulling it directly would trade away current DSpark scheduler correctness for a
launch recipe. The right move is to port the 1M/NVFP4 pieces into our current
fork intentionally.

## Pull-Back Plan

1. Port `nvfp4_ds_mla` support as source changes, not Dockerfile text patches.
   Target files in our vLLM fork:
   - `vllm/config/cache.py`
   - `vllm/utils/torch_utils.py`
   - `vllm/v1/kv_cache_interface.py`
   - `vllm/models/deepseek_v4/attention.py`
   - `vllm/models/deepseek_v4/nvidia/flashmla.py`

2. Add separate `nvfp4_ds_mla` compose/env profiles in this control repo.
   Keep them isolated from the current 262k fp8 DSpark experiment.

3. Preserve the validated launch choices from the derived repo:
   - `KV_CACHE_DTYPE=nvfp4_ds_mla`
   - `MAX_MODEL_LEN=1048576`
   - `MAX_NUM_SEQS=1` or `6`
   - `MAX_NUM_BATCHED_TOKENS=8192`
   - `GPU_MEMORY_UTILIZATION=0.80`
   - `VLLM_USE_B12X_WO_PROJECTION=0`
   - `VLLM_DSV4_B12X_COMPRESSED_MLA=0`
   - `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`

Implemented local profile files:

- `presets/dspark-v4-flash-1m-nvfp4-mseq1.env`
- `presets/dspark-v4-flash-1m-nvfp4-mseq6.env`
- `presets/dspark-v4-flash-200k-nvfp4-mseq16.env`

The default `.env.dspark-experiment` remains `MAX_MODEL_LEN=262144` with
`KV_CACHE_DTYPE=fp8`.

4. Reuse its operational verification gates:
   - `/v1/models` must report `max_model_len: 1048576`.
   - Logs must report a KV pool around `2,044,166` tokens.
   - Run a short p256/p512, g64/g256 speed probe before any full-context test.

5. Add a stricter next-stage 1M validation:
   - boot and advertised-length check
   - short decode speed probe
   - long prefill smoke below, near, and at the 1M envelope
   - retrieval/correctness prompt once prefill is stable

## Risks

- The name `nvfp4_ds_mla` is misleading in Stage C because DeepSeek V4 still
  uses a 584-byte envelope. Treat this as a memory/launch profile first.
- The true 416-byte layout remains unresolved; the derived note says it failed
  past roughly 411 real prompt tokens.
- The derived result is single-stream oriented and scheduler-off. It does not
  answer the current hardware-aware prefix scheduling goal.
- The build script uses node sync/build automation that should be audited before
  reuse. We should port source and env concepts, not blindly run the scripts.

## Immediate Gain

The clear benefit to pull back is a dedicated 1M context-window verification
lane. It gives us a proven parameter set for advertising a 1M window on the two
DGX Spark nodes, while our main branch continues improving DSpark decode
scheduling and metrics.

## Pull-Back Result

Implemented in our branches on 2026-06-29:

- Source-ported `nvfp4_ds_mla` Stage C support into the vLLM fork instead of
  Dockerfile text patching.
- Added isolated runtime profiles:
  - `presets/dspark-v4-flash-1m-nvfp4-mseq1.env`
  - `presets/dspark-v4-flash-1m-nvfp4-mseq6.env`
  - `presets/dspark-v4-flash-200k-nvfp4-mseq16.env`
- Kept `.env.dspark-experiment` on the default 262k fp8 lane.
- Made `KV_CACHE_DTYPE` configurable in the entrypoint/compose path.

Runtime validation:

| profile | health | served max model len | KV pool | full-window concurrency | request |
| --- | --- | ---: | ---: | ---: | --- |
| `1M / max_num_seqs=1` | pass | `1048576` | `2,110,064` | `2.01x` | pass |
| `1M / max_num_seqs=6` | pass | `1048576` | `2,068,655` | `1.97x` | pass |
| `200k / max_num_seqs=16` | pass | `200000` | `788,856` | `3.94x` | pass |

The `200k / max_num_seqs=16` profile is a shared-pool interactive concurrency
lane. It does not mean sixteen simultaneous full-200k requests fit; the engine
reported room for about four full-window requests.

Remaining gap: Stage C still uses DeepSeek V4's padded 584-byte sparse-MLA
envelope. The true compact 416-byte NVFP4 kernel remains unresolved and should
stay out of the default speed path until a source-level implementation passes
long-context correctness tests.

## Focused Pull-Back Follow-Up

Reviewed again on 2026-06-29 against the current `vllm-dspark-unholy` branch.
The community repo's important request-stability concept is now represented in
our newer code path without copying its older overlay:

- Main-KV writes use our direct selected-row Triton store path with
  `request_indices`.
- Draft reads pass the same persistent request slot tensor into
  `dspark_sparse_attention`, so condensation cannot silently switch a victim
  request to another row.
- Added a direct unit regression for the draft-read side of the condense bug:
  selected rows `[2, 0]` must read persistent rows `[2, 0]`, not compact rows
  `[0, 1]`.
- Added source-level Stage C tests for `nvfp4_ds_mla` page size and FlashMLA
  cache shape, keeping the 1M lane pinned to the padded 584-byte DeepSeek V4
  sparse-MLA envelope.
- Widened DSpark direct-store warmup coverage to include the validated
  concurrency lanes `1/4/8/16` as available, so c8/c16 clean-image runs do not
  pay a first-live-request Triton compile for the selected-row store kernel.

Focused Docker pytest command:

```bash
scripts/run-dspark-tests-in-docker.sh \
  tests/model_executor/test_flashinfer_autotune_cache.py::test_dspark_warmup_request_counts_cover_concurrency_lanes \
  tests/v1/spec_decode/test_dspark.py::test_dspark_sparse_attention_reads_request_stable_rows_after_condense \
  tests/v1/spec_decode/test_dspark.py::test_deepseek_v4_nvfp4_ds_mla_uses_stage_c_padded_page_size \
  tests/v1/spec_decode/test_dspark.py::test_deepseek_v4_flashmla_nvfp4_ds_mla_uses_padded_shape \
  -q
```

Result: `6 passed, 1 skipped`. The skip is the expected FlashMLA-extension
import skip in the dev test container; the serving image carries that module.

## Current Pull-Back Evidence

Latest request-stability smoke on the preserved 262k/fp8 lane:
`experiments/dspark-benchmarks/request_stability_pullback_20260629_143731.json`.

Configuration:

- Runtime image: clean no-cache image verified to contain request-stable slot
  wiring.
- Served model: `deepseek-v4-flash-dspark`
- Served max model length: `262144`
- `max_num_seqs=8`
- Smoke generation cap: `max_tokens=1024`

Results:

| check | result | aggregate tok/s | acceptance | accepted / draft |
| --- | --- | ---: | ---: | ---: |
| static c8 | pass, 8/8 requests | `129.995` | `36.74%` | `1.837` |
| staggered c8 | pass, 8/8 requests | `126.810` | `36.91%` | `1.846` |
| condense/churn | pass, byte-for-byte victim match | n/a | `48.75%` | `2.437` |

The condense test ran `52` churn requests while the victim request stayed
byte-for-byte equal to its reference and to the expected deterministic payload.
This is the practical validation that persistent DSpark main-KV rows survive
request start/finish churn and batch condensation.

Matched single-stream baseline on the same live image:
`single_stream_interactive_262k_window_pullback_before_raggedwarmup_20260629_144112_run*.json`.

| metric | mean | stdev | notes |
| --- | ---: | ---: | --- |
| server decode tok/s | `63.019` | `0.970` | three 1024-token runs |
| draft acceptance | `73.31%` | `1.73%` | greedy code-completion prompt |
| accepted / draft | `3.666` | `0.086` | stable quality signal |
| estimated cycle ms | `74.033` | `0.257` | derived from server tok/s and accepted/draft |

Interpretation:

- The community repo's major correctness idea is now pulled into our newer
  code path: stable request slots are used by both DSpark main-KV writes and
  draft reads.
- The community repo does not contain a newer decode-speed breakthrough beyond
  that slot stability and the 1M/NVFP4 launch lane.
- Remaining useful pull-back material is operational: keep the 1M and 200k
  profiles isolated, retain the KV-pool/concurrency documentation, and use
  their c16/c32-style benchmark tables as comparison baselines when running the
  `200k/max_num_seqs=16` profile.
- Do not wholesale copy the overlay. It is older than our current branch for
  direct selected-row stores, ragged batches, metrics, STS scheduling work, and
  GPU/offload-oriented fast paths.
