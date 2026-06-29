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

2. Add a separate `1m-nvfp4-dspark` compose/env profile in this control repo.
   Keep it isolated from the current 262k fp8 DSpark experiment.

3. Preserve the validated launch choices from the derived repo:
   - `MAX_MODEL_LEN=1048576`
   - `MAX_NUM_SEQS=1`
   - `MAX_NUM_BATCHED_TOKENS=8192`
   - `GPU_MEMORY_UTILIZATION=0.80`
   - `VLLM_USE_B12X_WO_PROJECTION=0`
   - `VLLM_DSV4_B12X_COMPRESSED_MLA=0`
   - `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`

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
