# DSpark Draft Quality Worklog

Date: 2026-06-27

Active goal: improve DeepSeek V4 Flash DSpark single-stream decode speed by
raising true draft quality through reference-parity fixes and grounded profiling.

## Current Grounding

- Fixed-length salted baseline: `37.64` server decode tok/s mean, `10.02%` CV,
  `14.28%` draft acceptance.
- Confidence threshold `0.50` async bridge: `38.20` server decode tok/s mean,
  `41.76%` CV, `30.06%` draft acceptance.
- Clean threshold-off pre-trim baseline: `29.60` server decode tok/s mean,
  `6.83%` CV, `11.10%` draft acceptance, `0.55` accepted tokens per draft.
- Rejected-context trim fix, threshold off, clean rebuilt runtime image:
  `72.87` server decode tok/s mean, `28.47%` CV, `55.31%` draft acceptance,
  `2.77` accepted tokens per draft.
- Interpretation: the main draft-quality issue was not confidence thresholding.
  vLLM's padded speculative path forwards rejected verification suffixes through
  the target model, and DSpark must not cache those hidden states as accepted
  target context.

## Implemented In This Pass

- vLLM DSpark draft wrapper now returns draft logits with draft ids and
  confidence so profiling can inspect the Markov-corrected draft distribution.
- DSpark proposer has opt-in greedy draft-probability export via
  `VLLM_DSPARK_EXPORT_DRAFT_PROBS=1`.
- Export is deliberately disabled for non-greedy requests until DSpark
  implements left-to-right probabilistic Markov sampling. Returning
  probabilities for argmax-sampled non-greedy drafts would make standard
  rejection sampling mathematically inconsistent.
- DSpark proposer now trims rejected target-context suffix hidden states before
  `prefill_main()`, using `num_rejected_tokens_gpu` and the target forward query
  boundaries from `CommonAttentionMetadata`.
- The fix matches the DeepSpec/DeepSeek reference behavior: only accepted
  target-context hidden states advance the DSpark internal context cache.

## Real-Model Benchmark

Configuration:

- Model: `deepseek-v4-flash-dspark`
- Model path:
  `/home/pieter/.cache/huggingface-dspark/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7`
- Served max model length: `262144`
- Benchmark shape: single stream, `512` target prompt tokens, `256` max decode
  tokens, `temperature=0.0`, `ignore_eos=true`, unique cache salt per run.
- Runtime image: clean rebuilt `vllm-dspark-runtime:local` on head and worker.
- DSpark confidence threshold: `0.0`.

Results from
`single_stream_interactive_262k_window_threshold0_trimfix_threshold0_trimfix_20260627_214524_run*.json`:

| run | server tok/s | draft acceptance | accepted / draft |
| --- | ---: | ---: | ---: |
| 1 | `85.32` | `67.46%` | `3.37` |
| 2 | `48.92` | `33.89%` | `1.69` |
| 3 | `84.37` | `64.59%` | `3.23` |

Aggregate:

- Server decode speed: `72.87` tok/s mean, `20.74` stdev, `28.47%` CV.
- Client approximate decode speed: `54.16` tok/s mean.
- Time to first content: `2.48s` mean.
- Draft acceptance: `55.31%` mean.
- Accepted tokens per draft: `2.77` mean.
- Improvement versus clean threshold-off pre-trim run: `+146.16%`.
- Improvement versus earlier salted threshold-off run: `+93.57%`.
- Improvement versus confidence-threshold `0.50` async post-JIT run:
  `+90.79%`.

## Verification

- Passed: `uv run pytest --hypothesis-show-statistics -q` in
  `experiments/dspark_harness` (`13 passed`, 5 grounded Hypothesis properties
  at 1000 examples each).
- Passed: Ruff on edited vLLM files with external cache:
  `RUFF_CACHE_DIR=/tmp/vllm-dspark-ruff-cache .venv/bin/python -m ruff check ...`
- Passed: focused DSpark vLLM tests in `vllm-dspark-dev:local` with current
  branch modules bind-mounted over the installed package:
  `31 passed, 3 skipped`.
- Passed: real-model single-stream benchmark repeated 3 times with a possible
  `262144` token context window; recorded significant decode speed increase.
- Passed: benchmark script AST parse and `dspark_quality_summary` smoke test.
- Blocked on the host: direct vLLM pytest import needs compiled `vllm._C`; this
  checkout's local uv venv has Python deps but not built vLLM CUDA extensions.
- Also blocked on the host: repo-owned generated/cache files are root-owned
  (`vllm/_version.py`, `.ruff_cache`, many `__pycache__` dirs), so editable
  rebuild and bytecode writes need cleanup or container execution.

## Remaining Shortcuts

- DSpark non-greedy drafting still uses greedy Markov token selection. Full
  reference parity requires sampling each Markov-corrected step left-to-right
  with the request sampling temperature, while retaining the corrected logits
  as draft probabilities for standard rejection sampling.
- DSpark draft KV numeric parity does not reproduce the released reference's
  in-place FP8 quant-dequant on the no-RoPE KV slice. This may affect exact
  logit parity but should be profiled before enabling because a Python fallback
  would slow decode.
- DSpark confidence scheduling is static-threshold based. The paper uses
  calibrated cumulative survival and hardware-aware capacity ranking.
- Variable-prefix scheduling is CPU/shape-management heavy enough that the
  `0.50` threshold acceptance gain was mostly consumed by overhead.
- Rejected-context trimming currently syncs query boundaries and rejection
  counts to CPU and requires uniform effective per-request lengths. This is
  acceptable for the current single-stream experiment, but it must become a
  GPU-side gather/pack path before multi-request DSpark serving is production
  ready.

## Custom Kernel Opportunities

- GPU-side confidence prefix selection and draft-length packing to remove CPU
  scheduling and route-packing overhead.
- Bucketed or graph-captured variable-prefix verification paths for lengths
  `0..5`, avoiding dynamic shape churn.
- Fused DSpark Markov head loop: apply low-rank Markov bias, select/sample the
  next token, and optionally emit top-k/probability diagnostics without full
  Python-side per-position control.
- Reference-parity DSpark KV FP8 quant-dequant for no-RoPE dimensions inside the
  sparse-attention path, ideally fused with KV projection/cache insertion.
- Fused draft-probability/top-k quality probe for profiling so quality metrics
  do not require a full-vocabulary softmax during performance runs.
- GPU-side rejected-context suffix trim and accepted-context gather/pack for
  DSpark `prefill_main()` updates.
- Add FlashInfer sparse MLA decode autotune buckets for observed DSpark decode
  shapes that currently fall back to untuned tactics during startup/graph
  capture.
- Extend DSpark warmup/graph capture coverage for first-request Triton kernels
  observed in logs: `_build_prefill_chunk_metadata_kernel`,
  `_pack_topk_routes_prefix_kernel`, `_pack_topk_routes_post_prefix_kernel`,
  `eagle_prepare_next_token_padded_kernel`, `rejection_greedy_sample_kernel`,
  and `eagle_prepare_inputs_padded_kernel`.

## Next Step

Keep the rejected-context trim as the correctness baseline, then target the
largest remaining decode-speed opportunity: move DSpark context trimming and
draft-length routing onto GPU and graph-capture the hot single-stream draft path
so the quality gain is not diluted by CPU synchronization and dynamic shape
management.
