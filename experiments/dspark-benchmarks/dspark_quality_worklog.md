# DSpark Draft Quality Worklog

Date: 2026-06-27

Active goal: improve DeepSeek V4 Flash DSpark single-stream decode speed by
raising true draft quality through reference-parity fixes and grounded profiling.

Updated completion gate: do not consider the current optimization goal complete
until repeated real-model single-stream decode is at least `25%` faster than the
`72.87` tok/s rejected-context-trim baseline. That requires at least
`91.09` server decode tok/s mean, plus evidence that the hot kernels are more
optimized or better graph-captured.

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
- DSpark warmup expansion for sparse-MLA decode, short prefill, route-pack
  prompt buckets, padded speculative prep, and rejection sampling:
  `78.86` server decode tok/s mean, `1.06%` CV, `56.42%` draft acceptance,
  `2.82` accepted tokens per draft.
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
- DeepSeek V4 warmup now includes DSpark uniform sparse-MLA decode autotune
  shapes, short single-prefill attention shapes, B12X route-pack prompt buckets
  observed in the benchmark, padded speculative prep kernels, and greedy
  rejection-sampler shapes.
- The runtime image overlays `kernel_warmup.py` into the packaged vLLM install,
  so these warmups are present in the experimental container rather than only in
  the source checkout.

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

## Warmup Route-Pack Benchmark

Configuration is the same as the real-model benchmark above, with a clean
rebuilt `vllm-dspark-runtime:local` image on head and worker. Results from
`single_stream_interactive_262k_window_routepack_20260627_231441_run*.json`:

| run | server tok/s | draft acceptance | accepted / draft | TTFC |
| --- | ---: | ---: | ---: | ---: |
| 1 | `79.83` | `57.27%` | `2.86` | `4.28s` |
| 2 | `78.38` | `56.12%` | `2.81` | `1.89s` |
| 3 | `78.37` | `55.88%` | `2.79` | `1.97s` |

Aggregate:

- Server decode speed: `78.86` tok/s mean, `0.84` stdev, `1.06%` CV.
- Client approximate decode speed: `58.66` tok/s mean.
- Time to first content: `2.71s` mean. Run 1 paid route-pack JIT and is not a
  steady-state TTFC sample.
- Draft acceptance: `56.42%` mean.
- Accepted tokens per draft: `2.82` mean.
- Improvement versus rejected-context trim benchmark: `+8.22%`.
- Improvement versus clean threshold-off pre-trim run: `+166.42%`.
- This is a checkpoint, not goal completion: it remains below the `91.09`
  tok/s `+25%` target versus the `72.87` tok/s trim baseline.

Log evidence after JIT monitor activation:

- First small request still triggered one-time
  `eagle_prepare_next_token_padded_kernel` JIT on both ranks.
- Benchmark run 1 still triggered one-time `_pack_topk_routes_prefix_kernel`
  and `_pack_topk_routes_post_prefix_kernel` JIT on both ranks.
- No route-pack JIT repeated in runs 2 or 3; despite the run-1 compile, the
  three-run throughput is now repeatable.

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
- Passed: clean rebuilt experimental container after warmup expansion; `/health`
  returned 200 on `0.0.0.0:8000`; benchmark repeated 3 times and recorded
  `78.86` server decode tok/s mean.
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
- Padded speculative next-token warmup still misses the exact runtime
  specialization for `eagle_prepare_next_token_padded_kernel`. The next pass
  should invoke the real drafter/runner prep path with a representative
  `InputBatch`, not just the raw Triton kernel.
- B12X route-pack warmup by token count is insufficient for the runtime
  specialization. The next pass should prewarm through the actual B12X
  workspace/binding path or trace the runtime top-k dtype/workspace capacity
  before launching direct route-pack kernels.

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
  still observed in logs: `_pack_topk_routes_prefix_kernel`,
  `_pack_topk_routes_post_prefix_kernel`, and
  `eagle_prepare_next_token_padded_kernel`.
- Add a route-pack specialization tracer around B12X W4A16 MoE to record
  runtime `topk_ids` dtype, token count, `top_k`, selected block size,
  `MAX_PACKED_ROUTES`, `MAX_ROUTE_BLOCKS`, and expert-map use.

## Next Step

Keep the rejected-context trim and expanded warmups as the current speed
baseline, then target the remaining one-time JITs with runtime-shape tracing.
After that, move DSpark context trimming and draft-length routing onto GPU and
graph-capture the hot single-stream draft path so the quality gain is not
diluted by CPU synchronization and dynamic shape management.
