# DSpark Draft Quality Worklog

Date: 2026-06-27

Active goal: improve DeepSeek V4 Flash DSpark single-stream decode speed by
raising true draft quality through reference-parity fixes and grounded profiling.

Latest live gate, 2026-06-28: keep the goal open until the corrected interactive
single-stream benchmark records a repeated real-model mean above `100` server
decode tok/s. Do not downshift this gate because the benchmark parser was
corrected. The paper reports much larger frontier gains in favorable serving
regimes, so `>100` tok/s remains an appropriate local target for this hardware
once the implementation is optimized. The local path must reduce
`Tdraft + Tverify` while preserving accepted length.

Updated completion gate: do not consider the current optimization goal complete
until repeated real-model single-stream decode is at least `25%` faster than the
`72.87` tok/s rejected-context-trim baseline. That requires at least
`91.09` server decode tok/s mean, plus evidence that the hot kernels are more
optimized or better graph-captured.

## Current Grounding

- Measurement correction, 2026-06-28: the benchmark stream parser now counts
  `delta.reasoning` as generated text. Older high-80s server decode tok/s
  results only started the decode clock when final `content` arrived, missing
  streamed reasoning tokens and inflating interactive tok/s. The corrected
  parser is the grounded metric for interactive inference speed.
- Corrected-parser paper-default run:
  `53.17` server decode tok/s mean, `3.87%` CV, `54.28%` draft acceptance,
  `2.71` accepted tokens per draft.
- Corrected-parser warmed repeat:
  `52.36` server decode tok/s mean, `20.15%` CV, `50.45%` draft acceptance,
  `2.52` accepted tokens per draft.
- Position-0 diagnostic run with corrected parser and diagnostics enabled:
  `57.88` server decode tok/s mean, `11.60%` CV, `61.01%` draft acceptance,
  `3.05` accepted tokens per draft. This is not the performance gate because
  the diagnostic runs the confidence head and copies scalar debug data, but it
  provides the current quality signal.
- Position-0 target-argmax diagnostic after three benchmark runs:
  `193` samples, `80.8%` target-argmax match, average diagnostic confidence
  `0.314`, matched confidence `0.315`, missed confidence `0.313`, and `94`
  confidence scalars normalized from logit-like values. This means the first
  draft token is often target-greedy-correct, but the captured confidence signal
  is not calibrated enough to drive the paper scheduler yet.
- Position-0 diagnostic cache-alias fix, 2026-06-28: the diagnostic-only
  cached confidence tensor was a detached GPU view into model output storage.
  That storage can be reused before the runner consumes the debug tensor, which
  explained the earlier logit-like confidence values. The diagnostic cache now
  clones the confidence tensor before handing it to the verifier-side logger.
- Position-0 target-argmax diagnostic after the clone fix:
  `paper_pos0clone_20260628_022224` measured `57.73` server decode tok/s mean,
  `2.96%` CV, `61.21%` draft acceptance, and `3.06` accepted tokens per draft
  with diagnostics enabled. Final logs reported `193` samples, `84.5%`
  target-argmax match, average confidence `0.977`, matched confidence `0.984`,
  missed confidence `0.942`, and `0` normalized confidence scalars.
- Updated interpretation: the confidence path is clean and paper-consistent,
  and position-0 quality is healthy. The remaining speed gap is dominated by
  execution overhead in `Tdraft + Tverify`, not by low draft quality or a missing
  confidence-threshold trick.
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
- Fast draft-output path, threshold off, clean rebuilt runtime image:
  `72.68` server decode tok/s mean, `23.88%` CV, `52.49%` draft acceptance,
  `2.62` accepted tokens per draft. This reduced returned-logit/confidence
  work and graph memory but did not improve the speed gate.
- Interpretation: the main draft-quality issue was not confidence thresholding.
  vLLM's padded speculative path forwards rejected verification suffixes through
  the target model, and DSpark must not cache those hidden states as accepted
  target context.
- Paper-refresh interpretation: in steady runs, position-wise conditional
  acceptance does not show DFlash-like suffix collapse. The next acceptance
  breakthrough is more likely first-token draft quality or reference numeric
  parity than tail pruning alone.

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

## Fast Draft-Output Benchmark

Configuration is the same as the real-model benchmark above. Results from
`single_stream_interactive_262k_window_fastoutputs_20260627_234455_run*.json`:

| run | server tok/s | draft acceptance | accepted / draft |
| --- | ---: | ---: | ---: |
| 1 | `87.54` | `65.00%` | `3.25` |
| 2 | `48.33` | `31.52%` | `1.58` |
| 3 | `82.17` | `60.95%` | `3.05` |

Aggregate:

- Server decode speed: `72.68` tok/s mean, `17.36` stdev, `23.88%` CV.
- Draft acceptance: `52.49%` mean.
- Accepted tokens per draft: `2.62` mean.
- Improvement versus the `72.87` tok/s rejected-context-trim baseline:
  `-0.26%`; this does not clear the `91.09` tok/s speed gate.
- Graph memory evidence improved: startup estimated CUDA graph memory fell
  from about `0.33 GiB` before the fast-output path to `0.17 GiB` after it.

Position-wise conditional acceptance, computed from the same prefix-survival
counter shape used in the DSpark paper:

| run | pos0 | pos1 | pos2 | pos3 | pos4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `81.67%` | `89.80%` | `86.36%` | `86.84%` | `93.94%` |
| 2 | `58.59%` | `67.24%` | `61.54%` | `79.17%` | `84.21%` |
| 3 | `77.78%` | `87.76%` | `86.05%` | `91.89%` | `85.29%` |

Mean conditional acceptance: `[72.68%, 81.60%, 77.98%, 85.97%, 87.81%]`.
Run 2 is mostly a first-token acceptance failure, not monotonic suffix decay.

Invalid kernel experiment:

- `_DSPARK_SCORE_K_BLOCK=16` was tested on the real model and rejected.
- Three measured runs produced near-zero accepted drafts (`0`, `0`, `1`
  accepted tokens), so the score block was restored to `8`.

## Reference KV Quant-Dequant Probe

Configuration is the same as the real-model benchmark above, with
`VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=1` enabled in the clean runtime image.
This applies a reference-parity FP8 quant-dequant pass to the no-RoPE draft KV
slice after DSpark KV projection.

Results from
`single_stream_interactive_262k_window_refkv_refkv_clean_20260628_003833_run*.json`:

| run | server tok/s | draft acceptance | accepted / draft | pos0 conditional |
| --- | ---: | ---: | ---: | ---: |
| 1 | `60.29` | `43.75%` | `2.19` | `77.50%` |
| 2 | `83.17` | `64.92%` | `3.25` | `85.25%` |
| 3 | `62.01` | `43.46%` | `2.17` | `75.31%` |

Aggregate:

- Server decode speed: `68.49` tok/s mean, `12.74` stdev, `18.60%` CV.
- Draft acceptance: `50.71%` mean.
- Accepted tokens per draft: `2.54` mean.
- First-token conditional acceptance: `79.35%` mean.
- This does not clear the `>90` tok/s goal and regresses versus the route-pack
  checkpoint (`78.86` tok/s, `56.42%` acceptance).
- First run JIT-compiled `_dspark_quant_dequant_nope_kernel`, so the exact run-1
  latency is not steady-state. Runs 2-3 still do not show a speed or quality win.
- Decision: keep the reference KV path as an explicit runtime flag for future
  correctness A/Bs, but do not bake it into the clean image by default.

## Clean No-RefKV Rebuild Benchmark

The clean runtime image was rebuilt so that
`VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT` remains a runtime flag instead of a
baked default. The clean service was restarted from `vllm-dspark-runtime:clean`
on both nodes and `/health` returned 200 on `0.0.0.0:8000`.

Results from
`single_stream_interactive_262k_window_clean_norefkv_20260628_004804_run*.json`:

| run | server tok/s | draft acceptance | accepted / draft | pos0 conditional |
| --- | ---: | ---: | ---: | ---: |
| 1 | `93.54` | `68.62%` | `3.43` | `86.21%` |
| 2 | `77.79` | `60.94%` | `3.05` | `76.56%` |
| 3 | `89.50` | `67.80%` | `3.39` | `86.44%` |

Aggregate:

- Server decode speed: `86.94` tok/s mean, `8.18` stdev, `9.41%` CV.
- Draft acceptance: `65.78%` mean.
- Accepted tokens per draft: `3.29` mean.
- First-token conditional acceptance: `83.07%` mean.
- This was the best old-parser baseline after rejecting the reference-KV
  default. It is useful for acceptance comparison, but the interactive decode
  speed number is not comparable to corrected-parser runs because the old
  parser missed `delta.reasoning` stream chunks.

## Rejected MTP-Length Sweep

The DSpark paper and V4 production notes fix the released V4 Flash/Pro draft
block at `gamma = 5` with the Markov head. Reducing `MTP_NUM_TOKENS` in our
vLLM integration is not equivalent to the paper scheduler, because the DSpark
draft model still computes the full configured `dspark_block_size=5` internally
and only verifies a shorter suffix. This sweep is therefore rejected as a
diversion from the paper path.

Results below are kept for draft-length and acceptance comparison. The tok/s
columns were captured with the old parser and should not be used as the
interactive speed gate.

Results:

| config | server tok/s mean | stdev | CV | draft acceptance | accepted / draft | pos0 conditional | suffix conditional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MTP_NUM_TOKENS=5` | `86.94` | `8.18` | `9.41%` | `65.78%` | `3.29` | `83.07%` | `87.39%` |
| `MTP_NUM_TOKENS=4` | `87.03` | `5.41` | `6.22%` | `70.14%` | `2.81` | `84.83%` | `87.51%` |
| `MTP_NUM_TOKENS=3` | `73.38` | `3.69` | `5.03%` | `77.57%` | `2.33` | `87.52%` | `88.49%` |

Interpretation: shorter verification raises per-token acceptance rates but
reduces accepted tokens per draft and does not reduce draft compute. The paper
solution is not changing gamma; it is confidence-scheduled target verification
using calibrated cumulative survival and a profiled hardware step curve.

## Paper-Grounded Scheduler Checkpoint

Implementation change:

- `DeepSeekV4DSparkModel.draft()` can now return confidence scores without
  returning full draft logits. This keeps the paper's confidence head available
  while using the cheaper Markov greedy argmax path when draft probabilities are
  not being exported.
- `DSparkProposer` now has explicit scheduler modes:
  `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off|threshold|hardware|auto`.
- The `hardware` mode wires Algorithm 1 from the paper into the runtime path,
  using cumulative prefix survival and a profiled
  `VLLM_DSPARK_SPS_CURVE=<batch_tokens>:<steps_per_second>,...` table.
- `VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1` is the default for the
  synchronous lossless Algorithm 1 path. Turning it off is reserved for a future
  async two-step scheduling barrier like the paper's Section 5.2 production
  adaptation.

This is the next clean benchmark target. It keeps gamma at `5` and evaluates
whether paper-style confidence scheduling can improve repeated decode speed
without the old full-logit confidence overhead.

## Corrected Stream-Parser Baseline

The benchmark now treats `delta.reasoning`, `delta.reasoning_content`, and
`delta.content` as generated stream text. This matches what the user experiences
interactively and fixes the older timing artifact where reasoning streamed
before the benchmark started the content clock.

Configuration:

- Runtime image: clean rebuilt `vllm-dspark-runtime:clean` overlay on head and
  worker.
- Endpoint: `http://127.0.0.1:8000`, bound to `0.0.0.0:8000`.
- Served max model length: `262144`.
- Benchmark shape: single stream, `512` target prompt tokens, `256` max decode
  tokens, `temperature=0.0`, `ignore_eos=true`, unique cache salt per run.
- DSpark settings: `MTP_NUM_TOKENS=5`,
  `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`, no local-argmax override, no
  reference-KV quant-dequant.

Results from
`single_stream_interactive_262k_window_paper_default_20260628_014117_run*.json`:

| run | server tok/s | approximate tok/s | draft acceptance | accepted / draft | pos0 conditional | suffix conditional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `51.97` | `49.73` | `54.00%` | `2.70` | `80.00%` | `73.25%` |
| 2 | `55.54` | `53.80` | `57.01%` | `2.85` | `82.09%` | `82.25%` |
| 3 | `51.99` | `49.95` | `51.83%` | `2.59` | `78.87%` | `72.77%` |

Aggregate:

- Server decode speed: `53.17` tok/s mean, `2.06` stdev, `3.87%` CV.
- Client approximate decode speed: `51.16` tok/s mean.
- Draft acceptance: `54.28%` mean.
- Accepted tokens per draft: `2.71` mean.
- First-token conditional acceptance: `80.32%` mean.
- Suffix conditional acceptance: `76.09%` mean.

Warmed repeat from
`single_stream_interactive_262k_window_paper_grounded_20260628_014441_run*.json`:

| run | server tok/s | approximate tok/s | draft acceptance | accepted / draft | pos0 conditional | suffix conditional |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `62.00` | `60.79` | `68.14%` | `3.41` | `84.75%` | `88.74%` |
| 2 | `53.97` | `51.64` | `48.53%` | `2.43` | `69.33%` | `83.68%` |
| 3 | `41.09` | `35.61` | `34.68%` | `1.73` | `51.06%` | `82.50%` |

Aggregate:

- Server decode speed: `52.36` tok/s mean, `10.55` stdev, `20.15%` CV.
- Client approximate decode speed: `49.35` tok/s mean.
- Draft acceptance: `50.45%` mean.
- Accepted tokens per draft: `2.52` mean.
- First-token conditional acceptance: `68.38%` mean.
- Suffix conditional acceptance: `84.97%` mean.

Interpretation:

- The old `86.94` tok/s no-refkv number is an old-parser artifact for
  interactive speed. The preview confirms it captured final answer content and
  missed earlier reasoning text.
- The paper-grounded signal is still useful: suffix conditional acceptance is
  generally healthy when position 0 survives. The repeatability problem is
  dominated by first-token draft quality.
- The next grounded diagnostic should measure position-0 draft/target agreement
  and confidence against the paper's Markov-head assumption before changing
  scheduler policy or draft length.

## Position-0 Quality Diagnostic

Implementation change:

- Added `VLLM_DSPARK_POSITION0_DIAGNOSTICS=1`, default off.
- When enabled, DSpark requests confidence scores without returning full draft
  logits. This keeps the diagnostic cheaper than draft-probability export.
- The runner compares draft token 0 against the target-model argmax for the
  matching verification row and logs aggregated:
  target-argmax match rate, average confidence, average confidence on matches,
  and average confidence on misses.
- This directly targets the paper's first conditional survival variable
  `c_1`: if position 0 is weak or overconfident, scheduler tweaks cannot
  recover large single-stream speedups.

Verification:

- Passed: Ruff on edited vLLM files with external cache after adding the
  diagnostic.
- Passed: `py_compile` on `dspark.py`, `dspark_proposer.py`,
  `gpu_model_runner.py`, `envs.py`, and `test_dspark.py`.
- Passed: focused DSpark vLLM tests in `vllm-dspark-dev:local`:
  `44 passed, 6 skipped`.

Next diagnostic run:

- Rebuild the clean runtime overlay.
- Restart with `VLLM_DSPARK_POSITION0_DIAGNOSTICS=1` for a diagnostic pass.
- Run the corrected single-stream benchmark and inspect logs for position-0
  match/confidence. If misses have high confidence, prioritize reference-parity
  and calibration. If misses have low confidence, prioritize paper scheduler
  integration and GPU-side variable-prefix execution. If match rate is high but
  tok/s remains low, prioritize kernel-native/graph-captured draft execution.

Diagnostic result, 2026-06-28:

- First attempt crashed the engine because the diagnostic path assumed
  confidence values were already probabilities; real decode exposed negative
  logit-like values. The accumulator now normalizes out-of-range diagnostic
  scalars with sigmoid and records `confidence_logits_normalized`; production
  confidence scheduling remains strict.
- Focused validation after the fix: `py_compile`, Ruff, and
  `scripts/run-dspark-tests-in-docker.sh` all passed
  (`45 passed, 6 skipped`).
- Clean runtime overlay was rebuilt on head and worker, restarted with
  `VLLM_DSPARK_POSITION0_DIAGNOSTICS=1`, and `/health` returned 200 on
  `0.0.0.0:8000`.
- Corrected-parser benchmark:

| Run | Server tok/s | Approx tok/s | TTFC s | Draft acc | Accepted/draft | Pos0 acc | Suffix acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `50.26` | `49.28` | `2.77` | `50.14%` | `2.51` | `76.71%` | `79.01%` |
| 2 | `62.95` | `61.96` | `0.44` | `68.62%` | `3.43` | `87.93%` | `87.58%` |
| 3 | `60.41` | `59.23` | `0.44` | `64.26%` | `3.21` | `81.97%` | `86.66%` |

Aggregate:

- Server decode speed: `57.88` tok/s mean, `6.71` stdev, `11.60%` CV.
- Client approximate decode speed: `56.82` tok/s mean.
- Draft acceptance: `61.01%` mean.
- Accepted tokens per draft: `3.05` mean.
- First-token conditional acceptance: `82.20%` mean.
- Suffix conditional acceptance: `84.42%` mean.

Position-0 diagnostic logs:

- Both TP ranks reported the same final aggregate:
  `samples=193`, `target_argmax_match_rate=0.808`, `avg_confidence=0.314`,
  `avg_confidence_matched=0.315`, `avg_confidence_missed=0.313`,
  `confidence_logits_normalized=94`.

Interpretation:

- Position 0 is not catastrophically weak; the target-greedy agreement is
  roughly in line with the benchmark's first-token conditional acceptance.
- The confidence signal is currently not paper-ready: it is too low, nearly
  uncorrelated with match outcome, and about half of observed values arrive as
  logit-like values. Using it for the hardware-aware prefix scheduler would
  likely prune good drafts.
- The next paper-grounded implementation step is confidence-path parity and
  calibration: trace why the runner sometimes observes raw confidence logits,
  ensure all scheduler-facing confidence rows are post-STS/post-sigmoid
  probabilities, and only then enable cumulative-survival scheduling.

## Verification

- Passed: `uv run pytest --hypothesis-show-statistics -q` in
  `experiments/dspark_harness` (`13 passed`, 5 grounded Hypothesis properties
  at 1000 examples each).
- Passed: Ruff on edited vLLM files with external cache:
  `RUFF_CACHE_DIR=/tmp/vllm-dspark-ruff-cache .venv/bin/python -m ruff check ...`
- Passed: focused DSpark vLLM tests in `vllm-dspark-dev:local` with current
  branch modules bind-mounted over the installed package:
  `31 passed, 3 skipped`.
- Passed: focused DSpark vLLM tests after adding fast-confidence and
  hardware-scheduler wiring:
  `39 passed, 6 skipped`.
- Passed: real-model single-stream benchmark repeated 3 times with a possible
  `262144` token context window; recorded significant decode speed increase.
- Passed: clean rebuilt experimental container after warmup expansion; `/health`
  returned 200 on `0.0.0.0:8000`; benchmark repeated 3 times and recorded
  `78.86` server decode tok/s mean.
- Passed: clean rebuilt fast-output container; `/health` returned 200 on
  `0.0.0.0:8000`; benchmark repeated 3 times and recorded `72.68` server
  decode tok/s mean. This is correctness/graph-memory progress, not a speed win.
- Passed: clean rebuilt reference-KV container; `/health` returned 200 on
  `0.0.0.0:8000`; benchmark repeated 3 times and recorded `68.49` server
  decode tok/s mean. This is negative evidence, not a speed win.
- Passed: clean rebuilt no-refkv container; `/health` returned 200 on
  `0.0.0.0:8000`; old-parser benchmark repeated 3 times and recorded `86.94`
  server decode tok/s mean. This is superseded for interactive speed by the
  corrected-parser runs above.
- Passed: corrected stream-parser benchmark repeated 3 times twice. Grounded
  interactive server decode means were `53.17` tok/s and `52.36` tok/s.
- Passed: focused DSpark vLLM tests after adding opt-in position-0 quality
  diagnostics: `44 passed, 6 skipped`.
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
- The DeepSpec paper and released model card both evaluate or recommend
  `temperature=1.0`; the current speed benchmark is greedy
  `temperature=0.0`, which intentionally avoids draft-probability export.
- DSpark draft KV numeric parity does not reproduce the released reference's
  in-place FP8 quant-dequant on the no-RoPE KV slice. This may affect exact
  logit parity. The initial Triton implementation is available behind
  `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT`, but the first real-model A/B
  regressed both speed and acceptance, so it should remain opt-in.
- Confidence scheduling currently receives already-sigmoided probabilities
  from the model wrapper. The paper's STS calibration is a logit-space
  temperature-scaling procedure; if calibration scalars become available, apply
  them before sigmoid.
- Position-0 diagnostics deliberately force the confidence head and CPU scalar
  logging. They are useful for quality checks but must stay disabled for
  performance-gate benchmarks.
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

- Reference-parity probe for DSpark draft KV `act_quant(..., inplace=True)` on
  no-RoPE dimensions, preferably fused into sparse attention or KV projection.
  This is now a draft-quality candidate, not just a numeric cleanup.
- GPU-side first-token quality diagnostics: record draft/target top-1 agreement
  and confidence for position 0 without full-vocabulary softmax in hot runs.
  The CPU/logging diagnostic now exists behind
  `VLLM_DSPARK_POSITION0_DIAGNOSTICS=1`; a lower-overhead Prometheus or
  GPU-resident variant is still a custom-kernel opportunity.
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

## Markov-Fusion A/B, 2026-06-28

Implementation:

- Added `VLLM_DSPARK_FUSED_MARKOV_ARGMAX`, default off.
- The fused path preserves the paper's Markov equation
  `B(x_{k-1}, .) = W1[x_{k-1}] W2`, but computes only local top-1 of
  `base_logits + B` instead of materializing the full Markov-bias vector.
- Added focused unit coverage for the torch reference and draft fast-path
  dispatch. Host pytest still cannot import this checkout without built
  `vllm._C`; the installed-container smoke test passed.

Result from
`single_stream_interactive_262k_window_paper_fused_markov_argmax_code_completion_20260628_045637_run*.json`:

| run | server tok/s | draft acceptance | accepted / draft | pos0 conditional | suffix conditional |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `54.79` | `59.08%` | `2.95` | `83.08%` | `81.05%` |
| 2 | `58.35` | `61.27%` | `3.06` | `90.48%` | `76.39%` |
| 3 | `59.16` | `63.23%` | `3.16` | `93.55%` | `81.08%` |

Aggregate:

- Server decode speed: `57.43` tok/s mean, `1.90` stdev.
- Draft acceptance: `61.19%` mean.
- Accepted tokens per draft: `3.06` mean.
- First-token conditional acceptance: `89.03%` mean.
- Suffix conditional acceptance: `79.51%` mean.

Interpretation:

- This did not beat the best corrected code-completion baseline
  (`58.20` tok/s mean with replicated Markov W1 and local argmax).
- The idea remains paper-grounded, but this first custom kernel likely loses to
  the existing vendor-optimized rank-256 projection despite avoiding a full
  materialized Markov-logit tensor. Keep it flag-gated and off for performance
  benchmarks until the kernel is redesigned.
- Since accepted length is already about `4.06` tokens including the bonus,
  `>100` tok/s requires reducing cycle time materially; perfect acceptance at
  the current roughly `70 ms` cycle would still cap below `90` tok/s.

## B12X Target-Kernel A/B Plan

- Next A/B disables the regressing fused Markov path and enables the fork's
  target-side B12X decode knobs:
  `VLLM_USE_B12X_MHC=1`, `VLLM_USE_B12X_SPARSE_INDEXER=1`, and
  `B12X_W4A16_TC_DECODE=1`.
- This is grounded in the DSpark equation as a `Tverify` attack rather than a
  drafter-quality change. It should preserve DSpark's released V4 setup:
  `gamma = 5`, Markov head, fixed lossless verification, and no confidence
  pruning for the single-stream benchmark.
- Shortcut/risk: older unholy-fusion docs listed B12X MHC and sparse indexer as
  unstable in the historical stable run. This pass is experimental only and
  must be reverted or kept opt-in unless repeated real-model benchmarks improve.

Result from
`single_stream_interactive_262k_window_paper_b12x_mhc_sparse_code_completion_20260628_051009_run*.json`:

| run | server tok/s | draft acceptance | accepted / draft | pos0 conditional | suffix conditional |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `49.23` | `59.69%` | `2.98` | `86.15%` | `80.92%` |
| 2 | `49.96` | `57.88%` | `2.89` | `84.85%` | `80.19%` |
| 3 | `56.32` | `69.31%` | `3.47` | `96.55%` | `83.39%` |

Aggregate:

- Server decode speed: `51.84` tok/s mean, `3.18` stdev.
- Draft acceptance: `62.29%` mean.
- Accepted tokens per draft: `3.11` mean.
- First-token conditional acceptance: `89.18%` mean.
- Suffix conditional acceptance: `81.50%` mean.

Interpretation:

- B12X MHC/sparse-indexer did not improve this DSpark single-stream path; it
  regressed versus the best corrected code-completion baseline (`58.20` tok/s)
  and the fused-Markov A/B (`57.43` tok/s).
- Acceptance stayed healthy, so the regression is a cycle-time/kernel choice
  problem rather than a DSpark draft-quality issue.
- Decision: revert these env flags for the working DSpark experiment. Keep the
  result as negative evidence and do not spend more time on broad target-kernel
  toggles without stage timing.

## Stage Timing Diagnostic, 2026-06-28

Implementation:

- Added `VLLM_DSPARK_STAGE_TIMING`, off by default, plus
  `VLLM_DSPARK_STAGE_TIMING_LOG_EVERY`.
- The diagnostic records CUDA events for `context_prepare`, `prefill_main`,
  `graph_prepare`, `draft`, `postprocess`, and wall `total`.
- This path synchronizes CUDA, so benchmark tok/s from timing-enabled runs is
  not valid for the speed gate. Use it only to locate the latency.
- Clean overlay images were rebuilt on both nodes before the run.

Run:
`single_stream_interactive_262k_window_paper_stage_timing_code_completion_20260628_052148_run1.json`
with `MAX_TOKENS=128` and timing enabled.

Observed request metrics:

- Server decode speed: `47.99` tok/s. Diagnostic only, not a gate result.
- Draft acceptance: `50.56%`.
- Accepted tokens per draft: `2.53`.
- First-token conditional acceptance: `83.33%`.
- Suffix conditional acceptance: `76.14%`.
- Prompt tokens: `411`; output tokens: `128`.

Stage averages from logs:

- Head, over `35` proposals: `context_prepare=0.034ms`,
  `prefill_main=1.318ms`, `graph_prepare=0.050ms`, `draft=7.531ms`,
  `postprocess=0.016ms`, `total=67.840ms`.
- Worker, over `35` proposals: `context_prepare=0.034ms`,
  `prefill_main=1.138ms`, `graph_prepare=0.049ms`, `draft=7.558ms`,
  `postprocess=0.016ms`, `total=68.734ms`.

Interpretation:

- The fixed DSpark draft block is not the largest remaining cost: the measured
  hot draft stage is about `7.5ms`, with target-layer prefill around
  `1.1ms` to `1.3ms`.
- Proposer wall total is about `68ms`. The first CUDA sync in the proposer is
  effectively waiting on the previous target verification step, so `Tverify`
  is now the dominant term in the paper equation
  `L = (Tdraft + Tverify) / tau`.
- Accepted length is healthy enough that `>100` tok/s requires a material
  cycle-time reduction, not only another small Markov-head optimization.

Next target:

- Test explicit target attention/backend choice
  `VLLM_ATTENTION_BACKEND=B12X_MLA_SPARSE` as a direct `Tverify` attack.
- If that does not improve repeated speed, inspect the target-layer feature
  capture path around DSpark layers `[40, 41, 42]`, because the implementation
  may be forcing extra hidden-state materialization around verification.

## Next Step

Use the DSpark paper equation `L = (Tdraft + Tverify) / tau` as the next gate.
The clone-fixed diagnostic shows `tau` and first-token quality are not the
current bottleneck, and the stage timing shows the DSpark draft block is already
small relative to target verification wait time. The next pass should reduce
`Tverify` while preserving the paper's released V4 setup: `gamma = 5`, Markov
head, and lossless verification. Candidate work should prioritize target-side
attention/backend selection, kernel-native verification hot paths, and lighter
target-layer feature capture for DSpark. Keep only changes that improve
repeated real-model server decode mean toward the `>100` tok/s target.

## B12X Compressed-MLA Decode A/B, 2026-06-28

Implementation:

- Added an opt-in source-level DeepSeek V4 SM120 decode path behind
  `VLLM_DSV4_B12X_COMPRESSED_MLA=1`.
- The path calls `b12x.attention.mla.compressed_api.compressed_mla_decode_forward`
  from the model's own SM120 decode implementation instead of relying on the
  generic attention-backend selector, which DeepSeek V4 bypasses.
- The path graph-captured successfully after adapting singleton-rank index
  tensors for b12x workspace policy checks.

Result from
`single_stream_interactive_262k_window_paper_b12x_compressed_mla_code_completion_20260628_055843_run*.json`:

| run | server tok/s | draft acceptance | accepted / draft | pos0 conditional | suffix conditional |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `2.17` | `0.82%` | `0.04` | `4.08%` | `0.00%` |
| 2 | `2.25` | `1.70%` | `0.09` | `8.51%` | `0.00%` |
| 3 | `2.11` | `0.40%` | `0.02` | `1.99%` | `0.00%` |

Aggregate:

- Server decode speed: `2.18` tok/s mean, `0.07` stdev.
- Draft acceptance: `0.97%` mean.
- Accepted tokens per draft: `0.05` mean.
- First-token conditional acceptance: `4.86%` mean.
- Suffix conditional acceptance: `0.00%` mean.

Interpretation:

- This is not a speed/backend regression alone; target verification outputs are
  no longer parity-compatible with the DSpark draft distribution, so standard
  verification rejects almost every draft token.
- The current b12x compressed-MLA path must stay off by default. It is useful
  only as a future verifier-parity debugging target with direct output
  comparison against the existing FlashInfer SM120 wrapper.
- This is the clearest evidence so far that target-side kernel work cannot
  substitute an implementation unless it preserves verification numerics.
Further speed work should first follow the paper path: keep the released
DSpark Markov draft distribution intact, profile target step capacity, and
use confidence-scheduled verification or target-feature capture reductions
that retain verifier parity.

## Deferred Target-Layer Capture, 2026-06-28

Hypothesis:

- The paper's DSpark latency equation is `L = (Tdraft + Tverify) / tau`.
- Stage timing showed `Tverify` dominates the remaining single-stream cycle
  time, while draft quality is already healthy.
- DeepSeek V4 target verification captures DSpark hidden features from target
  layers `[40, 41, 42]`. The previous implementation called `hc_post(...)`
  immediately after each captured layer, breaking the fused `hc_post_pre`
  chain at the consecutive target layers.
- Deferring capture for layers that have a following decoder layer should
  preserve verifier hidden states while reducing target-side materialization
  overhead.

Implementation:

- Added `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE`, enabled in the experiment
  env and passed through Docker compose.
- For target layers with a next decoder layer, the next layer copies the
  previous layer output from the already-materialized fused `hc_post_pre`
  residual into the DSpark hidden buffer.
- The released config has `dspark_target_layer_ids=[40, 41, 42]` and
  `num_hidden_layers=43`, so layers `40` and `41` are deferred. Layer `42` is
  the final target layer and keeps the immediate capture fallback.
- The failed `VLLM_DSV4_B12X_COMPRESSED_MLA` path remains disabled.

Validation so far:

- Python compile and Ruff passed for the edited vLLM files.
- Clean overlay image rebuild completed on both Spark nodes.
- Fresh server startup logs on both ranks show:
  `DeepSeek V4 DSpark deferred target-layer capture enabled for layers (40, 41).`

Next step:

- When the clean server finishes loading and graph capture, run the smoke
  request, then three repeated corrected single-stream code-completion
  benchmarks.
- Keep the `>100` tok/s gate. If this change only gives a small improvement,
  the next paper-aligned target remains calibrated confidence scheduling or a
  verifier-parity-preserving target kernel path.

Result from
`single_stream_interactive_262k_window_paper_deferred_capture_code_completion_20260628_062246_run*.json`:

| run | server tok/s | draft acceptance | accepted / draft | pos0 conditional | suffix conditional |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `45.66` | `40.71%` | `2.04` | `78.57%` | `65.55%` |
| 2 | `43.07` | `37.11%` | `1.86` | `80.00%` | `59.56%` |
| 3 | `44.06` | `38.85%` | `1.94` | `78.16%` | `57.74%` |

Aggregate:

- Server decode speed: `44.26` tok/s mean, `1.30` stdev.
- Draft acceptance: `38.89%` mean.
- Accepted tokens per draft: `1.94` mean.
- First-token conditional acceptance: `78.91%` mean.
- Suffix conditional acceptance: `60.95%` mean.
- Delta versus corrected baseline `58.20` tok/s: `-23.95%`.

Interpretation:

- The deferred-capture path is not parity-preserving. It starts and graph
  captures, but the hidden features consumed by the DSpark draft model differ
  enough to cut acceptance from about `62%` to about `39%`.
- This confirms the same lesson as the failed b12x compressed-MLA path: target
  hot-path shortcuts must preserve verifier/draft feature semantics exactly.
- Decision: do not keep this path enabled for the working DSpark experiment.
  Next work should either fix the capture equivalence with a direct tensor
  parity check or return to the paper's confidence-scheduled verification,
  which changes verification length without perturbing target hidden states.

## Clean Paper-Mode Baseline And B12X WO Projection A/B, 2026-06-28

Corrected clean baseline:

- Result set:
  `single_stream_interactive_262k_window_paper_forceknob_nothink_code_completion_20260628_064425_run*.json`.
- Workload: `SCENARIO=code_completion`, `PROMPT_TOKENS=512`,
  `MAX_TOKENS=256`, `THINKING=false`, `STABLE_PROMPT=1`.
- Server-side decode speed: `59.15` tok/s mean, `4.84` stdev.
- Draft acceptance: `63.48%` mean.
- Accepted tokens per draft: `3.17` mean.
- First-token conditional acceptance: `90.58%` mean.
- Suffix conditional acceptance: `80.35%` mean.

Candidate:

- Tested `VLLM_USE_B12X_WO_PROJECTION=1` as an isolated verifier-side
  output-projection optimization. No code change was required for this A/B.
- Result set:
  `single_stream_interactive_262k_window_paper_b12x_wo_code_completion_20260628_065704_run*.json`.

| run | server tok/s | draft acceptance | accepted / draft | pos0 conditional | suffix conditional |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `60.35` | `66.33%` | `3.32` | `85.00%` | `84.92%` |
| 2 | `61.41` | `67.80%` | `3.39` | `84.75%` | `84.67%` |
| 3 | `60.14` | `64.92%` | `3.25` | `91.80%` | `78.63%` |

Aggregate:

- Server decode speed: `60.64` tok/s mean, `0.68` stdev.
- Draft acceptance: `66.35%` mean.
- Accepted tokens per draft: `3.32` mean.
- First-token conditional acceptance: `87.18%` mean.
- Suffix conditional acceptance: `82.74%` mean.
- Delta versus corrected clean baseline: `+2.5%` server-side decode speed.

Interpretation:

- The path is parity-compatible enough for the benchmark: acceptance is close
  to the corrected clean baseline and suffix quality did not collapse.
- The speed gain is real but too small for the `>100` tok/s gate. This suggests
  WO projection is not the dominant verifier bottleneck on this workload.
- The JIT monitor reported late compilation of `_pack_topk_routes_*` and
  `eagle_prepare_next_token_padded_kernel` during the first two benchmark
  repeats. Warmup coverage for route packing and next-token preparation should
  be added before judging very small deltas.
- Keep this path as an optional optimization candidate. It does not change the
  next major direction: reduce verifier cycle time with a larger hot-path win,
  or implement confidence-scheduled verification exactly as described in the
  paper once calibrated confidence outputs are available.

## Forced Verification-Length Curve And Exact Capture A/B, 2026-06-28

Paper grounding:

- DSpark's latency target is `L = (Tdraft + Tverify) / tau`.
- The paper's confidence scheduler is a hardware-aware throughput allocator.
  It helps most when target verification capacity is saturated; the paper also
  states that verifying extra tokens has minimal opportunity cost under light
  system load.
- Our benchmark is deliberately single-stream. The useful test is therefore
  whether shorter verification prefixes reduce the per-cycle time enough to
  offset fewer accepted tokens.

Forced-prefix curve:

- Runtime: clean rebuilt `vllm-dspark-runtime:clean`, B12X WO projection on,
  confidence scheduler off, code-completion workload, three non-warmup repeats
  per forced length.
- Result prefix:
  `single_stream_interactive_262k_window_paper_sps_curve_b12xwo_20260628_070550_len*`.

| forced length | server tok/s mean | stdev | accepted / draft | cycle ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `9.70` | `0.25` | `3.33` | `445.33` |
| 1 | `19.16` | `0.86` | `0.95` | `101.97` |
| 2 | `28.93` | `0.20` | `1.80` | `96.91` |
| 3 | `36.39` | `0.55` | `2.39` | `93.23` |
| 4 | `41.46` | `1.31` | `2.97` | `95.79` |
| 5 | `62.33` | `2.22` | `3.42` | `70.94` |

Interpretation:

- Forced pruning is not a single-stream speed win on this hardware. Full
  DSpark-5 has both the best throughput and the shortest observed cycle.
- The corrected `>100` tok/s gate cannot be reached by acceptance alone at the
  current cycle time. With `gamma=5`, perfect acceptance emits at most six
  tokens per cycle; at `70.94 ms`, that ceiling is about `84.6 tok/s`.
- Therefore the next high-gain path must reduce verifier-cycle time materially
  or safely increase effective proposal length beyond the released DSpark block.

Exact deferred target-layer capture:

- Implementation adds
  `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT=1`. It preserves the fused
  `hc_post_pre` chain while separately computing the exact immediate
  `hc_post(...)` value for the DSpark hidden-state buffer.
- Both ranks were verified to log:
  `DeepSeek V4 DSpark deferred target-layer capture enabled for layers (40, 41) with exact capture.`
- Result set:
  `single_stream_interactive_262k_window_paper_deferred_exact_b12xwo_code_completion_20260628_075309_run*.json`.

| run | server tok/s | draft acceptance | accepted / draft | pos0 conditional | suffix conditional |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `59.15` | `63.55%` | `3.18` | `91.94%` | `78.26%` |
| 2 | `60.03` | `64.92%` | `3.25` | `90.16%` | `80.71%` |
| 3 | `60.52` | `66.00%` | `3.30` | `93.33%` | `80.19%` |

Aggregate:

- Server decode speed: `59.90` tok/s mean, `0.69` stdev.
- Draft acceptance: `64.82%` mean.
- Accepted tokens per draft: `3.24` mean.
- Cycle time estimate: `70.80 ms` mean.
- Delta versus B12X WO projection baseline `60.64` tok/s: `-1.2%`.

Decision:

- Exact capture is parity-compatible but not faster. It is useful as a
  reference for correctness, but should remain opt-in and disabled by default.
- The next DSpark speed milestone should target actual verifier hot-path
  kernels: avoid duplicate hidden materialization/copy for DSpark target
  features, warm or fuse `eagle_prepare_next_token_padded_kernel`, and profile
  sparse MLA/MoE decode shapes under the exact single-stream DSpark cycle.

## Direct DSpark HC-Post Mean Kernel, 2026-06-28

Implementation:

- Added `dspark_hc_post_mean(...)`, a Triton helper for exact deferred capture.
- It computes `MHCPostOp(x, residual, post, comb).mean(dim=1)` directly into
  the DSpark hidden buffer slice, instead of materializing the full
  `[tokens, hc_mult, hidden]` tensor and reducing it.
- Formula used:
  `mean_j(out_j) = mean_j(post_j) * x + sum_i(mean_j(comb_ij) * residual_i)`.
- CPU and CUDA reference checks passed against `mhc_post_torch(...).mean(dim=1)`
  with bf16 tolerance (`max_abs=0.015625` on CUDA).

Result set:
`single_stream_interactive_262k_window_paper_deferred_mean_kernel_b12xwo_code_completion_20260628_080616_run*.json`.

| run | server tok/s | draft acceptance | accepted / draft | cycle ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | `58.24` | `62.22%` | `3.11` | `70.59` |
| 2 | `60.21` | `64.92%` | `3.25` | `70.52` |
| 3 | `58.74` | `62.22%` | `3.11` | `69.99` |

Aggregate:

- Server decode speed: `59.06` tok/s mean, `1.02` stdev.
- Cycle time estimate: `70.37 ms` mean.
- Late JIT warnings appeared for route-pack and
  `eagle_prepare_next_token_padded_kernel`, so a second warmed pass was run.

Warmed result set:
`single_stream_interactive_262k_window_paper_deferred_mean_kernel_b12xwo_warm_code_completion_20260628_080704_run*.json`.

- Server decode speed: `60.42` tok/s mean, `1.25` stdev.
- Draft acceptance: `64.49%` mean.
- Accepted tokens per draft: `3.22` mean.
- Cycle time estimate: `69.93 ms` mean.

Interpretation:

- The kernel is correctness-preserving and shaves roughly `0.9 ms` from the
  exact-capture cycle estimate, but the gain is far below the `>100` tok/s
  requirement and does not beat the best B12X WO projection baseline.
- Keep the kernel as a useful opt-in/custom-kernel building block. It confirms
  that DSpark feature capture is not the dominant verifier bottleneck.
- Next A/B should move to target sparse-MLA/backend selection or MoE decode
  kernels, where a larger fraction of `Tverify` lives.

## Corrected Goal Gate And Verifier Timing, 2026-06-28

Paper grounding:

- The DSpark paper defines per-token latency as
  `L = (Tdraft + Tverify) / tau`.
- With the released DeepSeek V4 Flash DSpark config, `gamma=5`; perfect
  full-block acceptance emits at most six tokens per verification cycle.
- At the current best observed cycle time around `71 ms`, even perfect
  acceptance would cap throughput around `85 tok/s`. The corrected `>100 tok/s`
  gate therefore requires materially reducing cycle time, not just nudging
  acceptance quality.

Diagnostic iteration timing:

- Added opt-in `VLLM_DSPARK_ITER_TIMING=1`, implemented with CUDA
  synchronization and intended only for profiling.
- Warmed diagnostic result:
  `single_stream_interactive_262k_window_paper_iter_timing_b12xwo_warmed_code_completion_20260628_083444_run1.json`.
- Server decode speed: `61.31 tok/s`.
- Draft acceptance: `68.03%`.
- Accepted tokens per draft: `3.40`.

Approximate warmed per-iteration timing, from late cumulative log deltas:

| stage | time |
| --- | ---: |
| target forward | `~61.0 ms` |
| draft propose | `~8.6 ms` |
| target postprocess logits | `~2.5 ms` |
| total iteration | `~73.9 ms` |

Interpretation:

- The verifier target forward is the dominant wall-time term. Proposer-only
  changes cannot close the gap to `>100 tok/s`.
- The next grounded step is finer-grained target-forward profiling so kernel
  work targets the largest verifier components rather than broad toggles.
- Late first-request JIT remains visible for `_pack_topk_routes_*` and
  `eagle_prepare_next_token_padded_kernel`. Warmup coverage is a useful TODO,
  but it is not enough by itself for the corrected throughput gate.

## B12X W4A16 TC Decode A/B, 2026-06-28

Candidate:

- Tested `B12X_W4A16_TC_DECODE=1` as an isolated target-kernel toggle, while
  leaving DSpark confidence scheduling off and B12X WO projection on.
- Result set:
  `single_stream_interactive_262k_window_paper_b12x_w4a16_tc_decode_b12xwo_code_completion_20260628_084507_run*.json`.

| run | server tok/s | draft acceptance | accepted / draft | cycle ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | `53.00` | `63.55%` | `3.18` | `78.82` |
| 2 | `56.96` | `67.93%` | `3.40` | `77.18` |
| 3 | `54.89` | `65.33%` | `3.27` | `77.73` |

Aggregate:

- Server decode speed: `54.95 tok/s` mean, `1.98` stdev.
- Draft acceptance: `65.60%` mean.
- Accepted tokens per draft: `3.28` mean.
- First-token conditional acceptance: `89.98%` mean.
- Suffix conditional acceptance: `81.35%` mean.
- Cycle time estimate: `77.91 ms` mean.

Decision:

- Negative result versus the B12X WO baseline `60.64 tok/s`. Acceptance stayed
  similar, but the verification cycle slowed down.
- Keep `B12X_W4A16_TC_DECODE=0`.
- Keep `VLLM_USE_B12X_WO_PROJECTION=1` as the best known corrected baseline
  while pursuing target-forward kernel profiling.

## Torch Profile And Direction Review, 2026-06-28

Torch profiler run:

- Result set:
  `single_stream_interactive_262k_window_paper_torch_profile_b12xwo_code_completion_20260628_090846_run1.json`.
- Profiler overhead slowed the request to `42.31 tok/s`, so this run is
  diagnostic only and should not be compared as a speed result.
- Saved profiler summaries:
  `experiments/dspark-benchmarks/profiles/paper_torch_profile_b12xwo_code_completion_20260628_090846/head_profiler_out_0.txt`
  and
  `experiments/dspark-benchmarks/profiles/paper_torch_profile_b12xwo_code_completion_20260628_090846/worker_profiler_out_0.txt`.

Main profiler signal:

- The largest raw CUDA bucket is the B12X fused MoE W4A16 kernel family
  (`~275 ms`, `413` calls in the profiled request).
- Dense FP8/FP4 GEMM buckets and NCCL all-reduce are also visible, but much
  smaller than the overall target-forward scope.
- `gpu_model_runner: draft` and postprocess scopes are second-order relative
  to target verification.
- Late JIT warnings remain visible for `_pack_topk_routes_*` and
  `eagle_prepare_next_token_padded_kernel`, but they are not large enough to
  explain the corrected `>100 tok/s` gap.

B12X FP8 GEMM A/B:

- Tried `VLLM_USE_B12X_FP8_GEMM=1` after the profiler showed dense GEMM time.
- Startup failed during DSpark dummy draft execution inside
  `DeepSeekV4DSparkAttention.forward_dspark`, at
  `torch.ops.vllm.deepseek_v4_fp8_einsum(...)`.
- Root error:
  `DeepGEMM ... layout.hpp:39): t.dim() == N`.
- A follow-up patch marked DSpark `wo_a` to skip generic block-FP8 packing, but
  the retry failed with the same assertion.
- Decision: generic B12X FP8 GEMM is currently incompatible with DSpark draft
  attention's custom `wo_a` FP8 einsum layout. Keep
  `VLLM_USE_B12X_FP8_GEMM=0` and do not continue this A/B without a focused
  layout/parity fix.

Direction review:

- Read `experiments/dspark-benchmarks/dspark_direction_review.md`.
- The key correction is strategic: DSpark's paper result is a per-user
  throughput gain against the **MTP-1 baseline** under live concurrency, not a
  single-stream absolute `>100 tok/s` target.
- The current single-stream DSpark-variant baseline is self-referential. We
  have not yet measured the paper's MTP-1 reference point even though the repo
  has a validated preset in
  `docs/deepseek-v4-mtp1-fullgraph-validated-preset.md`.
- The single-stream confidence scheduler is structurally idle: with one active
  request, the best observed policy is to verify the full DSpark block, which
  the forced-length curve already showed.
- Therefore the next grounded milestone should be:
  1. measure MTP-1 single-stream using the same corrected benchmark parser,
  2. add in-harness warmup/discard and confidence intervals,
  3. run a DSpark-5 vs MTP-1 concurrency sweep,
  4. only return to custom kernels against the concurrent per-user throughput
     metric.

Status:

- We have not proven the limit of DSpark itself.
- We have hit the limit of the current single-stream metric: at `gamma=5`,
  `~71 ms` verification cycles cap ideal throughput around `85 tok/s` even with
  perfect acceptance.
- Further single-stream proposer knobs are unlikely to be high-leverage until
  target-forward parity-preserving kernels change materially.

## Revised Single-Stream Direction And MTP-1 Anchor, 2026-06-28

Revised direction:

- Read the revised `experiments/dspark-benchmarks/dspark_direction_review.md`.
- The corrected target workload is **single-stream coding latency**. Do not
  redirect the main goal to concurrency benchmarks.
- The gate is now speedup over MTP-1/no-spec on a warmed single-stream coding
  workload, with `>100 tok/s` kept as a stretch target.
- The next high-leverage implementation levers are:
  1. async two-step draft/verify pipelining,
  2. whole-cycle CUDA graph capture,
  3. tau stability/reference parity.
- Stop spending mainline time on the single-stream load-aware confidence
  scheduler, `SPS(B)`, and more proposer/kernel A/Bs unless they support those
  levers.

Harness updates:

- `scripts/dspark_single_stream_benchmark.py` now supports
  `--warmup-requests`; discarded warmup requests run before `metrics_before`.
- `scripts/summarize_dspark_benchmarks.py` now reports CV and a 95% confidence
  half-width, and handles non-DSpark rows with missing suffix-position metrics.

MTP-1 single-stream coding baseline:

- Server: unholy-fusion MTP-1, `deepseek-v4-flash`, `MAX_MODEL_LEN=262144`,
  `MTP_NUM_TOKENS=1`.
- Scenario: `code_completion`, `PROMPT_TOKENS=512`, `MAX_TOKENS=256`,
  `THINKING=false`, `WARMUP_REQUESTS=1`.
- Result set:
  `single_stream_interactive_262k_window_mtp1_paper_code_completion_20260628_101450_run*.json`.

| run | server tok/s | MTP acceptance | accepted / draft | cycle ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | `39.64` | `93.94%` | `0.94` | `48.93` |
| 2 | `40.34` | `96.15%` | `0.96` | `48.62` |
| 3 | `39.66` | `94.66%` | `0.95` | `49.08` |

Aggregate:

- Server decode speed: `39.88 tok/s` mean, `0.40` stdev, `1.0%` CV,
  `95% CI ±1.00 tok/s`.
- MTP accepted tokens per draft: `0.949` mean.
- Cycle time estimate: `48.88 ms` mean.

Interpretation:

- Against the best corrected DSpark B12X-WO baseline (`60.64 tok/s`), DSpark is
  already about `1.52x` MTP-1 (`+52%`) on this warmed coding workload.
- This confirms the revised review's baseline concern: DSpark was being judged
  against internal variants instead of the actual MTP-1 alternative.
- The remaining single-stream speed work should target cycle reduction
  (`Tdraft + Tverify`) via pipelining and graph capture, then tau stability.

No-spec single-stream coding baseline:

- Added `entrypoints/entrypoint.unholy-nospec.sh`, an experimental wrapper that
  reuses the unholy-fusion entrypoint but removes only the two
  `--speculative-config` lines at container startup.
- Confirmed startup logs show `speculative_config=None` and base
  `DeepseekV4ForCausalLM`.
- Scenario: same as MTP-1 (`code_completion`, `PROMPT_TOKENS=512`,
  `MAX_TOKENS=256`, `THINKING=false`, `WARMUP_REQUESTS=1`).
- Result set:
  `single_stream_interactive_262k_window_nospec_paper_code_completion_20260628_102239_run*.json`.

| run | server tok/s |
| --- | ---: |
| 1 | `26.34` |
| 2 | `26.32` |
| 3 | `26.33` |

Aggregate:

- Server decode speed: `26.33 tok/s` mean, `0.009` stdev, `0.03%` CV,
  `95% CI ±0.023 tok/s`.

Speedup anchors using best corrected DSpark B12X-WO baseline (`60.64 tok/s`):

- DSpark vs MTP-1: `1.52x` (`+52%`).
- DSpark vs no-spec: `2.30x` (`+130%`).

Next measurement:

- Rerun DSpark B12X-WO with the same in-harness warmup parser to get an
  apples-to-apples DSpark anchor before implementing async draft/verify
  pipelining.

Warmed DSpark apples-to-apples anchor:

- Removed the experimental DSpark `wo_a.b12x_skip_generic_block_fp8_linear`
  line from the overlay image. That line was from the failed generic B12X FP8
  GEMM A/B and broke baseline startup by preventing the DSpark custom
  `deepseek_v4_fp8_einsum` path from seeing the expected scale layout.
- Rebuilt `vllm-dspark-runtime:clean` on both nodes and verified the installed
  `dspark.py` no longer contains the bad line.
- Server started successfully with `VLLM_USE_B12X_FP8_GEMM=0`.
- Scenario: same as MTP-1/no-spec (`code_completion`, `PROMPT_TOKENS=512`,
  `MAX_TOKENS=256`, `THINKING=false`, `WARMUP_REQUESTS=1`).
- Result set:
  `single_stream_interactive_262k_window_dspark_b12xwo_warm_paper_code_completion_20260628_103658_run*.json`.

| run | server tok/s | draft acceptance | accepted / draft | cycle ms | pos0 | suffix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `63.87` | `69.82%` | `3.49` | `70.32` | `91.23%` | `82.82%` |
| 2 | `65.19` | `72.14%` | `3.61` | `70.67` | `94.64%` | `85.35%` |
| 3 | `61.18` | `65.00%` | `3.25` | `69.47` | `93.33%` | `81.06%` |

Aggregate:

- Server decode speed: `63.41 tok/s` mean, `2.04` stdev, `3.2%` CV,
  `95% CI ±5.08 tok/s`.
- Draft acceptance: `68.99%` mean.
- Accepted tokens per draft: `3.45` mean.
- Cycle time estimate: `70.15 ms` mean.

Grounded speedup:

- DSpark vs MTP-1 (`39.88 tok/s`): `1.59x` (`+59%`).
- DSpark vs no-spec (`26.33 tok/s`): `2.41x` (`+141%`).

Implication:

- DSpark is already close to the paper's lower bound versus MTP-1 in the
  single-stream coding regime, and far above no-spec.
- The throughput ceiling is now plainly cycle-time limited: `~70 ms` cycle with
  `gamma=5`.
- Next implementation target is paper-grounded single-stream cycle reduction:
  preserve full-block verification under light load, keep tau/reference parity
  intact, and remove execution overhead through graph capture and kernel-native
  DSpark draft stages. Treat draft/verify overlap as an experimental systems
  optimization, not as paper Section 5.2 itself.

## Paper Refresh Correction, 2026-06-28

I re-read `DSpark_paper.txt` after the revised direction note. The single-stream
metric correction is right, but one mechanism label needs correction:

- Paper Section 5.2 is the **asynchronous hardware-aware prefix scheduler** for
  production ZOS/CUDA-graph compatibility. It uses confidence outputs from two
  steps prior to choose a capacity limit, while current candidates are still
  ranked by their actual up-to-date cumulative confidence scores.
- Section 5.2 does **not** describe a draft/verify overlap pipeline. For our
  single-stream workload, that scheduler is mostly inert because the paper also
  states extra verification tokens have minimal opportunity cost under light
  load.
- Therefore the aligned single-stream goal is:
  1. keep full `gamma=5` verification unless confidence/tau data proves a
     quality issue,
  2. optimize `L = (Tdraft + Tverify) / tau` by reducing launch/Python/kernel
     overhead with graph capture and kernel-native DSpark draft stages,
  3. stabilize tau through reference parity, especially Markov/non-greedy
     sampling and KV/hidden-state parity,
  4. only pursue draft/verify overlap behind an experimental flag after proving
     that the required data dependencies are available early enough to reuse the
     optimistic draft without changing target-distribution correctness.

Corrected active gate:

- Primary: repeated warmed single-stream coding decode speedup versus MTP-1
  (`39.88 tok/s`) with confidence interval.
- Current DSpark anchor: `63.41 tok/s`, `1.59x` MTP-1, `2.41x` no-spec.
- Stretch: `>100 tok/s`, but only by stacking higher tau with lower cycle time;
  do not treat scheduler work or proposer-only A/Bs as the main path.

## DeepGEMM MegaMoE Backend A/B, 2026-06-28

Hypothesis:

- The torch profile points at target-side B12X fused MoE W4A16 kernels as the
  largest verifier-cycle bottleneck, so a faster verifier MoE backend would be a
  high-leverage `Tverify` win if it preserved verifier numerics.

Runtime A/B:

- Launched the DSpark server with transient
  `VLLM_EXTRA_ARGS='--moe-backend deep_gemm_mega_moe --enable-expert-parallel'`
  against `.env.dspark-experiment`.
- Startup accepted both flags and assigned tensor/expert-parallel ranks
  (`TP0_EP0`, `TP1_EP1`).
- Startup then failed during `finalize_mega_moe_weights` after shard loading
  with:
  `NotImplementedError: DeepGEMM MegaMoE requires SM100 GPUs.`

Decision:

- `deep_gemm_mega_moe` is not a viable runtime flag on the DGX Spark / SM121
  path as currently implemented.
- Do not keep cycling on this backend unless we explicitly take on source work
  to make the MegaMoE path support SM121 and prove output parity.
- The aligned next path remains parity-preserving `Tverify` work on the current
  supported B12X verifier path: graph capture, decode-shape autotune, and fused
  DSpark feature capture.

## B12X MoE Scratch Plan Cache, 2026-06-28

Hypothesis:

- The torch profile identified B12X fused MoE as the largest verifier-side CUDA
  bucket. The B12X backend is the only currently supported SM120 W4A16/parity
  path for this checkpoint; FlashInfer BF16 MXFP4 is not available for SM120 and
  MXFP8 activation paths would change verifier numerics.
- The B12X `apply(...)` path recomputed `plan_tp_moe_scratch(...)` for identical
  static decode shapes, and `workspace_shapes(...)` can ask for the same plan
  multiple times before the kernel launch. Caching identical plan objects is
  parity-preserving because it does not change inputs, weights, routing, scratch
  contents, or the B12X kernel call.

Implementation:

- Added a per-`B12xExperts` `_fp4_moe_plan_cache` keyed by token count, expert
  shape, K/N/top-k, device, dtype, activation, quant mode, source format,
  W13 layout, router-weight mode, and SwiGLU limit.
- Routed both `workspace_shapes(...)` and `apply(...)` through the cache.
- Updated `docker/Dockerfile.dspark-runtime-overlay` so
  `vllm/model_executor/layers/fused_moe/b12x_moe.py` is copied into
  `vllm-dspark-runtime:clean` and py-compiled during image build.
- Added `tests/kernels/moe/test_b12x_moe.py` to verify identical-key reuse with
  a monkeypatched planner.

Validation:

- `RUFF_CACHE_DIR=/tmp/ruff-cache-vllm uv run --no-sync ruff check
  vllm/model_executor/layers/fused_moe/b12x_moe.py
  tests/kernels/moe/test_b12x_moe.py` passed.
- `PYTHONPYCACHEPREFIX=/tmp/vllm-pycache python3 -m py_compile ...` passed.
- Full pytest on the host is blocked by the local checkout lacking the compiled
  `vllm._C` extension; run the focused test inside a built vLLM image or a
  built host environment before upstreaming.
- Rebuilt `vllm-dspark-runtime:clean` on both nodes and verified the installed
  image file contains `_get_or_plan_fp4_moe_scratch`.

Benchmark plan:

- Restart the DSpark server on the rebuilt image.
- Run the same warmed single-stream `code_completion` benchmark three times and
  compare against the current DSpark anchor (`63.41 tok/s`, `1.59x` MTP-1).

Benchmark result:

- Rebuilt and restarted `vllm-dspark-runtime:clean` on both nodes.
- Confirmed startup selected the B12X MXFP4 backend and served
  `deepseek-v4-flash-dspark` with `max_model_len=262144`.
- Scenario: same warmed single-stream coding gate
  (`code_completion`, `PROMPT_TOKENS=512`, `MAX_TOKENS=256`,
  `THINKING=false`, `WARMUP_REQUESTS=1`).
- Result set:
  `single_stream_interactive_262k_window_dspark_b12x_plan_cache_paper_code_completion_20260628_110648_run*.json`.

| run | server tok/s | draft acceptance | accepted / draft | cycle ms | pos0 | suffix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `57.88` | `61.59%` | `3.08` | `70.48` | `93.65%` | `75.66%` |
| 2 | `58.46` | `63.55%` | `3.18` | `71.46` | `91.94%` | `79.12%` |
| 3 | `56.96` | `60.63%` | `3.03` | `70.77` | `85.94%` | `77.38%` |

Aggregate:

- Server decode speed: `57.77 tok/s` mean, `0.76` stdev, `1.3%` CV,
  `95% CI ±1.88 tok/s`.
- Draft acceptance: `61.92%` mean.
- Accepted tokens per draft: `3.10` mean.
- Cycle time estimate: `70.91 ms` mean.

Decision:

- This is a negative hot-decode result versus the warmed DSpark anchor
  (`63.41 tok/s`), despite low variance.
- The plan cache is likely outside the steady-state bottleneck because the hot
  decode loop is graph-replayed/kernel-bound, not Python planner-bound.
- Removed the plan-cache patch from the default runtime overlay after re-reading
  the direction review: negative or non-improving kernel/proposer experiments
  should not be baked into the clean path.
- Next aligned Lever 1 target: remove the remaining inference-time JIT misses
  logged after startup (`_pack_topk_routes_small_prefix_kernel`,
  `_pack_topk_routes_prefix_kernel`, `_pack_topk_routes_post_prefix_kernel`,
  `eagle_prepare_next_token_padded_kernel`) and then re-run the warmed gate.

## B12X Route-Pack Exact-Workspace Warmup, 2026-06-28

Alignment:

- The updated direction review keeps the primary path on parity-preserving
  `Tverify` reduction and explicitly names the JIT-escaping route-pack kernels
  as part of the whole-cycle graph/capture coverage problem.
- This change is verifier-parity neutral: it only runs dummy route-pack calls
  during startup warmup to force Triton specialization before the measured
  request. It does not alter live routing, logits, target verification, or
  draft acceptance math.

Implementation:

- Extended the B12X route-pack warmup prompt-token list to include exact shapes
  seen by the grounded coding benchmark (`411`, `415`, `416`) in addition to the
  previous round buckets (`512`, `513`, `1024`).
- For each warmed token count, call `pack_topk_routes_by_expert(...)` twice:
  once with default internal workspace and once with exact caller-owned
  `packed_route_indices`, `block_expert_ids`, `packed_route_count`, and
  `expert_offsets` buffers. The second call mirrors the real W4A16 MoE path,
  where exact workspace capacity can select different Triton specializations
  than the capacity-rounded planner path.

Validation:

- `RUFF_CACHE_DIR=/tmp/ruff-cache-vllm uv run --no-sync ruff check
  vllm/model_executor/warmup/kernel_warmup.py` passed.
- `PYTHONPYCACHEPREFIX=/tmp/vllm-pycache python3 -m py_compile
  vllm/model_executor/warmup/kernel_warmup.py` passed.

Benchmark plan:

- Rebuild the clean runtime image on both nodes with only this aligned warmup
  change.
- Restart DSpark and check startup plus first measured request logs for the
  route-pack/eagle JIT warnings.
- Re-run the same warmed single-stream `code_completion` benchmark three times
  and compare against the `63.41 tok/s` DSpark anchor and `39.88 tok/s` MTP-1.

Follow-up implementation:

- The first route-pack-only warmup still allowed first-request JIT warnings on
  the smoke prompt for `_pack_topk_routes_small_prefix_kernel` and
  `eagle_prepare_next_token_padded_kernel`.
- Extended the short prefill warmup list to include `12` and `16` tokens, which
  covers the smoke/chat-template prompt neighborhood.
- Warmed route-pack with both 2D top-k route IDs and flattened route IDs because
  the B12X package can specialize those routes separately.
- Warmed `eagle_prepare_next_token_padded_kernel` for sampled widths `1..6`
  instead of only the full DSpark width. The live first request can specialize
  on the accepted/sample width, not only the configured speculative width.

Validation:

- `RUFF_CACHE_DIR=/tmp/ruff-cache-vllm uv run --no-sync ruff check
  vllm/model_executor/warmup/kernel_warmup.py` passed.
- `PYTHONPYCACHEPREFIX=/tmp/vllm-pycache python3 -m py_compile
  vllm/model_executor/warmup/kernel_warmup.py` passed.
- Rebuilt `vllm-dspark-runtime:clean` on both nodes.
- Smoke request succeeded after startup and produced no
  `JIT compilation during inference` warnings for `_pack_topk` or
  `eagle_prepare` on either rank.

Benchmark result:

- Scenario: same warmed single-stream coding gate
  (`code_completion`, `PROMPT_TOKENS=512`, `MAX_TOKENS=256`,
  `THINKING=false`, `WARMUP_REQUESTS=1`).
- Result set:
  `single_stream_interactive_262k_window_dspark_routepack_eagle_warmup2_paper_code_completion_20260628_113246_run*.json`.

| run | server tok/s | draft acceptance | accepted / draft | cycle ms | pos0 | suffix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `64.44` | `69.82%` | `3.49` | `69.70` | `92.98%` | `82.37%` |
| 2 | `59.55` | `62.26%` | `3.11` | `69.07` | `87.10%` | `78.34%` |
| 3 | `59.89` | `62.26%` | `3.11` | `68.67` | `88.71%` | `80.64%` |

Aggregate:

- Server decode speed: `61.29 tok/s` mean, `2.73` stdev, `4.5%` CV,
  `95% CI ±6.78 tok/s`.
- Draft acceptance: `64.78%` mean.
- Accepted tokens per draft: `3.24` mean.
- Cycle time estimate: `69.15 ms` mean.

Decision:

- This successfully removes first-request route-pack/EAGLE JIT misses, which is
  useful for startup determinism and benchmark cleanliness.
- It does not improve steady decode speed versus the current DSpark anchor
  (`63.41 tok/s`); measured mean is `-3.3%` below anchor and within the broader
  observed DSpark variance band.
- Keep this warmup patch only if we value clean first-token behavior and lower
  cold-run noise; do not count it toward the active decode-speed goal.
- Next aligned target: confirm whether the DSpark target verifier path is fully
  CUDA-graph replayed during decode. If any part of target verify still runs
  eager or outside the captured graph, graphing that path is the next largest
  parity-preserving `Tverify` lever. If graph replay is already complete, move
  to actual-shape sparse-MLA autotune coverage and DSpark feature-materialization
  fusion.

## DSpark Decode Graph / Sparse-MLA Diagnostic, 2026-06-28

Purpose:

- Re-check the main Lever 1 assumption before implementing another patch:
  whether warmed DSpark target verification is still leaking out of CUDA graph
  replay, and whether sparse MLA decode is still using untuned tactics.
- This run used `--cudagraph-metrics` plus `VLLM_DSPARK_ITER_TIMING=1`.
  Timing numbers from this mode are diagnostic only because CUDA is
  synchronized around stages.

Runtime graph evidence:

```text
CUDAGraph Config Settings:
- Mode: FULL_AND_PIECEWISE
- Capture sizes: [1, 2, 4, 8]

CUDAGraph Stats:
| Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |
| 6               | 6             | 0            | FULL         | 31    |
| 415             | 415           | 0            | NONE         | 1     |
```

Interpretation:

- The DSpark verifier decode step runs at width `6` and is already `FULL`
  CUDA-graph replayed in the measured decode loop.
- The `415` token `NONE` entry is the one-off prefill for the benchmark prompt,
  not warmed decode generation.
- Therefore, decode graph capture is not the current missing speed lever.

Sparse-MLA autotune evidence:

- Startup logs include DSpark uniform-decode sparse MLA autotune shapes.
- The FlashInfer autotune cache in the clean container contains entries for the
  DSpark decode shapes around `512` and `2048`.
- The prior torch profile shows FlashInfer sparse MLA at roughly `1.2%` self
  CUDA in the diagnostic profile, while B12X W4A16/MXFP4 MoE is roughly `39.6%`
  self CUDA.

Decision:

- Whole-cycle decode graph capture and sparse-MLA decode autotune are already
  covered enough that they are unlikely to produce the next large speed step.
- The remaining high-value paths are:
  - bit-equivalent B12X MXFP4 MoE reduction, because it is the dominant
    verifier-side CUDA bucket;
  - reference-parity / τ work, because current accepted tokens per draft are
    around `3.2` out of a maximum `5`, so raising acceptance compounds with the
    fixed verifier cycle.
- Do not count the graph diagnostic speed (`55.41 tok/s`) as a benchmark result;
  it was run with synchronization-heavy diagnostics enabled.

## Reference KV Quant-Dequant A/B, 2026-06-28

Purpose:

- Test the paper-aligned acceptance lever identified from the suffix decay:
  reproduce the released reference's in-place FP8 quant-dequant on the draft
  no-RoPE KV slice and check whether it raises `τ`.
- The flag was wired through compose as
  `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT`.
- Workload was the same warmed single-stream coding gate:
  `code_completion`, `PROMPT_TOKENS=512`, `MAX_TOKENS=256`,
  `THINKING=false`, `WARMUP_REQUESTS=1`.

Baseline, flag disabled:

```text
single_stream_interactive_262k_window_dspark_kvq_flagwired_baseline_code_completion_20260628_115455_run*.json
server_tok_s mean: 58.48
draft_acceptance mean: 0.6531
accepted_per_draft mean: 3.2654
suffix conditional mean: 0.7915
cycle_ms mean: 72.96
```

KV quant-dequant enabled, first pass:

- Result set:
  `single_stream_interactive_262k_window_dspark_reference_kvq_code_completion_20260628_120300_run*.json`.
- A live JIT warning fired during this pass:
  `_dspark_quant_dequant_nope_kernel`.
- Treat this pass as diagnostic only.
- Mean: `57.79 tok/s`, draft acceptance `0.6091`, accepted/draft `3.0457`,
  suffix conditional `0.7808`, cycle `70.00 ms`.

KV quant-dequant enabled, post-JIT clean pass:

```text
single_stream_interactive_262k_window_dspark_reference_kvq_postjit_code_completion_20260628_120422_run*.json
server_tok_s mean: 60.75
draft_acceptance mean: 0.6421
accepted_per_draft mean: 3.2105
suffix conditional mean: 0.7823
cycle_ms mean: 69.31
```

Interpretation:

- The flag reduces measured cycle time (`72.96 ms` to `69.31 ms`) and gives a
  small speed improvement in the clean post-JIT pass (`58.48` to
  `60.75 tok/s`), but it does not improve draft quality.
- Draft acceptance, accepted tokens per draft, and suffix conditional
  acceptance all move down versus the disabled baseline.
- This is not the desired `τ` / reference-parity win. A true paper-aligned
  acceptance lever should raise accepted tokens per draft even if it adds a
  fusable draft-side tax.

Decision:

- Keep the compose wiring for the flag so we can reproduce the experiment.
- Leave the experiment default disabled (`0`).
- Do not count this as progress toward the decode-speed goal because the
  observed speed gain comes with worse draft quality and is small relative to
  run-to-run variance.
- Before shelving the idea permanently, validate the scale formula and group
  size against the released reference implementation. If the formula is wrong,
  this A/B re-measured a bug rather than the reference behavior.

## Claude Review + B12X Actionability Check, 2026-06-28

Review loop:

- Started a persistent reviewer session in tmux:
  `dspark-claude-review`, using
  `claude --resume 9059f32e-3e69-4db2-9982-4468f6c5073d --add-dir /home/pieter/Code/vllm-dspark-unholy`.
- Claude agreed with the clean profile interpretation: target-side B12X W4A16
  MoE is the only bucket large enough to plausibly produce a `>25%`
  single-stream speed gain by itself.
- Claude's caution: do not write a new custom MoE kernel first. A
  bit-equivalent tactic/tile change inside B12X is acceptable if exposed, but a
  new kernel with different accumulation or quantization order is parity-risky.

Container-side B12X hook check:

- `TPMoEScratchCaps` has no tactic, tile, split-K, warp, or scheduler fields.
  vLLM only passes shape/quant/source/layout data into
  `plan_tp_moe_scratch`.
- B12X W4A16 env hooks found:
  `B12X_W4A16_TC_DECODE`, `B12X_TIMING` / `VLLM_B12X_TIMING`,
  cutover knobs, dynamic knobs, max-active-cluster overrides, and
  `B12X_MOE_TILE_MN`.
- `B12X_MOE_TILE_MN` appears to apply to the non-W4A16 static/micro path, not
  the packed W4A16 path we use for DeepSeek V4 Flash.
- The relevant W4A16 selector is internal Python:
  `b12x/moe/fused/w4a16/kernel.py::_select_tile_config`.
  The first inspection used an oversized assumed MoE shape. The model config
  grounds the real shape as `hidden_size=4096`, `moe_intermediate_size=2048`,
  and `num_experts_per_tok=6`. For the DSpark decode target rows (`m=6`,
  `topk=6`, `block_size_m=8`, `scale_format=e8m0_k32`), it selects
  `(tile_k=64, tile_n=128, cta_threads=128, blocks_per_sm=3)`.
- The only obvious existing W4A16 tactic env, `B12X_W4A16_TC_DECODE=1`, was
  already tested and regressed throughput (`54.95 tok/s`), so it is not the
  next speed path.

Profiler harness added:

- Added `entrypoints/entrypoint.dspark-ncu-launch.sh` to launch only the
  selected role under `ncu --mode=launch` while preserving the DSpark rewrite
  from the unholy entrypoint.
- Added `compose/docker-compose.dspark-ncu.yml` to mount host Nsight tools and
  grant profiler capabilities only for the profiling launch.
- Added `scripts/dspark_ncu_attach_moe.sh` to attach to the launched head
  container and collect a bounded B12X W4A16 kernel sample.
- Added ignore rules for Nsight reports and generated DSpark profile/stdout
  artifacts.
- Syntax and compose service validation passed; files were synced to the
  worker. The current live server was not restarted for this.

Current decision:

- Claude follow-up after the hook check: run the MoE NCU pass once because the
  harness makes it cheap and it directly decides whether B12X selector work is
  worth touching.
- If NCU shows the W4A16 MoE kernel is already near bandwidth saturation,
  pivot to NCCL/all-reduce timing and draft/verify overlap. Do not sweep
  B12X tile configs.
- If NCU shows meaningful bandwidth headroom, try only a controlled,
  env-gated scheduling override in the B12X W4A16 selector. Avoid a broad
  `(tile_k, tile_n, cta_threads, blocks_per_sm)` sweep, and do not change
  `tile_k` first because it is the highest parity-risk dimension.
- Any selector override must have strict output-difference and acceptance
  validation before repeated speed benchmarks.
- If we avoid touching B12X internals, the remaining safe levers are smaller:
  NCCL/all-reduce tuning and draft/verify overlap. Claude estimates these
  stack to less than the `>25%` target unless draft cost is larger than the
  current torch profile suggests.

Shortcuts / caveats to catch later:

- The NCU harness is head-rank focused. It is enough for per-rank MoE kernel
  bandwidth but not a full two-rank communication profile.
- The NCU harness has been syntax/config validated, but the first live
  `ncu --mode=launch` attempt was not usable: the head passed the attach gate,
  but then stalled before spawning its local worker subprocess. The attach also
  reported no profiled kernels. The profiled server was stopped and the normal
  DSpark server was relaunched.
- The attach helper initially passed `--target-processes` in attach mode; NCU
  rejects that option there. It is now only present on the launch side.
- The B12X selector analysis used installed package inspection and inferred
  decode shape; verify exact shape from B12X timing or NCU metadata before
  committing to a tile override.
- Any B12X tile override must be gated and validated for output parity before
  benchmark gating.

Follow-up reviewer guidance and harness, 2026-06-28:

- Claude's post-failure recommendation was not to keep fighting live
  `ncu --mode=attach`. The broken path is vLLM process-tree specific, not a
  reason to abandon the MoE question.
- Two independent next measurements are now the grounded path:
  1. NSYS timeline on the real server to check NCCL serialization, launch gaps,
     and the size of any draft-only idle block. This gates a parity-zero
     communication/launch-overhead win.
  2. Standalone B12X W4A16 NCU microbench to check whether the dominant fused
     MoE kernel has real bandwidth/occupancy headroom. This gates any
     B12X-selector work.
- Added `scripts/dspark_b12x_moe_microbench.py`, a synthetic packed-W4A16
  single-process benchmark. It constructs the packed tensor shapes directly
  instead of loading DeepSeek weights, then calls the installed
  `run_w4a16_moe` path with `m=6`, `topk=6`, `hidden=4096`,
  `intermediate=2048`, `num_experts=256`, `scale_format=e8m0_k32`,
  `weight_layout=packed`, and direct top-k routes. This targets the same fused
  W4A16 path as the real profile, not the modelopt small-M micro path.
- Added `scripts/dspark_ncu_b12x_moe_microbench.sh`, a one-shot Docker/Nsight
  wrapper. It mounts host Nsight tools into `vllm-dspark-runtime:clean`, exports
  reports under `experiments/dspark-benchmarks/profiles/`, and refuses a full
  GPU-memory run while `vllm-spark-head` is running unless
  `ALLOW_WITH_SERVER=1` is set.
- Validation:
  - `bash -n scripts/dspark_ncu_b12x_moe_microbench.sh` passed.
  - `PYTHONPYCACHEPREFIX=/tmp/dspark_pycache python3 -m py_compile
    scripts/dspark_b12x_moe_microbench.py` passed. The repo has an unrelated
    root-owned `scripts/__pycache__`, so pycache output must be redirected.
  - Container dry-run passed with
    `PROFILE=0 NO_GPUS=1 MICROBENCH_ARGS='--dry-run --tiny'
    bash scripts/dspark_ncu_b12x_moe_microbench.sh`.
  - Full-shape dry-run passed and confirmed the target selector:
    `moe_block_size=8`, `tile_k=64`, `tile_n=128`,
    `cta_threads=128`, `blocks_per_sm=3`.
  - Full-shape synthetic static allocation estimate is `3.19 GiB`. The first
    attempted NCU run used the wrong `7168/9216/topk=8` shape, estimated
    `25.10 GiB`, and failed with a packed tensor descriptor overflow. That run
    is rejected as a shape-assumption bug, not a B12X limitation.
- NCU gate from Claude:
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed >= ~85%`: MoE is
    effectively HBM-saturated; do not override selector knobs, pivot to
    NCCL/launch/draft timing.
  - `< ~65%` plus low occupancy or relevant stall signal: implement one
    env-gated scheduling override for `tile_n`, `blocks_per_sm`, or
    `cta_threads`. Do not change `tile_k` first because it changes K-reduction
    chunking and is the highest parity-risk knob.
  - `65-85%`: try only the knob implicated by NCU, then validate output parity,
    draft acceptance, and repeated real-model decode speed.

Corrected B12X NCU result:

- Profile report:
  `experiments/dspark-benchmarks/profiles/b12x_moe_full_corrected_memproxy_20260628_1323/b12x_moe_full_corrected_memproxy_20260628_1323.ncu-rep`.
- Corrected shape:
  `m=6`, `topk=6`, `hidden=4096`, `intermediate=2048`,
  `num_experts=256`, `scale_format=e8m0_k32`, packed W4A16.
- Fused launch:
  `tile_k=64`, `tile_n=128`, `cta_threads=128`, `blocks_per_sm=3`,
  `grid=144`, `block=128`, `regs/thread=80`, shared memory `27.648 KiB`.
- Explicit fused-kernel NCU metrics, two samples:
  - `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`:
    `10.15%` mean.
  - `gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed`:
    `10.13%` mean.
  - `smsp__warps_active.avg.pct_of_peak_sustained_active`: `24.61%`.
  - `smsp__issue_active.avg.pct_of_peak_sustained_active`: `17.50%`.
  - `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active`:
    `9.32`.
  - `smsp__average_warps_issue_stalled_not_selected_per_issue_active`:
    `0.11`.
  - Fused kernel duration: `2.11 ms` mean under NCU replay.
- Decision: this is not HBM-saturated. It clears Claude's "headroom" gate for
  one controlled scheduling experiment. The signal points to latency/occupancy
  and long-scoreboard behavior, not memory bandwidth saturation.
- Candidate selector configs available for `moe_block_size=8` are:
  `(tile_k=128, tile_n=128, cta_threads=256)`,
  `(tile_k=64, tile_n=128, cta_threads=128)` current, and
  `(tile_k=128, tile_n=64, cta_threads=128)`.
- Added `--force-tile-config TILE_K,TILE_N,CTA_THREADS` to the standalone
  microbench so candidate configs can be A/B tested before any runtime patch.
  Added `--force-blocks-per-sm N` for the parity-preferred first experiment.
  If no tile is provided, this first resolves the current selector tile and
  then overrides only `blocks_per_sm`.
  Dry-run selector outcomes for the corrected shape:
  - `(128,128,256)` -> `blocks_per_sm=1`.
  - `(64,128,128)` current -> `blocks_per_sm=3`.
  - `(128,64,128)` -> `blocks_per_sm=2`.
  - `(64,128,128)` plus `--force-blocks-per-sm 4` -> `blocks_per_sm=4`.
- Claude's recommendation after seeing the corrected NCU profile:
  1. First try blocks-per-SM only with the current tile. This should be
     bit-equivalent if it only changes CTA scheduling, so require exact output
     equality in the standalone binding test. Re-NCU should show active warps
     up and long-scoreboard down.
  2. If that is capped or ineffective, try `(128,64,128)` because it halves
     `tile_n` and may relieve shared-memory pressure, but it also changes
     `tile_k` and therefore the fp32 K-reduction chunking. Treat this as
     numerically-close, not bit-exact.
  3. Try `(128,128,256)` only after those two because it drops to one
     block-per-SM in the selector dry-run.
  Live gate for keeping a candidate: repeated warmed corrected-parser tok/s at
  least `5%` above baseline with accepted/draft, position-0 acceptance, and
  suffix conditional acceptance within baseline noise.
- Next implementation should be an env-gated override, with the first try
  selected by NCU/Claude. Avoid baking any selector change until it passes:
  1. microbench output finite/same-shape and no runtime error,
  2. unit-level bit/close check for the affected W4A16 path where practical,
  3. real-model DSpark acceptance within baseline noise,
  4. repeated corrected single-stream benchmark speed improvement.

Selector override experiment outcome, 2026-06-28:

- Implemented an env-gated vLLM B12X W4A16 selector override in the runtime
  image, default off:
  - `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM`
  - `VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M`
  - `VLLM_B12X_W4A16_FORCE_TILE_CONFIG`
- The hook is installed before B12X W4A16 preparation / CUDA graph capture and
  is gated to small decode shapes by `problem_m`. Large prefill/warmup shapes
  keep the upstream selector.
- Validation:
  - `ruff check` and `ruff format --check` passed on edited vLLM files.
  - `py_compile` passed.
  - Containerized selector test passed: `5 passed`.
  - Real installed-package probe on both head and worker:
    `VLLM_B12X_W4A16_FORCE_TILE_CONFIG=128,64,128` returns
    `decode (128, 64, 128, 2)` and `large (64, 128, 128, 3)`.
- Candidate 1, current tile `(64,128,128)` plus
  `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=4`, failed before live testing:
  - Current selector timing-only synthetic: `4.53 ms/iter`.
  - Forced `blocks_per_sm=4` did not finish even timing-only after more than
    60 seconds, and NCU runs also failed to complete promptly.
  - Decision: treat as pathological / capped by the real shared-memory launch
    regime, not a viable live candidate.
- Candidate 2, forced tile `(128,64,128)` with B12X-computed
  `blocks_per_sm=2`, passed synthetic timing but failed live quality/speed:
  - Synthetic timing-only: `2.89 ms/iter` vs current `4.53 ms/iter`.
  - NCU report:
    `experiments/dspark-benchmarks/profiles/b12x_moe_tile128_64_ncu_quick_20260628_1400/b12x_moe_tile128_64_ncu_quick_20260628_1400.ncu-rep`.
  - NCU mechanism signal:
    long-scoreboard dropped from about `9.32` to about `4.96`, issue-active
    rose from about `17.50%` to about `18.12%`, but the kernel remained
    shared-memory limited at `2` blocks/SM with `35.84 KiB` shared memory per
    block.
  - Live warmed code-completion benchmark files:
    `single_stream_interactive_262k_window_b12x_tile128_64_code_completion_20260628_141235_run{1,2,3}.json`.
  - Live decode tok/s: `41.51`, `50.36`, `46.28`; mean `46.05`.
  - Recent warmed code-completion anchors:
    `paper_deferred_mean_kernel_b12xwo_warm_code_completion_20260628_080704`
    mean `60.42` tok/s, and
    `paper_sps_curve_b12xwo_20260628_070550_len5_20260628_074317` mean
    `62.33` tok/s.
  - Candidate acceptance collapsed:
    accepted/draft mean `2.08` vs anchors about `3.22-3.42`; acceptance-rate
    mean `0.416` vs anchors about `0.645-0.684`; suffix conditional mean
    `0.617` vs anchors about `0.80-0.82`.
  - Decision: changing `tile_k` to `128` is not acceptance-safe. The kernel
    timing win is eaten by lower DSpark draft acceptance and extra target work.
- The live server was restored to the default selector path after the failed
  candidate. `/health` returned `200` and logs show `Application startup
  complete` with no selector override activation.
- Claude's post-result recommendation:
  1. Stop MoE tile/kernel work for now. Current-tile `blocks_per_sm` is
     pathological, and tile changes are acceptance-unsafe.
  2. Run a reference-acceptance diagnostic against the DeepSpec/DeepSeek
     reference implementation on the identical code-completion prompts. If the
     reference achieves much higher accepted/draft than our ~`3.2-3.4` anchor,
     the biggest remaining win is draft-quality / tau parity.
  3. In parallel, run NSYS on the real server to validate whether NCCL
     all-reduce is serialized with compute. NCCL comm/comp overlap is the
     likely parity-zero `5-7%` win.
  4. If tau is already tapped and NCCL is not enough, deliberately re-scope
     draft/verify overlap as the remaining significant parity-preserving lever.

## Realistic Coding-Session Harness + Near-Ceiling Milestone (2026-06-28)

Built a realistic coding-session inference harness
(`scripts/dspark_coding_session_*`): a read-only agentic multi-turn loop that
drives the served model to investigate this project, reproducing the real coding
session the synthetic short-prompt benchmark misses. It streams per-turn
metrics to stderr and supports early-stop (`--stop-file`,
`--early-stop-acceptance-floor`, `--early-stop-max-context`). Per-turn and
per-session decode-speed and spec-decode acceptance/tau come from `/metrics`
diffing (reusing the single-stream benchmark helpers); the summarizer emits a
context-length-binned acceptance curve.

3-session validation (10 turns each, context growing to ~80k tokens):

- mean acceptance `0.627`, mean accepted/draft `3.14`, mean tau `4.14`,
  mean decode `58.2 tok/s`, CV `5-7%` on acceptance/decode (A/B-ready).
- The tau decay reproduces: per-turn tau falls `4.36 -> ~2.8` as context grows
  `0.1k -> 80k` tokens.

Key analytical findings from the harness data:

- **The decode cycle is context-independent.** cycle = tau / decode_tps is flat
  `~70 ms` from `0.1k` to `80k` context. The observed decode_tps drop is
  entirely tau-driven (decode_tps / tau is constant `~14.3`); the per-token
  forward does not slow with context. Sparse-MLA is already windowed, so there
  is no large-context forward lever and a forward profile at 80k would just
  re-show the small-context MoE-43% split.
- **tau decay is a suffix effect, not a backbone collapse.** Per-position
  conditional acceptance vs context bucket:

  | context | pos0 | pos1 | pos2 | pos3 | pos4 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 0-2k | 0.878 | 0.842 | 0.841 | 0.831 | 0.807 |
  | 8-20k | 0.892 | 0.858 | 0.863 | 0.824 | 0.846 |
  | 20-40k | 0.893 | 0.874 | 0.861 | 0.820 | 0.834 |
  | 40-80k | 0.861 | 0.834 | 0.822 | 0.799 | 0.768 |

  pos0 (draft backbone first-token) is stable `~0.86-0.89` at every context (no
  out-of-distribution collapse); only the suffix softens mildly at `40-80k`
  (pos4 `0.834 -> 0.768`). pos0 is at its greedy T=0 ceiling (~target-argmax
  match rate); raising it needs a better draft, not an integration fix.

Conclusion: DSpark single-stream is near its ceiling at `~60 tok/s`
(`~1.5x MTP-1`, `2.3x no-spec`). The forward is flat and MoE-tapped; tau is
healthy with only mild long-context suffix decay. Remaining single-stream
levers are modest: NCCL all-reduce overlap (`~5-7%`, parity-zero, only if
currently serialized) and possibly a small suffix-tau reference-parity bump if
the DeepSpec reference holds a higher suffix at long context. The `>25%` goal is
not reachable single-stream with the forward flat and tau healthy; it lives in
re-scoping to draft/verify overlap or concurrency.

Next-step options recorded for the next pass: (1) NSYS timeline to confirm
whether NCCL all-reduces are serialized (decides the one certain parity-zero
win); (2) DeepSpec reference acceptance at `~40-80k` context to check suffix-tau
headroom; (3) deliberate re-scope of draft/verify overlap for the `>25%` goal.

## Mixed Prefill+Decode Stability Fix, 2026-06-29

Problem:

- Concurrent DSpark serving crashed at c=4 when the scheduler mixed a new
  request prefill with an existing request's speculative verify/decode step.
- The failing batch had flattened target rows `421 = 415 + 6` for
  `batch_size=2`, so `DSparkProposer._view_by_request()` raised:
  `DSpark currently requires uniform flattened per-request inputs`.
- This was not a model-quality issue; it was a proposer batching assumption.

Implementation:

- Added a ragged target-context preparation path in
  `vllm/v1/spec_decode/dspark_proposer.py`.
- The proposer now uses `query_start_loc` to split target hidden states and
  positions by request, then groups requests with equal target-context lengths.
- Each group is passed to `prefill_main()` as a full-batch tensor so DSpark's
  internal main-KV cache row identity is preserved. Requests not in that group
  get placeholder rows with `num_rejected_tokens == group_len`.
- Updated `DeepSeekV4DSparkAttention.store_main_kv()` so a row can have zero
  valid tokens (`valid_lengths.clamp(min=0)`), allowing placeholder rows to be
  true no-ops.
- Kept the old uniform path intact for single-stream and naturally rectangular
  batches.

Validation:

- Local syntax: `py_compile` passed with `PYTHONPYCACHEPREFIX=/tmp/...`.
- Lint: `ruff check` passed with `RUFF_CACHE_DIR=/tmp/...`.
- Clean runtime image rebuilt on head and worker:
  `vllm-dspark-runtime:clean`.
- Installed overlay file hashes matched across nodes for
  `dspark_proposer.py` and `dspark.py`.
- Focused runtime-image tests:
  `5 passed, 54 deselected`.
- Runtime-image `test_dspark.py -k "not triton"`:
  `55 passed, 4 deselected`.
- Full `test_dspark.py` in the clean runtime image reached `55 passed`; the
  remaining 4 CUDA Triton kernel tests failed because the thin runtime image has
  no C compiler for Triton's driver helper build. This is an image capability
  gap, not a DSpark proposer regression.

Server validation:

- Restarted TP=2 DSpark stack on `vllm-dspark-runtime:clean`.
- Raised `.env.dspark-experiment` to `MAX_NUM_SEQS=8` so c=4/c=8 tests are not
  hidden behind the old `MAX_NUM_SEQS=2` workaround.
- Startup completed with `max_num_seqs=8`, `max_model_len=262144`,
  `method=dspark`, and CUDA graph capture through `largest=96`.
- Post-benchmark log scan found no `ValueError`, no uniform-reshape exception,
  no traceback, and no NCCL watchdog fallout. `/health` remained `200`.

Benchmark results:

Result sets:

- c=4 smoke:
  `concurrent_interactive_262k_window_c4_mixedfix_mseq8_c4_20260629_013005_run1.json`
- c=8 smoke:
  `concurrent_interactive_262k_window_c8_mixedfix_mseq8_c8_20260629_013034_run1.json`
- c=4 repeat:
  `concurrent_interactive_262k_window_c4_mixedfix_mseq8_c4_3x_20260629_013113_run*.json`
- c=8 repeat:
  `concurrent_interactive_262k_window_c8_mixedfix_mseq8_c8_3x_20260629_013223_run*.json`

Three-run repeat summary:

| concurrency | per-user server decode tok/s | aggregate decode tok/s | acceptance | accepted/draft | first-token | suffix conditional | mean TTFC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `31.78 ± 1.86` | `127.10 ± 7.45` | `61.81% ± 2.20%` | `3.09 ± 0.11` | `90.34% ± 2.12%` | `79.17% ± 1.96%` | `1.00s ± 0.01s` |
| 8 | `24.27 ± 2.06` | `194.16 ± 16.46` | `62.39% ± 0.51%` | `3.12 ± 0.03` | `90.99% ± 0.55%` | `78.36% ± 0.59%` | `1.70s ± 0.04s` |

Shortcut / follow-up notes:

- The mixed-batch path is stable but not yet optimal. It groups ragged chunks in
  Python and reads small query/rejection metadata on CPU even when
  `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`.
- Placeholder rows preserve DSpark cache row identity, but they still run dummy
  projection work inside `prefill_main()`. A future DSpark-specific ragged
  main-KV update API or kernel should remove that overhead.
- The next performance work should keep this stability gate in place and avoid
  returning to `MAX_NUM_SEQS=2` as a workaround.
