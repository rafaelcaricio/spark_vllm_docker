# DSpark Confidence Threshold 0.50 Async-Bridge Benchmark

Date: 2026-06-27

Goal status: not complete. The implementation is functionally active and
measured, but decode inference speed has not increased significantly versus the
fixed-length baseline.

Server configuration:

- Model: `deepseek-v4-flash-dspark`
- Max model length: `262144`
- Prompt target: 512 local chat-template tokens
- Decode target: 256 server-counted generation tokens
- Single stream, `temperature=0`, `ignore_eos=true`
- `VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.50`
- Async scheduler bridge enabled: DSpark confidence prefix lengths are passed
  back through `ModelRunnerOutput.draft_token_lengths` and used to size the
  next async speculative placeholder list.

Important caveat:

- The first corrected three-run set (`conf050_asyncbridge_20260627_203916`) was
  partially contaminated by first-use Triton JIT for variable-prefix route
  packing. The post-JIT set below is the primary comparison.

## Post-JIT Repeated Runs

| Run | Server decode tok/s | End-to-end s | TT first content s | Decode span s | Drafts | Draft tokens | Accepted | Draft tokens/draft | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 52.55 | 9.03 | 4.18 | 4.85 | 103 | 395 | 157 | 3.83 | 39.75% |
| 2 | 21.02 | 15.67 | 3.55 | 12.13 | 178 | 585 | 77 | 3.29 | 13.16% |
| 3 | 41.01 | 9.45 | 3.23 | 6.22 | 109 | 397 | 148 | 3.64 | 37.28% |

Aggregate:

- Mean server decode speed: `38.20 tok/s`
- Sample stddev: `15.95 tok/s`
- CV: `41.76%`
- Mean time to first byte: `0.421 s`
- Mean time to first content: `3.653 s`
- Mean decode span after first content: `7.733 s`
- Mean end-to-end latency: `11.386 s`
- Mean draft tokens per draft: `3.59` versus fixed-length `5.00`
- Mean accepted tokens: `127.33`
- Mean draft tokens: `459.00`
- Mean acceptance rate: `30.06%`
- Accepted tokens by position total: `{0: 172, 1: 84, 2: 55, 3: 38, 4: 33}`

Comparison against fixed-length salted baseline from
`repeat_profile_20260627_194827_summary.md`:

- Fixed-length mean server decode speed: `37.64 tok/s`
- Fixed-length decode speed CV: `10.02%`
- Fixed-length mean accepted rate: `14.28%`
- Fixed-length accepted totals by position: `{0: 161, 1: 75, 2: 40, 3: 29, 4: 14}`

Interpretation:

- The async bridge is effective: thresholding now changes the actual scheduled
  verification length, not just CPU-side diagnostics.
- Static threshold `0.50` prunes about 28% of draft-token verification work
  (`3.59 / 5.00`) and improves mean accepted draft rate and accepted tokens by
  position.
- It is not yet a clear decode-speed win. Mean speed is roughly baseline-level,
  but variability is much worse. This points to confidence calibration,
  threshold selection, and variable-prefix scheduling overhead as the next
  profiling targets.

## Why This Is Not Done

- The speed delta is too small to claim an improvement: `38.20 tok/s` versus
  the fixed-length baseline of `37.64 tok/s` is within the observed run-to-run
  noise.
- Variance regressed badly: CV rose from `10.02%` to `41.76%`, which makes the
  path less predictable for interactive serving.
- The confidence threshold improved pruning and acceptance metrics, but those
  gains were offset by the cost of variable-prefix scheduling, route packing,
  rejection sampling shape changes, or confidence miscalibration.
- Run 2 shows the failure mode clearly: scheduled draft length was lower, but
  accepted tokens dropped and decode span stretched to `12.13 s`.

## How We Proved The Current Path

- Added `VLLM_DSPARK_CONFIDENCE_THRESHOLD` and validated that `0.50` prunes the
  DSpark block before verification.
- Bridged per-request draft lengths from the proposer through
  `ModelRunnerOutput.draft_token_lengths`.
- Updated async scheduler placeholder sizing so the next verification step uses
  the pruned length instead of always scheduling five draft tokens.
- Rebuilt both runtime images and verified the image contains the new scheduler,
  output, env, and proposer code.
- Ran three post-JIT salted single-stream repetitions against the real
  DeepSeek V4 Flash DSpark server with the 262144-token context window enabled.

## Next Step

Run a grounded threshold/cost sweep before implementing more kernels:

- Test `VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.20`, `0.35`, and `0.50` with three
  post-JIT salted repetitions each.
- Keep the success gate strict: mean server decode speed must exceed the
  fixed-length baseline by a meaningful margin and variance must not explode.
- Record scheduled draft length, prune rate, accepted tokens, accepted tokens
  by position, TTFC, decode span, and server-side generation tok/s for each
  threshold.
- If no threshold crosses the speed gate, the next implementation target should
  be variable-prefix overhead: bucket pruned draft lengths or move confidence
  prefix selection and speculative input preparation into GPU-side kernels.
