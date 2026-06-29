# DSpark Screenshot Scaling Verification

Date: 2026-06-29

Source prompt: screenshot reporting one TP=2 DSpark replica at
`MAX_NUM_SEQS=16` with a static batch scaling curve:

| concurrency | screenshot aggregate | screenshot per-stream | screenshot acceptance |
| ---: | ---: | ---: | ---: |
| 1 | `52.1` tok/s | `52.1` | `0.59-0.64` |
| 2 | `82.6` tok/s | `41.3` | `0.60` |
| 4 | `123.9` tok/s | `31.0` | `0.58-0.63` |
| 8 | `212.3` tok/s | `26.5` | `0.58-0.62` |
| 16 | `301.2` tok/s | `18.8` | `0.59-0.61` |

## Local Configuration

Stack:

- Image: `vllm-dspark-runtime:clean`
- Model: `deepseek-v4-flash-dspark`
- TP: `2`
- `MAX_MODEL_LEN=262144`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `MAX_NUM_SEQS=16`
- `MTP_NUM_TOKENS=5`
- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off` for the reproduction sweep
- Prompt shape: `PROMPT_TOKENS=512`, `MAX_TOKENS=256`, `THINKING=false`
- Repeats: `3` per concurrency

Startup verified c16 graph coverage:

- `cudagraph_capture_sizes` up to `192`
- graph profiling logged `PIECEWISE=25 (largest=192), FULL=13 (largest=96)`

## Local Scheduler-Off Sweep

These values are server-side generation metrics from the benchmark JSON. The
`per-stream` column is the harness `per_user_decode_tokens_per_second_server_metrics`;
`aggregate` is the corresponding batch aggregate.

| concurrency | aggregate tok/s | per-stream tok/s | acceptance | accepted/draft | mean scheduled draft length | TTFC mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `57.77 ± 2.06` | `57.77` | `0.6501` | `3.250` | `5.000` | `0.49s` |
| 2 | `86.14 ± 5.84` | `43.07` | `0.6176` | `3.088` | `5.000` | `0.67s` |
| 4 | `132.24 ± 7.70` | `33.06` | `0.6268` | `3.134` | `5.000` | `1.02s` |
| 8 | `197.19 ± 10.58` | `24.65` | `0.6229` | `3.114` | `5.000` | `1.80s` |
| 16 | `319.37 ± 6.72` | `19.96` | `0.6233` | `3.116` | `5.000` | `3.32s` |

Interpretation:

- The screenshot's main result reproduces on our current code path by changing
  the serving regime to `MAX_NUM_SEQS=16`.
- We slightly exceed the screenshot at c16: local `319.37` aggregate tok/s vs
  screenshot `301.2`.
- Acceptance remains stable near `0.62`, matching the screenshot's conclusion
  that DSpark batching quality is numerically fine under higher concurrency.
- This does not improve single interactive stream latency. Per-stream speed
  falls from `57.77` at c1 to `19.96` at c16 while aggregate throughput rises.

## Hardware Scheduler Check

Focused c16 run with:

- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=hardware`
- `VLLM_DSPARK_SPS_CURVE=8:7.235357,12:7.749286,16:6.334372,20:6.767292,24:7.280485,48:5.396963`
- `VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=0`

Result:

| mode | aggregate tok/s | per-stream tok/s | acceptance | accepted/draft | mean scheduled draft length | prune rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scheduler off | `319.37 ± 6.72` | `19.96` | `0.6233` | `3.116` | `5.000` | `0.0000` |
| hardware scheduler | `282.43 ± 31.52` | `17.65` | `0.6183` | `3.088` | `4.995` | `0.0010` |

Draft-length histogram over the three hardware-scheduler c16 runs:

- length 2: `2`
- length 3: `3`
- length 4: `3`
- length 5: `3014`

Interpretation:

- The current hardware scheduler still does not prune enough to help, even at
  c16.
- It regresses aggregate throughput by about `11.6%` versus scheduler-off.
- This means our current local SPS curve is not sufficient for c16 policy
  decisions. It only reaches B=48, while c16 full DSpark verification operates
  in the B=96 region.

## Conclusion

The screenshot brings a real insight, but it is mostly a serving-configuration
insight for our current branch:

- `MAX_NUM_SEQS=16` is viable on the current DSpark code.
- Static full-length DSpark verification scales to roughly `319 tok/s` aggregate
  at c16 on this TP=2 setup.
- We do not need to assume unknown external code changes to reproduce the
  screenshot's frontier result.
- The next useful scheduler work is not more policy tweaking against the old
  B<=48 curve. It is a c16 forced-length SPS profile to cover B=32,48,64,80,96,
  then a scheduler retest with that c16-grounded curve.

## Artifacts

- Scheduler-off JSON pattern:
  `experiments/dspark-benchmarks/concurrent_interactive_262k_window_c*_screenshot_scaling_verify_off_mseq16_*_20260629_064107_run*.json`
- Hardware-scheduler JSON pattern:
  `experiments/dspark-benchmarks/concurrent_interactive_262k_window_c16_screenshot_scaling_verify_hardware_mseq16_c16_20260629_065910_run*.json`
- Server env capture:
  `experiments/dspark-benchmarks/screenshot_scaling_verify_off_20260629_064107_server_env.txt`
  and
  `experiments/dspark-benchmarks/screenshot_scaling_verify_hardware_20260629_065910_server_env.txt`
