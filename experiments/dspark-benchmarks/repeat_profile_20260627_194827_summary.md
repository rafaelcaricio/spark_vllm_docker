# DSpark Single-Stream Repeatability Profile - 2026-06-27

Server:

- Endpoint: `http://127.0.0.1:8010`
- Served model: `deepseek-v4-flash-dspark`
- Advertised max model length: `262144`
- Request shape: single stream, chat completion, `prompt_tokens=512`,
  `max_tokens=256`, `temperature=0`, `ignore_eos=true`

Primary repeatability set:

- Prompt text was identical across runs.
- Each request used a different `cache_salt` so prefix-cache hits did not mask
  prefill/TTFC behavior.
- Each run generated exactly 256 server-counted tokens.

| Run | Prompt Tokens Local / Server | Decode Tok/s | TT First Byte | TT First Content | Decode Span | End-to-End | Drafts | Draft Tokens | Accepted | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `salted_repeat_20260627_194827_run1` | 488 / 492 | 35.07 | 0.417s | 4.107s | 7.272s | 11.379s | 159 | 795 | 96 | 12.08% |
| `salted_repeat_20260627_194827_run2` | 488 / 492 | 35.89 | 0.418s | 3.988s | 7.105s | 11.093s | 158 | 790 | 99 | 12.53% |
| `salted_repeat_20260627_194827_run3` | 488 / 492 | 41.97 | 0.417s | 3.501s | 6.075s | 9.577s | 136 | 680 | 124 | 18.24% |

Aggregate:

| Metric | Mean | Stddev | CV | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Server decode tok/s | 37.64 | 3.77 | 10.02% | 35.07 | 41.97 |
| Client decode tok/s after first content | 37.79 | 3.79 | 10.02% | 35.21 | 42.14 |
| Time to first byte | 0.417s | 0.001s | 0.16% | 0.417s | 0.418s |
| Time to first content | 3.866s | 0.321s | 8.30% | 3.501s | 4.107s |
| Decode span after first content | 6.817s | 0.648s | 9.50% | 6.075s | 7.272s |
| End-to-end latency | 10.683s | 0.968s | 9.07% | 9.577s | 11.379s |
| Draft acceptance | 14.28% | 3.43% | 24.03% | 12.08% | 18.24% |

Accepted tokens by draft position across the three salted runs:

| Draft Position | Accepted Tokens |
| ---: | ---: |
| 0 | 161 |
| 1 | 75 |
| 2 | 40 |
| 3 | 29 |
| 4 | 14 |

Comparison set:

- A prior six-run cache-busting set changed the prompt marker per run.
- It averaged 33.27 server decode tok/s with 17.90% CV and 13.58% acceptance
  with 32.82% CV.
- Because prompt text varied, use it as stress/noise evidence rather than the
  primary repeatability benchmark.

Grounded read:

- First-byte latency is stable; the server is not intermittently stalling before
  streaming starts.
- Decode throughput is repeatable enough for directional work, but still noisy
  enough that kernel changes should be compared over repeated runs.
- Acceptance rate varies more than raw decode speed. The next profiling pass
  should split DSpark draft time by stage and should also improve reference
  parity/draft quality, not only fuse kernels.
- Accepted tokens remain front-loaded by draft position, so the current
  five-token DSpark block is not being fully used.

Artifacts:

- `single_stream_interactive_262k_window_salted_repeat_20260627_194827_run1.json`
- `single_stream_interactive_262k_window_salted_repeat_20260627_194827_run2.json`
- `single_stream_interactive_262k_window_salted_repeat_20260627_194827_run3.json`
- Matching `.metrics.before.txt` and `.metrics.after.txt` files for each run.
