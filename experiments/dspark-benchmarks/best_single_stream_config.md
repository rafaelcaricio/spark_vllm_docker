# DSpark Best Single-Stream Decode Config — Runbook + Validated Numbers

Date: 2026-06-28
Status: **validated, reproducible, server left running on this config.**

## Best config (the highest single-stream decode speed found)

Served model: `deepseek-v4-flash-dspark` (DeepSeek-V4-Flash DSpark, γ=5, Markov head).
Runtime image: `vllm-dspark-runtime:clean` (clean overlay = warmup2 route-pack patch + DSpark
target-forward/dspark source overlay; **default B12X W4A16 MoE selector**, no experimental patches).
Topology: TP=2 across 2× DGX Spark (head `192.168.250.12`, worker `192.168.250.13`), `mp` backend,
`MAX_NUM_SEQS=1`, `MAX_NUM_BATCHED_TOKENS=8192`, `max_model_len=262144`, `kv_cache_dtype=fp8`,
`enforce_eager=False` (FULL cudagraph at decode).

Decisive env (`.env.dspark-experiment`):

```
MTP_NUM_TOKENS=5                              # gamma=5 (the released DSpark-5 block)
VLLM_USE_B12X_MOE=1                           # B12X W4A16 MoE
VLLM_USE_B12X_WO_PROJECTION=1                 # verifier output-projection opt (best corrected baseline)
VLLM_DSPARK_CONFIDENCE_SCHEDULER=off          # scheduler is concurrency-only; inert single-stream
VLLM_DSPARK_LOCAL_ARGMAX=1                    # vocab-parallel local argmax draft path
VLLM_DSPARK_REPLICATE_MARKOV_W1=1             # replicated Markov W1
VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0             # fused Markov lost to vendor projection
VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0      # KVQ neutral on quality (tested)
VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=0         # MoE selector override OFF (blocks_per_sm-only shmem-hangs; tile changes collapse acceptance)
VLLM_B12X_W4A16_FORCE_TILE_CONFIG=            # empty (default selector)
B12X_W4A16_TC_DECODE=0                        # regressed
VLLM_DSV4_B12X_COMPRESSED_MLA=0              # broke verifier numerics (0.97% acceptance)
VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0       # changed captured features -> acceptance collapse
VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT=0
VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1   # default (scheduler off anyway)
```

## Reproducibility (validated 2026-06-28)

Benchmark: `scripts/run_dspark_benchmark_repeats.sh`, `SCENARIO=code_completion`,
`PROMPT_TOKENS=512`, `MAX_TOKENS=256`, `temperature=0.0`, `thinking=false`, `ignore_eos`,
`WARMUP_REQUESTS=1`, unique `cache_salt` per run, single stream.

5-run result (server-decode tok/s via `/metrics` `generation_tokens` delta):

| run | tok/s | accept | acc/draft | TTFC |
| --- | ---: | ---: | ---: | ---: |
| 1 | 58.35 | 0.613 | 3.06 | 0.48s |
| 2 | 60.43 | 0.663 | 3.32 | 0.48s |
| 3 | 60.67 | 0.663 | 3.32 | 0.43s |
| 4 | 59.68 | 0.649 | 3.25 | 0.48s |
| 5 | 62.42 | 0.690 | 3.45 | 0.42s |

**Aggregate: 60.31 tok/s mean, stdev 1.33, CV 2.2%** (range 58.35–62.42).
- Acceptance 0.66, accepted/draft 3.28 → **τ ≈ 4.28**.
- TTFC 0.46s. Cycle ≈ τ/tps ≈ **70.9 ms**.
- **1.51× MTP-1** (39.88 tok/s), **2.29× no-spec** (26.33 tok/s).

## How to run / reproduce

```bash
# server (already up on this config; to restart head-first, then worker, absolute paths):
ENV=/home/pieter/Code/bjk110_spark-vllm-docker/.env.dspark-experiment
DC=/home/pieter/Code/bjk110_spark-vllm-docker/docker-compose.yml
docker compose --env-file "$ENV" -f "$DC" --profile head up -d      # master first
sleep 25
ssh 192.168.250.13 "docker compose --env-file $ENV -f $DC --profile worker up -d"
# poll until /health -> 200 (~5 min: model load + graph capture)

# reproducibility benchmark (5 warmed code_completion repeats):
SCENARIO=code_completion THINKING=false \
  bash scripts/run_dspark_benchmark_repeats.sh best_repro 5
uv run python scripts/summarize_dspark_benchmarks.py best_repro
```

## Settled this session (why this is the ceiling for in-scope single-stream)

Every algorithmic single-stream lever was tested and accounted for:

- **Forward (Tverify): flat ~70.9 ms to 80k-token context** — sparse-MLA is windowed; no large-context
  forward slowdown. MoE ~43% (tapped: W4A16 tile changes collapse acceptance; `blocks_per_sm`-only
  shmem-hangs; b12x selector frozen). Dense ~30% (vendor). NCCL ~5–7% (latency-bound; FlashInfer
  allreduce failed-closed on SM121/ws=2).
- **τ: at true ceiling.** Draft path audited paper-faithful (parallel backbone base logits + rank-256
  Markov bias on previous sampled token + target head reuse). pos0 stable ~0.88 across context;
  suffix decay on hard/novel content is **inherent** to the released DSpark-5, not a parity bug.
- **Confidence pruning: dead.** γ=3 at clean FULL@4 (uniform_decode_query_len=4) = **56.6 tok/s,
  −6% vs γ=5** — cycle drops 70.9→61.2 ms (−13.7%) but τ drops 4.28→3.47 (−18.7%) faster. Matches
  the paper: light load → verify the full block. (The old forced-length curve was cudagraph-contaminated;
  this is the clean measurement.)
- **NCCL/compute overlap:** existing DBO / shared-expert-stream / multi-stream-GEMM infra is gated
  off *by design* below batch 32/256/1024; single-stream decode is batch 6.
- **KVQ, fused-Markov, compressed-MLA, deferred-capture, TC_DECODE:** all neutral or regressing
  (kept flag-gated, off).

**Realistic coding-session note:** the controlled `code_completion` gate (~60 tok/s, τ≈4.3) is the
*easy-content* ceiling. A real coding session runs ~40 tok/s / τ≈3.0 because novel reasoning is harder
to draft (suffix collapse); see the `detail` corpus harness (`scripts/dspark_coding_session_*`).

## Next lever (out of current scope, evidence-supported)

Single-stream in-scope work is exhausted. The only remaining lever that still improves single-stream
coding decode is **draft/verify overlap** (hide Tdraft ~8.6 ms behind Tverify; lossless; stays
uniform/FULL-safe at γ=5): expected cycle 70.9→~61 ms → ~+12% tok/s, gated on the `detail` harness
with acceptance/τ preserved.
