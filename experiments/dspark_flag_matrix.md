# DSpark Flag Matrix

Purpose: keep one canonical DSpark serving lane and prevent every experiment
from becoming a permanent configuration dimension.

## Canonical 262k Speed Lane

Use `presets/dspark-v4-flash-262k-canonical.env` for normal 262k DSpark speed
work. It keeps only the flags that define the currently validated path:

- `MTP_NUM_TOKENS=5`: released DSpark gamma/block size.
- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`: single-stream speed lane verifies the
  full block; confidence scheduling remains a separate concurrency experiment.
- `VLLM_DSPARK_LOCAL_ARGMAX=1`: settled faster draft token selection path.
- `VLLM_DSPARK_REPLICATE_MARKOV_W1=1`: settled draft-side Markov W1 placement.
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`: keep rejected-context masking on
  GPU.
- `VLLM_DSPARK_DRAFT_STREAM=1`: default DSpark evolution lane. The current
  implementation is draft-stream/deferred-fence bookkeeping overlap, not full
  draft/verify overlap. Keep it in the experiment lane because it is the hook
  for earlier launch points, stream/event fencing, and optimistic
  double-buffered drafts, but do not call it production-stable until async
  lifecycle and CUDA graph stream-safety validation pass.
- `VLLM_TRITON_MLA_SPARSE=1`: current sparse MLA lane.
- `VLLM_USE_B12X_MOE=1`: current target MoE kernel path.
- `VLLM_USE_B12X_WO_PROJECTION=1`: best corrected verifier projection path.

Mixed/condensed request shapes use the compact ragged DSpark path by default;
the old ragged/pad compatibility switch was removed.

## Active Experiment Knobs

These should be shell overrides or short-lived benchmark lanes, not default
preset entries:

- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=hardware`,
  `VLLM_DSPARK_SPS_CURVE=...`,
  `VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=...`: concurrency scheduler lane.
- `VLLM_DSPARK_FORCE_DRAFT_LENGTH=N`: forced-length curve and paper-alignment
  tests.
- `VLLM_DSPARK_STS_TEMPERATURES=...`: only after fitting from diagnostic data.

## Diagnostics

Diagnostics stay off in speed presets and should be enabled for bounded runs:

- `VLLM_DSPARK_STAGE_TIMING`
- `VLLM_DSPARK_ITER_TIMING`
- `VLLM_DSPARK_TARGET_TIMING`
- `VLLM_DSPARK_COLLECT_CONFIDENCE_DIAGNOSTICS`
- `VLLM_DSPARK_STS_CALIBRATION_DIAGNOSTICS`
- `VLLM_DSPARK_POSITION0_DIAGNOSTICS`
- `VLLM_CUSTOM_SCOPES_FOR_PROFILING`

## Settled Off

Keep these available for historical reproduction or targeted debugging, but do
not include them in the canonical preset:

- `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0`: tested and lost to the vendor
  projection path.
- `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0`: tested neutral for quality.
- `VLLM_DSV4_B12X_COMPRESSED_MLA=0`: broke verifier numerics/acceptance.
- `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0` and
  `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT=0`: changed captured features
  and collapsed acceptance.
- `VLLM_USE_B12X_FP8_GEMM=0`, `VLLM_USE_B12X_MHC=0`,
  `VLLM_USE_B12X_SPARSE_INDEXER=0`: not promoted for the current DSpark lane.
- `B12X_W4A16_TC_DECODE=0`: regressed in testing.
- `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM`,
  `VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M`,
  `VLLM_B12X_W4A16_FORCE_TILE_CONFIG`: tile/selector overrides belong only in
  dedicated NCU experiments.

## Long-Context Lanes

Keep 1M/NVFP4 capacity work isolated in its existing presets:

- `presets/dspark-v4-flash-1m-nvfp4-mseq1.env`
- `presets/dspark-v4-flash-1m-nvfp4-mseq6.env`
- `presets/dspark-v4-flash-200k-nvfp4-mseq16.env`

Those profiles validate capacity and concurrency envelopes; they should not
silently change the 262k fp8 speed lane.

They still default `VLLM_DSPARK_DRAFT_STREAM=1` so pipeline-scaffold work
evolves consistently across the DSpark lanes. Override it to `0` only for
explicit A/B control runs.
