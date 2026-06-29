#!/usr/bin/env bash
# Attach Nsight Compute to a DSpark server launched with
# entrypoint.dspark-ncu-launch.sh and collect a bounded B12X W4A16 MoE sample.
set -euo pipefail

container="${CONTAINER:-vllm-spark-head}"
port="${DSPARK_NCU_PORT:-49152}"
label="${LABEL:-dspark_ncu_moe_$(date +%Y%m%d_%H%M%S)}"
out_dir="${OUT_DIR_CONTAINER:-/cache/huggingface/dspark-profiles/${label}}"
out_base="${out_dir}/${label}"
kernel_regex="${KERNEL_REGEX:-.*b12x.*w4a16.*}"
launch_count="${LAUNCH_COUNT:-8}"
ncu_set="${NCU_SET:-speedOfLight}"

echo "[dspark-ncu] attaching to ${container} on port ${port}"
echo "[dspark-ncu] kernel regex: ${kernel_regex}"
echo "[dspark-ncu] output base: ${out_base}"

docker exec \
  -e DSPARK_NCU_ATTACH_OUT_DIR="${out_dir}" \
  -e DSPARK_NCU_ATTACH_OUT_BASE="${out_base}" \
  -e DSPARK_NCU_ATTACH_KERNEL_REGEX="${kernel_regex}" \
  -e DSPARK_NCU_ATTACH_LAUNCH_COUNT="${launch_count}" \
  -e DSPARK_NCU_ATTACH_PORT="${port}" \
  -e DSPARK_NCU_ATTACH_SET="${ncu_set}" \
  "${container}" \
  bash -lc '
    set -euo pipefail
    export PATH="/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:${PATH:-}"
    mkdir -p "${DSPARK_NCU_ATTACH_OUT_DIR}"
    ncu \
      --mode=attach \
      --hostname 127.0.0.1 \
      --port "${DSPARK_NCU_ATTACH_PORT}" \
      --kernel-name-base demangled \
      --kernel-name "regex:${DSPARK_NCU_ATTACH_KERNEL_REGEX}" \
      --launch-count "${DSPARK_NCU_ATTACH_LAUNCH_COUNT}" \
      --set "${DSPARK_NCU_ATTACH_SET}" \
      --clock-control none \
      --cache-control none \
      --print-summary per-kernel \
      --export "${DSPARK_NCU_ATTACH_OUT_BASE}" \
      --force-overwrite
  '

echo "[dspark-ncu] done; report should be under ${out_base}.ncu-rep"
