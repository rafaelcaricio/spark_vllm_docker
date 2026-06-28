#!/usr/bin/env bash
# Run the standalone B12X W4A16 DSpark MoE microbench under Nsight Compute.
set -euo pipefail

image="${IMAGE:-vllm-dspark-runtime:clean}"
label="${LABEL:-b12x_moe_microbench_$(date +%Y%m%d_%H%M%S)}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${OUT_DIR:-${repo_root}/experiments/dspark-benchmarks/profiles/${label}}"
profile="${PROFILE:-1}"
kernel_regex="${KERNEL_REGEX:-.*W4A16FusedMoeKernel.*}"
launch_skip="${NCU_LAUNCH_SKIP:-8}"
launch_count="${NCU_LAUNCH_COUNT:-12}"
ncu_set="${NCU_SET:-speedOfLight}"
ncu_bin="${NCU_BIN:-/usr/local/cuda/bin/ncu}"
ncu_metrics="${NCU_METRICS:-}"
microbench_args="${MICROBENCH_ARGS:---m 6 --topk 6 --hidden-size 4096 --intermediate-size 2048 --num-experts 256 --warmup 8 --iterations 64}"

mkdir -p "${out_dir}"

if docker ps --format '{{.Names}}' | grep -qx 'vllm-spark-head'; then
  if [[ "${ALLOW_WITH_SERVER:-0}" != "1" && "${microbench_args}" != *"--dry-run"* && "${microbench_args}" != *"--tiny"* ]]; then
    echo "[dspark-ncu-moe] refusing full GPU microbench while vllm-spark-head is running." >&2
    echo "[dspark-ncu-moe] stop the DSpark server first, or set ALLOW_WITH_SERVER=1 for an intentional run." >&2
    exit 2
  fi
fi

docker_args=(
  run --rm
  --ipc=host
  --cap-add SYS_ADMIN
  --security-opt seccomp=unconfined
  -v /usr/local/cuda/bin:/usr/local/cuda/bin:ro
  -v /usr/local/cuda-13.0/bin:/usr/local/cuda-13.0/bin:ro
  -v /opt/nvidia/nsight-compute:/opt/nvidia/nsight-compute:ro
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro
  -v "${repo_root}/scripts:/workspace/scripts:ro"
  -v "${out_dir}:/profiles"
  -e PATH="/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/opt/env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  -e B12X_W4A16_TC_DECODE="${B12X_W4A16_TC_DECODE:-0}"
)

if [[ "${NO_GPUS:-0}" != "1" ]]; then
  docker_args+=(--gpus "${GPUS:-all}")
fi

cmd=(/opt/env/bin/python /workspace/scripts/dspark_b12x_moe_microbench.py)
# shellcheck disable=SC2206
cmd+=(${microbench_args})

if [[ "${profile}" == "1" ]]; then
  report="/profiles/${label}"
  ncu_collection_args=(--set "${ncu_set}" --section SchedulerStats --section WarpStateStats)
  if [[ -n "${ncu_metrics}" ]]; then
    ncu_collection_args=(--metrics "${ncu_metrics}")
  fi
  echo "[dspark-ncu-moe] image=${image}"
  echo "[dspark-ncu-moe] output=${out_dir}"
  echo "[dspark-ncu-moe] kernel_regex=${kernel_regex}"
  docker "${docker_args[@]}" "${image}" bash -lc "
    set -euo pipefail
    '${ncu_bin}' \
      --target-processes all \
      --kernel-name-base demangled \
      --kernel-name 'regex:${kernel_regex}' \
      --launch-skip '${launch_skip}' \
      --launch-count '${launch_count}' \
      ${ncu_collection_args[*]} \
      --clock-control none \
      --cache-control none \
      --print-summary per-kernel \
      --export '${report}' \
      --force-overwrite \
      ${cmd[*]} | tee '/profiles/${label}.stdout.txt'
  "
else
  docker "${docker_args[@]}" "${image}" "${cmd[@]}" | tee "${out_dir}/${label}.stdout.txt"
fi

echo "[dspark-ncu-moe] done: ${out_dir}"
