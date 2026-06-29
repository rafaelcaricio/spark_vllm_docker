#!/usr/bin/env bash
set -euo pipefail

label_prefix="${1:-hardware_sps_profile}"
runs="${RUNS:-1}"
profile_points="${PROFILE_POINTS:-4:0 4:1 4:2 4:3 4:4 4:5 8:3 8:4 8:5}"
repo_dir="${REPO_DIR:-/home/pieter/Code/bjk110_spark-vllm-docker}"
worker_host="${WORKER_HOST:-192.168.250.13}"
env_file="${ENV_FILE:-.env.dspark-experiment}"
compose_files=(
  "-f" "docker-compose.yml"
  "-f" "compose/docker-compose.dspark-experiment.yml"
)
health_url="${HEALTH_URL:-http://127.0.0.1:8000/health}"
start_worker_delay="${START_WORKER_DELAY:-20}"
timestamp="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

bench_env_base=(
  "SCENARIO=${SCENARIO:-code_completion}"
  "PROMPT_TOKENS=${PROMPT_TOKENS:-512}"
  "MAX_TOKENS=${MAX_TOKENS:-256}"
  "THINKING=${THINKING:-false}"
  "STABLE_PROMPT=${STABLE_PROMPT:-1}"
  "WARMUP_BATCHES=${WARMUP_BATCHES:-1}"
)

server_env_base=(
  "VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0"
  "VLLM_DSPARK_CONFIDENCE_SCHEDULER=hardware"
  "VLLM_DSPARK_SPS_CURVE="
  "VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=${VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP:-0}"
  "VLLM_DSPARK_POSITION0_DIAGNOSTICS=${VLLM_DSPARK_POSITION0_DIAGNOSTICS:-0}"
  "VLLM_DSPARK_STAGE_TIMING=${VLLM_DSPARK_STAGE_TIMING:-0}"
)

quote_words() {
  local out=()
  local word
  for word in "$@"; do
    out+=("$(printf '%q' "${word}")")
  done
  printf '%s ' "${out[@]}"
}

run_remote() {
  local cmd="$1"
  ssh -o BatchMode=yes "${worker_host}" "bash -lc $(printf '%q' "${cmd}")"
}

compose_cmd() {
  printf 'docker compose --env-file %q ' "${env_file}"
  quote_words "${compose_files[@]}"
}

stop_stack() {
  run_remote "cd $(printf '%q' "${repo_dir}") && $(compose_cmd) --profile worker down --remove-orphans"
  (cd "${repo_dir}" && $(compose_cmd) --profile head down --remove-orphans)
}

start_stack_for_length() {
  local length="$1"
  local env_words=("${server_env_base[@]}" "VLLM_DSPARK_FORCE_DRAFT_LENGTH=${length}")
  local env_prefix
  env_prefix="$(quote_words "${env_words[@]}")"

  (cd "${repo_dir}" && env ${env_prefix} $(compose_cmd) --profile head up -d)
  sleep "${start_worker_delay}"
  run_remote "cd $(printf '%q' "${repo_dir}") && env ${env_prefix} $(compose_cmd) --profile worker up -d"
}

wait_for_health() {
  local deadline=$((SECONDS + ${HEALTH_TIMEOUT_SECONDS:-900}))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "${health_url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  docker logs --tail 160 vllm-dspark-head >&2 || true
  run_remote "docker logs --tail 160 vllm-dspark-worker" >&2 || true
  return 1
}

run_point() {
  local concurrency="$1"
  local length="$2"
  local label="${label_prefix}_c${concurrency}_len${length}_${timestamp}"

  echo "=== DSpark SPS profile c=${concurrency} forced_len=${length} ==="
  stop_stack
  start_stack_for_length "${length}"
  wait_for_health

  env "${bench_env_base[@]}" CONCURRENCY="${concurrency}" TIMESTAMP="${timestamp}" \
    bash scripts/run_dsv4_concurrent_benchmark_repeats.sh "${label}" "${runs}"
}

cd "${repo_dir}"

for point in ${profile_points}; do
  IFS=: read -r concurrency length <<<"${point}"
  if [[ -z "${concurrency}" || -z "${length}" ]]; then
    echo "Invalid PROFILE_POINTS entry: ${point}" >&2
    exit 1
  fi
  run_point "${concurrency}" "${length}"
done

curve_json="experiments/dspark-benchmarks/${label_prefix}_${timestamp}_sps_curve.json"
uv run python scripts/build_dspark_sps_curve.py \
  "concurrent_interactive_262k_window_c*_${label_prefix}_*_len*_$(printf '%s' "${timestamp}")_run*.json" \
  --output-json "${curve_json}"

echo "=== SPS curve written to ${curve_json} ==="
