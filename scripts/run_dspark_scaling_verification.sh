#!/usr/bin/env bash
set -euo pipefail

label_prefix="${1:-c16_scaling_verify}"
runs="${RUNS:-3}"
concurrencies="${CONCURRENCIES:-1 2 4 8 16}"
repo_dir="${REPO_DIR:-/home/pieter/Code/bjk110_spark-vllm-docker}"
worker_host="${WORKER_HOST:-192.168.250.13}"
env_file="${ENV_FILE:-.env.dspark-experiment}"
health_url="${HEALTH_URL:-http://127.0.0.1:8000/health}"
start_worker_delay="${START_WORKER_DELAY:-20}"
timestamp="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
max_num_seqs="${MAX_NUM_SEQS:-16}"
scheduler="${VLLM_DSPARK_CONFIDENCE_SCHEDULER:-off}"
sps_curve="${VLLM_DSPARK_SPS_CURVE:-}"
early_stop="${VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP:-1}"
start_only="${START_ONLY:-0}"

compose_files=(
  "-f" "docker-compose.yml"
  "-f" "compose/docker-compose.dspark-experiment.yml"
)

bench_env_base=(
  "SCENARIO=${SCENARIO:-code_completion}"
  "PROMPT_TOKENS=${PROMPT_TOKENS:-512}"
  "MAX_TOKENS=${MAX_TOKENS:-256}"
  "THINKING=${THINKING:-false}"
  "STABLE_PROMPT=${STABLE_PROMPT:-1}"
  "WARMUP_BATCHES=${WARMUP_BATCHES:-1}"
)

server_env_base=(
  "MAX_NUM_SEQS=${max_num_seqs}"
  "VLLM_DSPARK_CONFIDENCE_THRESHOLD=${VLLM_DSPARK_CONFIDENCE_THRESHOLD:-0.0}"
  "VLLM_DSPARK_CONFIDENCE_SCHEDULER=${scheduler}"
  "VLLM_DSPARK_SPS_CURVE=${sps_curve}"
  "VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=${early_stop}"
  "VLLM_DSPARK_FORCE_DRAFT_LENGTH=${VLLM_DSPARK_FORCE_DRAFT_LENGTH:-}"
  "VLLM_DSPARK_STS_TEMPERATURES=${VLLM_DSPARK_STS_TEMPERATURES:-}"
  "VLLM_DSPARK_COLLECT_CONFIDENCE_DIAGNOSTICS=${VLLM_DSPARK_COLLECT_CONFIDENCE_DIAGNOSTICS:-0}"
  "VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY=${VLLM_DSPARK_CONFIDENCE_DIAGNOSTICS_LOG_EVERY:-0}"
  "VLLM_DSPARK_POSITION0_DIAGNOSTICS=${VLLM_DSPARK_POSITION0_DIAGNOSTICS:-0}"
  "VLLM_DSPARK_STAGE_TIMING=${VLLM_DSPARK_STAGE_TIMING:-0}"
)

shell_quote() {
  local value="$1"
  printf "'%s'" "${value//\'/\'\\\'\'}"
}

quote_words() {
  local out=()
  local word
  for word in "$@"; do
    out+=("$(shell_quote "${word}")")
  done
  printf '%s ' "${out[@]}"
}

run_remote() {
  local cmd="$1"
  ssh -o BatchMode=yes "${worker_host}" "bash -lc $(printf '%q' "${cmd}")"
}

compose_cmd() {
  printf 'docker compose --env-file %s ' "$(shell_quote "${env_file}")"
  quote_words "${compose_files[@]}"
}

stop_stack() {
  run_remote "cd $(shell_quote "${repo_dir}") && $(compose_cmd) --profile worker down --remove-orphans"
  (
    cd "${repo_dir}"
    docker compose --env-file "${env_file}" "${compose_files[@]}" \
      --profile head down --remove-orphans
  )
}

start_stack() {
  local env_prefix
  env_prefix="$(quote_words "${server_env_base[@]}")"

  (
    cd "${repo_dir}"
    env "${server_env_base[@]}" docker compose --env-file "${env_file}" \
      "${compose_files[@]}" --profile head up -d
  )
  sleep "${start_worker_delay}"
  run_remote "cd $(shell_quote "${repo_dir}") && env ${env_prefix} $(compose_cmd) --profile worker up -d"
}

wait_for_health() {
  local deadline=$((SECONDS + ${HEALTH_TIMEOUT_SECONDS:-900}))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "${health_url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  docker logs --tail 180 vllm-dspark-head >&2 || true
  run_remote "docker logs --tail 180 vllm-dspark-worker" >&2 || true
  return 1
}

run_sweep() {
  local concurrency
  for concurrency in ${concurrencies}; do
    local label="${label_prefix}_${scheduler}_mseq${max_num_seqs}_c${concurrency}"
    echo "=== scaling verification ${label} runs=${runs} ==="
    env "${bench_env_base[@]}" CONCURRENCY="${concurrency}" TIMESTAMP="${timestamp}" \
      bash scripts/run_dsv4_concurrent_benchmark_repeats.sh "${label}" "${runs}"
    uv run python scripts/summarize_dspark_benchmarks.py \
      "concurrent_interactive_262k_window_c${concurrency}_${label}_${timestamp}_run*.json"
  done
}

cd "${repo_dir}"

echo "=== restarting DSpark stack: MAX_NUM_SEQS=${max_num_seqs} scheduler=${scheduler} ==="
stop_stack
start_stack
wait_for_health

docker inspect vllm-dspark-head --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | sort \
  | grep -E 'MAX_NUM_SEQS|MAX_MODEL_LEN|MAX_NUM_BATCHED|DSPARK|B12X|SERVED_MODEL_NAME' \
  > "experiments/dspark-benchmarks/${label_prefix}_${scheduler}_${timestamp}_server_env.txt"

if [[ "${start_only}" == "1" ]]; then
  echo "=== DSpark stack started; START_ONLY=1 ==="
  exit 0
fi

run_sweep

echo "=== scaling verification complete: label=${label_prefix} timestamp=${timestamp} ==="
