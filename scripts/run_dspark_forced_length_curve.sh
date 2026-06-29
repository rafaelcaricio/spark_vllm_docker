#!/usr/bin/env bash
set -euo pipefail

label_prefix="${1:-paper_sps_curve}"
runs="${RUNS:-3}"
lengths="${LENGTHS:-0 1 2 3 4 5}"
repo_dir="${REPO_DIR:-/home/pieter/Code/bjk110_spark-vllm-docker}"
worker_host="${WORKER_HOST:-192.168.250.13}"
env_file="${ENV_FILE:-.env.dspark-experiment}"
worker_session="${WORKER_SESSION:-dspark-worker}"
head_session="${HEAD_SESSION:-dspark-head}"
health_url="${HEALTH_URL:-http://127.0.0.1:8000/health}"
bench_env=(
  "SCENARIO=${SCENARIO:-code_completion}"
  "PROMPT_TOKENS=${PROMPT_TOKENS:-512}"
  "MAX_TOKENS=${MAX_TOKENS:-1024}"
  "THINKING=${THINKING:-false}"
  "STABLE_PROMPT=${STABLE_PROMPT:-1}"
)

server_env_base=(
  "VLLM_DSPARK_CONFIDENCE_THRESHOLD=${VLLM_DSPARK_CONFIDENCE_THRESHOLD:-0.0}"
  "VLLM_DSPARK_CONFIDENCE_SCHEDULER=${VLLM_DSPARK_CONFIDENCE_SCHEDULER:-off}"
  "VLLM_DSPARK_POSITION0_DIAGNOSTICS=${VLLM_DSPARK_POSITION0_DIAGNOSTICS:-0}"
  "VLLM_DSPARK_STAGE_TIMING=${VLLM_DSPARK_STAGE_TIMING:-0}"
  "VLLM_DSV4_B12X_COMPRESSED_MLA=${VLLM_DSV4_B12X_COMPRESSED_MLA:-0}"
  "VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=${VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE:-0}"
  "VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT=${VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT:-0}"
  "VLLM_USE_B12X_WO_PROJECTION=${VLLM_USE_B12X_WO_PROJECTION:-0}"
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

stop_stack() {
  docker compose --env-file "${env_file}" --profile head down --remove-orphans
  run_remote "cd $(printf '%q' "${repo_dir}") && docker compose --env-file $(printf '%q' "${env_file}") --profile worker down --remove-orphans"
}

start_stack_for_length() {
  local length="$1"
  local env_words=("${server_env_base[@]}" "VLLM_DSPARK_FORCE_DRAFT_LENGTH=${length}")
  local env_prefix
  env_prefix="$(quote_words "${env_words[@]}")"

  run_remote "cd $(printf '%q' "${repo_dir}") && (tmux has-session -t $(printf '%q' "${worker_session}") 2>/dev/null && tmux kill-session -t $(printf '%q' "${worker_session}") || true) && tmux new-session -d -s $(printf '%q' "${worker_session}") $(printf '%q' "cd ${repo_dir} && env ${env_prefix} docker compose --env-file ${env_file} --profile worker up 2>&1 | tee /tmp/${worker_session}.log")"

  if tmux has-session -t "${head_session}" 2>/dev/null; then
    tmux kill-session -t "${head_session}"
  fi
  tmux new-session -d -s "${head_session}" \
    "cd ${repo_dir} && env ${env_prefix} docker compose --env-file ${env_file} --profile head up 2>&1 | tee /tmp/${head_session}.log"
}

wait_for_health() {
  local deadline=$((SECONDS + ${HEALTH_TIMEOUT_SECONDS:-900}))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "${health_url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  docker logs --tail 120 vllm-spark-head >&2 || true
  run_remote "docker logs --tail 120 vllm-spark-worker" >&2 || true
  return 1
}

run_benchmarks_for_length() {
  local length="$1"
  local label="${label_prefix}_len${length}"

  echo "=== warmup forced length ${length} ==="
  env "${bench_env[@]}" MAX_TOKENS=64 \
    bash scripts/run_dspark_benchmark_repeats.sh "${label}_warmup" 1

  echo "=== benchmark forced length ${length} (${runs} runs) ==="
  env "${bench_env[@]}" \
    bash scripts/run_dspark_benchmark_repeats.sh "${label}" "${runs}"

  uv run python scripts/summarize_dspark_benchmarks.py \
    "single_stream_interactive_262k_window_${label}_[0-9]*_run*.json"
}

cd "${repo_dir}"

for length in ${lengths}; do
  echo "=== DSpark forced verification length ${length} ==="
  stop_stack
  start_stack_for_length "${length}"
  wait_for_health
  run_benchmarks_for_length "${length}"
done

echo "=== forced-length curve complete ==="
