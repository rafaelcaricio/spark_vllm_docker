#!/usr/bin/env bash
set -euo pipefail

label="${1:-target_graph_profile}"
repo_dir="${REPO_DIR:-/home/pieter/Code/bjk110_spark-vllm-docker}"
worker_host="${WORKER_HOST:-192.168.250.13}"
env_file="${ENV_FILE:-.env.dspark-experiment}"
health_url="${HEALTH_URL:-http://127.0.0.1:8000/health}"
timestamp="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
profile_name="${label}_${timestamp}"
container_profile_dir="${CONTAINER_PROFILE_DIR:-/cache/huggingface/dspark-profiles/${profile_name}}"
host_profile_dir="${HOST_PROFILE_DIR:-${repo_dir}/experiments/dspark-benchmarks/profiles/${profile_name}}"
max_tokens="${MAX_TOKENS:-1024}"
prompt_tokens="${PROMPT_TOKENS:-512}"
warmup_requests="${WARMUP_REQUESTS:-1}"

compose_files=(
  "-f" "docker-compose.yml"
  "-f" "compose/docker-compose.dspark-experiment.yml"
)

profiler_json="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${container_profile_dir}\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"ignore_frontend\":true,\"delay_iterations\":${PROFILE_DELAY_ITERATIONS:-3},\"max_iterations\":${PROFILE_MAX_ITERATIONS:-24},\"warmup_iterations\":${PROFILE_WARMUP_ITERATIONS:-1},\"active_iterations\":${PROFILE_ACTIVE_ITERATIONS:-8}}"

server_env=(
  "VLLM_EXTRA_ARGS=--profiler-config=${profiler_json}"
  "VLLM_CUSTOM_SCOPES_FOR_PROFILING=${VLLM_CUSTOM_SCOPES_FOR_PROFILING:-1}"
  "VLLM_DSPARK_ITER_TIMING=${VLLM_DSPARK_ITER_TIMING:-0}"
  "VLLM_DSPARK_STAGE_TIMING=${VLLM_DSPARK_STAGE_TIMING:-0}"
  "VLLM_DSPARK_TARGET_TIMING=${VLLM_DSPARK_TARGET_TIMING:-0}"
)

bench_env=(
  "SCENARIO=${SCENARIO:-code_completion}"
  "PROMPT_TOKENS=${prompt_tokens}"
  "MAX_TOKENS=${max_tokens}"
  "THINKING=${THINKING:-false}"
  "STABLE_PROMPT=${STABLE_PROMPT:-1}"
  "WARMUP_REQUESTS=${warmup_requests}"
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
  (
    cd "${repo_dir}"
    docker compose --env-file "${env_file}" "${compose_files[@]}" \
      --profile head down --remove-orphans
  )
  run_remote "cd $(shell_quote "${repo_dir}") && $(compose_cmd) --profile worker down --remove-orphans"
}

start_stack() {
  local env_prefix
  env_prefix="$(quote_words "${server_env[@]}")"
  (
    cd "${repo_dir}"
    env "${server_env[@]}" docker compose --env-file "${env_file}" \
      "${compose_files[@]}" --profile head up -d
  )
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
  docker logs --tail 180 vllm-dspark-head >&2 || docker logs --tail 180 vllm-spark-head >&2 || true
  run_remote "docker logs --tail 180 vllm-dspark-worker || docker logs --tail 180 vllm-spark-worker" >&2 || true
  return 1
}

start_profile() {
  curl -fsS -X POST "${health_url%/health}/start_profile" >/dev/null
}

stop_profile() {
  curl -fsS -X POST "${health_url%/health}/stop_profile" >/dev/null
}

collect_profiles() {
  mkdir -p "${host_profile_dir}/head" "${host_profile_dir}/worker"
  docker cp "vllm-dspark-head:${container_profile_dir}/." "${host_profile_dir}/head/" \
    2>/dev/null || docker cp "vllm-spark-head:${container_profile_dir}/." "${host_profile_dir}/head/" \
    2>/dev/null || true
  run_remote "mkdir -p $(shell_quote "${host_profile_dir}/worker") && docker cp vllm-dspark-worker:$(shell_quote "${container_profile_dir}")/. $(shell_quote "${host_profile_dir}/worker")/ 2>/dev/null || docker cp vllm-spark-worker:$(shell_quote "${container_profile_dir}")/. $(shell_quote "${host_profile_dir}/worker")/ 2>/dev/null || true"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete \
      "${worker_host}:${host_profile_dir}/worker/" \
      "${host_profile_dir}/worker/" || true
  else
    scp -r "${worker_host}:${host_profile_dir}/worker/." \
      "${host_profile_dir}/worker/" || true
  fi
}

cd "${repo_dir}"
mkdir -p "${host_profile_dir}"

echo "=== restarting DSpark with torch profiler: ${profile_name} ==="
stop_stack
start_stack
wait_for_health

echo "=== warmup request before profiler ==="
env "${bench_env[@]}" MAX_TOKENS=64 \
  bash scripts/run_dspark_benchmark_repeats.sh "${label}_preprofile_warmup" 1

echo "=== start profiler ==="
start_profile

echo "=== profiled benchmark max_tokens=${max_tokens} ==="
env "${bench_env[@]}" TIMESTAMP="${timestamp}" \
  bash scripts/run_dspark_benchmark_repeats.sh "${label}_profiled" 1

echo "=== stop profiler ==="
stop_profile
collect_profiles

echo "=== profile output: ${host_profile_dir} ==="
find "${host_profile_dir}" -maxdepth 3 -type f | sort
