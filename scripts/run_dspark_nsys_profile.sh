#!/usr/bin/env bash
set -euo pipefail

label="${1:-draft_timeline_nsys}"
repo_dir="${REPO_DIR:-/home/pieter/Code/bjk110_spark-vllm-docker}"
worker_host="${WORKER_HOST:-192.168.250.13}"
env_file="${ENV_FILE:-.env.dspark-experiment}"
health_url="${HEALTH_URL:-http://127.0.0.1:8000/health}"
timestamp="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
profile_name="${label}_${timestamp}"
container_profile_dir="${CONTAINER_PROFILE_DIR:-/tmp/dspark-profiles/${profile_name}}"
host_profile_dir="${HOST_PROFILE_DIR:-${repo_dir}/experiments/dspark-benchmarks/profiles/${profile_name}}"
max_tokens="${MAX_TOKENS:-1024}"
prompt_tokens="${PROMPT_TOKENS:-512}"
warmup_requests="${WARMUP_REQUESTS:-1}"
restore_stack="${RESTORE_STACK:-1}"

compose_files=(
  "-f" "docker-compose.yml"
  "-f" "compose/docker-compose.dspark-experiment.yml"
)

nsys_compose_files=(
  "-f" "docker-compose.yml"
  "-f" "compose/docker-compose.dspark-experiment.yml"
  "-f" "compose/docker-compose.dspark-nsys.yml"
)

cuda_profiler_json="{\"profiler\":\"cuda\"}"

server_env=(
  "ENTRYPOINT_FILE=./entrypoints/entrypoint.dspark-nsys-launch.sh"
  "VLLM_EXTRA_ARGS=--profiler-config=${cuda_profiler_json}"
  "DSPARK_NSYS_LABEL=${profile_name}"
  "DSPARK_NSYS_OUT_DIR=${container_profile_dir}"
  "DSPARK_NSYS_LAUNCH=${DSPARK_NSYS_LAUNCH:-1}"
  "DSPARK_NSYS_LAUNCH_WORKER=${DSPARK_NSYS_LAUNCH_WORKER:-1}"
  "DSPARK_NSYS_ROLE=${DSPARK_NSYS_ROLE:-all}"
  "DSPARK_NSYS_CAPTURE_RANGE=${DSPARK_NSYS_CAPTURE_RANGE:-cudaProfilerApi}"
  "DSPARK_NSYS_CAPTURE_RANGE_END=${DSPARK_NSYS_CAPTURE_RANGE_END:-stop}"
  "DSPARK_NSYS_TRACE=${DSPARK_NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn}"
  "DSPARK_NSYS_CUDA_GRAPH_TRACE=${DSPARK_NSYS_CUDA_GRAPH_TRACE:-graph}"
  "DSPARK_NSYS_TRACE_FORK_BEFORE_EXEC=${DSPARK_NSYS_TRACE_FORK_BEFORE_EXEC:-true}"
  "DSPARK_NSYS_SAMPLE=${DSPARK_NSYS_SAMPLE:-none}"
  "DSPARK_NSYS_BACKTRACE=${DSPARK_NSYS_BACKTRACE:-none}"
  "DSPARK_NSYS_STATS=${DSPARK_NSYS_STATS:-true}"
  "DSPARK_NSYS_EXPORT=${DSPARK_NSYS_EXPORT:-sqlite}"
  "DSPARK_NSYS_PYTORCH=${DSPARK_NSYS_PYTORCH:-none}"
  "VLLM_CUSTOM_SCOPES_FOR_PROFILING=${VLLM_CUSTOM_SCOPES_FOR_PROFILING:-0}"
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
  local files=("$@")
  printf 'docker compose --env-file %s ' "$(shell_quote "${env_file}")"
  quote_words "${files[@]}"
}

stop_stack() {
  (
    cd "${repo_dir}"
    docker compose --env-file "${env_file}" "${nsys_compose_files[@]}" \
      --profile head down --remove-orphans
  )
  run_remote "cd $(shell_quote "${repo_dir}") && $(compose_cmd "${nsys_compose_files[@]}") --profile worker down --remove-orphans"
}

start_profiled_stack() {
  local env_prefix
  env_prefix="$(quote_words "${server_env[@]}")"
  (
    cd "${repo_dir}"
    env "${server_env[@]}" docker compose --env-file "${env_file}" \
      "${nsys_compose_files[@]}" --profile head up -d
  )
  run_remote "cd $(shell_quote "${repo_dir}") && env ${env_prefix} $(compose_cmd "${nsys_compose_files[@]}") --profile worker up -d"
}

start_normal_stack() {
  (
    cd "${repo_dir}"
    docker compose --env-file "${env_file}" "${compose_files[@]}" \
      --profile head up -d
  )
  run_remote "cd $(shell_quote "${repo_dir}") && $(compose_cmd "${compose_files[@]}") --profile worker up -d"
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

finalize_profiled_containers() {
  local timeout="${NSYS_STOP_TIMEOUT_SECONDS:-120}"
  docker stop -t "${timeout}" vllm-dspark-head >/dev/null 2>&1 || true
  run_remote "docker stop -t $(shell_quote "${timeout}") vllm-dspark-worker >/dev/null 2>&1 || true"
}

collect_profiles() {
  mkdir -p "${host_profile_dir}/head" "${host_profile_dir}/worker"
  docker cp "vllm-dspark-head:${container_profile_dir}/." "${host_profile_dir}/head/" \
    2>/dev/null || true
  run_remote "mkdir -p $(shell_quote "${host_profile_dir}/worker") && docker cp vllm-dspark-worker:$(shell_quote "${container_profile_dir}")/. $(shell_quote "${host_profile_dir}/worker")/ 2>/dev/null || true"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete \
      "${worker_host}:${host_profile_dir}/worker/" \
      "${host_profile_dir}/worker/" || true
  else
    scp -r "${worker_host}:${host_profile_dir}/worker/." \
      "${host_profile_dir}/worker/" || true
  fi
}

sync_worker_profile_files() {
  rsync -a \
    compose/docker-compose.dspark-nsys.yml \
    "${worker_host}:${repo_dir}/compose/docker-compose.dspark-nsys.yml"
  rsync -a \
    entrypoints/entrypoint.dspark-nsys-launch.sh \
    "${worker_host}:${repo_dir}/entrypoints/entrypoint.dspark-nsys-launch.sh"
}

restore_normal_stack() {
  if [ "${restore_stack}" != "1" ]; then
    return 0
  fi
  echo "=== restoring normal non-nsys DSpark stack ==="
  stop_stack || true
  start_normal_stack
  wait_for_health
}

on_exit() {
  local rc=$?
  restore_normal_stack || true
  exit "${rc}"
}

cd "${repo_dir}"
mkdir -p "${host_profile_dir}"
sync_worker_profile_files

trap on_exit EXIT

echo "=== restarting DSpark under Nsight Systems: ${profile_name} ==="
stop_stack
start_profiled_stack
wait_for_health

echo "=== warmup request before nsys capture ==="
env "${bench_env[@]}" MAX_TOKENS=64 \
  bash scripts/run_dspark_benchmark_repeats.sh "${label}_prensys_warmup" 1

echo "=== start CUDA profiler capture range ==="
start_profile

echo "=== profiled benchmark max_tokens=${max_tokens} ==="
env "${bench_env[@]}" TIMESTAMP="${timestamp}" \
  bash scripts/run_dspark_benchmark_repeats.sh "${label}_nsys_profiled" 1

echo "=== stop CUDA profiler capture range ==="
stop_profile || true

finalize_profiled_containers
collect_profiles

echo "=== nsys output: ${host_profile_dir} ==="
find "${host_profile_dir}" -maxdepth 3 -type f | sort
