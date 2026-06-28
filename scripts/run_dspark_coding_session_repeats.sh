#!/usr/bin/env bash
# Repeat the DSpark realistic coding-session benchmark N times.
# Each session gets a unique cache_salt (KV isolation) and a warmup request.
# Generated JSON/stdout land under experiments/dspark-benchmarks/ (gitignored).
set -euo pipefail

label="${1:?usage: run_dspark_coding_session_repeats.sh <label> [runs]}"
runs="${2:-3}"

model_dir="${MODEL_DIR:-/home/pieter/.cache/huggingface-dspark/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7}"
model="${MODEL:-deepseek-v4-flash-dspark}"
base_url="${BASE_URL:-http://127.0.0.1:8000}"
out_dir="${OUT_DIR:-experiments/dspark-benchmarks}"
warmup_requests="${WARMUP_REQUESTS:-1}"
max_tokens="${MAX_TOKENS:-1024}"
timestamp="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${out_dir}"

for i in $(seq 1 "${runs}"); do
  out="${out_dir}/coding_session_${label}_${timestamp}_run${i}.json"
  run_marker="${label}-${timestamp}-run${i}"
  echo "=== ${label} coding-session run ${i}/${runs} -> ${out} ==="
  early_stop_args=()
  [ -n "${STOP_FILE:-}" ] && early_stop_args+=(--stop-file "${STOP_FILE}")
  [ -n "${EARLY_STOP_MIN_TURNS:-}" ] && early_stop_args+=(--early-stop-min-turns "${EARLY_STOP_MIN_TURNS}")
  [ -n "${EARLY_STOP_ACCEPTANCE_FLOOR:-}" ] && early_stop_args+=(--early-stop-acceptance-floor "${EARLY_STOP_ACCEPTANCE_FLOOR}")
  [ -n "${EARLY_STOP_MAX_CONTEXT:-}" ] && early_stop_args+=(--early-stop-max-context "${EARLY_STOP_MAX_CONTEXT}")
  uv run --with transformers --with sentencepiece --with protobuf \
    python scripts/dspark_coding_session_benchmark.py \
      --base-url "${base_url}" \
      --model "${model}" \
      --model-dir "${model_dir}" \
      --max-tokens "${max_tokens}" \
      --temperature 0.0 \
      --thinking false \
      --warmup-requests "${warmup_requests}" \
      --cache-salt "${run_marker}" \
      --output-json "${out}" \
      "${early_stop_args[@]}" \
    > "${out%.json}.stdout.txt" 2> "${out%.json}.stderr.txt"
  echo "    live progress: tail -f ${out%.json}.stderr.txt"
done

echo "=== summarizing ${runs} run(s) ==="
uv run --with transformers --with sentencepiece --with protobuf \
  python scripts/summarize_dspark_coding_session.py \
    --out-dir "${out_dir}" "${label}" --timestamp "${timestamp}" || true
