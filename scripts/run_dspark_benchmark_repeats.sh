#!/usr/bin/env bash
set -euo pipefail

label="${1:?usage: run_dspark_benchmark_repeats.sh <label> [runs]}"
runs="${2:-3}"

model_dir="${MODEL_DIR:-/home/pieter/.cache/huggingface-dspark/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7}"
model="${MODEL:-deepseek-v4-flash-dspark}"
base_url="${BASE_URL:-http://127.0.0.1:8000}"
out_dir="${OUT_DIR:-experiments/dspark-benchmarks}"
prompt_tokens="${PROMPT_TOKENS:-512}"
max_tokens="${MAX_TOKENS:-1024}"
temperature="${TEMPERATURE:-0.0}"
thinking="${THINKING:-default}"
endpoint="${ENDPOINT:-chat}"
scenario="${SCENARIO:-context_confirm}"
timestamp="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
stable_prompt="${STABLE_PROMPT:-1}"
prompt_suffix_base="${PROMPT_SUFFIX:-${label}-${timestamp}}"
warmup_requests="${WARMUP_REQUESTS:-1}"

mkdir -p "${out_dir}"

for i in $(seq 1 "${runs}"); do
  out="${out_dir}/single_stream_interactive_262k_window_${label}_${timestamp}_run${i}.json"
  run_marker="${label}-${timestamp}-run${i}"
  prompt_suffix="${prompt_suffix_base}"
  stable_prompt_args=()
  if [[ "${stable_prompt}" == "1" ]]; then
    stable_prompt_args+=(--stable-prompt)
  else
    prompt_suffix="${run_marker}"
  fi
  echo "=== ${label} run ${i}/${runs} -> ${out} ==="
  uv run --with transformers --with sentencepiece --with protobuf \
    python scripts/dspark_single_stream_benchmark.py \
      --base-url "${base_url}" \
      --model "${model}" \
      --model-dir "${model_dir}" \
      --prompt-tokens "${prompt_tokens}" \
      --max-tokens "${max_tokens}" \
      --temperature "${temperature}" \
      --endpoint "${endpoint}" \
      --scenario "${scenario}" \
      --thinking "${thinking}" \
      --ignore-eos \
      --prompt-suffix "${prompt_suffix}" \
      --cache-salt "${run_marker}" \
      --warmup-requests "${warmup_requests}" \
      "${stable_prompt_args[@]}" \
      --output-json "${out}" \
    > "${out%.json}.stdout.txt"
done
