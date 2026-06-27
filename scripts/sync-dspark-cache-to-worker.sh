#!/usr/bin/env bash
set -euo pipefail

WORKER_HOST="${WORKER_HOST:-192.168.250.13}"
CACHE_DIR="${CACHE_DIR:-/home/pieter/.cache/huggingface-dspark}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-DSpark}"
DOWNLOAD_LOG="${DOWNLOAD_LOG:-/home/pieter/Code/DeepSpec/logs/deepseek-v4-flash-dspark.download.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"

echo "$(date -Is) waiting for ${MODEL_ID} download to finish"
while ! grep -q "HF_DOWNLOAD_EXIT_CODE=" "${DOWNLOAD_LOG}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

if ! grep -q "HF_DOWNLOAD_EXIT_CODE=0" "${DOWNLOAD_LOG}"; then
  echo "$(date -Is) not copying: download failed or was interrupted" >&2
  tail -n 80 "${DOWNLOAD_LOG}" >&2 || true
  exit 1
fi

echo "$(date -Is) verifying local cache ${CACHE_DIR}"
hf cache verify "${MODEL_ID}" \
  --cache-dir "${CACHE_DIR}" \
  --fail-on-missing-files \
  --format human

echo "$(date -Is) creating worker cache directory on ${WORKER_HOST}"
ssh -o BatchMode=yes "${WORKER_HOST}" "mkdir -p '${CACHE_DIR}'"

echo "$(date -Is) syncing ${CACHE_DIR}/ to ${WORKER_HOST}:${CACHE_DIR}/"
rsync -aH --info=progress2 \
  --exclude=.locks/ \
  --exclude="*.incomplete" \
  "${CACHE_DIR}/" \
  "${WORKER_HOST}:${CACHE_DIR}/"

echo "$(date -Is) worker copy complete"
ssh -o BatchMode=yes "${WORKER_HOST}" \
  "du -sh '${CACHE_DIR}' && find '${CACHE_DIR}/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots' -maxdepth 2 \\( -name config.json -o -name model.safetensors.index.json \\) -print 2>/dev/null"
