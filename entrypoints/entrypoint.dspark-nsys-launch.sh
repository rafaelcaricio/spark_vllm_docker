#!/usr/bin/env bash
# Experimental DSpark entrypoint with optional Nsight Systems launch support.
#
# This mirrors entrypoint.dspark.sh, then optionally wraps the selected role in
# `nsys profile`. The intended capture trigger is vLLM's CUDA profiler wrapper:
# start with /start_profile and stop with /stop_profile.
set -euo pipefail

SRC="${DSPARK_BASE_ENTRYPOINT:-/entrypoint.unholy.sh}"
DST="/tmp/entrypoint.dspark.generated.sh"

if [ ! -f "${SRC}" ]; then
  echo "[dspark-nsys] ERROR: base entrypoint not found: ${SRC}" >&2
  exit 1
fi

/opt/env/bin/python - "${SRC}" "${DST}" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text()
needle = '\\"method\\":\\"mtp\\"'
if needle not in text:
    raise SystemExit("[dspark-nsys] ERROR: MTP speculative-config anchor not found")
text = text.replace(needle, '\\"method\\":\\"dspark\\"')
text = text.replace("[unholy]", "[dspark]")
text = text.replace("unholy-fusion", "dspark-experiment")
dst.write_text(text)
dst.chmod(0o755)
PY

profile_role="${DSPARK_NSYS_ROLE:-all}"
role="${ROLE:-}"
should_profile=0
if [ "${DSPARK_NSYS_LAUNCH:-0}" = "1" ]; then
  if [ "${profile_role}" = "all" ] || [ "${role}" = "${profile_role}" ]; then
    should_profile=1
  fi
fi

if [ "${should_profile}" = "1" ]; then
  export PATH="/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/opt/nvidia/nsight-systems/2025.3.2/bin:${PATH:-}"
  if ! command -v nsys >/dev/null 2>&1; then
    echo "[dspark-nsys] ERROR: nsys not found; mount Nsight Systems or disable DSPARK_NSYS_LAUNCH" >&2
    exit 1
  fi

  out_dir="${DSPARK_NSYS_OUT_DIR:-/cache/huggingface/dspark-profiles/nsys}"
  label="${DSPARK_NSYS_LABEL:-dspark_nsys}"
  mkdir -p "${out_dir}"

  out_base="${out_dir}/${label}_${role}_%h_%p"
  trace="${DSPARK_NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn}"
  sample="${DSPARK_NSYS_SAMPLE:-none}"
  backtrace="${DSPARK_NSYS_BACKTRACE:-none}"
  capture_range="${DSPARK_NSYS_CAPTURE_RANGE:-cudaProfilerApi}"
  capture_end="${DSPARK_NSYS_CAPTURE_RANGE_END:-stop}"
  cuda_graph_trace="${DSPARK_NSYS_CUDA_GRAPH_TRACE:-graph}"
  trace_fork="${DSPARK_NSYS_TRACE_FORK_BEFORE_EXEC:-true}"
  stats="${DSPARK_NSYS_STATS:-true}"
  export_formats="${DSPARK_NSYS_EXPORT:-sqlite}"
  pytorch="${DSPARK_NSYS_PYTORCH:-none}"

  echo "[dspark-nsys] launching ROLE=${role} under nsys profile"
  echo "[dspark-nsys] output=${out_base}"
  echo "[dspark-nsys] capture=${capture_range} end=${capture_end} trace=${trace}"

  nsys_args=(
    profile
    "--output=${out_base}"
    "--force-overwrite=true"
    "--trace=${trace}"
    "--sample=${sample}"
    "--backtrace=${backtrace}"
    "--capture-range=${capture_range}"
    "--capture-range-end=${capture_end}"
    "--cuda-graph-trace=${cuda_graph_trace}"
    "--trace-fork-before-exec=${trace_fork}"
    "--stats=${stats}"
    "--export=${export_formats}"
    "--show-output=true"
  )
  if [ "${pytorch}" != "none" ]; then
    nsys_args+=("--pytorch=${pytorch}")
  fi

  exec nsys "${nsys_args[@]}" bash "${DST}"
fi

exec bash "${DST}"
