#!/usr/bin/env bash
# Experimental DSpark entrypoint with optional Nsight Compute launch support.
#
# This mirrors entrypoint.dspark.sh, but can run the head rank under
# `ncu --mode=launch` so a later `ncu --mode=attach` can collect a bounded
# warmed-decode profile without profiling startup.
set -euo pipefail

SRC="${DSPARK_BASE_ENTRYPOINT:-/entrypoint.unholy.sh}"
DST="/tmp/entrypoint.dspark.generated.sh"

if [ ! -f "${SRC}" ]; then
  echo "[dspark-ncu] ERROR: base entrypoint not found: ${SRC}" >&2
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
    raise SystemExit("[dspark-ncu] ERROR: MTP speculative-config anchor not found")
text = text.replace(needle, '\\"method\\":\\"dspark\\"')
text = text.replace("[unholy]", "[dspark]")
text = text.replace("unholy-fusion", "dspark-experiment")
dst.write_text(text)
dst.chmod(0o755)
PY

profile_role="${DSPARK_NCU_ROLE:-head}"
if [ "${DSPARK_NCU_LAUNCH:-0}" = "1" ] && [ "${ROLE:-}" = "${profile_role}" ]; then
  export PATH="/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:${PATH:-}"
  if ! command -v ncu >/dev/null 2>&1; then
    echo "[dspark-ncu] ERROR: ncu not found; mount host Nsight tools or disable DSPARK_NCU_LAUNCH" >&2
    exit 1
  fi

  port="${DSPARK_NCU_PORT:-49152}"
  target_processes="${DSPARK_NCU_TARGET_PROCESSES:-all}"
  echo "[dspark-ncu] launching ROLE=${ROLE} under ncu --mode=launch port=${port}"
  exec ncu \
    --mode=launch \
    --port "${port}" \
    --target-processes "${target_processes}" \
    --forward-signals \
    bash "${DST}"
fi

exec bash "${DST}"
