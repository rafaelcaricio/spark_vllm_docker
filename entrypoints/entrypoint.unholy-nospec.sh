#!/usr/bin/env bash
# Experimental no-spec wrapper for the unholy-fusion DeepSeek V4 Flash path.
#
# It reuses the known-good unholy entrypoint and removes only the MTP
# speculative-config lines from the generated runtime script.
set -euo pipefail

SRC="${UNHOLY_BASE_ENTRYPOINT:-/entrypoint.unholy.sh}"
DST="/tmp/entrypoint.unholy-nospec.generated.sh"

if [ ! -f "${SRC}" ]; then
  echo "[unholy-nospec] ERROR: base entrypoint not found: ${SRC}" >&2
  exit 1
fi

/opt/env/bin/python - "${SRC}" "${DST}" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
lines = src.read_text().splitlines(keepends=True)
filtered = [
    line for line in lines
    if "--speculative-config" not in line
]
removed = len(lines) - len(filtered)
if removed != 2:
    raise SystemExit(
        f"[unholy-nospec] ERROR: expected to remove 2 speculative-config "
        f"lines, removed {removed}"
    )
text = "".join(filtered).replace("[unholy]", "[unholy-nospec]")
dst.write_text(text)
dst.chmod(0o755)
PY

exec bash "${DST}"
