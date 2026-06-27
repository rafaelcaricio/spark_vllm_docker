#!/usr/bin/env bash
# Experimental DSpark entrypoint wrapper.
#
# Reuses the known-good unholy-fusion GB10 entrypoint and rewrites only the
# speculative method from MTP to DSpark at container startup.
set -euo pipefail

SRC="${DSPARK_BASE_ENTRYPOINT:-/entrypoint.unholy.sh}"
DST="/tmp/entrypoint.dspark.generated.sh"

if [ ! -f "${SRC}" ]; then
  echo "[dspark] ERROR: base entrypoint not found: ${SRC}" >&2
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
    raise SystemExit("[dspark] ERROR: MTP speculative-config anchor not found")
text = text.replace(needle, '\\"method\\":\\"dspark\\"')
text = text.replace("[unholy]", "[dspark]")
text = text.replace("unholy-fusion", "dspark-experiment")
dst.write_text(text)
dst.chmod(0o755)
PY

exec bash "${DST}"
