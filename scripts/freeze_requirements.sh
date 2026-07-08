#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python - <<'PY'
import platform
import sys

if platform.system() != "Linux":
    raise SystemExit("Refusing to write requirements-frozen-linux.txt outside Linux.")
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "Refusing to write requirements-frozen-linux.txt outside Python 3.10. "
        f"Current version: {sys.version.split()[0]}"
    )
PY

python -m pip list --format=freeze | sort > requirements-frozen-linux.txt
echo "Wrote requirements-frozen-linux.txt from $(python --version 2>&1)"
