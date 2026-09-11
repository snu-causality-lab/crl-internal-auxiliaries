#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python run_all.py \
  --ours \
  --synthetic \
  --dgp-list a \
  --seeds 0 1 \
  --max-epochs 1 \
  --result-root ./results_smoke \
  --accelerator cpu
