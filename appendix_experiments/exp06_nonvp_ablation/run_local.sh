#!/bin/bash
# Smoke test for exp06 (VP vs non-VP mixing ablation).
# Runs graph a, seed 0, both mixings, 5 epochs only.

set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-python}
RESULT_DIR="appendix_experiments/exp06_nonvp_ablation/results_smoke"

${PYTHON} appendix_experiments/exp06_nonvp_ablation/run_nonvp.py \
    --dgp a \
    --mixing both \
    --seeds 0 \
    --max-epochs 5 \
    --result-dir "${RESULT_DIR}" \
    --accelerator "${ACCELERATOR:-cpu}" \
    --device "${DEVICE:-0}" \
    --stop-on-error

${PYTHON} appendix_experiments/exp06_nonvp_ablation/aggregate.py \
    --result-dir "${RESULT_DIR}"

echo "[exp06/local] smoke run done. See ${RESULT_DIR}/nonvp_ablation_summary.csv"
