#!/bin/bash
# Smoke test for exp01 (non-linear SCM).
# Runs a single seed=0, single DGP=a, both SCMs, 5 epochs only.
# No sbatch; defaults to CPU for laptop/login-node smoke testing.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
cd "${REPO}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Use the user's currently active python; adjust if necessary.
PYTHON="${PYTHON:-python}"

RESULT_DIR="appendix_experiments/exp01_nonlinear_scm/results_smoke"
mkdir -p "${RESULT_DIR}"

${PYTHON} appendix_experiments/exp01_nonlinear_scm/run_nonlinear.py \
    --dgp a \
    --scm both \
    --seeds 0 \
    --max-epochs 5 \
    --result-dir "${RESULT_DIR}" \
    --accelerator "${ACCELERATOR:-cpu}" \
    --device "${DEVICE:-0}" \
    --compute-nonlinear-metrics \
    --stop-on-error

${PYTHON} appendix_experiments/exp01_nonlinear_scm/aggregate.py \
    --result-dir "${RESULT_DIR}"

echo "[exp01/local] smoke run done. See ${RESULT_DIR}/nonlinear_scm_summary.csv"
