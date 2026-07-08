#!/bin/bash
# Smoke test for exp04 (beta-range ablation).
# Runs a single seed=0, single DGP=c_real, all 3 beta configs, 5 epochs only.
# No sbatch; defaults to CPU for laptop/login-node smoke testing.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
cd "${REPO}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Use the user's currently active python; adjust if necessary.
PYTHON="${PYTHON:-python}"

RESULT_DIR="appendix_experiments/exp04_beta_ablation/results_smoke"
mkdir -p "${RESULT_DIR}"

${PYTHON} appendix_experiments/exp04_beta_ablation/run_beta_ablation.py \
    --dgp c_real \
    --beta all \
    --seeds 0 \
    --max-epochs 5 \
    --result-dir "${RESULT_DIR}" \
    --accelerator "${ACCELERATOR:-cpu}" \
    --device "${DEVICE:-0}" \
    --stop-on-error

${PYTHON} appendix_experiments/exp04_beta_ablation/aggregate.py \
    --result-dir "${RESULT_DIR}"

echo "[exp04/local] smoke run done. See ${RESULT_DIR}/beta_ablation_summary.csv"
