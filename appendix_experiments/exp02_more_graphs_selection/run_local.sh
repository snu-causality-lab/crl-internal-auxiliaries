#!/bin/bash
# Smoke test for exp02 (extra-graphs sweep).
# Runs one graph (c_dense), seed=0, 5 epochs.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
cd "${REPO}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PYTHON="${PYTHON:-python}"

RESULT_DIR="appendix_experiments/exp02_more_graphs_selection/results_smoke"
mkdir -p "${RESULT_DIR}"

${PYTHON} appendix_experiments/exp02_more_graphs_selection/run_selection_ablation.py \
    --only c_dense \
    --seeds 0 \
    --max-epochs 5 \
    --result-dir "${RESULT_DIR}" \
    --accelerator "${ACCELERATOR:-cpu}" \
    --device "${DEVICE:-0}" \
    --stop-on-error

${PYTHON} appendix_experiments/exp02_more_graphs_selection/aggregate.py \
    --result-dir "${RESULT_DIR}"

echo "[exp02/local] smoke run done. See ${RESULT_DIR}/selection_ablation_summary.csv"
