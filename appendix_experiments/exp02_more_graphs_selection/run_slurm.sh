#!/bin/bash
#SBATCH --job-name=crl_exp02_selection
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
# 6 DGPs (including c_real baseline) x 5 seeds = 30 tasks
#SBATCH --array=0-29%8

set -euo pipefail

cd "${CRL_REPO_DIR:-$PWD}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CRL_CONDA_ENV:-crl}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# c_real is the paper's baseline (low density); the rest are progressively
# denser / larger graphs from extra_graphs.py.
DGPS=(c_real c_dense c_deep c_chain c_obs_chain c_hub6)
SEEDS=(0 1 2 3 4)

NDGPS=${#DGPS[@]}
NSEEDS=${#SEEDS[@]}

task=${SLURM_ARRAY_TASK_ID}
seed_idx=$(( task % NSEEDS ))
dgp_idx=$(( (task / NSEEDS) % NDGPS ))

DGP=${DGPS[$dgp_idx]}
SEED=${SEEDS[$seed_idx]}

echo "[exp02] task=${task} DGP=${DGP} SEED=${SEED}"

RESULT_DIR=appendix_experiments/exp02_more_graphs_selection/results
STATUS_FILE="${RESULT_DIR}/status/task_${SLURM_ARRAY_TASK_ID}.tsv"

python appendix_experiments/exp02_more_graphs_selection/run_selection_ablation.py \
    --only "${DGP}" \
    --seeds "${SEED}" \
    --max-epochs 20 \
    --result-dir "${RESULT_DIR}" \
    --status-file "${STATUS_FILE}" \
    --accelerator gpu \
    --device 0 \
    --stop-on-error

echo "[exp02] task=${task} done"
