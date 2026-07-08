#!/bin/bash
#SBATCH --job-name=crl_exp04_beta
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
# 3 beta x 2 DGPs x 5 seeds = 30 tasks (0..29)
#SBATCH --array=0-29%8

# Submit with:   sbatch appendix_experiments/exp04_beta_ablation/run_slurm.sh
# For the full paper run (25-seed additional-ablation setting), change the
# array upper bound to 0-149%8 and set SEEDS=(0..24) below.

set -euo pipefail

cd "${CRL_REPO_DIR:-$PWD}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CRL_CONDA_ENV:-crl}"
fi

# Thread pinning. FrEIA/PyTorch otherwise grabs all cores and noticeably
# slows the sweep when 8 concurrent jobs share a node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Avoid CUDA allocator fragmentation with many small batches.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Beta configurations (label in first array; bounds read by run_beta_ablation.py).
BETAS=(paper symmetric small)
# Four-node multi-observable DGPs used for the beta-range sweep.
DGPS=(c c_real)
SEEDS=(0 1 2 3 4)

NBETAS=${#BETAS[@]}   # 3
NDGPS=${#DGPS[@]}     # 2
NSEEDS=${#SEEDS[@]}   # 5

task=${SLURM_ARRAY_TASK_ID}
# Task decomposition (matches the task description):
#   seed_idx = task % 5
#   dgp_idx  = (task / 5) % 2
#   beta_idx = (task / 10) % 3
seed_idx=$(( task % NSEEDS ))
dgp_idx=$(( (task / NSEEDS) % NDGPS ))
beta_idx=$(( (task / (NSEEDS * NDGPS)) % NBETAS ))

BETA=${BETAS[$beta_idx]}
DGP=${DGPS[$dgp_idx]}
SEED=${SEEDS[$seed_idx]}

echo "[exp04] task=${task} BETA=${BETA} DGP=${DGP} SEED=${SEED}"

RESULT_DIR=appendix_experiments/exp04_beta_ablation/results
STATUS_FILE="${RESULT_DIR}/status/task_${SLURM_ARRAY_TASK_ID}.tsv"

python appendix_experiments/exp04_beta_ablation/run_beta_ablation.py \
    --dgp "${DGP}" \
    --beta "${BETA}" \
    --seeds "${SEED}" \
    --max-epochs 20 \
    --result-dir "${RESULT_DIR}" \
    --status-file "${STATUS_FILE}" \
    --accelerator gpu \
    --device 0 \
    --stop-on-error

echo "[exp04] task=${task} done"
