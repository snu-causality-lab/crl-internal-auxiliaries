#!/bin/bash
#SBATCH --job-name=crl_exp06_nonvp
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
# 3 DGPs x 2 mixings x 25 seeds = 150 tasks (0..149)
#SBATCH --array=0-149%16

# Submit with: sbatch appendix_experiments/exp06_nonvp_ablation/run_slurm.sh

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

DGPS=(a b c_real)
MIXINGS=(volumepreserving nonvolumepreserving)
SEEDS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)

NDGPS=${#DGPS[@]}     # 3
NMIXINGS=${#MIXINGS[@]}  # 2
NSEEDS=${#SEEDS[@]}   # 25

task=${SLURM_ARRAY_TASK_ID}
seed_idx=$(( task % NSEEDS ))
mixing_idx=$(( (task / NSEEDS) % NMIXINGS ))
dgp_idx=$(( (task / (NSEEDS * NMIXINGS)) % NDGPS ))

DGP=${DGPS[$dgp_idx]}
MIXING=${MIXINGS[$mixing_idx]}
SEED=${SEEDS[$seed_idx]}

echo "[exp06] task=${task} DGP=${DGP} MIXING=${MIXING} SEED=${SEED}"

RESULT_DIR=appendix_experiments/exp06_nonvp_ablation/results
STATUS_FILE="${RESULT_DIR}/status/task_${SLURM_ARRAY_TASK_ID}.tsv"

python appendix_experiments/exp06_nonvp_ablation/run_nonvp.py \
    --dgp "${DGP}" \
    --mixing "${MIXING}" \
    --seeds "${SEED}" \
    --max-epochs 20 \
    --result-dir "${RESULT_DIR}" \
    --status-file "${STATUS_FILE}" \
    --accelerator gpu \
    --device 0 \
    --stop-on-error

echo "[exp06] task=${task} done"
