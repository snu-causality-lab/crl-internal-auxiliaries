#!/bin/bash
#SBATCH --job-name=crl_exp01_nonlinear
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
# 3 DGPs x 2 SCMs x 5 seeds = 30 tasks (0..29)
#SBATCH --array=0-29%8

# Submit with:   sbatch appendix_experiments/exp01_nonlinear_scm/run_slurm.sh
# For the full paper run (25-seed additional-ablation setting), change the
# array to 0-149%8 and set SEEDS=(0..24) below.

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

DGPS=(a b c_real)
SCMS=(linear location-scale)
SEEDS=(0 1 2 3 4)

NDGPS=${#DGPS[@]}     # 3
NSCMS=${#SCMS[@]}     # 2
NSEEDS=${#SEEDS[@]}   # 5

task=${SLURM_ARRAY_TASK_ID}
seed_idx=$(( task % NSEEDS ))
scm_idx=$(( (task / NSEEDS) % NSCMS ))
dgp_idx=$(( (task / (NSEEDS * NSCMS)) % NDGPS ))

DGP=${DGPS[$dgp_idx]}
SCM=${SCMS[$scm_idx]}
SEED=${SEEDS[$seed_idx]}

echo "[exp01] task=${task} DGP=${DGP} SCM=${SCM} SEED=${SEED}"

RESULT_DIR=appendix_experiments/exp01_nonlinear_scm/results
STATUS_FILE="${RESULT_DIR}/status/task_${SLURM_ARRAY_TASK_ID}.tsv"

python appendix_experiments/exp01_nonlinear_scm/run_nonlinear.py \
    --dgp "${DGP}" \
    --scm "${SCM}" \
    --seeds "${SEED}" \
    --max-epochs 20 \
    --result-dir "${RESULT_DIR}" \
    --status-file "${STATUS_FILE}" \
    --accelerator gpu \
    --device 0 \
    --compute-nonlinear-metrics \
    --stop-on-error

echo "[exp01] task=${task} done"
