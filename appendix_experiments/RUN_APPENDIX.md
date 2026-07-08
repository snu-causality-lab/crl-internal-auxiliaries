# UAI 2026 CRL Appendix Experiment Runbook

Set these environment variables if your cluster setup differs from the
defaults:

```bash
export CRL_REPO_DIR=/path/to/crl-internal-auxiliaries
export CRL_CONDA_ENV=crl
```

## Experiments to run

| Exp | Array | Tasks | GPU-hr | Appendix use |
|-----|-------|-------|--------|--------------|
| 01 | 3 DGPs x 2 SCMs x 5 seeds | 30 | ~3-4 pilot | Nonlinear-SCM ablation; optional GBR-DCI / kernel-ridge R2 metrics. |
| 02 | 6 DGPs x 5 seeds | 30 | ~3-4 pilot | Extra selection graphs for Algorithm 1. |
| 04 | 3 beta configs x 2 DGPs x 5 seeds | 30 | ~3-4 pilot | Beta-range robustness check. |
| 06 | 3 DGPs x 2 mixings x 25 seeds | 150 | ~18-24 full | VP vs non-VP data-generating mixing ablation. |

Pilot bundle: 90 tasks. Full exp06: 150 tasks. Combined if all are submitted:
240 tasks. The paper's additional-ablation tables (exp01, exp04, exp06) use
seeds `0..24` (25 seeds); the paper's main experiments use seeds `0..19`
(20 repetitions); pilot scripts use seeds `0..4`.

## Step-by-step

### 1. Activate environment and sanity check

```bash
conda env list | grep -qE "^\s*${CRL_CONDA_ENV:-crl}\s" && echo "ENV OK" || echo "ENV MISSING"
```

If output is `ENV MISSING`, create it first:

```bash
cd "$CRL_REPO_DIR"
conda create -n "${CRL_CONDA_ENV:-crl}" python=3.10 -y
conda activate "${CRL_CONDA_ENV:-crl}"
if [ -f requirements-frozen-linux.txt ]; then
  pip install -r requirements-frozen-linux.txt
else
  pip install -r requirements.txt
fi
```

Then activate and verify imports:

```bash
conda activate "${CRL_CONDA_ENV:-crl}"
python - <<'PY'
import torch, pytorch_lightning, normflows, FrEIA
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("pl", pytorch_lightning.__version__)
print("normflows/FrEIA imported OK")
PY
```

Expect `cuda_available True` on GPU nodes. Resolve missing imports before
launching arrays; otherwise every task will fail.

### 2. Smoke test

```bash
cd "$CRL_REPO_DIR"
bash appendix_experiments/exp01_nonlinear_scm/run_local.sh
bash appendix_experiments/exp02_more_graphs_selection/run_local.sh
bash appendix_experiments/exp04_beta_ablation/run_local.sh
bash appendix_experiments/exp06_nonvp_ablation/run_local.sh
```

If any smoke test fails, inspect that experiment's `results_smoke/` directory
and the corresponding stdout log before launching Slurm arrays.

### 3. Launch Slurm arrays

```bash
cd "$CRL_REPO_DIR"

JOB01=$(sbatch --parsable appendix_experiments/exp01_nonlinear_scm/run_slurm.sh)
JOB02=$(sbatch --parsable appendix_experiments/exp02_more_graphs_selection/run_slurm.sh)
JOB04=$(sbatch --parsable appendix_experiments/exp04_beta_ablation/run_slurm.sh)
JOB06=$(sbatch --parsable appendix_experiments/exp06_nonvp_ablation/run_slurm.sh)

echo "exp01=$JOB01 exp02=$JOB02 exp04=$JOB04 exp06=$JOB06"
```

### 4. Monitor

```bash
squeue -u $USER
squeue -j $JOB01,$JOB02,$JOB04,$JOB06
tail -f slurm-crl_exp06_nonvp-*_0.out
```

Typical synthetic task runtime is 6-8 min on RTX-A6000 / A100. Pilot 30-task
arrays finish in roughly 25-40 min with `%8` concurrency. Exp06 uses 150 tasks
with `%16` concurrency.

### 5. Aggregate when done

```bash
cd "$CRL_REPO_DIR"

python appendix_experiments/exp01_nonlinear_scm/aggregate.py \
  --result-dir appendix_experiments/exp01_nonlinear_scm/results

python appendix_experiments/exp02_more_graphs_selection/aggregate.py \
  --result-dir appendix_experiments/exp02_more_graphs_selection/results

python appendix_experiments/exp04_beta_ablation/aggregate.py \
  --result-dir appendix_experiments/exp04_beta_ablation/results

python appendix_experiments/exp06_nonvp_ablation/aggregate.py \
  --result-dir appendix_experiments/exp06_nonvp_ablation/results
```

Each aggregator writes `*_summary.csv` and `*_summary.tex`.
The array scripts also write per-task status files under
`results/status/task_<array_id>.tsv`; these are for failure inspection and do
not need to be merged before aggregation.

## Scale-up

The pilot scripts use 5 seeds per cell. To upgrade pilots:

- exp01: edit `run_slurm.sh` to `SEEDS=(0..24)` and `#SBATCH --array=0-149%8`
  for the 25-seed additional-ablation setting reported in the paper.
- exp02: edit to `SEEDS=(0..19)` and `#SBATCH --array=0-119%8` for the 20-seed
  main-experiment convention (this is a selection check, not an App. E table).
- exp04: edit to `SEEDS=(0..24)` and `#SBATCH --array=0-149%8` for the 25-seed
  additional-ablation setting reported in the paper.
- exp06: already runs the 25-seed additional-ablation setting.

## Failure recovery

- Individual task failure: inspect the `slurm-*.out`/`slurm-*.err` files, then
  re-submit only the failing task with
  `sbatch --array=N appendix_experiments/exp0X_*/run_slurm.sh`.
- Missing dependency: `conda activate crl; pip install -r requirements-frozen-linux.txt`
  on Linux, falling back to `requirements.txt` only if the frozen file is not
  present.
- OOM: reduce the batch size in the relevant `run_slurm.sh` python command,
  e.g. add `--batch-size 512`.

## Files in this folder

- `README.md` -- overall description of the additional appendix runs.
- `exp01_nonlinear_scm/` -- nonlinear SCM sweep and optional nonlinear metrics.
- `exp02_more_graphs_selection/` -- denser graphs for Algorithm 1.
- `exp04_beta_ablation/` -- beta-range ablation.
- `exp06_nonvp_ablation/` -- VP vs non-VP data-generating mixing ablation.
- `RUN_APPENDIX.md` -- this file.
