# UAI 2026 CRL Additional Appendix Experiments

Supplementary runs for the UAI 2026 paper
*On Causal Representation Learning with Internal Auxiliaries*
These scripts reproduce additional experiments used in the final paper and
appendix.

## Relationship to the main codebase

Most additional runs are wrappers around the paper code. Two appendix
experiments required small core additions: `--mixing=nonvolumepreserving` for
the VP-vs-non-VP ablation and `COMPUTE_NONLINEAR_METRICS=1` for GBR-DCI /
kernel-ridge R2 validation. Baseline paper runs remain unchanged unless those
options are explicitly used.

Seed conventions differ by purpose. The paper's main runs use the contiguous
block `0..19`. Pilot sweeps use `0..4`. The appendix nonlinear-metric and
VP-vs-non-VP tables use `0..24`.

## Experiments

| Exp | Status | Compute | Appendix use |
|-----|--------|---------|--------------|
| 01  | active | ~30 GPU-hr full | Non-linear SCM via `--scm location-scale`, with optional GBR-DCI / kernel-ridge R2 validation. |
| 02  | active | ~15 GPU-hr | Extra denser/larger DGPs with Algorithm 1 selection enabled; a robustness sweep, not a with-vs-without selection toggle. |
| 04  | active | ~12-16 GPU-hr full | Beta-coefficient range ablation: paper range vs. sign-varying vs. small magnitude. |
| 06  | active | ~18-24 GPU-hr full | VP vs. non-VP data-generating mixing ablation using GIN vs. Glow-style coupling blocks. |

All experiments share the same environment as the main paper:

- `torch==2.5.1`, `pytorch-lightning==1.9.5`, `normflows==1.7.3`,
  `FrEIA==0.2`, Python 3.10.
- `conda activate crl` on the cluster, or set `CRL_CONDA_ENV` before submitting
  Slurm jobs if the environment has a different name.

## Running on the cluster

Each experiment directory has a `run_slurm.sh` that is directly `sbatch`-able.
Launch from the repository root, or set
`CRL_REPO_DIR=/path/to/crl-internal-auxiliaries` before launch.

For smoke testing on a laptop or a login node, each experiment directory
(`exp01`, `exp02`, `exp04`, `exp06`) has a `run_local.sh` that runs a single
`seed=0`, `--max-epochs` reduced (5 for synthetic), single DGP subset. This
exercises the active code paths without burning hours.

## Outputs

Each experiment writes to its own `results/` sub-directory. The aggregation
script produces `*_summary.csv` and a small LaTeX snippet suitable for
checking the appendix tables.

## Appendix Mapping

- exp01: nonlinear-SCM and nonlinear-metric validation.
- exp02: denser/larger graph robustness for Algorithm 1 with selection enabled.
- exp04: beta-range robustness.
- exp06: VP vs. non-VP data-generating mixing sensitivity.

For release use, aggregate each experiment's `results/` directory and compare
the generated TeX snippets against the paper appendix tables.
