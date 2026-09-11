# On Causal Representation Learning with Internal Auxiliaries

Code accompanying [On Causal Representation Learning with Internal Auxiliaries](https://proceedings.mlr.press/v337/kim26e.html),
published at UAI 2026 (PMLR 337:3061–3082). See
[Reproducing Experiments](#reproducing-experiments) for the included pipelines.

## Overview

The paper studies causal representation learning when observed sources also
enter the mixing process as **internal** auxiliaries. Its identifiability
result covers the selected-source case (`O = C`) under the stated variability,
support, common-reparameterization, block-matching, and volume-preserving
assumptions. The experimental method also handles observed-but-unselected
sources through a structural prediction penalty.

The released pipeline uses GIN (General Incompressible-flow Networks) coupling
blocks and graph-specific, preconfigured auxiliary indices. It does not execute
the paper's graph-selection or Bayes-ball algorithms (Algorithms 1 and 2).

## Requirements

- Python 3.10 (tested; recommended for reproducible installs)
- CUDA-compatible GPU (recommended)
- Install the minimal direct dependencies:

```bash
conda create -n crl python=3.10 -y
conda activate crl
pip install -r requirements.txt
```

For Linux/CUDA release reproduction, use the frozen lockfile inside the same
Python 3.10 environment:

```bash
pip install -r requirements-frozen-linux.txt
```

After validating a fresh Linux/CUDA deployment, regenerate the lockfile from
that server environment with:

```bash
bash scripts/freeze_requirements.sh
```

## Repository Structure

```
crl-internal-auxiliaries/
├── README.md                          # This file
├── LICENSE                            # MIT license
├── LICENSES/                          # Third-party license texts
├── CITATION.cff                       # Citation metadata
├── THIRD-PARTY-NOTICES.md             # Third-party code notices
├── requirements.txt                   # Python dependencies
├── requirements-frozen-linux.txt      # Frozen Linux/CUDA environment
├── run_all.py                         # Unified experiment runner
├── scripts/
│   ├── smoke_test.sh                  # Quick CPU smoke test
│   └── freeze_requirements.sh         # Regenerate Linux/CUDA pip freeze lock
│
├── config.py                          # DGP configurations for proposed method (Ours)
├── config_GIN.py                      # DGP configurations for GIN baseline
├── config_iVAE.py                     # DGP configurations for iVAE baseline
│
├── example_flow.py                    # Run Ours on Flow dataset
├── example_pendulum.py                # Run Ours on Pendulum dataset
├── example_GIN.py                     # Run GIN on synthetic data
├── example_GIN_flow.py                # Run GIN on Flow dataset
├── example_GIN_pendulum.py            # Run GIN on Pendulum dataset
├── example_iVAE.py                    # Run iVAE on synthetic data
├── example_iVAE_flow.py               # Run iVAE on Flow dataset
├── example_iVAE_pendulum.py           # Run iVAE on Pendulum dataset
│
├── data_generator/                    # Synthetic data generation
│   ├── scm.py                         #   Structural Causal Models (Linear, Location-Scale)
│   ├── mixing_function.py             #   Mixing functions (Linear, Nonlinear, VP, Non-VP ablation)
│   ├── noise_generator.py             #   Gaussian noise with environment shifts
│   ├── multi_env_gdp.py               #   Multi-environment data generating process
│   ├── data_module.py                 #   PyTorch Lightning data module
│   └── ...
│
├── data_flows/                        # Flow dataset
│   ├── flow.py                        #   Script to generate Flow images
│   ├── flow_noise_snapshot.tar.gz      #   Paper-reproduction Flow snapshot
│   └── data_module.py                 #   Data loader for image datasets
│
├── data_pendulum/                     # Pendulum dataset
│   └── pendulum_dataset.py            #   Script to generate Pendulum images
│
├── model/                             # Proposed model implementation
│   ├── our_model.py                   #   Main model (CauCAModel, NonlinearCauCAModel)
│   ├── encoder.py                     #   Volume-preserving encoder (GIN-based)
│   ├── dci.py                         #   DCI metric (Disentanglement, Completeness)
│   ├── mcc.py                         #   MCC metric (Mean Correlation Coefficient)
│   ├── image_compress.py              #   Image encoder/decoder for image datasets
│   └── normalizing_flow/              #   Base distributions and flow utilities
│       ├── distribution.py            #     Graph-aware prior distribution
│       ├── nonparametric_distribution.py
│       └── utils.py                   #     Flow construction utilities
│
├── experiments/                       # Experiment entry points
│   ├── Ours/main.py                   #   Main script for proposed method
│   ├── GIN/                           #   GIN baseline
│   │   ├── main.py                    #     Main script
│   │   ├── GIN.py                     #     GIN model
│   │   ├── net.py                     #     Network utilities
│   │   └── dci.py                     #     DCI metric
│   └── iVAE/                          #   iVAE baseline
│       ├── main.py                    #     Main script
│       ├── wrappers.py                #     iVAE wrapper (Lightning module)
│       ├── nets.py                    #     Network architectures
│       ├── dci.py                     #     DCI metric
│       └── mcc.py                     #     MCC metric
│
├── plots/                             # Plotting scripts for paper figures
│
└── appendix_experiments/              # Additional appendix experiment wrappers
```

## DGP Configurations → Paper Figures

The Data Generating Process (DGP) names in the config files correspond to the paper's figures:

| DGP Name  | Paper Reference      | Description                                        |
|-----------|---------------------|----------------------------------------------------|
| `a`       | Fig. 1(a) dependency graph | 5 nodes, four conditionally independent targets; auxiliary also enters mixing |
| `b`       | Fig. 1(c)            | 5 nodes, single observable source (ISA setting)     |
| `b_aug`   | —                    | 5-node graph variant with the same selected source as `b` |
| `c`       | App. E beta ablation | 4 nodes, multiple observables                       |
| `c_aug`   | —                    | 4-node variant of `c` with an added edge to the observed-but-unselected node |
| `c_real`  | Fig. 1(d)            | 4 nodes; selected source `s4`, observed-but-unselected source `s2` |

DGP `a` is not an external-auxiliary control: all five sources enter the
mixing map. For `c_real`, code indices `[0, 1, 2, 3]` correspond to paper
sources `[s1, s3, s2, s4]`.

## Reproducing Experiments

Run the commands below from the repository root; the repository is not installed
as a Python package.

The public runners cover quantitative synthetic/image experiments and the
listed appendix sweeps. They do not include the separate image compression,
reconstruction, and traversal experiment. For the proposed method and GIN,
main image runs apply the VP flow at the full image dimension.

### Step 1: Generate Image Datasets

Synthetic data (graphs a, b, c, etc.) is generated on-the-fly during training. For **image-based** experiments (Flow and Pendulum), you must first prepare the datasets.

For the Flow experiments reported in the paper, use the bundled fixed snapshot:

```bash
mkdir -p data_flows/causal_data
tar -xzf data_flows/flow_noise_snapshot.tar.gz -C data_flows/causal_data
```

The Flow generator remains available for fresh simulations, but regenerated
images need not match the paper figures bit-for-bit. Use the bundled snapshot
for paper reproduction.

Generate Pendulum images with:

```bash
# Generate Pendulum dataset (pendulum shadow images)
cd data_pendulum
python pendulum_dataset.py
cd ..
```

This will create image files under:
- `data_flows/causal_data/flow_noise/{train,test}/`
- `data_pendulum/causal_data/pendulum/{train,test}/`

To generate a fresh Flow simulation instead of using the bundled snapshot:

```bash
cd data_flows
python flow.py
cd ..
```

### Step 2: Run Experiments

You can use the **unified runner** or the convenience scripts.

#### Option A: Unified Runner (Recommended)

```bash
# Run ALL experiments (all models × all datasets)
python run_all.py --all

# Run specific experiment groups
python run_all.py --synthetic    # Synthetic data only (Ours + GIN + iVAE)
python run_all.py --flow         # Flow dataset only
python run_all.py --pendulum     # Pendulum dataset only

# Run a specific model on a specific dataset
python run_all.py --ours --synthetic
python run_all.py --gin --flow
python run_all.py --ivae --pendulum
```

#### Option B: Convenience Scripts

These scripts provide convenience entry points for individual model/dataset
combinations. Use `run_all.py` when you want one consistent interface for the
full paper-evaluation runs or custom smoke-test settings.

```bash
# Proposed method (Ours)
python run_all.py --ours --synthetic   # Synthetic data
python example_flow.py                 # Flow dataset
python example_pendulum.py             # Pendulum dataset

# GIN baseline
python example_GIN.py                  # Small synthetic example
python example_GIN_flow.py             # Flow dataset
python example_GIN_pendulum.py         # Pendulum dataset

# iVAE baseline
python example_iVAE.py                 # Small synthetic example
python example_iVAE_flow.py            # Flow dataset
python example_iVAE_pendulum.py        # Pendulum dataset
```

The paper-evaluation scripts run **20 seeds** (0–19) by default. `run_all.py`
uses the paper epoch settings unless `--max-epochs` is provided: synthetic runs
use 20 epochs, Ours Flow uses 50, image GIN uses 40, and image iVAE uses 80.
Ours Pendulum follows the corresponding example script setting of 80 epochs.
Additional appendix wrappers document their own pilot/full seed ranges in
`appendix_experiments/`. Image runs use batch size 1024 by default; if host RAM
or accelerator memory is insufficient, rerun with a smaller override such as
`--batch-size 128`.

#### Issue Reproducers and Regression Tests

The [issue reproduction guide](REPRODUCING_ISSUES.md) compares the original
release with the fixes for smoke-output collisions and iVAE automatic device
selection. These checks use temporary sentinel files or a small forward pass;
they do not train models or rerun the paper experiments.

```bash
python -B -m unittest discover -s tests -v
```

Tests requiring the ML environment or accelerator hardware skip explicitly
when unavailable. The one-epoch smoke test below is a separate training check.

#### Quick Smoke Test

After installing dependencies, run a short CPU-only check:

```bash
bash scripts/smoke_test.sh
```

This runs the proposed method on DGP `a` for one epoch and one seed. It is
intended only to verify that imports, data generation, training, and result
writing work in the current environment. Its output is isolated under
`results_smoke/result_ours_synthetic/a/`.

To keep other short runs separate from paper results, set `--result-root`
(default: the current directory):

```bash
python run_all.py --ours --synthetic --dgp-list a --seeds 0 1 --max-epochs 1 --accelerator cpu --result-root ./results_check
```

### Additional Appendix Experiments

The `appendix_experiments/` directory contains wrappers for the paper's
additional ablations and an extra graph robustness sweep:

- `exp01_nonlinear_scm`: compares `--scm linear` with
  `--scm location-scale`. Passing `--compute-nonlinear-metrics` sets
  `COMPUTE_NONLINEAR_METRICS=1` and writes optional GBR-DCI /
  kernel-ridge R2 metrics.
- `exp02_more_graphs_selection`: additional denser/larger graph cases with
  preconfigured selected auxiliary sets; no automatic graph-selection step
  or with-vs-without selection comparison.
- `exp04_beta_ablation`: beta-coefficient range ablation.
- `exp06_nonvp_ablation`: compares VP data-generating mixing with
  `--mixing nonvolumepreserving`, a Glow-style non-VP coupling-flow
  ablation.

Baseline paper runs are unchanged unless these flags or wrappers are used.

### Plotting Paper Figures

After the corresponding `result_*` directories are generated, the plotting
scripts in `plots/` recreate the paper figure panels:

| Script | Inputs | Output |
|--------|--------|--------|
| `plots/plot_4_2_dci.py` | `result_{ours,GIN,iVAE}_synthetic/{a,b,c_real}` | `plots/combined_architeccture.pdf` |
| `plots/plot_4_2_mcc.py` | `result_{ours,GIN,iVAE}_synthetic/{a,b,c_real}` | `plots/{a,b,c_real}_combined_plot_abl.pdf` |
| `plots/plot_4_2_dci_high.py` | `result_{ours,GIN,iVAE}_{pendulum,flow}/c_real` | `plots/combined_architeccture_img.pdf` |
| `plots/plot_4_2_mcc_high.py` | `result_{ours,GIN,iVAE}_{pendulum,flow}/c_real` | `plots/{pendulum,flow}_combined_plot_abl.pdf` |
| `plots/plot_4_1_dci.py` | `result_ours_flow/c_real` and separately generated `result_ours_flow_noselect/c_real` | `plots/combined_selection_abl_flow.pdf` |

The released runners do not provide a command to generate the Flow
no-selection comparison arm. The selection-ablation plot requires previously
generated `result_ours_flow_noselect/c_real` results before running
`plots/plot_4_1_dci.py`.

Example plotting commands:

```bash
python plots/plot_4_2_dci.py
python plots/plot_4_2_mcc.py
python plots/plot_4_2_dci_high.py
python plots/plot_4_2_mcc_high.py
python plots/plot_4_1_dci.py   # requires result_ours_flow_noselect/c_real
```

### Step 3: Interpret Results

`run_all.py` saves results under `result_*/` directories inside
`--result-root`, with the following structure:

```
result_<model>_<dataset>/
└── <dgp_name>/
    ├── <seed>_mcc.csv             # matched MCC correlation matrix
    ├── <seed>_mcc_meta.json       # MCC score and matching indices
    └── <seed>_dci.csv             # MCC-based DCI-style metrics
```

- **MCC score**: Mean Correlation Coefficient — higher is better (reported in the paper's bar charts)
- **MCC-based DCI-style metrics**:
  - `disentanglement`: How well each estimated latent captures only one true factor
  - `completeness`: How well each true factor is captured by only one estimated latent
  The matched MCC matrix is stored as `[true source, estimated latent]`; the
  DCI helper receives its transpose so these names follow the paper convention.
  Scores are weighted by each coordinate's or factor's total absolute
  correlation, with a `1e-11` offset for entropy evaluation.
- **Optional nonlinear metrics**: When `COMPUTE_NONLINEAR_METRICS=1`, the
  proposed-method script also writes `<seed>_nonlinear_metrics.csv` with
  GBR-DCI and kernel-ridge R2 summaries.

### Evaluation matching convention (proposed method vs. baselines)

In the code, `selected_idx` denotes `C`, while `observed_idx` denotes the
observed-but-unselected set `O minus C`. All methods evaluate the same unobserved
target sources after excluding both sets. For `c_real`, selected `[3]` and
observed-but-unselected `[2]` leave target indices `[0, 1]`.

The learned matching candidates differ:

```python
excluded_idx = set(selected_idx) | set(observed_idx)
target_idx = sorted(set(range(v.shape[1])) - excluded_idx)
candidate_idx = sorted(set(range(z.shape[1])) - excluded_idx)

# Proposed method: permutation across the candidate set is allowed.
mean_corr_coef(v[:, target_idx], z[:, candidate_idx])
# GIN / iVAE: match against the full learned representation.
mean_corr_coef(v[:, target_idx], z)
```

For synthetic runs, the true and learned dimensions agree, so the proposed
method's candidate indices equal the target indices. For image runs, its
learned representation has the image dimension, yielding many more candidates.
This convention does not guarantee source-to-slot alignment. For a fixed
representation, enlarging the candidate set cannot reduce optimized MCC,
but need not improve the DCI-style scores computed after matching.

## Hardware Requirements

- **Synthetic experiments**: Can run on a single GPU (NVIDIA GPU with ≥4GB VRAM)
- **Image experiments** (Flow, Pendulum): Recommended ≥8GB VRAM and enough host
  RAM to load the PNG tensors into memory; reduce `--batch-size` if memory is
  tight
- **CPU fallback**: Modify `--accelerator=cpu` in the experiment commands (significantly slower)
- **iVAE device selection**: Fixed distribution tensors are initialized on the
  device selected by Lightning, including when `--accelerator auto` is used.

## Notes

- The implementation of the volume-preserving encoder is built upon the [CauCA](https://github.com/akekic/causal-component-analysis) codebase.
- The Pendulum and Flow dataset generators are adapted from CausalVAE (Yang et al., 2021).
- The DCI-style score follows the DisentanglementLib entropy formulation with MCC-based importance values.
- The graph prior predicts observed-but-unselected values from the full
  learned representation at each node's parent indices and its own slot.
  Parentless nodes use their own slot directly. This prediction penalty
  does not guarantee source-to-slot alignment.
- Third-party code notices are listed in `THIRD-PARTY-NOTICES.md`.

## Citation

If you use this code, please cite the accompanying paper. Citation metadata is
provided in [CITATION.cff](CITATION.cff), including the preferred UAI paper citation.

## License

This repository is released under the MIT License; see `LICENSE`.
