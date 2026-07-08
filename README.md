# On Causal Representation Learning with Internal Auxiliaries

This repository provides the implementation for reproducing the experiments in
the UAI 2026 paper *On Causal Representation Learning with Internal
Auxiliaries*.

## Overview

The paper addresses the challenge of causal representation learning (CRL) when observable sources act as **internal** auxiliaries in the mixing process. Standard identifiability proofs (e.g., multi-environment ICA/ISA) break down in this setting because the auxiliary information is entangled with the latent variables through the mixing function. Our framework achieves identifiability through:

1. **Volume-preserving mixing assumption** — using GIN (General Incompressible-flow Networks) coupling blocks
2. **Graph-aware variable selection** — selecting which observable sources to use as auxiliaries based on the causal graph structure

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
| `a`       | Fig. 1(a)            | 5 nodes, single auxiliary (ICA setting)             |
| `b`       | Fig. 1(c)            | 5 nodes, single observable source (ISA setting)     |
| `b_aug`   | Fig. 1(c) augmented  | 5 nodes, augmented observable                       |
| `c`       | App. F beta ablation | 4 nodes, multiple observables                       |
| `c_aug`   | —                    | 4 nodes, augmented multiple observables             |
| `c_real`  | Fig. 1(d)            | 4 nodes; selected source `s4`, observed-but-unselected source `s2` |

## Reproducing Experiments

Run the commands below from the repository root; the repository is not installed
as a Python package.

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

#### Quick Smoke Test

After installing dependencies, run a short CPU-only check:

```bash
bash scripts/smoke_test.sh
```

This runs the proposed method on DGP `a` for one epoch and one seed. It is
intended only to verify that imports, data generation, training, and result
writing work in the current environment.

### Additional Appendix Experiments

The `appendix_experiments/` directory contains reproducible wrappers for
additional experiments reported in the paper appendix:

- `exp01_nonlinear_scm`: compares `--scm linear` with
  `--scm location-scale`. Passing `--compute-nonlinear-metrics` sets
  `COMPUTE_NONLINEAR_METRICS=1` and writes optional GBR-DCI /
  kernel-ridge R2 metrics.
- `exp02_more_graphs_selection`: additional denser/larger graph cases with
  the selection step enabled.
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

The Flow selection-ablation plot is part of the paper figures, but its
no-selection result directory is not produced by the default `run_all.py`
pipeline. Generate or provide `result_ours_flow_noselect/c_real` before running
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

Results are saved under `result_*/` directories with the following structure:

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
- **Optional nonlinear metrics**: When `COMPUTE_NONLINEAR_METRICS=1`, the
  proposed-method script also writes `<seed>_nonlinear_metrics.csv` with
  GBR-DCI and kernel-ridge R2 summaries.

### Evaluation matching convention (proposed method vs. baselines)

All MCC/DCI scores are computed on the **unobserved target sources** — the
coordinates that remain after removing the selected and observed sources
(`target_idx = all_idx - selected_idx - observed_idx`). For DGP `c_real`
(`selected_idx=[3]`, `observed_idx=[2]`) the target sources are `{0, 1}`, and
this target set is identical for the proposed method, GIN, and iVAE.

The two families differ only in **which learned coordinates the target sources
are matched against**:

- **Proposed method**: the graph constraint confines each target source to the
  corresponding learned target block, so the true target block is matched against
  the *same-indexed* learned target block (the metric still resolves the
  within-block permutation) —
  `mean_corr_coef(v[:, target_idx], z[:, target_idx])`.
- **GIN / iVAE baselines**: the target sources are not pre-aligned to particular
  learned coordinates, so the true target block is matched against the *full*
  learned representation and the metric selects the best-correlated coordinates —
  `mean_corr_coef(v[:, target_idx], z)`.

Matching against the full learned representation gives the baselines the larger
search space, so this is a deliberately **more lenient** evaluation for the
baselines and does not disadvantage them relative to the proposed method.

## Hardware Requirements

- **Synthetic experiments**: Can run on a single GPU (NVIDIA GPU with ≥4GB VRAM)
- **Image experiments** (Flow, Pendulum): Recommended ≥8GB VRAM and enough host
  RAM to load the PNG tensors into memory; reduce `--batch-size` if memory is
  tight
- **CPU fallback**: Modify `--accelerator=cpu` in the experiment commands (significantly slower)

## Notes

- The implementation of the volume-preserving encoder is built upon the [CauCA](https://github.com/akekic/causal-component-analysis) codebase.
- The Pendulum and Flow dataset generators are adapted from CausalVAE (Yang et al., 2021).
- The DCI-style score follows the DisentanglementLib entropy formulation with MCC-based importance values.
- The graph-prior implementation uses graph predecessor indices as fixed
  learned-coordinate slots for each node-wise predictor.
- Third-party code notices are listed in `THIRD-PARTY-NOTICES.md`.

## Citation

If you use this code, please cite the accompanying paper. Citation metadata is
provided in `CITATION.cff`.

## License

This repository is released under the MIT License; see `LICENSE`.
