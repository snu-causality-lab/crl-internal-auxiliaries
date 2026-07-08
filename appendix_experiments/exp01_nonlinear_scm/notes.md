# exp01_nonlinear_scm

## Motivation

- Motivation: check whether results rely on the default linear SCM.
- Goal: check whether recovery remains stable under non-linear latent
  mechanisms.

## What we do

The main training script `experiments/Ours/main.py` already exposes a CLI
flag `--scm {linear, location-scale}`. The `location-scale` option triggers
`data_generator.scm.LocationScaleSCM`, which builds mechanisms of the form
`v_i = snr * f_loc(pa_i) + f_scale(u_i)` where `f_loc` and `f_scale` are
random non-linear compositions of `leaky_tanh` layers (see
`data_generator/utils.py:108-153`). The core training path is unchanged for
the SCM sweep; optional nonlinear metrics are enabled only when
`COMPUTE_NONLINEAR_METRICS=1` is set through
`run_nonlinear.py --compute-nonlinear-metrics`.

The sweep covers the three synthetic DGPs the paper emphasizes
(`a`, `b`, `c_real`) and two SCMs (`linear`, `location-scale`) with seeds
`0..4` (pilot), `0..19` (paper convention), or `0..24` (appendix
nonlinear-metric tables). The seed range is contiguous, matching
`example_GIN.py`.

## Expected output

`results/{scm}/{dgp}/{seed}_dci.csv` (DCI disentanglement + completeness,
written by `model/our_model.test_epoch_end`) and
`results/{scm}/{dgp}/{seed}_mcc.csv` (matched correlation matrix) and
`{seed}_mcc_meta.json` (MCC score and matching indices). `aggregate.py`
compiles these into
`nonlinear_scm_summary.{csv,tex}`. If nonlinear metrics are enabled, each seed
also writes `{seed}_nonlinear_metrics.csv`; the aggregator adds GBR-DCI and
kernel-ridge R2 columns automatically.

## Runtime estimate

- One run (20 epochs, synthetic, batch 1024, k_flows=8) takes ~6-8 min on a
  single RTX-A6000 / A100 based on the paper's training budget.
- Pilot (3 DGPs x 2 SCMs x 5 seeds = 30 runs): ~3-4 GPU-hr.
- Full (3 DGPs x 2 SCMs x 25 seeds = 150 runs): ~15-20 GPU-hr.

## How to interpret

Appendix claim: DCI disentanglement and MCC under `scm=location-scale`
should be comparable to those under `scm=linear` (the paper's default). If
the gap is within ~0.05 for DCI disentanglement and ~0.03 for MCC, our
identifiability story carries over to non-linear SCMs. A larger degradation
would be a real finding to disclose.

`aggregate.py` writes a LaTeX booktabs snippet for the paper appendix
(table has columns `DGP`, `SCM`, `DCI disent.`,
`DCI compl.`, `MCC`, `n`).

## Author follow-up

- After running, compare `results/nonlinear_scm_summary.tex` with the paper
  appendix if the table is retained.
- Use the final manuscript as the reference for appendix text and tables.
