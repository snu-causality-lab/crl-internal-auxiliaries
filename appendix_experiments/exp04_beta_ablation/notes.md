# exp04_beta_ablation

## Motivation

- Motivation: check the synthetic SCM setting in Sec. 5 / Fig. 3, which fixes
  `beta_ij ~ Uniform[0.5, 1.0]` (positive only, |beta| >= 0.5), which
  does not exercise negative or small-magnitude coefficients.

This ablation supports the appendix table for beta-range
robustness on the paper's four-node multi-observable synthetic graph settings.

## What we do

The training script `experiments/Ours/main.py` already exposes three
CLI flags that thread through `data_generator.multi_env_gdp` into
`LinearSCM`:

- `--scm-coeffs-low`  (default: `0.5`)
- `--scm-coeffs-high` (default: `1.0`)
- `--scm-coeffs-min-abs-value` (default: `0.5`)

We sweep three `(label, low, high, min_abs)` configurations:

| label       | low   | high | min_abs | note                                 |
|-------------|-------|------|---------|--------------------------------------|
| `paper`     |  0.5  | 1.0  | 0.5     | paper Sec.5 / Fig.3 caption (baseline) |
| `symmetric` | -1.0  | 1.0  | 0.5     | `multi_env_gdp.py` default (negatives) |
| `small`     |  0.1  | 0.3  | 0.05    | small magnitude, positive              |

Note on `min_abs`: `data_generator.utils.sample_coeffs` asserts
`min_abs < max(|low|, |high|)` and rejection-samples until every
coefficient has `|beta| >= min_abs`. For the `small` config we lower
`min_abs` to `0.05 < 0.3` so the assert passes and the accept region
already covers the entire `Uniform[0.1, 0.3]` support (no rejections).

DGPs swept: `c` and `c_real` from `config.DGP` (4-node, both use
`selected_idx=[3]` with `observed_idx=[2]`). Seeds: `0..4` (pilot),
contiguous range.

The public training entry point already exposes these flags; this wrapper just
sweeps them via subprocess.

## Expected output

`results/{beta_label}/{dgp}/{seed}_dci.csv` (DCI disentanglement +
completeness, written by `model/our_model.test_epoch_end`) and
`results/{beta_label}/{dgp}/{seed}_mcc.csv` (matched correlation matrix) and
`{seed}_mcc_meta.json` (MCC score and matching indices). `aggregate.py`
compiles these into
`beta_ablation_summary.{csv,tex}` with columns
`DGP | beta range | DCI disent. | DCI compl. | MCC | n`.

## Runtime estimate

- One run (20 epochs, synthetic, batch 1024, k_flows=8) takes ~6-8 min
  on a single RTX-A6000 / A100 based on the paper's training budget.
- Pilot (3 beta x 2 DGPs x 5 seeds = 30 runs): **~3-4 GPU-hr**.
- Full (3 beta x 2 DGPs x 25 seeds = 150 runs): ~15-20 GPU-hr.

## How to interpret

Appendix claim: DCI disentanglement and MCC should degrade
*gracefully* (not catastrophically) when beta crosses into negatives
(`symmetric`) or becomes small (`small`). If all three configs land
within ~0.05 DCI disent. / ~0.03 MCC of each other, the positivity
and magnitude assumption in the paper's synthetic setup is incidental
rather than load-bearing.

If `small` collapses (which would be unsurprising: identifying signs
and low-SNR edges is genuinely harder), we report this honestly as a
scope statement -- the paper's method targets the moderate-signal
regime and `small` falls below that threshold.

`aggregate.py` writes a LaTeX booktabs snippet for checking against the
paper appendix.

## Author follow-up

- After running, copy `results/beta_ablation_summary.tex` into the
  paper appendix if the table is retained.
- Use the three-row summary to report the `paper` baseline number,
  `symmetric` delta, and `small` delta if space permits.
- Note only if needed: the public `experiments/Ours/main.py` defaults to the
  paper range. The lower-level `make_multi_env_dgp` factory retains broader
  library defaults, but released experiment scripts pass explicit paper-range
  values.
