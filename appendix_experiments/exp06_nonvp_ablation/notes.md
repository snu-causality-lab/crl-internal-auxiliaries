# exp06_nonvp_ablation

## Motivation

- Motivation: good likelihood under a VP estimator does not by itself justify the
  VP assumption on the data-generating mixing process.

## What we do

This sweep changes only the data-generating mixing function:

| label | data-generating mixing |
|-------|------------------------|
| `volumepreserving` | 8-block FrEIA `GINCouplingBlock` flow |
| `nonvolumepreserving` | 8-block FrEIA `GLOWCouplingBlock` flow |

The two settings use the same fully connected subnet constructor, clamp value
2.0, random permutations, optimizer settings, SCM, graphs, and training budget.
The estimator remains the paper's VP flow model; only the synthetic data
generation changes.

## Runs

The appendix table uses graphs `a`, `b`, and `c_real`, both mixing settings,
seeds `0..24`, and 20 epochs:

```bash
python appendix_experiments/exp06_nonvp_ablation/run_nonvp.py \
    --dgp a b c_real \
    --mixing both \
    --seeds 0-24 \
    --max-epochs 20 \
    --result-dir appendix_experiments/exp06_nonvp_ablation/results
```

For a smoke test, run:

```bash
bash appendix_experiments/exp06_nonvp_ablation/run_local.sh
```

The smoke test uses one graph and one seed only. It verifies that both VP and
non-VP code paths execute; it does not reproduce the 25-seed appendix table.

## Outputs

`results/{mixing}/{dgp}/{seed}_dci.csv` contains DCI disentanglement and
completeness. The matched MCC score is stored in the corresponding
`{seed}_mcc_meta.json` file. `aggregate.py` compiles these into
`nonvp_ablation_summary.{csv,tex}`.

## Interpretation

Use the aggregate table to compare VP and non-VP data-generating mixing across
graphs and metrics. This is an empirical recovery-sensitivity check for the
current pipeline, not a claim that VP is necessary for identifiability in
general.
