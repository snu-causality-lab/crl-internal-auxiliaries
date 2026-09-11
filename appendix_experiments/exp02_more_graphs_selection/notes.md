# exp02_more_graphs_selection

## Motivation

- Goal: evaluate the proposed training method on denser or larger graphs
  using preconfigured selected auxiliary sets.

## What we do

Five additional synthetic DGPs are defined in ``extra_graphs.py``
(``c_dense``, ``c_deep``, ``c_chain``, ``c_obs_chain``, ``c_hub6``).
They vary the topology of ``c_real`` (Fig. 1d), including denser graphs
and longer chains. Each graph uses the ``selected_idx`` in its DGP spec.
The paper's ``c_real`` baseline from ``config.DGP`` is included with
``--include-baseline``. These runs evaluate recovery with fixed auxiliary
sets; they do not execute Algorithm 1 or Bayes-ball.

No edit is made to ``config.py``. Instead, the checked-in wrapper
``_wrapper.py`` registers ``EXTRA_DGPS`` into ``config.DGP`` for that
subprocess and then invokes ``experiments.Ours.main`` via ``runpy``.

## Scope of the comparison

This sweep does not compare the chosen set ``C`` with conditioning on all
observed sources (``C = O``). An empty ``selected_idx`` instead removes
all conditioning sources and is unsupported by the current covariance /
multivariate-normal training term. That limitation is separate from a
selection-vs-all-observed comparison.

## Expected output

* ``results/{dgp}/{seed}_dci.csv``, ``{seed}_mcc.csv``, and
  ``{seed}_mcc_meta.json``
* ``results/selection_ablation_summary.{csv,tex}``

## Runtime estimate

Single run ~6-8 min on A100. Sweep including the baseline (6 DGPs x 5 seeds = 30 runs)
~3-4 GPU-hr.

## How to interpret

Compare DCI / MCC across graph configurations and seeds to assess empirical
recovery with the specified auxiliary sets. Differences do not isolate the
benefit of the selection algorithm, since both the graphs and selected sets
vary. This sweep is not a reported App. E table.

## Author follow-up

- Inspect ``selection_ablation_summary.csv`` / ``.tex`` alongside each
  graph's configured auxiliary set.
- If a seed crashes (GPU OOM, numerical blow-up, etc.), it gets logged
  to the run's ``*_errors.log`` file and the sweep continues unless
  ``--stop-on-error`` is passed.
