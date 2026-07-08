# exp02_more_graphs_selection

## Motivation

- Motivation: check robustness beyond the limited set of graph structures in Fig. 3.
- Goal: check whether Algorithm 1 with selection enabled remains stable on
  denser or larger graphs.

## What we do

Five additional synthetic DGPs are defined in ``extra_graphs.py``
(``c_dense``, ``c_deep``, ``c_chain``, ``c_obs_chain``, ``c_hub6``).
All of them extend the style of ``c_real`` (Fig. 1d) with more nodes
and/or denser parent sets. We run Algorithm 1 with selection enabled
(its default configuration) on each graph and compare to the paper's
``c_real`` baseline from ``config.DGP``. The hypothesis: the method
continues to deliver strong DCI / MCC as graphs grow denser.

No edit is made to ``config.py``. Instead, the checked-in wrapper
``_wrapper.py`` registers ``EXTRA_DGPS`` into ``config.DGP`` for that
subprocess and then invokes ``experiments.Ours.main`` via ``runpy``.

## Why not with-vs-without selection?

The obvious experiment -- toggle ``selected_idx = []`` on the same
graph -- is not realisable in the current codebase. Algorithm 1's loss
slices the latent by ``selected_idx`` and then computes a covariance /
multivariate normal density on the slice; with an empty index set the
call crashes in ``training_step`` (``torch.cov`` / ``MultivariateNormal``
on empty tensors). Rather than fight the training step, we reframe the
study: run Algorithm 1 on a spectrum of graph densities and show that
it holds up from 4-node sparse (``c_real``) through 6-node dense
(``c_hub6``).

## Expected output

* ``results/{dgp}/{seed}_dci.csv``, ``{seed}_mcc.csv``, and
  ``{seed}_mcc_meta.json``
* ``results/selection_ablation_summary.{csv,tex}`` (paper-ready).

## Runtime estimate

Single run ~6-8 min on A100. Full sweep (6 DGPs x 5 seeds = 30 runs)
~3-4 GPU-hr.

## How to interpret

If Algorithm 1's DCI / MCC remain strong on the denser graphs
(``c_hub6``, ``c_deep``, ``c_dense``), this supports the robustness of
the selection-enabled method beyond the Fig. 3 graphs. It does not claim
to measure a with-vs-without selection delta. If the baseline
(``c_real``) and the dense graphs give similar numbers, we still report
them and note that Fig. 3's setting is representative.

## Author follow-up

- Run the sweep, then inspect ``selection_ablation_summary.csv`` /
  ``.tex`` and compare the table against the paper appendix.
- If a seed crashes (GPU OOM, numerical blow-up, etc.), it gets logged
  to the run's ``*_errors.log`` file and the sweep continues unless
  ``--stop-on-error`` is passed.
