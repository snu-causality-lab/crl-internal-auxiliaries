# Third-Party Notices

This repository includes or adapts small portions of third-party code. The
upstream notices below are preserved in the relevant source files.

## DisentanglementLib

Files:

- `model/dci.py`
- `experiments/GIN/dci.py`
- `experiments/iVAE/dci.py`

These files adapt the DisentanglementLib implementation of disentanglement and
completeness metrics. DisentanglementLib is licensed under the Apache License,
Version 2.0. See the headers in the files above for the original notice and
`LICENSES/Apache-2.0.txt` for the license text.

## CausalVAE Dataset Generators

Files:

- `data_flows/flow.py`
- `data_pendulum/pendulum_dataset.py`

These dataset generators are adapted from CausalVAE code by Huawei
Technologies Co., Ltd. and are distributed under the MIT License as indicated
in their source headers. The MIT license terms are also included in this
repository's top-level `LICENSE`.

## CauCA

Files:

- `model/encoder.py`
- `model/normalizing_flow/`
- portions of `model/our_model.py`

The volume-preserving encoder and causal-flow implementation build on the
Causal Component Analysis (CauCA) codebase:
https://github.com/akekic/causal-component-analysis

CauCA is distributed under the MIT License; the MIT license terms are also
included in this repository's top-level `LICENSE`.

## iVAE-Style MCC Utilities

Files:

- `model/mcc.py`
- `experiments/iVAE/mcc.py`

These files include MCC and assignment utilities adapted from public nonlinear
ICA/iVAE evaluation code, including the auction-based matching routine used for
MCC-style evaluation.

## External Dependencies

Python packages installed through `requirements.txt` or
`requirements-frozen-linux.txt` remain under their own upstream licenses.
