# Reproduce the reported issues without training

These checks diagnose [smoke-output collisions (#1)](https://github.com/snu-causality-lab/crl-internal-auxiliaries/issues/1)
and [iVAE automatic device placement (#3)](https://github.com/snu-causality-lab/crl-internal-auxiliaries/issues/3).
They do not generate datasets, optimize models, or recompute paper results.
The documentation corrections are tracked in [#2](https://github.com/snu-causality-lab/crl-internal-auxiliaries/issues/2).

Run the commands from this checkout's root. To compare the original release,
create a separate checkout once:

```bash
git worktree add --detach ../crl-release-repro a5e73e0866820a8f8955ca2be3a47b97b1f5f2c9
```

## Smoke result collision

Python 3.10 or newer is sufficient; no third-party packages are needed.

```bash
python scripts/reproduce_smoke_output_collision.py --repo ../crl-release-repro
python scripts/reproduce_smoke_output_collision.py --repo .
```

The first command intentionally exits **1** on the released bug:
`status=collision`, `paper_result_overwritten=true`. The second exits **0**:
`status=isolated`, `paper_result_overwritten=false`.

The reproducer intercepts the launcher's subprocess calls, extracts the actual
DCI CSV write block, and writes synthetic sentinel metrics only inside a
temporary directory. Original and smoke destinations are derived from their
real command arguments. Existing experiment files are not accessed.

The regression tests include an append-mode control: identical paths alone
are not treated as evidence that historical CSV content disappeared.

## iVAE automatic device placement

Use an environment with the repository's Torch/Lightning dependencies
installed. The tested versions were PyTorch 2.5.1 and Lightning 1.9.5 on
Python 3.11.15/macOS with an actual MPS accelerator.

```bash
python scripts/reproduce_ivae_auto_device.py --accelerator auto --main-source ../crl-release-repro/experiments/iVAE/main.py
python scripts/reproduce_ivae_auto_device.py --accelerator auto
python scripts/reproduce_ivae_auto_device.py --accelerator cpu
```

On the tested MPS machine, the original initializer exits **1** with CPU
sampling tensors and MPS model parameters. The corrected initializer exits
**0** with a finite ELBO and matching devices. Explicit CPU and MPS controls
pass before and after the fix. CUDA was not runtime-tested here.

The script reads the actual model/Trainer initialization statements from the
selected source file. Its default adapter uses the real `nets.iVAE` and
`Normal` implementation with hidden width 8 (489 parameters), followed by
Lightning's actual module transfer and one ELBO evaluation on two fixed input
rows. `--full-model` also checks the original wrapper's normal hidden width;
it uses substantially more memory. Both sizes reproduced the original failure
and passed after the fix on MPS.

An `auto` run that selects CPU is explicitly identified as a CPU-only control,
not accelerator regression coverage. Missing dependencies produce exit **2**.

## Regression checks

```bash
python -B -m unittest discover -s tests -v
```

The standard-library environment runs seven launcher/CSV tests and explicitly
skips four ML tests. With the repository dependencies and MPS available, all
eleven tests pass. Device tests cover CPU, actual automatic and explicit
accelerator selection, and preservation of Python/NumPy/Torch random states
when the Trainer is constructed earlier. Hardware-dependent tests skip if
the relevant accelerator is unavailable.

The GitHub workflow runs the standard-library checks on Python 3.10 and 3.12.
Its success does not assert accelerator coverage; that is provided by the
separate hardware-dependent checks above.
