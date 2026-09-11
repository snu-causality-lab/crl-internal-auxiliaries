"""Run the real iVAE initialization and one ELBO evaluation, without training.

The initialization statements are read from experiments/iVAE/main.py rather
than reimplementing its device policy. No data module, Trainer.fit/test,
optimizer, checkpoint, or result writer is invoked. A small adapter calls the
real nets.iVAE with hidden_dim=8 instead of the wrapper's default 4096; its
Normal sampling and ELBO code are unchanged. Use --full-model to check the
normal iVAEWrapper as well. Inputs are two fixed zero-valued rows.

Example:
    python scripts/reproduce_ivae_auto_device.py --accelerator auto

To compare old initialization against the same unchanged model implementation:
    git show a5e73e0:experiments/iVAE/main.py > /tmp/ivae-main-before.py
    python scripts/reproduce_ivae_auto_device.py --main-source /tmp/ivae-main-before.py

Exit codes: 0 = finite ELBO; 1 = runtime failure/nonfinite ELBO; 2 = unavailable
dependencies. An auto run on CPU cannot reproduce the accelerator mismatch;
the JSON report explicitly identifies it as a CPU-only control.
"""

import argparse
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = REPO_ROOT / "experiments" / "iVAE" / "main.py"


def initialization_block(source_path):
    """Extract only the existing model/Trainer construction before fit()."""
    tree = ast.parse(Path(source_path).read_text(), filename=str(source_path))
    main_guard = next(
        node for node in tree.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "__name__ == '__main__'"
    )
    fit_index = next(
        index for index, node in enumerate(main_guard.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "trainer.fit"
    )
    initialization_names = {"_tensor_device", "_devices", "model", "trainer"}
    start = fit_index
    while start:
        assigned = {
            node.id for node in ast.walk(main_guard.body[start - 1])
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        if not assigned.intersection(initialization_names):
            break
        start -= 1
    nodes = main_guard.body[start:fit_index]
    assigned = {
        node.id for item in nodes for node in ast.walk(item)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    if assigned != initialization_names:
        raise ValueError("Cannot identify the complete iVAE initialization block")
    return compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec")


def run_probe(accelerator="auto", device=0, main_source=MAIN_SOURCE, full_model=False):
    # Imports stay optional for the standard-library test suite.
    import pytorch_lightning as pl
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    if full_model:
        from experiments.iVAE.wrappers import iVAEWrapper
    else:
        from experiments.iVAE.nets import iVAE

        def iVAEWrapper(*, data_dim, latent_dim, selected_idx, device, **unused):
            # Only replace the wrapper's fixed hidden width. Use the real
            # network/distributions and Lightning's actual module transfer.
            model = pl.LightningModule()
            model.encoder = iVAE(
                data_dim=data_dim, latent_dim=latent_dim,
                aux_dim=len(selected_idx), hidden_dim=8, device=device,
            )
            return model

    args = SimpleNamespace(
        accelerator=accelerator, device=device, model="nonlinear",
        max_epochs=1, wandb=False, check_val_every_n_epoch=1,
        lr=0.0001, lr_scheduler=None, lr_min=0.0,
        dgp="device_probe", training_seed=123, result_dir="unused_device_probe",
        mode="synthetic",
    )
    namespace = dict(
        args=args, pl=pl, iVAEWrapper=iVAEWrapper,
        data_dim=3, latent_dim=2, selected_idx=[2], observed_idx=[],
    )
    report = dict(
        main_source=str(main_source), torch_version=torch.__version__,
        lightning_version=pl.__version__, accelerator=accelerator,
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
        training_ran=False, full_model=full_model,
    )
    try:
        exec(initialization_block(main_source), namespace)
        model, trainer = namespace["model"], namespace["trainer"]
        report["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        report["constructor_device"] = str(namespace["_tensor_device"])
        target = trainer.strategy.root_device
        report["trainer_device"] = str(target)
        report["accelerator_exercised"] = target.type != "cpu"
        # This is the same registered-module move used by Lightning setup.
        trainer.strategy.model = model
        trainer.strategy.model_to_device()
        report["parameter_device"] = str(next(model.parameters()).device)
        report["sampling_device"] = str(model.encoder.encoder_dist._dist.loc.device)
        report["decoder_variance_device"] = str(model.encoder.decoder_var.device)
        model.eval()
        with torch.no_grad():
            elbo, _ = model.encoder.elbo(
                torch.zeros((2, 3), device=target),
                torch.zeros((2, 1), device=target),
            )
        report["finite_elbo"] = bool(torch.isfinite(elbo).item())
        report["status"] = "passed" if report["finite_elbo"] else "failed"
        if accelerator == "auto" and target.type == "cpu":
            report["note"] = "CPU-only control; no accelerator mismatch was tested."
    except RuntimeError as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu", "mps"), default="auto")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--main-source", type=Path, default=MAIN_SOURCE)
    parser.add_argument("--full-model", action="store_true", help="Use the real wrapper's 4096-wide model instead of the small adapter.")
    args = parser.parse_args()
    try:
        report = run_probe(args.accelerator, args.device, args.main_source, args.full_model)
    except ImportError as error:
        print(json.dumps({"status": "unavailable", "error": str(error)}, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
