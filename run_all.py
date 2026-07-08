"""
Unified experiment runner for all models and datasets.

Usage:
    python run_all.py --all                                # Run everything
    python run_all.py --synthetic                          # All models, synthetic
    python run_all.py --flow                               # All models, Flow image
    python run_all.py --pendulum                           # All models, Pendulum image
    python run_all.py --ours --synthetic                   # Ours on synthetic only
    python run_all.py --gin --flow                         # GIN on Flow only
    python run_all.py --seeds 0 5                          # Custom seed range [0, 5)
    python run_all.py --dgp-list a b                       # Subset of synthetic DGPs
    python run_all.py --ours --synthetic \
        --max-epochs 2 --accelerator cpu                   # Smoke test on CPU

Result directory layout (consumed by plots/*.py):
    result_<model>_<mode>/<dgp>/<seed>_dci.csv
    result_<model>_<mode>/<dgp>/<seed>_mcc.csv
    result_<model>_<mode>/<dgp>/<seed>_mcc_meta.json

    <model> in {ours, GIN, iVAE}
    <mode>  in {synthetic, flow, pendulum}
    <dgp>   in {a, b, c_real} for synthetic; c_real for image modes
"""

import argparse
import os
import shlex
import subprocess
import sys


PYTHON = sys.executable

PAPER_EPOCHS = {
    ("ours", "synthetic"): 20,
    ("ours", "flow"): 50,
    ("ours", "pendulum"): 80,
    ("gin", "synthetic"): 20,
    ("gin", "flow"): 40,
    ("gin", "pendulum"): 40,
    ("ivae", "synthetic"): 20,
    ("ivae", "flow"): 80,
    ("ivae", "pendulum"): 80,
}


def run_command(command):
    print(f"\n{'=' * 80}")
    print(f"Running: {shlex.join(command)}")
    print(f"{'=' * 80}")
    subprocess.run(command, check=True)


def _epochs(args, model, mode):
    return args.max_epochs if args.max_epochs is not None else PAPER_EPOCHS[(model, mode)]


def _batch_size(args):
    return [] if args.batch_size is None else [f"--batch-size={args.batch_size}"]


def _extra(args, model, mode):
    """Common extra flags appended to every run."""
    return [
        f"--max-epochs={_epochs(args, model, mode)}",
        f"--accelerator={args.accelerator}",
        f"--device={args.device}",
    ] + _batch_size(args)


def run_ours_synthetic(seeds, dgp_list, args):
    result_dir = "./result_ours_synthetic"
    os.makedirs(result_dir, exist_ok=True)
    for seed in seeds:
        for dgp_name in dgp_list:
            os.makedirs(os.path.join(result_dir, dgp_name), exist_ok=True)
            cmd = [
                PYTHON,
                "-m",
                "experiments.Ours.main",
                f"--dgp={dgp_name}",
                f"--seed={seed}",
                f"--training-seed={seed}",
                f"--result-dir={result_dir}",
                "--lr-scheduler=cosine",
            ] + _extra(args, "ours", "synthetic")
            run_command(cmd)


def run_ours_flow(seeds, args):
    result_dir = "./result_ours_flow"
    os.makedirs(os.path.join(result_dir, "c_real"), exist_ok=True)
    for seed in seeds:
        cmd = [
            PYTHON,
            "-m",
            "experiments.Ours.main",
            "--dgp=c_real",
            f"--seed={seed}",
            f"--training-seed={seed}",
            f"--result-dir={result_dir}",
            "--lr-scheduler=cosine",
            "--mode=flow",
            "--check-val-every-n-epoch=50",
            "--lr=0.001",
            "--k-flows=8",
        ] + _extra(args, "ours", "flow")
        run_command(cmd)


def run_ours_pendulum(seeds, args):
    result_dir = "./result_ours_pendulum"
    os.makedirs(os.path.join(result_dir, "c_real"), exist_ok=True)
    for seed in seeds:
        cmd = [
            PYTHON,
            "-m",
            "experiments.Ours.main",
            "--dgp=c_real",
            f"--seed={seed}",
            f"--training-seed={seed}",
            f"--result-dir={result_dir}",
            "--mode=pendulum",
            "--lr-scheduler=cosine",
            "--lr=0.001",
        ] + _extra(args, "ours", "pendulum")
        run_command(cmd)


def run_gin_synthetic(seeds, dgp_list, args):
    result_dir = "./result_GIN_synthetic"
    os.makedirs(result_dir, exist_ok=True)
    for seed in seeds:
        for dgp_name in dgp_list:
            os.makedirs(os.path.join(result_dir, dgp_name), exist_ok=True)
            cmd = [
                PYTHON,
                "-m",
                "experiments.GIN.main",
                f"--dgp={dgp_name}",
                f"--seed={seed}",
                f"--training-seed={seed}",
                f"--result-dir={result_dir}",
            ] + _extra(args, "gin", "synthetic")
            run_command(cmd)


def run_gin_flow(seeds, args):
    result_dir = "./result_GIN_flow"
    os.makedirs(os.path.join(result_dir, "c_real"), exist_ok=True)
    for seed in seeds:
        cmd = [
            PYTHON,
            "-m",
            "experiments.GIN.main",
            "--dgp=c_real",
            f"--seed={seed}",
            f"--training-seed={seed}",
            f"--result-dir={result_dir}",
            "--mode=flow",
            "--lr=0.001",
        ] + _extra(args, "gin", "flow")
        run_command(cmd)


def run_gin_pendulum(seeds, args):
    result_dir = "./result_GIN_pendulum"
    os.makedirs(os.path.join(result_dir, "c_real"), exist_ok=True)
    for seed in seeds:
        cmd = [
            PYTHON,
            "-m",
            "experiments.GIN.main",
            "--dgp=c_real",
            f"--seed={seed}",
            f"--training-seed={seed}",
            f"--result-dir={result_dir}",
            "--mode=pendulum",
            "--lr=0.001",
        ] + _extra(args, "gin", "pendulum")
        run_command(cmd)


def run_ivae_synthetic(seeds, dgp_list, args):
    result_dir = "./result_iVAE_synthetic"
    os.makedirs(result_dir, exist_ok=True)
    for seed in seeds:
        for dgp_name in dgp_list:
            os.makedirs(os.path.join(result_dir, dgp_name), exist_ok=True)
            cmd = [
                PYTHON,
                "-m",
                "experiments.iVAE.main",
                f"--dgp={dgp_name}",
                f"--seed={seed}",
                f"--training-seed={seed}",
                f"--result-dir={result_dir}",
                "--batch-size=32",
                "--lr=0.0001",
            ] + _extra(args, "ivae", "synthetic")
            run_command(cmd)


def run_ivae_flow(seeds, args):
    result_dir = "./result_iVAE_flow"
    os.makedirs(os.path.join(result_dir, "c_real"), exist_ok=True)
    for seed in seeds:
        cmd = [
            PYTHON,
            "-m",
            "experiments.iVAE.main",
            "--dgp=c_real",
            f"--seed={seed}",
            f"--training-seed={seed}",
            f"--result-dir={result_dir}",
            "--mode=flow",
            "--lr=0.0001",
        ] + _extra(args, "ivae", "flow")
        run_command(cmd)


def run_ivae_pendulum(seeds, args):
    result_dir = "./result_iVAE_pendulum"
    os.makedirs(os.path.join(result_dir, "c_real"), exist_ok=True)
    for seed in seeds:
        cmd = [
            PYTHON,
            "-m",
            "experiments.iVAE.main",
            "--dgp=c_real",
            f"--seed={seed}",
            f"--training-seed={seed}",
            f"--result-dir={result_dir}",
            "--mode=pendulum",
            "--lr=0.0001",
        ] + _extra(args, "ivae", "pendulum")
        run_command(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified experiment runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    parser.add_argument("--ours", action="store_true", help="Run proposed method")
    parser.add_argument("--gin", action="store_true", help="Run GIN baseline")
    parser.add_argument("--ivae", action="store_true", help="Run iVAE baseline")

    # Dataset selection
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic data")
    parser.add_argument("--flow", action="store_true", help="Run on Flow dataset")
    parser.add_argument("--pendulum", action="store_true", help="Run on Pendulum dataset")

    # Convenience
    parser.add_argument("--all", action="store_true", help="Run all experiments")

    # Configuration
    parser.add_argument(
        "--seeds", nargs=2, type=int, default=[0, 20], metavar=("START", "END"),
        help="Seed range [START, END). Default: 0 20 (matches paper).",
    )
    parser.add_argument(
        "--dgp-list", nargs="+", default=["a", "b", "c_real"],
        help="DGP names for synthetic experiments. Default: a b c_real.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="Override max training epochs per run; by default, use the paper settings.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size per run; by default, use each entry point's paper setting.",
    )
    parser.add_argument(
        "--accelerator", type=str, default="gpu", choices=["gpu", "cpu", "mps", "auto"],
        help="Lightning accelerator. Use cpu or mps for local smoke tests on Mac.",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU index when accelerator=gpu (ignored otherwise).",
    )

    args = parser.parse_args()

    if args.all:
        args.ours = args.gin = args.ivae = True
        args.synthetic = args.flow = args.pendulum = True

    if not (args.ours or args.gin or args.ivae):
        args.ours = args.gin = args.ivae = True
    if not (args.synthetic or args.flow or args.pendulum):
        args.synthetic = args.flow = args.pendulum = True

    seeds = list(range(args.seeds[0], args.seeds[1]))
    dgp_list = args.dgp_list

    print(f"Seeds: {seeds}")
    print(f"DGPs: {dgp_list}")
    print(f"Models: ours={args.ours}, gin={args.gin}, ivae={args.ivae}")
    print(f"Datasets: synthetic={args.synthetic}, flow={args.flow}, pendulum={args.pendulum}")
    epoch_label = args.max_epochs if args.max_epochs is not None else "paper defaults"
    print(f"Epochs: {epoch_label}, accelerator: {args.accelerator}, device: {args.device}")

    if args.ours and args.synthetic:
        run_ours_synthetic(seeds, dgp_list, args)
    if args.ours and args.flow:
        run_ours_flow(seeds, args)
    if args.ours and args.pendulum:
        run_ours_pendulum(seeds, args)

    if args.gin and args.synthetic:
        run_gin_synthetic(seeds, dgp_list, args)
    if args.gin and args.flow:
        run_gin_flow(seeds, args)
    if args.gin and args.pendulum:
        run_gin_pendulum(seeds, args)

    if args.ivae and args.synthetic:
        run_ivae_synthetic(seeds, dgp_list, args)
    if args.ivae and args.flow:
        run_ivae_flow(seeds, args)
    if args.ivae and args.pendulum:
        run_ivae_pendulum(seeds, args)

    print("\n" + "=" * 80)
    print("All selected experiments completed!")
    print("=" * 80)
