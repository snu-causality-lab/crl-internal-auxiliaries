"""Sweep runner for exp01 (non-linear SCM).

Runs ``experiments.Ours.main`` with ``--scm location-scale`` (nonlinear
mechanisms built from leaky_tanh, already implemented in
``data_generator/scm.py:LocationScaleSCM``) and compares against the paper's
default ``--scm linear`` for a small set of synthetic DGPs (``a``, ``b``,
``c_real``).

The SCM sweep itself only launches ``python -m experiments.Ours.main`` with
different flags. When ``--compute-nonlinear-metrics`` is passed, the
subprocess also enables the release-only ``COMPUTE_NONLINEAR_METRICS=1`` hook
in ``model/our_model.py``.

Usage
-----
    python appendix_experiments/exp01_nonlinear_scm/run_nonlinear.py \
        --dgp a b c_real \
        --scm both \
        --seeds 0-4 \
        --max-epochs 20 \
        --result-dir appendix_experiments/exp01_nonlinear_scm/results

The seeds range is inclusive on both ends ("0-4" means 0,1,2,3,4). Only
contiguous ranges are accepted -- the paper uses the contiguous block
``range(0, 20)`` and we want to preserve that convention.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Defaults match the paper: synthetic DGPs, 20 epochs, GIN encoder.
DEFAULT_DGPS = ["a", "b", "c_real"]
VALID_SCMS = ["linear", "location-scale"]


def _parse_seed_range(spec: str) -> list[int]:
    """Parse a contiguous seed range such as ``0-4`` or a single int.

    Raises ValueError if the range is not contiguous. We deliberately refuse
    comma-separated lists to avoid accidental non-adjacent subsets.
    """
    spec = spec.strip()
    if "-" in spec:
        lo_s, hi_s = spec.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        if hi < lo:
            raise ValueError(f"Seed range {spec!r} is reversed")
        return list(range(lo, hi + 1))
    # single seed
    return [int(spec)]


def _scm_choices(choice: str) -> list[str]:
    if choice == "both":
        return list(VALID_SCMS)
    if choice in VALID_SCMS:
        return [choice]
    raise ValueError(f"Unknown --scm value {choice!r}")


def _run_one(
    *,
    dgp: str,
    scm: str,
    seed: int,
    max_epochs: int,
    batch_size: int,
    lr: float,
    k_flows: int,
    result_dir: Path,
    accelerator: str,
    device: int,
    python_exe: str,
    compute_nonlinear_metrics: bool,
    extra_args: list[str],
    dry_run: bool,
) -> int:
    """Launch one training run and return the subprocess exit code."""
    # Keep the paper's directory layout: ``result_dir/{dgp}/{seed}_dci.csv``
    # is how ``model.our_model.test_epoch_end`` writes results. We therefore
    # use a distinct ``result_dir`` per (scm_type, dgp) so the CSVs don't
    # collide, and we create the per-DGP subfolder beforehand.
    run_result_dir = result_dir / scm

    cmd = [
        python_exe,
        "-m",
        "experiments.Ours.main",
        f"--dgp={dgp}",
        f"--seed={seed}",
        f"--training-seed={seed}",
        f"--max-epochs={max_epochs}",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
        f"--k-flows={k_flows}",
        f"--scm={scm}",
        f"--result-dir={run_result_dir}",
        f"--accelerator={accelerator}",
        f"--device={device}",
    ]
    cmd.extend(extra_args)

    log_path = run_result_dir / dgp / f"{seed}_stdout.log"
    print("[exp01] Launching:", " ".join(cmd))
    print("[exp01] Logging to:", log_path)

    env = os.environ.copy()
    if compute_nonlinear_metrics:
        env["COMPUTE_NONLINEAR_METRICS"] = "1"
        print("[exp01] COMPUTE_NONLINEAR_METRICS=1")

    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        logf.write("# Command: " + " ".join(cmd) + "\n")
        if compute_nonlinear_metrics:
            logf.write("# Env: COMPUTE_NONLINEAR_METRICS=1\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep --scm {linear, location-scale} across DGPs/seeds "
                    "for the paper's synthetic mode."
    )
    ap.add_argument(
        "--dgp",
        nargs="+",
        default=DEFAULT_DGPS,
        help="Which DGP keys from config.DGP to sweep (default: a b c_real).",
    )
    ap.add_argument(
        "--scm",
        choices=["linear", "location-scale", "both"],
        default="both",
        help="Which SCM type(s) to run (default: both).",
    )
    ap.add_argument(
        "--seeds",
        default="0-19",
        help="Contiguous seed range, e.g. '0-4' or '0-19' (default: 0-19).",
    )
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--k-flows", type=int, default=8)
    ap.add_argument(
        "--result-dir",
        type=Path,
        default=Path("appendix_experiments/exp01_nonlinear_scm/results"),
        help=(
            "Root results directory. Per-run CSVs end up under "
            "{result_dir}/{scm}/{dgp}/{seed}_*"
        ),
    )
    ap.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help=(
            "Optional per-invocation status TSV. Useful for Slurm arrays; "
            "defaults to {result-dir}/run_status.tsv."
        ),
    )
    ap.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for the subprocess (default: sys.executable).",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the sweep on the first non-zero exit. Default: log and continue.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    ap.add_argument(
        "--compute-nonlinear-metrics",
        action="store_true",
        help=(
            "Set COMPUTE_NONLINEAR_METRICS=1 for each run, producing "
            "{seed}_nonlinear_metrics.csv with GBR-DCI and kernel-ridge R2."
        ),
    )
    ap.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args appended verbatim to every sub-call.",
    )
    args = ap.parse_args()

    seeds = _parse_seed_range(args.seeds)
    scms = _scm_choices(args.scm)
    result_dir = args.result_dir.resolve()
    status_path = (
        args.status_file.resolve() if args.status_file is not None
        else result_dir / "run_status.tsv"
    )
    if not args.dry_run:
        result_dir.mkdir(parents=True, exist_ok=True)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(status_path, "w") as sf:
            sf.write("dgp\tscm\tseed\treturncode\n")

    total = len(args.dgp) * len(scms) * len(seeds)
    done = 0
    for dgp in args.dgp:
        for scm in scms:
            for seed in seeds:
                rc = _run_one(
                    dgp=dgp,
                    scm=scm,
                    seed=seed,
                    max_epochs=args.max_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    k_flows=args.k_flows,
                    result_dir=result_dir,
                    accelerator=args.accelerator,
                    device=args.device,
                    python_exe=args.python,
                    compute_nonlinear_metrics=args.compute_nonlinear_metrics,
                    extra_args=args.extra,
                    dry_run=args.dry_run,
                )
                done += 1
                print(f"[exp01] {done}/{total} dgp={dgp} scm={scm} "
                      f"seed={seed} rc={rc}")
                if not args.dry_run:
                    with open(status_path, "a") as sf:
                        sf.write(f"{dgp}\t{scm}\t{seed}\t{rc}\n")
                if rc != 0 and args.stop_on_error:
                    sys.exit(rc)

    if args.dry_run:
        print("[exp01] Dry run complete; no files were written.")
    else:
        print("[exp01] Sweep complete; status file at", status_path)


if __name__ == "__main__":
    main()
