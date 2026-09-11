"""Density / complexity sweep with preconfigured auxiliary sets.

This experiment evaluates the proposed training method on denser or larger
graphs, reporting DCI / MCC with each graph's fixed auxiliary set. It does
not execute the paper's graph-selection algorithm or Bayes-ball, and does
not measure a with-vs-without selection effect.

For every graph in ``extra_graphs.EXTRA_DGPS`` (and optionally the
paper's ``c_real`` baseline via ``--include-baseline``) we launch the
main entry point ``python -m experiments.Ours.main`` as a subprocess,
with ``selected_idx`` taken from the DGP spec. The extra DGPs are
registered at runtime via the checked-in wrapper ``_wrapper.py`` so
``config.py`` itself is not edited.

Usage
-----
    python appendix_experiments/exp02_more_graphs_selection/\
run_selection_ablation.py --seeds 0-4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # repository root


def _parse_seed_range(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec:
        lo_s, hi_s = spec.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        if hi < lo:
            raise ValueError(f"Seed range {spec!r} is reversed")
        return list(range(lo, hi + 1))
    return [int(spec)]


WRAPPER_PATH = HERE / "_wrapper.py"


def _ensure_wrapper() -> Path:
    """Return the checked-in subprocess wrapper path."""
    if not WRAPPER_PATH.exists():
        raise FileNotFoundError(f"Missing exp02 wrapper: {WRAPPER_PATH}")
    return WRAPPER_PATH


def _run_one(
    *,
    dgp: str,
    seed: int,
    max_epochs: int,
    batch_size: int,
    lr: float,
    k_flows: int,
    result_dir: Path,
    accelerator: str,
    device: int,
    python_exe: str,
    wrapper: Path,
    dry_run: bool,
) -> int:
    dgp_dir = result_dir / dgp

    cmd = [
        python_exe,
        str(wrapper),
        f"--dgp={dgp}",
        f"--seed={seed}",
        f"--training-seed={seed}",
        f"--max-epochs={max_epochs}",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
        f"--k-flows={k_flows}",
        f"--result-dir={result_dir}",
        f"--accelerator={accelerator}",
        f"--device={device}",
    ]
    log_path = dgp_dir / f"{seed}_stdout.log"
    print(f"[exp02] run {dgp} seed={seed} -> {log_path}")
    if dry_run:
        return 0
    dgp_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        logf.write("# Command: " + " ".join(cmd) + "\n")
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONPATH": (
                    f"{REPO}:{os.environ.get('PYTHONPATH', '')}"
                ),
            },
        )
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run exp02 graph robustness with preconfigured auxiliary sets."
    )
    ap.add_argument("--seeds", default="0-4",
                    help="Contiguous seed range (default 0-4).")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--k-flows", type=int, default=8)
    ap.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--result-dir",
        type=Path,
        default=Path("appendix_experiments/exp02_more_graphs_selection/results"),
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
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the sweep on the first non-zero exit. Default: log and continue.",
    )
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="If given, restrict to this subset of extra DGP names.",
    )
    ap.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also run the paper's ``c_real`` DGP (from config.DGP) as a "
             "low-density baseline point.",
    )
    args = ap.parse_args()

    # Load the DGP list without importing ``config`` at module top (so that
    # this script can be imported in places where ``config.py`` is not on
    # ``sys.path``).
    sys.path.insert(0, str(HERE))
    from extra_graphs import EXTRA_DGPS  # noqa: E402

    wrapper = _ensure_wrapper()
    result_dir = args.result_dir.resolve()
    status_path = (
        args.status_file.resolve() if args.status_file is not None
        else result_dir / "run_status.tsv"
    )
    err_log = status_path.with_name(status_path.stem + "_errors.log")
    if not args.dry_run:
        result_dir.mkdir(parents=True, exist_ok=True)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        err_log.write_text("")  # reset
        with open(status_path, "w") as sf:
            sf.write("dgp\tseed\treturncode\n")

    seeds = _parse_seed_range(args.seeds)
    if args.only is None:
        targets = list(EXTRA_DGPS.keys())
        if args.include_baseline:
            targets = ["c_real"] + targets
    else:
        targets = args.only

    total = len(targets) * len(seeds)
    done = 0
    for dgp in targets:
        if dgp not in EXTRA_DGPS and dgp != "c_real":
            raise KeyError(
                f"Unknown DGP {dgp!r} (not in EXTRA_DGPS or the "
                f"``c_real`` baseline)"
            )
        for seed in seeds:
            rc = _run_one(
                dgp=dgp,
                seed=seed,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                k_flows=args.k_flows,
                result_dir=result_dir,
                accelerator=args.accelerator,
                device=args.device,
                python_exe=args.python,
                wrapper=wrapper,
                dry_run=args.dry_run,
            )
            done += 1
            print(f"[exp02] {done}/{total} dgp={dgp} seed={seed} rc={rc}")
            if not args.dry_run:
                with open(status_path, "a") as sf:
                    sf.write(f"{dgp}\t{seed}\t{rc}\n")
            if rc != 0 and not args.dry_run:
                with open(err_log, "a") as ef:
                    ef.write(f"RC={rc} dgp={dgp} seed={seed}\n")
            if rc != 0 and args.stop_on_error:
                sys.exit(rc)

    if args.dry_run:
        print("[exp02] Dry run complete; no files were written.")
    else:
        print("[exp02] sweep complete. status:", status_path,
              "errors:", err_log)


if __name__ == "__main__":
    main()
