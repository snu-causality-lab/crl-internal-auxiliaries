"""Sweep runner for exp06 (VP vs non-VP mixing ablation).

This sweep checks whether VP-in-estimator is a fair proxy for VP-in-data:
a likelihood-good fit by a VP method does not imply the data-generating
mixing function is VP.

This sweep keeps everything in the paper's pipeline fixed except the
data-generating mixing g: VP coupling (GINCouplingBlock) vs non-VP
coupling (GLOWCouplingBlock), same depth, same subnet, same
permutations. Identifiability metrics (DCI / MCC) are then compared.
If the VP assumption matters, non-VP runs should degrade.

Layout mirrors ``exp01_nonlinear_scm/run_nonlinear.py`` so the cluster
sbatch driver and aggregator look identical.

Usage
-----
    python appendix_experiments/exp06_nonvp_ablation/run_nonvp.py \\
        --dgp a b c_real \\
        --mixing both \\
        --seeds 0-24 \\
        --max-epochs 20 \\
        --result-dir appendix_experiments/exp06_nonvp_ablation/results
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_DGPS = ["a", "b", "c_real"]
VALID_MIXINGS = ["volumepreserving", "nonvolumepreserving"]


def _parse_seed_range(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec:
        lo_s, hi_s = spec.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        if hi < lo:
            raise ValueError(f"Seed range {spec!r} is reversed")
        return list(range(lo, hi + 1))
    return [int(spec)]


def _mixing_choices(choice: str) -> list[str]:
    if choice == "both":
        return list(VALID_MIXINGS)
    if choice in VALID_MIXINGS:
        return [choice]
    raise ValueError(f"Unknown --mixing value {choice!r}")


def _run_one(
    *,
    dgp: str,
    mixing: str,
    seed: int,
    max_epochs: int,
    batch_size: int,
    lr: float,
    k_flows: int,
    result_dir: Path,
    accelerator: str,
    device: int,
    python_exe: str,
    extra_args: list[str],
    dry_run: bool,
) -> int:
    run_result_dir = result_dir / mixing

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
        f"--scm=linear",
        f"--mixing={mixing}",
        f"--result-dir={run_result_dir}",
        f"--accelerator={accelerator}",
        f"--device={device}",
    ]
    cmd.extend(extra_args)

    log_path = run_result_dir / dgp / f"{seed}_stdout.log"
    print("[exp06] Launching:", " ".join(cmd))
    print("[exp06] Logging to:", log_path)

    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        logf.write("# Command: " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep --mixing {volumepreserving, nonvolumepreserving} "
                    "for VP-vs-nonVP data-gen ablation."
    )
    ap.add_argument("--dgp", nargs="+", default=DEFAULT_DGPS)
    ap.add_argument(
        "--mixing",
        choices=["volumepreserving", "nonvolumepreserving", "both"],
        default="both",
    )
    ap.add_argument("--seeds", default="0-24")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--k-flows", type=int, default=8)
    ap.add_argument(
        "--result-dir",
        type=Path,
        default=Path("appendix_experiments/exp06_nonvp_ablation/results"),
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
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    seeds = _parse_seed_range(args.seeds)
    mixings = _mixing_choices(args.mixing)
    result_dir = args.result_dir.resolve()
    status_path = (
        args.status_file.resolve() if args.status_file is not None
        else result_dir / "run_status.tsv"
    )
    if not args.dry_run:
        result_dir.mkdir(parents=True, exist_ok=True)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(status_path, "w") as sf:
            sf.write("dgp\tmixing\tseed\treturncode\n")

    total = len(args.dgp) * len(mixings) * len(seeds)
    done = 0
    for dgp in args.dgp:
        for mixing in mixings:
            for seed in seeds:
                rc = _run_one(
                    dgp=dgp,
                    mixing=mixing,
                    seed=seed,
                    max_epochs=args.max_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    k_flows=args.k_flows,
                    result_dir=result_dir,
                    accelerator=args.accelerator,
                    device=args.device,
                    python_exe=args.python,
                    extra_args=args.extra,
                    dry_run=args.dry_run,
                )
                done += 1
                print(f"[exp06] {done}/{total} dgp={dgp} mixing={mixing} "
                      f"seed={seed} rc={rc}")
                if not args.dry_run:
                    with open(status_path, "a") as sf:
                        sf.write(f"{dgp}\t{mixing}\t{seed}\t{rc}\n")
                if rc != 0 and args.stop_on_error:
                    sys.exit(rc)

    if args.dry_run:
        print("[exp06] Dry run complete; no files were written.")
    else:
        print("[exp06] Sweep complete; status file at", status_path)


if __name__ == "__main__":
    main()
