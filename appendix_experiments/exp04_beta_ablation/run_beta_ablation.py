"""Sweep runner for exp04 (beta-coefficient range ablation).

The paper's synthetic SCM fixes ``beta_ij ~ Uniform[0.5, 1.0]``
(positive-only, |beta| >= 0.5). The
identifiability claims should be robust to (i) negative coefficients,
(ii) small-magnitude coefficients. The training script already exposes
``--scm-coeffs-low``/``--scm-coeffs-high``/``--scm-coeffs-min-abs-value``
CLI flags which thread through ``data_generator.multi_env_gdp`` into
``LinearSCM``. This script launches the existing
``python -m experiments.Ours.main`` subprocess with different flags and
organises the per-seed output directories.

Usage
-----
    python appendix_experiments/exp04_beta_ablation/run_beta_ablation.py \
        --dgp c c_real \
        --beta all \
        --seeds 0-4 \
        --max-epochs 20 \
        --result-dir appendix_experiments/exp04_beta_ablation/results

The seed range is inclusive on both ends ("0-4" means 0,1,2,3,4). Only
contiguous ranges are accepted -- the paper uses the contiguous block
``range(0, 20)`` and we want to preserve that convention.

Beta configurations
-------------------
Three (label, low, high, min_abs) tuples are swept:

* ``paper``      low=0.5, high=1.0, min_abs=0.5   (paper Sec.5 / Fig.3 range)
* ``symmetric``  low=-1.0, high=1.0, min_abs=0.5  (multi_env_gdp.py default)
* ``small``      low=0.1, high=0.3, min_abs=0.05  (small magnitude, positive)

``min_abs`` must satisfy ``min_abs < max(|low|, |high|)`` because
``data_generator.utils.sample_coeffs`` asserts this (and rejection-samples
otherwise). For the ``small`` config, ``0.05 < 0.3`` holds and every
sampled coefficient is already >= 0.05 since ``low = 0.1 > 0.05``, so no
rejection loop kicks in.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Paper Sec.5 / Fig.3 caption range is (label, low, high, min_abs_value).
BETA_CONFIGS: dict[str, tuple[float, float, float]] = {
    # label    : (low,  high, min_abs_value)
    "paper":     (0.5,  1.0,  0.5),
    "symmetric": (-1.0, 1.0,  0.5),
    "small":     (0.1,  0.3,  0.05),
}

# Four-node multi-observable DGPs used for the beta-range sweep.
DEFAULT_DGPS = ["c", "c_real"]
VALID_BETAS = list(BETA_CONFIGS.keys())


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


def _beta_choices(choice: str) -> list[str]:
    if choice == "all":
        return list(VALID_BETAS)
    if choice in VALID_BETAS:
        return [choice]
    raise ValueError(
        f"Unknown --beta value {choice!r}; expected one of "
        f"{VALID_BETAS + ['all']}"
    )


def _run_one(
    *,
    dgp: str,
    beta_label: str,
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
    """Launch one training run and return the subprocess exit code."""
    # Keep layout similar to exp01:
    # ``{result_dir}/{beta_label}/{dgp}/{seed}_dci.csv``
    low, high, min_abs = BETA_CONFIGS[beta_label]
    run_result_dir = result_dir / beta_label

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
        "--scm=linear",
        f"--scm-coeffs-low={low}",
        f"--scm-coeffs-high={high}",
        f"--scm-coeffs-min-abs-value={min_abs}",
        f"--result-dir={run_result_dir}",
        f"--accelerator={accelerator}",
        f"--device={device}",
    ]
    cmd.extend(extra_args)

    log_path = run_result_dir / dgp / f"{seed}_stdout.log"
    print("[exp04] Launching:", " ".join(cmd))
    print("[exp04] Logging to:", log_path)

    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        logf.write("# Command: " + " ".join(cmd) + "\n")
        logf.write(
            f"# beta_label={beta_label} low={low} high={high} "
            f"min_abs={min_abs}\n"
        )
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Sweep SCM coefficient ranges (paper / symmetric / small) "
            "across DGPs and seeds for the paper's synthetic mode."
        )
    )
    ap.add_argument(
        "--dgp",
        nargs="+",
        default=DEFAULT_DGPS,
        help=(
            "Which DGP keys from config.DGP to sweep. "
            "Default: c c_real (four-node multi-observable settings)."
        ),
    )
    ap.add_argument(
        "--beta",
        choices=VALID_BETAS + ["all"],
        default="all",
        help="Which beta config(s) to run (default: all three).",
    )
    ap.add_argument(
        "--seeds",
        default="0-4",
        help="Contiguous seed range, e.g. '0-4' (default: 0-4 pilot).",
    )
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--k-flows", type=int, default=8)
    ap.add_argument(
        "--result-dir",
        type=Path,
        default=Path("appendix_experiments/exp04_beta_ablation/results"),
        help=(
            "Root results directory. Per-run CSVs end up under "
            "{result_dir}/{beta_label}/{dgp}/{seed}_*"
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
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args appended verbatim to every sub-call.",
    )
    args = ap.parse_args()

    seeds = _parse_seed_range(args.seeds)
    betas = _beta_choices(args.beta)
    result_dir = args.result_dir.resolve()
    status_path = (
        args.status_file.resolve() if args.status_file is not None
        else result_dir / "run_status.tsv"
    )
    if not args.dry_run:
        result_dir.mkdir(parents=True, exist_ok=True)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(status_path, "w") as sf:
            sf.write("dgp\tbeta\tlow\thigh\tmin_abs\tseed\treturncode\n")

    total = len(args.dgp) * len(betas) * len(seeds)
    done = 0
    for dgp in args.dgp:
        for beta_label in betas:
            low, high, min_abs = BETA_CONFIGS[beta_label]
            for seed in seeds:
                rc = _run_one(
                    dgp=dgp,
                    beta_label=beta_label,
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
                print(
                    f"[exp04] {done}/{total} dgp={dgp} beta={beta_label} "
                    f"seed={seed} rc={rc}"
                )
                if not args.dry_run:
                    with open(status_path, "a") as sf:
                        sf.write(
                            f"{dgp}\t{beta_label}\t{low}\t{high}\t{min_abs}\t"
                            f"{seed}\t{rc}\n"
                        )
                if rc != 0 and args.stop_on_error:
                    sys.exit(rc)

    if args.dry_run:
        print("[exp04] Dry run complete; no files were written.")
    else:
        print("[exp04] Sweep complete; status file at", status_path)


if __name__ == "__main__":
    main()
