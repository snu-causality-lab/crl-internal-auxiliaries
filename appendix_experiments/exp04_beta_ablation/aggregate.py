"""Aggregate exp04 (beta-range ablation) per-seed CSVs into a summary table.

``model/our_model.py`` writes two kinds of files per (dgp, seed):

* ``{seed}_dci.csv`` -- a DictWriter CSV with header
  ``disentanglement,completeness`` on one row of values.
* ``{seed}_mcc.csv`` and ``{seed}_mcc_meta.json`` -- the matched score matrix
  and its MCC metadata. Legacy ``{seed}_[...]score:NNN.csv`` files are still
  accepted as a fallback.

This script walks ``--result-dir`` (which should equal the directory the
sweep script wrote to), groups runs by ``(beta_label, dgp)``, and computes
mean/std of DCI disentanglement, DCI completeness, and MCC.

Outputs
-------
``results/beta_ablation_summary.csv``  (human/pandas friendly)
``results/beta_ablation_summary.tex``  (LaTeX booktabs snippet)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

# Keep in sync with BETA_CONFIGS in run_beta_ablation.py. The values here
# are used only to render the "beta range" column in the LaTeX table; the
# sweep script owns the actual sampling bounds.
BETA_RANGES: dict[str, tuple[float, float]] = {
    "paper":     (0.5, 1.0),
    "symmetric": (-1.0, 1.0),
    "small":     (0.1, 0.3),
}

FLOAT_PATTERN = r"(?:nan|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
SCORE_RE = re.compile(rf"score:(?P<score>{FLOAT_PATTERN})", re.IGNORECASE)


def _mean_std(xs: Iterable[float]) -> tuple[float, float, int]:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    if not vals:
        return float("nan"), float("nan"), 0
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0, 1
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return mean, math.sqrt(var), len(vals)


def _read_dci(csv_path: Path) -> tuple[float, float]:
    """Return (disentanglement, completeness) or (nan, nan) on error."""
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        return float(row["disentanglement"]), float(row["completeness"])
    except Exception as exc:
        print(f"[warn] Could not parse {csv_path}: {exc}")
        return float("nan"), float("nan")


def _score_from_filename(path: Path) -> float:
    m = SCORE_RE.search(path.name)
    if not m:
        return float("nan")
    return float(m.group("score"))


def _score_for_seed(seed_dir: Path, seed: str) -> float:
    meta_path = seed_dir / f"{seed}_mcc_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return float(json.load(f)["score"])
        except Exception as exc:
            print(f"[warn] Could not parse {meta_path}: {exc}")

    candidates = list(seed_dir.glob(f"{seed}_*score:*.csv"))
    if not candidates:
        return float("nan")
    ordered = sorted(
        candidates,
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )
    if len(ordered) > 1:
        print(
            f"[warn] Multiple score files for seed {seed}; "
            f"using newest: {ordered[0]}"
        )
    return _score_from_filename(ordered[0])


def _collect(
    result_dir: Path,
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Build ``{(beta_label, dgp): {'disent': [...], 'comp': [...], 'mcc': [...]}}``."""
    data: dict[tuple[str, str], dict[str, list[float]]] = {}
    # layout: {result_dir}/{beta_label}/{dgp}/{seed}_{dci,mcc}.csv plus mcc metadata
    for beta_dir in sorted(p for p in result_dir.iterdir() if p.is_dir()):
        beta_label = beta_dir.name
        if beta_label not in BETA_RANGES:
            # skip non-run artefact dirs like "aggregates"
            continue
        for dgp_dir in sorted(p for p in beta_dir.iterdir() if p.is_dir()):
            dgp = dgp_dir.name
            bucket = data.setdefault(
                (beta_label, dgp),
                {"disent": [], "comp": [], "mcc": []},
            )
            for dci_file in sorted(dgp_dir.glob("*_dci.csv")):
                disent, comp = _read_dci(dci_file)
                bucket["disent"].append(disent)
                bucket["comp"].append(comp)
                seed = dci_file.stem.split("_")[0]
                bucket["mcc"].append(_score_for_seed(dgp_dir, seed))
    return data


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        print("[warn] No rows to write to", path)
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _beta_range_str(beta_label: str) -> str:
    low, high = BETA_RANGES[beta_label]
    return f"[{low:g}, {high:g}]"


def _write_tex(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("% No data\n")
        return
    header = (
        "\\begin{tabular}{llcccc}\n\\toprule\n"
        "DGP & $\\beta$ range & DCI disent. & DCI compl. & MCC & $n$ \\\\\n"
        "\\midrule\n"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            f"{r['dgp']} & {r['beta_range']} & "
            f"{r['dci_disent_mean']}$\\pm${r['dci_disent_std']} & "
            f"{r['dci_comp_mean']}$\\pm${r['dci_comp_std']} & "
            f"{r['mcc_mean']}$\\pm${r['mcc_std']} & {r['n_seeds']} \\\\"
        )
    footer = "\n\\bottomrule\n\\end{tabular}\n"
    path.write_text(header + "\n".join(body_lines) + footer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--result-dir",
        type=Path,
        default=Path("appendix_experiments/exp04_beta_ablation/results"),
    )
    ap.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Output files prefix (default: {result-dir}/beta_ablation_summary).",
    )
    args = ap.parse_args()

    result_dir = args.result_dir.resolve()
    out_prefix = (
        args.out_prefix or result_dir / "beta_ablation_summary"
    ).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    data = _collect(result_dir)
    if not data:
        print(f"[exp04/aggregate] Nothing found under {result_dir}.")
        return

    # Sort rows by (dgp, beta_label) for stable ordering in the printed table.
    # Within dgp, keep the semantic order paper / symmetric / small.
    beta_order = {"paper": 0, "symmetric": 1, "small": 2}
    sorted_keys = sorted(
        data.keys(),
        key=lambda k: (k[1], beta_order.get(k[0], 99)),
    )

    rows = []
    for (beta_label, dgp) in sorted_keys:
        vals = data[(beta_label, dgp)]
        d_mean, d_std, n_d = _mean_std(vals["disent"])
        c_mean, c_std, n_c = _mean_std(vals["comp"])
        m_mean, m_std, n_m = _mean_std(vals["mcc"])
        n_seeds = max(n_d, n_c, n_m)
        rows.append(
            {
                "dgp": dgp,
                "beta_label": beta_label,
                "beta_range": _beta_range_str(beta_label),
                "dci_disent_mean": f"{d_mean:.4f}",
                "dci_disent_std": f"{d_std:.4f}",
                "dci_comp_mean": f"{c_mean:.4f}",
                "dci_comp_std": f"{c_std:.4f}",
                "mcc_mean": f"{m_mean:.4f}",
                "mcc_std": f"{m_std:.4f}",
                "n_seeds": str(n_seeds),
            }
        )

    _write_csv(out_prefix.with_suffix(".csv"), rows)
    _write_tex(out_prefix.with_suffix(".tex"), rows)
    print(
        "[exp04/aggregate] wrote",
        out_prefix.with_suffix(".csv"),
        "and",
        out_prefix.with_suffix(".tex"),
    )
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
