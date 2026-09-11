"""Aggregate exp02 (extra-graphs sweep) CSVs into a summary table.

Layout expected, produced by ``run_selection_ablation.py``:
    {result-dir}/{dgp}/{seed}_dci.csv
    {result-dir}/{dgp}/{seed}_mcc.csv
    {result-dir}/{dgp}/{seed}_mcc_meta.json

Legacy ``{seed}_*score:*.csv`` files are still accepted as a fallback.

One row per DGP: DCI disentanglement, DCI completeness and MCC score
(mean +/- std across seeds). Each graph uses a preconfigured auxiliary set;
the table summarizes recovery across graphs, not a with-vs-without
selection effect or execution of the paper's graph-selection algorithm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

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


def _read_dci(path: Path) -> tuple[float, float]:
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        return float(row["disentanglement"]), float(row["completeness"])
    except Exception as exc:
        print(f"[warn] Could not parse {path}: {exc}")
        return float("nan"), float("nan")


def _score_from_filename(path: Path) -> float:
    m = SCORE_RE.search(path.name)
    return float(m.group("score")) if m else float("nan")


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


def _collect(result_dir: Path) -> dict[str, dict[str, list[float]]]:
    data: dict[str, dict[str, list[float]]] = {}
    # ``{result_dir}/{dgp}/`` is the canonical layout; also walk one
    # level deeper in case the trainer nested a sub-folder.
    for dgp_dir in sorted(p for p in result_dir.iterdir() if p.is_dir()):
        dgp = dgp_dir.name
        if dgp in {"errors.log"}:
            continue
        bucket = data.setdefault(dgp,
                                 {"disent": [], "comp": [], "mcc": []})

        def _ingest(sdir: Path) -> None:
            for dci_file in sorted(sdir.glob("*_dci.csv")):
                disent, comp = _read_dci(dci_file)
                bucket["disent"].append(disent)
                bucket["comp"].append(comp)
                seed = dci_file.stem.split("_")[0]
                bucket["mcc"].append(_score_for_seed(sdir, seed))

        _ingest(dgp_dir)
        for nested in dgp_dir.iterdir():
            if nested.is_dir():
                _ingest(nested)
    return data


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("% No data\n")
        return
    header = (
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "DGP & DCI disent. & DCI compl. & MCC & $n$ \\\\\n"
        "\\midrule\n"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            f"{r['dgp']} & "
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
        default=Path("appendix_experiments/exp02_more_graphs_selection/results"),
    )
    ap.add_argument("--out-prefix", type=Path, default=None)
    args = ap.parse_args()
    result_dir = args.result_dir.resolve()
    out_prefix = (args.out_prefix
                  or result_dir / "selection_ablation_summary").resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    data = _collect(result_dir)
    if not data:
        print(f"[exp02/aggregate] Nothing under {result_dir}.")
        return

    rows: list[dict[str, str]] = []
    for dgp, vals in sorted(data.items()):
        d_mean, d_std, n_d = _mean_std(vals["disent"])
        c_mean, c_std, n_c = _mean_std(vals["comp"])
        m_mean, m_std, n_m = _mean_std(vals["mcc"])
        rows.append({
            "dgp": dgp,
            "dci_disent_mean": f"{d_mean:.4f}",
            "dci_disent_std": f"{d_std:.4f}",
            "dci_comp_mean": f"{c_mean:.4f}",
            "dci_comp_std": f"{c_std:.4f}",
            "mcc_mean": f"{m_mean:.4f}",
            "mcc_std": f"{m_std:.4f}",
            "n_seeds": str(max(n_d, n_c, n_m)),
        })

    _write_csv(out_prefix.with_suffix(".csv"), rows)
    _write_tex(out_prefix.with_suffix(".tex"), rows)
    print("[exp02/aggregate] wrote",
          out_prefix.with_suffix(".csv"),
          "and",
          out_prefix.with_suffix(".tex"))
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
