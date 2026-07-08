from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


LEGACY_SORT_RE = re.compile(r"array\(\[[\d,\s]+\]\), array\(\[([\d,\s]+)\]\)")
LEGACY_SCORE_RE = re.compile(r"score:([0-9.eE+-]+|nan)(?=\.csv$)", re.IGNORECASE)
SEED_RE = re.compile(r"^(\d+)_")


def _seed_from_name(path: Path) -> str | None:
    match = SEED_RE.match(path.name)
    return match.group(1) if match else None


def _read_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None, index_col=None, sep=r"\s+").values


def _legacy_sort_order(filename: str) -> list[int] | None:
    match = LEGACY_SORT_RE.search(filename)
    if not match:
        return None
    return [int(v) for v in match.group(1).split(",")]


def _choose_one_per_seed(paths: list[Path]) -> list[Path]:
    chosen: dict[str, Path] = {}
    for path in sorted(paths, key=lambda p: (p.name.endswith("_mcc.csv"), p.stat().st_mtime, p.name)):
        seed = _seed_from_name(path)
        if seed is not None:
            chosen[seed] = path
    return [chosen[k] for k in sorted(chosen, key=int)]


def load_mcc_arrays(directory: str | Path, model_name: str) -> list[np.ndarray]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Missing result directory: {directory}")

    new_paths = list(directory.glob("*_mcc.csv"))
    legacy_paths = [p for p in directory.glob("*.csv") if LEGACY_SCORE_RE.search(p.name)]
    arrays: list[np.ndarray] = []

    for path in _choose_one_per_seed(legacy_paths + new_paths):
        if path.name.endswith("_mcc.csv"):
            arrays.append(_read_matrix(path))
            continue

        sort_order = _legacy_sort_order(path.name)
        if sort_order is None:
            continue
        df = pd.DataFrame(_read_matrix(path))
        if "ours" not in model_name:
            df = df.iloc[:, sort_order]
        arrays.append(df.values)

    return arrays


def load_mcc_scores(directory: str | Path) -> list[float]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Missing result directory: {directory}")

    scores_by_seed: dict[str, float] = {}
    for meta_path in sorted(directory.glob("*_mcc_meta.json")):
        seed = _seed_from_name(meta_path)
        if seed is None:
            continue
        with open(meta_path) as f:
            scores_by_seed[seed] = float(json.load(f)["score"])

    for path in sorted(directory.glob("*.csv"), key=lambda p: (p.stat().st_mtime, p.name)):
        seed = _seed_from_name(path)
        match = LEGACY_SCORE_RE.search(path.name)
        if seed is not None and match and seed not in scores_by_seed:
            scores_by_seed[seed] = float(match.group(1))

    return [scores_by_seed[k] for k in sorted(scores_by_seed, key=int)]


def load_dci_frames(directory: str | Path, max_seeds: int = 20) -> list[pd.DataFrame]:
    """Load DCI CSVs in deterministic seed order."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Missing result directory: {directory}")

    frames_by_seed: dict[int, pd.DataFrame] = {}
    for path in sorted(directory.glob("*_dci.csv")):
        seed = _seed_from_name(path)
        if seed is None:
            continue
        seed_int = int(seed)
        if seed_int < max_seeds:
            frames_by_seed[seed_int] = pd.read_csv(path)

    return [frames_by_seed[k] for k in sorted(frames_by_seed)]
