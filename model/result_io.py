from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_int_list(values: Any) -> list[int]:
    return [int(v) for v in np.asarray(values).tolist()]


def save_mcc_result(
    result_dir: str | Path,
    seed: int,
    matrix: np.ndarray,
    score: float,
    match: tuple[np.ndarray, np.ndarray],
) -> None:
    """Save MCC outputs with Windows-safe, overwrite-stable filenames."""
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    np.savetxt(result_path / f"{seed}_mcc.csv", matrix, fmt="%.2f")
    metadata = {
        "score": float(score),
        "row_ind": _as_int_list(match[0]),
        "col_ind": _as_int_list(match[1]),
        "matrix": "matched_correlation",
    }
    with open(result_path / f"{seed}_mcc_meta.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
