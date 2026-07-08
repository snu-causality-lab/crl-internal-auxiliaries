"""Register exp02 extra DGPs before invoking the main training entry point.

This wrapper is intentionally checked in rather than generated at runtime so
Slurm array tasks never race while writing a shared source file. The production
``experiments.Ours.main`` module is unchanged unless this wrapper is used.
"""

from __future__ import annotations

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config as _config  # noqa: E402
from extra_graphs import EXTRA_DGPS  # noqa: E402


def main() -> None:
    # ``c_real`` already lives in config.DGP. The extra graph keys are
    # registered only for this wrapper process.
    for _name, _spec in EXTRA_DGPS.items():
        if _name not in _config.DGP:
            _config.DGP[_name] = _spec

    runpy.run_module("experiments.Ours.main", run_name="__main__")


if __name__ == "__main__":
    main()
