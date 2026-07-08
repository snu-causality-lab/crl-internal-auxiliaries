"""Backward-compatible synthetic MCC plotting entry point.

The current result layout is produced by run_all.py:
result_<model>_synthetic/<dgp>/...
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from result_utils import load_mcc_arrays, load_mcc_scores

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

plt.rcParams.update({"font.size": 12})

MODELS = {
    "Ours": "ours_synthetic",
    "GIN": "GIN_synthetic",
    "iVAE": "iVAE_synthetic",
}
DGP = "b"


def load_scores(model_dir, dgp):
    directory = _REPO_ROOT / f"result_{model_dir}" / dgp
    return load_mcc_scores(directory)


def load_mean_heatmap(model_dir, dgp):
    directory = _REPO_ROOT / f"result_{model_dir}" / dgp
    arrays = load_mcc_arrays(directory, model_dir)
    if not arrays:
        raise FileNotFoundError(f"No MCC score CSVs found in {directory}")
    return np.mean(np.array(arrays), axis=0)


def main():
    fig, axes = plt.subplots(1, len(MODELS) + 1, figsize=(9, 3))

    data = {label: load_scores(model_dir, DGP) for label, model_dir in MODELS.items()}
    sns.barplot(data=data, errorbar=("ci", 95), linestyle="none", ax=axes[0])
    axes[0].set_title("MCC SCORE, 95% CI")
    axes[0].set_xlabel("Models")
    axes[0].set_ylabel("MCC Score")

    for ax, (label, model_dir) in zip(axes[1:], MODELS.items()):
        mean_array = load_mean_heatmap(model_dir, DGP)
        sns.heatmap(mean_array, annot=True, vmin=0.1, vmax=0.9, ax=ax, cmap="GnBu")
        ax.set_title(label)
        ax.set_xlabel("Estimated Latents")
        ax.set_ylabel("True Latents")

    output_path = _SCRIPT_DIR / f"{DGP}_combined_plot.pdf"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
