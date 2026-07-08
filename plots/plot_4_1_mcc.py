from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from result_utils import load_mcc_arrays

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

plt.rcParams.update({"font.size": 11})

DGP = "c_real"
MODELS = {
    "Ours": "ours_synthetic",
    "GIN": "GIN_synthetic",
    "iVAE": "iVAE_synthetic",
}


def load_mean_heatmap(model_dir):
    directory = _REPO_ROOT / f"result_{model_dir}" / DGP
    arrays = load_mcc_arrays(directory, model_dir)
    if not arrays:
        raise FileNotFoundError(f"No MCC score CSVs found in {directory}")
    return np.mean(np.array(arrays), axis=0)


def main():
    fig, axes = plt.subplots(1, len(MODELS), figsize=(9, 3))
    for ax, (label, model_dir) in zip(axes, MODELS.items()):
        mean_array = load_mean_heatmap(model_dir)
        sns.heatmap(mean_array, annot=True, vmin=0.1, vmax=0.9, ax=ax, cmap="GnBu")
        ax.set_title(label)
        ax.set_xlabel("Estimated Latents")
        ax.set_ylabel("True Latents")

    output_path = _SCRIPT_DIR / f"{DGP}_combined_plot_abl_ns.pdf"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
