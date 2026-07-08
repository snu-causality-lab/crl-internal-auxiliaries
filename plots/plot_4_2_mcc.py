import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from result_utils import load_mcc_arrays

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

plt.rcParams.update({'font.size': 11})

def generate_mean_heatmap(ax, directory, model_name):
    combined_data = load_mcc_arrays(directory, model_name)
    if not combined_data:
        raise FileNotFoundError(f"No MCC files found in {directory}")
    mean_array = np.mean(np.array(combined_data), axis=0)
    if model_name == "ours_synthetic":
        tmp = "Ours"
    elif model_name == "GIN_synthetic":
        tmp = "GIN"
        mean_array = mean_array[:,:]
    elif model_name == "iVAE_synthetic":
        tmp = "iVAE"
        mean_array = mean_array[:,:]

    sns.heatmap(mean_array, annot=True , vmin=0.1, vmax=0.9, ax=ax, cmap='GnBu')

    ax.set_title(f"{tmp}")
    ax.set_xlabel("Estimated Latents")
    ax.set_ylabel("True Latents")


def generate_barplot(ax, data, DGPNAME, colors3):
    sns.barplot(data=data[DGPNAME], palette=colors3, errorbar=("ci", 95), linestyle="none", ax=ax)
    ax.set_title("MCC SCORE, 95% CI")
    ax.set_xlabel("Models")
    ax.set_ylabel("MCC Score")
    ax.set_xticklabels(["Ours", "GIN", "iVAE"])

MODELS = ['ours_synthetic','GIN_synthetic', 'iVAE_synthetic'] 
# DGPNAME = 'c_real'  

if __name__ == "__main__":

    DGP_list = ['a', 'b', 'c_real']
    result_list = MODELS
    colors3 = ['#FF3B30', '#007AFF', '#34C759']
    for DGPNAME in DGP_list:
   
        fig, axes = plt.subplots(1, len(MODELS), figsize=(9, 3)) 

        for i, model in enumerate(MODELS):
            csv_directory = str(_REPO_ROOT / f'result_{model}' / DGPNAME) + '/'
            generate_mean_heatmap(axes[i], csv_directory, model)


        output_path = str(_SCRIPT_DIR / f"{DGPNAME}_combined_plot_abl.pdf")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
