import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from result_utils import load_dci_frames

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

plt.rcParams.update({'font.size': 11})

mpl.rcParams['pdf.fonttype'] = 42  # Use Type 42 (TrueType) instead of Type 3
mpl.rcParams['ps.fonttype'] = 42  # Same setting for PostScript output
plt.rcParams.update({'ytick.labelsize': 7})
mpl.rcParams['text.latex.preamble'] = (
    r'\usepackage{helvet}'  # helvetica font
    r'\usepackage{sansmath}'  # math-font matching  helvetica
    r'\sansmath'  # actually tell tex to use it!
)

def generate_combined_violin_plots(result_dict, DGP_list):

    colors3 = ['#FF3B30', '#007AFF', '#34C759']
    combined_fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=False)

    for idx, (plot_name, result_list) in enumerate(result_dict.items()):
        ax = axes[idx]
        data = {}


        for dgp in DGP_list:
            data[dgp] = {}
            for result in result_list:
                data[dgp][result] = []
                directory = _REPO_ROOT / f'result_{result}' / 'c_real'
                data[dgp][result].extend(load_dci_frames(directory))


        if plot_name == 'pendulum':
            gin = pd.concat(data[plot_name]['GIN_pendulum'])
            ivae = pd.concat(data[plot_name]['iVAE_pendulum'])
            ours = pd.concat(data[plot_name]['ours_pendulum'])
        else:
            gin = pd.concat(data[plot_name]['GIN_flow'])
            ivae = pd.concat(data[plot_name]['iVAE_flow'])
            ours = pd.concat(data[plot_name]['ours_flow'])

        gin["Group"] = "GIN"  
        ours["Group"] = "Ours" 
        ivae["Group"] = "iVAE" 

        combined_data = pd.concat([ours, gin, ivae], ignore_index=True)
        #combined_data = combined_data.drop(columns=['informativeness_train', 'informativeness_test'])

        melted_data = combined_data.melt(id_vars="Group", var_name="Feature", value_name="Value")


        sns.boxplot(
            data=melted_data,
            x="Feature",  
            y="Value",    
            hue="Group",    
            palette=colors3,
            # errorbar=("ci", 95),
            dodge=0.2,
            #capsize=0.05,
            #marker='D',
            #linestyle="None",
            ax=ax,
            showfliers=False
        )

        if plot_name == 'c_real':
            tmp = 'c'
        else:
            tmp = plot_name 
        ax.set_title(f"DGP: {tmp}")
        if idx == 0:
            ax.legend(title="Model", loc='lower right', fontsize=8)
        else:
            ax.get_legend().remove()

        plt.tight_layout()
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.savefig(str(_SCRIPT_DIR / 'combined_architeccture_img.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.show()


if __name__ == "__main__":
    # result_dict = {
    # "a": ['ours_constmodi2', 'GIN_s3', 'iVAE_s3'],
    # "b": ['ours_constmodi2', 'GIN_s3', 'iVAE_s3'],
    # "c_real": ['ours_constmodi2', 'GIN_s3', 'iVAE_s3']
    # }
    result_dict = {
    "pendulum": ['ours_pendulum', 'GIN_pendulum', 'iVAE_pendulum'],
    "flow": ['ours_flow', 'GIN_flow', 'iVAE_flow'],
    }
    DGP_list = ['pendulum', 'flow']
    generate_combined_violin_plots(result_dict, DGP_list)
