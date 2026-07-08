import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from result_utils import load_dci_frames

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

plt.rcParams.update({'font.size': 12})

mpl.rcParams['pdf.fonttype'] = 42  # Use Type 42 (TrueType) instead of Type 3
mpl.rcParams['ps.fonttype'] = 42  # Same setting for PostScript output
plt.rcParams.update({'ytick.labelsize': 11})
mpl.rcParams['text.latex.preamble'] = (
    r'\usepackage{helvet}'  # helvetica font
    r'\usepackage{sansmath}'  # math-font matching  helvetica
    r'\sansmath'  # actually tell tex to use it!
)
DGPNAME = 'c_real'

def generate_combined_violin_plots(result_dict, DGP_list, DGPNAME):
    colors3 = ['#FF3B30', '#007AFF', '#34C759']
    combined_fig, axes = plt.subplots(1, 1, figsize=(4, 4), sharey=True)

    for idx, (plot_name, result_list) in enumerate(result_dict.items()):
        ax = axes
        data = {}

        for dgp in DGP_list:
            data[dgp] = {}
            for result in result_list:
                data[dgp][result] = []
                directory = _REPO_ROOT / f'result_{result}' / dgp
                if not directory.exists():
                    raise FileNotFoundError(
                        f"Missing optional selection-ablation result directory: {directory}. "
                        "This plot requires both with-selection and no-selection Flow runs."
                    )
                data[dgp][result].extend(load_dci_frames(directory))

        ours = pd.concat(data[DGPNAME][result_list[1]])
        oursns = pd.concat(data[DGPNAME][result_list[0]])

        ours["Group"] = "w/ selection"
        oursns["Group"] = "w/o selection"

        combined_data = pd.concat([ours, oursns], ignore_index=True)
        # combined_data = combined_data.drop(columns=['informativeness_train', 'informativeness_test'])

        melted_data = combined_data.melt(id_vars="Group", var_name="Feature", value_name="Value")

        sns.boxplot(
            data=melted_data,
            x="Feature",  
            y="Value",     
            hue="Group",    
            palette=colors3,
            #errorbar=("ci", 90),
            dodge=0.2,
            #capsize=0.05,
            #marker='D',
            #linestyle="None",
            ax=ax,
            showfliers=False
        )

      
        if DGPNAME == 'c_real':
            tmp = 'c'
        else:
            tmp = DGPNAME 
        ax.set_title(f"{plot_name}, DGP: Flow")
        if idx == 0:
            ax.legend(title="Model", loc="upper right", fontsize=10, frameon=True, framealpha=0.8)
        else:
            ax.get_legend().remove()

        plt.tight_layout()
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.savefig(str(_SCRIPT_DIR / 'combined_selection_abl_flow.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.show()


if __name__ == "__main__":
    # result_dict = {
    # "Ours": ['ours_constmodi3_ns', 'ours_constmodi2'],
    # "GIN": ['GIN2', 'GIN_s3'],
    # "iVAE": ['iVAE2', 'iVAE_s3']
    # }
    # Optional ablation comparing no-selection vs with-selection Flow runs.
    # The main run_all.py pipeline produces result_ours_flow; provide
    # result_ours_flow_noselect separately before running this plot.
    result_dict = {
        "Ours": ['ours_flow_noselect', 'ours_flow'],
    }
    # DGP_list = ['c', 'c_aug', 'c_real']
    DGP_list = ['c_real']
    generate_combined_violin_plots(result_dict, DGP_list, DGPNAME=DGPNAME)
