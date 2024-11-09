import pandas as pd 
import numpy as np 
import json 
import matplotlib.pyplot as plt 
import seaborn as sns 

def get_not_available_and_results_json(j_values, r_values, m_values, dir):
    """
    Get not_available_fnames: a list of strings
    and results dic where key is combstr and value is an array of tau values
    """
    temp_json_results_dir = f"{dir}/temp_json_results"
    dic = {}
    dic['param'] = {
        "num_of_datasets_per_combination": 50,
        "n_iter": 10000,
        "n_biomarkers": 10
    }
    not_available_fnames = []
    not_available_count = 0
    for j in j_values:
        for r in r_values:
            combstr = f"{int(j*r)}|{j}"
            if combstr not in dic:
                dic[combstr] = []
            for m in m_values:
                try:
                    with open(f"{temp_json_results_dir}/temp_results_{j}_{r}_{m}.json") as f:
                        d = json.load(f)
                    tau = list(d.values())[0][0]
                    dic[combstr].append(tau)
                except:
                    not_available_count += 1
                    fname = f"{j} {r} {m}"
                    not_available_fnames.append(fname)
                    dic[combstr].append(None)

    print(f"not available count for {dir}: {not_available_count}")
    return not_available_fnames, dic  

def plot_tau_synthetic(
        tau_file,
        j_values,
        r_values,
        num_of_datasets_per_combination,
        dir,
        plot_name = 'violin_plot',
    ):
    with open(tau_file) as f:
        tau = json.load(f)
    algorithm = dir.replace("_", " ")
    data = tau
    param = data['param']
    print(param)
    dict_list = []
    for j in j_values:
        for r in r_values:
            key = f"{int(j*r)}|{j}"
            for m in range(0, num_of_datasets_per_combination):
                dic = {}  # Create a new dictionary for each loop iteration
                dic["j"] = f"$J={j}$"
                dic['r'] = f"{int(r*100)}%"
                dic['tau'] = data[key][m]
                dict_list.append(dic)  # Append the new dictionary
    df = pd.DataFrame(dict_list)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the boxplot
    # Optionally, add a violin plot for better distribution visualization
    if 'box_plot' in plot_name:
        g = sns.boxplot(data=df, x="j", y="tau", hue="r", palette="bright", ax = ax)
    else:
        g = sns.violinplot(data=df, x="j", y="tau", hue="r", palette="bright", 
                           dodge=True, alpha=0.6, linewidth=0, ax = ax)
        
    g.set_ylim(-0.5, 1)

    # Set the x-axis label
    g.set_xlabel("Participant Size", fontsize=14)

    # Set the y-axis label
    g.set_ylabel("Kendall's Tau", fontsize=14)

    # Set the plot title
    g.set_title(f"Kendall's Tau values across different combinations in synthetic data ({algorithm})", fontsize=16)

    # Adjust the legend and move it outside the figure
    plt.legend(title="Healthy Ratio", title_fontsize='13', fontsize='10', 
            bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add a multi-line caption to the plot
    caption_text = (
        "Notes:\n"
        "\n"
        "This figure shows Kendall's Tau for different combinations of participant size and healthy ratio.\n"
        f"Each combination has {param['num_of_datasets_per_combination']} variants of datasets\n"
        f"The results are derived from our own implementation of {algorithm} based on synthetic data with 10 biomarkers.\n"
        f"Number of iterations: {param['n_iter']}."
    )
    ax.figure.text(
        0.05, -0.01, caption_text, ha='left', va='top',
        fontsize=12, wrap=True
    )

    # Adjust the layout to make room for the legend
    plt.tight_layout()  # Leave some space at the bottom for the caption
    # Show the plot
    # plt.show()
    plt.savefig(f'{dir}/{plot_name}.png', bbox_inches='tight')
    # Close the plot to avoid issues with subsequent plots
    plt.close()

if __name__ == "__main__":
    j_values = [50, 200, 500]
    r_values = [0.1, 0.25, 0.5, 0.75, 0.9]
    m_values = range(50)  # From 0 to 49 (inclusive)
    dirs = ['conjugate_priors', 'soft_kmeans', 'hard_kmeans']
    for dir in dirs:
        not_available_fnames, results_dic = get_not_available_and_results_json(
            j_values, r_values, m_values, dir)
        
        results_json_file = f'{dir}/results.json'
        not_available_fnames_file = f'{dir}/not_available_fnames.txt'

        with open(results_json_file, "w") as f:
            json.dump(results_dic, f, indent=2)
        with open(not_available_fnames_file, "w") as f:
            for item in not_available_fnames:
                f.write(f"{item}\n")
        
        ns = j_values
        rs = r_values
        num_of_datasets_per_combination = len(m_values)
        for plot_name in ['box_plot', 'violin_plot']:
            plot_tau_synthetic(
                results_json_file,
                j_values,
                r_values,
                num_of_datasets_per_combination,
                dir,
                plot_name = plot_name,
            )

        

    
