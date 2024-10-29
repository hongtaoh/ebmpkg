#!/usr/bin/env python3

import json 
import pandas as pd 
import constrained_kmeans_alg 
import new_utils 
from scipy.stats import kendalltau
import sys
import os

n_shuffle = 2
iterations = 10
burn_in = 2
thining = 2

# iterations = 10000
# burn_in = 5000
# thining = 100

base_dir = os.getcwd()
print(f"Current working directory: {base_dir}")
data_dir = os.path.join(base_dir, "data")
conjugate_priors_dir = os.path.join(base_dir, 'constrained_kmeans')
temp_results_dir = os.path.join(conjugate_priors_dir, "temp_json_results")
img_dir = os.path.join(conjugate_priors_dir, 'img')
results_file = os.path.join(conjugate_priors_dir, "results.json")

# Set all directories using absolute paths
# temp_results_dir = os.path.join(base_dir, "temp_json_results/conjugate_priors")
# img_dir = os.path.join(base_dir, 'img/conjugate_priors')
# results_file = os.path.join(base_dir, "results.json")

os.makedirs(conjugate_priors_dir, exist_ok=True)
os.makedirs(temp_results_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

print(f"Data directory: {data_dir}")
print(f"Temp results directory: {temp_results_dir}")
print(f"Image directory: {img_dir}")

if __name__ == "__main__":

    # Read parameters from command line arguments
    j = int(sys.argv[1])
    r = float(sys.argv[2])
    m = int(sys.argv[3])

    print(f"Processing with j={j}, r={r}, m={m}")

    combstr = f"{int(j*r)}|{j}"
    heatmap_folder = os.path.join(img_dir, str(j), combstr)

    # Ensure the heatmap folder exists
    if not os.path.exists(heatmap_folder):
        os.makedirs(heatmap_folder)
    
    filename = f"{combstr}_{m}"
    data_file = f"{data_dir}/{filename}.csv"
    data_we_have = pd.read_csv(data_file)
    n_biomarkers = len(data_we_have.biomarker.unique())

    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        sys.exit(1)  # Exit early if the file doesn't exist
    else:
        print(f"Data file found: {data_file}")

    # Define the temporary result file
    temp_result_file = os.path.join(temp_results_dir, f"temp_results_{j}_{r}_{m}.json")

    # temp_result_file = f"{temp_results_dir}/temp_results_{j}_{r}_{m}.json"
    
    dic = {}

    if combstr not in dic:
        dic[combstr] = []

    accepted_order_dicts = constrained_kmeans_alg.metropolis_hastings_constrained_kmeans(
        data_we_have,
        iterations,
        n_shuffle,
    )

    new_utils.save_heatmap(
        accepted_order_dicts,
        burn_in, 
        thining, 
        folder_name=heatmap_folder,
        file_name=f"{filename}", 
        title=f'heatmap of {filename}')
    
    most_likely_order_dic = new_utils.obtain_most_likely_order_dic(
        accepted_order_dicts, burn_in, thining)
    most_likely_order = list(most_likely_order_dic.values())
    tau, p_value = kendalltau(most_likely_order, range(1, n_biomarkers + 1))
    
    dic[combstr].append(tau)
    
    # Write the results to a unique temporary file inside the temp folder
    with open(temp_result_file, "w") as file:
        json.dump(dic, file, indent=4)
    print(f"{filename} is done! Results written to {temp_result_file}")
