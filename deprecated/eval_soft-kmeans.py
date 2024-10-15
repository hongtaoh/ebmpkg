#!/usr/bin/env python3

import json 
import pandas as pd 
import soft_kmeans_alg 
import new_utils 
import time 
from scipy.stats import kendalltau
import sys
import os
print("Current working directory:", os.getcwd())

n_shuffle = 2
iterations = 5000
burn_in = 2500
thining = 50
dic = {}

base_dir = os.getcwd()

data_dir = "data"
temp_results_dir = "temp_json_results"  # Folder for temp results
not_available_fnames_file = 'results/not_available_fnames.txt'
img_dir = 'img'
results_file = "results.json"
js = [50, 200, 500]
rs = [0.1, 0.25, 0.5, 0.75, 0.9]
num_of_datasets_per_combination = 50

if __name__ == "__main__":
    # Read parameters from command line arguments
    j = int(sys.argv[1])
    r = float(sys.argv[2])
    m = int(sys.argv[3])

    print(f"Processing with j={j}, r={r}, m={m}")

    combstr = f"{int(j*r)}|{j}"
    heatmap_folder = f"{img_dir}/{j}/{combstr}"
    # dic[combstr] = []
    
    filename = f"{combstr}_{m}"
    data_file = f"{data_dir}/{filename}.csv"
    data_we_have = pd.read_csv(data_file)
    n_biomarkers = len(data_we_have.biomarker.unique())

    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        sys.exit(1)  # Exit early if the file doesn't exist
    else:
        print(f"Data file found: {data_file}")

    # Create the temp results folder if it doesn't exist
    if not os.path.exists(temp_results_dir):
        os.makedirs(temp_results_dir)

    # Each job writes to its own temporary result file in the folder
    temp_result_file = f"{temp_results_dir}/temp_results_{j}_{r}_{m}.json"
    
    dic = {}

    if combstr not in dic:
        dic[combstr] = []

    accepted_order_dicts = soft_kmeans_alg.metropolis_hastings_soft_kmeans(
        data_we_have,
        iterations,
        n_shuffle,
    )

    # Ensure the heatmap folder exists
    if not os.path.exists(heatmap_folder):
        os.makedirs(heatmap_folder)

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
