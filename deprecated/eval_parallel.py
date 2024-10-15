import json 
import pandas as pd 
import numpy as np
import soft_kmeans_alg 
import new_utils 
import os
import json 
from scipy.stats import kendalltau
from concurrent.futures import ProcessPoolExecutor, as_completed

n_shuffle = 2
iterations = 1000
burn_in = 500
thinning = 20
data_dir = "data"
js = [50, 200, 500]
rs = [0.1, 0.25, 0.5, 0.75, 0.9]
num_of_datasets_per_combination = 50
result_json = 'json_files/results.json'

def process_file(filename, combstr, j, r):
    data_file = f"{data_dir}/{filename}.csv"
    data_we_have = pd.read_csv(data_file)
    n_biomarkers = len(data_we_have.biomarker.unique())

    accepted_order_dicts = soft_kmeans_alg.metropolis_hastings_soft_kmeans(
        data_we_have,
        iterations,
        n_shuffle,
    )

    heatmap_folder = f"img/{j}/{combstr}"
    new_utils.save_heatmap(accepted_order_dicts,
                        burn_in, thinning, folder_name=heatmap_folder,
                        file_name=f"{filename}", title=f'heatmap of {filename}')
    
    most_likely_order_dic = new_utils.obtain_most_likely_order_dic(
        accepted_order_dicts, burn_in, thinning)
    most_likely_order = list(most_likely_order_dic.values())
    tau, p_value = kendalltau(most_likely_order, range(1, n_biomarkers + 1))

    print(f"{filename} is done!")

    return combstr, tau 

if __name__ == "__main__":
    dic = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for j in js: 
            for r in rs:
                combstr = f"{int(j*r)}|{j}"
                to_test_m = np.random.choice(num_of_datasets_per_combination, 2, replace=False)
                for m in range(0, num_of_datasets_per_combination):
                    filename = f"{combstr}_{m}"
                    futures.append(executor.submit(process_file, filename, combstr, j, r))
        
        for future in as_completed(futures):
            combstr, tau = future.result()
            dic.setdefault(combstr, []).append(tau)

            with open("json_files/results.json", "w") as file:
                json.dump(dic, file, indent=4)
    
    print("All datasets are processed!")
                    

