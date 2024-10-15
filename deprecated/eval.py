import json 
import pandas as pd 
import soft_kmeans_alg 
import new_utils 
import json 
import time 
from scipy.stats import kendalltau

n_shuffle = 2
iterations = 20
burn_in = 10
thining = 2
dic = {}

data_dir = "data"
js = [50, 200, 500]
rs = [0.1, 0.25, 0.5, 0.75, 0.9]
num_of_datasets_per_combination = 50

if __name__ == "__main__":

    for j in js:
        for r in rs:
            combstr = f"{int(j*r)}|{j}"
            heatmap_folder = f"img/{j}/{combstr}"
            dic[combstr] = []
            for m in range(0, num_of_datasets_per_combination):
                filename = f"{combstr}_{m}"
                data_file = f"{data_dir}/{filename}.csv"
                data_we_have = pd.read_csv(data_file)
                n_biomarkers = len(data_we_have.biomarker.unique())
    
                accepted_order_dicts = soft_kmeans_alg.metropolis_hastings_soft_kmeans(
                    data_we_have,
                    iterations,
                    n_shuffle,
                )

                new_utils.save_heatmap(accepted_order_dicts,
                                    burn_in, thining, folder_name=heatmap_folder,
                                    file_name=f"{filename}", title=f'heatmap of {filename}')
                
                most_likely_order_dic = new_utils.obtain_most_likely_order_dic(
                    accepted_order_dicts, burn_in, thining)
                most_likely_order = list(most_likely_order_dic.values())
                tau, p_value = kendalltau(most_likely_order, range(1, n_biomarkers + 1))
            
                dic[combstr].append(tau)
                # write the JSON to a file
                with open("json_files/results.json", "w") as file:
                    json.dump(dic, file, indent=4)
                print(f"{filename} is done!")


