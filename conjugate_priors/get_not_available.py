import pandas as pd 
import numpy as np 
import json 

j_values = [50, 200, 500]
r_values = [0.1, 0.25, 0.5, 0.75, 0.9]
m_values = range(50)  # From 0 to 49 (inclusive)
temp_json_results_dir = 'temp_json_results'
not_available_fnames_file = 'not_available_fnames.txt'

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
                dic[combstr].append(np.nan)

print(f"not available: {not_available_count}")
with open('results.json', "w") as file:
    json.dump(dic, file, indent=4)

with open(not_available_fnames_file, "w") as f:
    for item in not_available_fnames:
        f.write(f"{item}\n")