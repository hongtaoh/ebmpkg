import os
import time
import random
import matplotlib.pyplot as plt
import concurrent.futures
import deprecated.utils as utils
import json
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.stats import mode
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':
    # iterations = 2000
    # burn_in = 1000
    # thining = 20
    # n_shuffle = 2
    # chen_real_order = [1, 3, 5, 2, 4]
    # # from 1 to 5
    S_ordering =[
        'HIP-FCI', 'PCC-FCI', 'HIP-GMI', 'FUS-GMI', 'FUS-FCI'
    ]
    real_theta_phi_file = 'real_theta_phi.json'

    ns = [50, 200, 500]
    rs = [0.1, 0.25, 0.5]
    chen_participant_size = [144, 500]
    algorithms = ['conjugate_priors', 'soft_kmeans', 'hard_kmeans']
    tau_json = "json_files/tau.json"
    ln_ll_json = "json_files/ln_ll.json"
    time_json = "json_files/time.json"

    # hashmap_of_tau_dicts = {}
    # hashmap_of_tau_dicts['synthetic'] = {}
    # hashmap_of_tau_dicts['synthetic']['conjugate_priors'] = {}
    # hashmap_of_tau_dicts['synthetic']['soft_kmeans'] = {}
    # hashmap_of_tau_dicts['synthetic']['hard_kmeans'] = {}
    # hashmap_of_tau_dicts['chen_data'] = {}
    # hashmap_of_tau_dicts['chen_data']['conjugate_priors'] = {}
    # hashmap_of_tau_dicts['chen_data']['soft_kmeans'] = {}
    # hashmap_of_tau_dicts['chen_data']['hard_kmeans'] = {}

    # # compare the likelihood of the most likely order
    # # and the real order
    # hashmap_of_ln_ll_dicts = {}
    # hashmap_of_ln_ll_dicts['synthetic'] = {}
    # hashmap_of_ln_ll_dicts['synthetic']['conjugate_priors'] = {}
    # hashmap_of_ln_ll_dicts['synthetic']['soft_kmeans'] = {}
    # hashmap_of_ln_ll_dicts['synthetic']['hard_kmeans'] = {}
    # hashmap_of_ln_ll_dicts['chen_data'] = {}
    # hashmap_of_ln_ll_dicts['chen_data']['conjugate_priors'] = {}
    # hashmap_of_ln_ll_dicts['chen_data']['soft_kmeans'] = {}
    # hashmap_of_ln_ll_dicts['chen_data']['hard_kmeans'] = {}

    # # time consumption dic
    # hashmap_of_time_dicts = {}
    # hashmap_of_time_dicts['synthetic'] = {}
    # hashmap_of_time_dicts['synthetic']['conjugate_priors'] = {}
    # hashmap_of_time_dicts['synthetic']['soft_kmeans'] = {}
    # hashmap_of_time_dicts['synthetic']['hard_kmeans'] = {}
    # hashmap_of_time_dicts['chen_data'] = {}
    # hashmap_of_time_dicts['chen_data']['conjugate_priors'] = {}
    # hashmap_of_time_dicts['chen_data']['soft_kmeans'] = {}
    # hashmap_of_time_dicts['chen_data']['hard_kmeans'] = {}

    for n in ns:
        for r in rs:
            comb_str = f"{int(r*n)}|{n}"

            participants_data = utils.generate_data_from_ebm(
                n_participants=n,
                S_ordering=S_ordering,
                real_theta_phi_file=real_theta_phi_file,
                healthy_ratio=r,
                output_dir = 'data/synthetic',
                seed=1234,
            )

    #         # Simulated data with conjugate priors
    #         utils.run_conjugate_priors(
    #             data_we_have=participants_data,
    #             data_source="Synthetic Data",
    #             iterations=iterations,
    #             log_folder_name=f"logs/synthetic/{comb_str}/conjugate_priors",
    #             img_folder_name=f"img/synthetic/{comb_str}/conjugate_priors",
    #             n_shuffle=n_shuffle,
    #             burn_in=burn_in,
    #             thining=thining,
    #             tau_dic=hashmap_of_tau_dicts['synthetic']['conjugate_priors'],
    #             ln_ll_dic=hashmap_of_ln_ll_dicts['synthetic']['conjugate_priors'],
    #             time_dic = hashmap_of_time_dicts['synthetic']['conjugate_priors'],
    #         )

    #         # Simulated data with kmeans
    #         utils.run_soft_kmeans(
    #             data_we_have=participants_data,
    #             data_source="Synthetic Data",
    #             iterations=iterations,
    #             n_shuffle=n_shuffle,
    #             log_folder_name=f"logs/synthetic/{comb_str}/soft_kmeans",
    #             img_folder_name=f"img/synthetic/{comb_str}/soft_kmeans",
    #             burn_in=burn_in,
    #             thining=thining,
    #             tau_dic=hashmap_of_tau_dicts['synthetic']['soft_kmeans'],
    #             ln_ll_dic=hashmap_of_ln_ll_dicts['synthetic']['soft_kmeans'],
    #             time_dic = hashmap_of_time_dicts['synthetic']['soft_kmeans']
    #         )
    #         # Soley kmeans
    #         utils.run_kmeans(
    #             data_we_have=participants_data,
    #             data_source="Synthetic Data",
    #             iterations=iterations,
    #             n_shuffle=n_shuffle,
    #             log_folder_name=f"logs/synthetic/{comb_str}/hard_kmeans",
    #             img_folder_name=f"img/synthetic/{comb_str}/hard_kmeans",
    #             burn_in=burn_in,
    #             thining=thining,
    #             tau_dic=hashmap_of_tau_dicts['synthetic']['hard_kmeans'],
    #             ln_ll_dic=hashmap_of_ln_ll_dicts['synthetic']['hard_kmeans'],
    #             time_dic = hashmap_of_time_dicts['synthetic']['hard_kmeans']
    #         )

    # """Chen Data
    # """
    # for par_size in chen_participant_size:
    #     participant_data = utils.process_chen_data(
    #         "data/Chen2016Data.xlsx",
    #         chen_real_order,
    #         participant_size=par_size,
    #         seed=None
    #     )

    #     # Chen data with conjugate priors
    #     utils.run_conjugate_priors(
    #         data_we_have=participant_data,
    #         data_source="Chen Data",
    #         iterations=iterations,
    #         log_folder_name=f"logs/chen_data/{par_size}/conjugate_priors",
    #         img_folder_name=f"img/chen_data/{par_size}/conjugate_priors",
    #         n_shuffle=n_shuffle,
    #         burn_in=burn_in,
    #         thining=thining,
    #         tau_dic=hashmap_of_tau_dicts['chen_data']['conjugate_priors'],
    #         ln_ll_dic=hashmap_of_ln_ll_dicts['chen_data']['conjugate_priors'],
    #         time_dic = hashmap_of_time_dicts['chen_data']['conjugate_priors']
    #     )
    #     # Chen data with soft kmeans
    #     utils.run_soft_kmeans(
    #         data_we_have=participant_data,
    #         data_source="Chen Data",
    #         iterations=iterations,
    #         n_shuffle=n_shuffle,
    #         log_folder_name=f"logs/chen_data/{par_size}/soft_kmeans",
    #         img_folder_name=f"img/chen_data/{par_size}/soft_kmeans",
    #         burn_in=burn_in,
    #         thining=thining,
    #         tau_dic=hashmap_of_tau_dicts['chen_data']['soft_kmeans'],
    #         ln_ll_dic=hashmap_of_ln_ll_dicts['chen_data']['soft_kmeans'],
    #         time_dic = hashmap_of_time_dicts['chen_data']['soft_kmeans']
    #     )
    #     # Chen data with kmeans only
    #     utils.run_kmeans(
    #         data_we_have=participant_data,
    #         data_source="Chen Data",
    #         iterations=iterations,
    #         n_shuffle=n_shuffle,
    #         log_folder_name=f"logs/chen_data/{par_size}/hard_kmeans",
    #         img_folder_name=f"img/chen_data/{par_size}/hard_kmeans",
    #         burn_in=burn_in,
    #         thining=thining,
    #         tau_dic=hashmap_of_tau_dicts['chen_data']['hard_kmeans'],
    #         ln_ll_dic=hashmap_of_ln_ll_dicts['chen_data']['hard_kmeans'],
    #         time_dic = hashmap_of_time_dicts['chen_data']['hard_kmeans']
    #     )

    # with open(tau_json, 'w') as fp:
    #     json.dump(hashmap_of_tau_dicts, fp)

    # with open(ln_ll_json, 'w') as fp:
    #     json.dump(hashmap_of_ln_ll_dicts, fp)

    # with open(time_json, 'w') as fp:
    #     json.dump(hashmap_of_time_dicts, fp)

    # for algorithm in algorithms:
    #     utils.plot_tau_synthetic(
    #         tau_json,
    #         algorithm,
    #         ns,
    #         rs
    #     )

    # utils.plot_tau_chen_data(
    #     tau_json,
    #     chen_participant_size,
    #     algorithms,
    # )

    # for algorithm in algorithms:
    #     utils.plot_ln_ll_synthetic(
    #         ln_ll_json,
    #         algorithm,
    #         ns,
    #         rs
    #     )

    # utils.plot_ln_ll_chen_data(
    #     ln_ll_json,
    #     chen_participant_size,
    #     algorithms,
    # )

    # for algorithm in algorithms:
    #     utils.plot_time_synthetic(
    #         time_json,
    #         algorithm,
    #         ns,
    #         rs,
    # )

    # utils.plot_time_chen_data(
    #     time_json,
    #     chen_participant_size,
    #     algorithms,
    # )

