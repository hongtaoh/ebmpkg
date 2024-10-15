import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import random
import math
import json
import time
import os
import scipy.stats as stats
from scipy.stats import kendalltau
import re
plt.rcParams['font.family'] = 'DejaVu Sans'
from typing import List, Optional, Tuple, Dict

def metropolis_hastings_kmeans(
    data_we_have,
    iterations,
    n_shuffle,
    log_folder_name,
):
    '''Implement the metropolis-hastings algorithm
    Inputs: 
        - data: data_we_have
        - iterations: number of iterations

    Outputs:
        - best_order: a numpy array
        - best_likelihood: a scalar 
    '''
    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_biomarkers = len(biomarkers)
    n_stages = n_biomarkers + 1
    non_diseased_participant_ids = data_we_have.loc[
        data_we_have.diseased == False].participant.unique()
    diseased_stages = np.arange(start=1, stop=n_stages, step=1)
    # obtain the iniial theta and phi estimates
    theta_phi_kmeans = get_theta_phi_estimates(
        data_we_have,
        biomarkers,
    )

    all_order_dicts = []
    all_current_accepted_likelihoods = []
    acceptance_count = 0
    all_current_acceptance_ratios = []
    all_current_accepted_order_dicts = []
    terminal_output_strings = []

    current_accepted_order = np.random.permutation(np.arange(1, n_stages))
    current_accepted_order_dict = dict(zip(biomarkers, current_accepted_order))
    current_accepted_likelihood = -np.inf

    for _ in range(iterations):
        new_order = current_accepted_order.copy()
        shuffle_order(new_order, n_shuffle)
        current_order_dict = dict(zip(biomarkers, new_order))

        all_participant_ln_likelihood, \
            hashmap_of_normalized_stage_likelihood_dicts = calculate_all_participant_ln_likelihood_and_update_hashmap(
                _,
                data_we_have,
                current_order_dict,
                n_participants,
                non_diseased_participant_ids,
                theta_phi_kmeans,
                diseased_stages,
            )

        # Log-Sum-Exp Trick
        max_likelihood = max(all_participant_ln_likelihood,
                             current_accepted_likelihood)
        prob_of_accepting_new_order = np.exp(
            (all_participant_ln_likelihood - max_likelihood) -
            (current_accepted_likelihood - max_likelihood)
        )

        # prob_of_accepting_new_order = np.exp(
        #     all_participant_ln_likelihood - current_accepted_likelihood)

        # it will definitly update at the first iteration
        if np.random.rand() < prob_of_accepting_new_order:
            acceptance_count += 1
            current_accepted_order = new_order
            current_accepted_likelihood = all_participant_ln_likelihood
            current_accepted_order_dict = current_order_dict

        all_current_accepted_likelihoods.append(current_accepted_likelihood)
        acceptance_ratio = acceptance_count*100/(_+1)
        all_current_acceptance_ratios.append(acceptance_ratio)
        all_order_dicts.append(current_order_dict)
        all_current_accepted_order_dicts.append(current_accepted_order_dict)

        if (_+1) % 10 == 0:
            formatted_string = (
                f"iteration {_ + 1} done, "
                f"current accepted likelihood: {current_accepted_likelihood}, "
                f"current acceptance ratio is {acceptance_ratio:.2f} %, "
                f"current accepted order is {current_accepted_order_dict}, "
            )
            terminal_output_strings.append(formatted_string)
            print(formatted_string)

    final_acceptance_ratio = acceptance_count/iterations

    save_output_strings(log_folder_name, terminal_output_strings)
    save_all_dicts(all_order_dicts, log_folder_name, "all_order")
    save_all_dicts(
        all_current_accepted_order_dicts,
        log_folder_name,
        "all_current_accepted_order_dicts")
    save_all_current_accepted(
        all_current_accepted_likelihoods,
        "all_current_accepted_likelihoods",
        log_folder_name)
    save_all_current_accepted(
        all_current_acceptance_ratios,
        "all_current_acceptance_ratios",
        log_folder_name)
    # save hashmap_of_estimated_theta_and_phi_dicts
    with open(f'{log_folder_name}/theta_phi_kmeans.json', 'w') as fp:
        json.dump(theta_phi_kmeans, fp)
    print("done!")
    return (
        current_accepted_order_dict,
        all_order_dicts,
        all_current_accepted_order_dicts,
        all_current_accepted_likelihoods,
        all_current_acceptance_ratios,
        final_acceptance_ratio,
        theta_phi_kmeans
    )


def obtain_most_likely_order_dic(all_current_accepted_order_dicts, burn_in, thining):
    """Obtain the most likely order based on all the accepted orders 
    Inputs:
        - all_current_accepted_order_dicts 
        - burn_in
        - thining
    Outputs:
        - a dictionary where key is biomarker and value is the most likely order for that biomarker
        Note that in this dic, the keys follow an order. The true order of the first biomarker 
        is 1, the second, 2. 
    """
    df = pd.DataFrame(all_current_accepted_order_dicts)
    biomarker_stage_probability_df = get_biomarker_stage_probability(
        df, burn_in, thining)
    dic = {}
    assigned_stages = set()

    for i, biomarker in enumerate(biomarker_stage_probability_df.index):
        # probability array for that biomarker
        prob_arr = np.array(biomarker_stage_probability_df.iloc[i, :])

        # Sort indices of probabilities in descending order
        sorted_indices = np.argsort(prob_arr)[::-1] + 1

        for stage in sorted_indices:
            if stage not in assigned_stages:
                dic[biomarker] = int(stage)
                assigned_stages.add(stage)
                break
        else:
            raise ValueError(
                f"Could not assign a unique stage for biomarker {biomarker}.")
    return dic

def estimate_params_exact(m0, n0, s0_sq, v0, data):
    '''This is to estimate means and vars based on conjugate priors
    Inputs:
        - data: a vector of measurements 
        - m0: prior estimate of $\mu$.
        - n0: how strongly is the prior belief in $m_0$ is held.
        - s0_sq: prior estimate of $\sigma^2$.
        - v0: prior degress of freedome, influencing the certainty of $s_0^2$.

    Outputs:
        - mu estiate, std estimate
    '''
    # Data summary
    sample_mean = np.mean(data)
    sample_size = len(data)
    sample_var = np.var(data, ddof=1)  # ddof=1 for unbiased estimator

    # Update hyperparameters for the Normal-Inverse Gamma posterior
    updated_m0 = (n0 * m0 + sample_size * sample_mean) / (n0 + sample_size)
    updated_n0 = n0 + sample_size
    updated_v0 = v0 + sample_size
    updated_s0_sq = (1 / updated_v0) * ((sample_size - 1) * sample_var + v0 * s0_sq +
                                        (n0 * sample_size / updated_n0) * (sample_mean - m0)**2)
    updated_alpha = updated_v0/2
    updated_beta = updated_v0*updated_s0_sq/2

    # Posterior estimates
    mu_posterior_mean = updated_m0
    sigma_squared_posterior_mean = updated_beta/updated_alpha

    mu_estimation = mu_posterior_mean
    std_estimation = np.sqrt(sigma_squared_posterior_mean)

    return mu_estimation, std_estimation

def get_theta_phi_conjugate_priors(biomarkers, data_we_have, theta_phi_kmeans):
    '''To get estimated parameters, returns a hashmap
    Input:
    - biomarkers: biomarkers 
    - data_we_have: participants data filled with initial or updated participant_stages
    - theta_phi_kmeans: the initial theta and phi values

    Output: 
    - a hashmap of dictionaries. Key is biomarker name and value is a dictionary.
    This dictionary contains the theta and phi mean/std values. 
    '''
    # empty list of dictionaries to store the estimates
    hashmap_of_means_stds_estimate_dicts = {}

    for biomarker in biomarkers:
        # Initialize dictionary outside the inner loop
        dic = {'biomarker': biomarker}
        for affected in [True, False]:
            data_full = data_we_have[(data_we_have.biomarker == biomarker) & (
                data_we_have.affected == affected)]
            if len(data_full) > 1:
                measurements = data_full.measurement
                s0_sq = np.var(measurements, ddof=1)
                m0 = np.mean(measurements)
                mu_estimate, std_estimate = estimate_params_exact(
                    m0=m0, n0=1, s0_sq=s0_sq, v0=1, data=measurements)
                if affected:
                    dic['theta_mean'] = mu_estimate
                    dic['theta_std'] = std_estimate
                else:
                    dic['phi_mean'] = mu_estimate
                    dic['phi_std'] = std_estimate
            # If there is only one observation or not observation at all, resort to theta_phi_kmeans
            # YES, IT IS POSSIBLE THAT DATA_FULL HERE IS NULL
            # For example, if a biomarker indicates stage of (num_biomarkers), but all participants' stages
            # are smaller than that stage; so that for all participants, this biomarker is not affected
            else:
                # print(theta_phi_kmeans)
                if affected:
                    dic['theta_mean'] = theta_phi_kmeans[biomarker]['theta_mean']
                    dic['theta_std'] = theta_phi_kmeans[biomarker]['theta_std']
                else:
                    dic['phi_mean'] = theta_phi_kmeans[biomarker]['phi_mean']
                    dic['phi_std'] = theta_phi_kmeans[biomarker]['phi_std']
        # print(f"biomarker {biomarker} done!")
        hashmap_of_means_stds_estimate_dicts[biomarker] = dic
    return hashmap_of_means_stds_estimate_dicts


def add_kj_and_affected_and_modify_diseased(data, participant_stages, n_participants):
    '''This is to fill up data_we_have. 
    Basically, add two columns: k_j, affected, and modify diseased column
    based on the initial or updated participant_stages
    Note that we assume here we've already got S_n

    Inputs:
        - data_we_have
        - participant_stages: np array 
        - participants: 0-99
    '''
    participant_stage_dic = dict(
        zip(np.arange(0, n_participants), participant_stages))
    data['k_j'] = data.apply(
        lambda row: participant_stage_dic[row.participant], axis=1)
    data['diseased'] = data.apply(lambda row: row.k_j > 0, axis=1)
    data['affected'] = data.apply(lambda row: row.k_j >= row.S_n, axis=1)
    return data

def compute_all_participant_ln_likelihood_and_update_participant_stages(
        n_participants,
        data,
        non_diseased_participant_ids,
        estimated_theta_phi,
        disease_stages,
        participant_stages,
):
    all_participant_ln_likelihood = 0
    for p in range(n_participants):
        # this participant data
        pdata = data[data.participant == p].reset_index(drop=True)

        """If this participant is not diseased (i.e., if we know k_j is equal to 0)
        We still need to compute the likelihood of this participant seeing this sequence of biomarker data
        but we do not need to estimate k_j like below

        We still need to compute the likelihood because we need to add it to all_participant_ln_likelihood
        """
        if p in non_diseased_participant_ids:
            this_participant_likelihood = compute_likelihood(
                pdata, k_j=0, theta_phi=estimated_theta_phi)
            this_participant_ln_likelihood = np.log(
                this_participant_likelihood + 1e-10)
        else:
            # initiaze stage_likelihood
            stage_likelihood_dict = {}
            for k_j in disease_stages:
                # even though data above has everything, it is filled up by random stages
                # we don't like it and want to know the true k_j. All the following is to update participant_stages
                participant_likelihood = compute_likelihood(
                    pdata, k_j, estimated_theta_phi)
                # update each stage likelihood for this participant
                stage_likelihood_dict[k_j] = participant_likelihood
            likelihood_sum = sum(stage_likelihood_dict.values())
            normalized_stage_likelihood = [
                l/likelihood_sum for l in stage_likelihood_dict.values()]
            normalized_stage_likelihood_dict = dict(
                zip(disease_stages, normalized_stage_likelihood))
            # print(normalized_stage_likelihood)
            sampled_stage = np.random.choice(
                disease_stages, p=normalized_stage_likelihood)
            participant_stages[p] = sampled_stage

            # use weighted average likelihood because we didn't know the exact participant stage
            # all above to calculate participant_stage is only for the purpous of calculate theta_phi
            this_participant_likelihood = weighted_average_likelihood(
                likelihood_sum)
            this_participant_ln_likelihood = np.log(
                this_participant_likelihood + 1e-10)
        """
        All the codes in between are calculating this_participant_ln_likelihood. 
        If we already know kj=0, then
        it's very simple. If kj is unknown, we need to calculate the likelihood of seeing 
        this sequence of biomarker
        data at different stages, and get the relative likelihood before 
        we get a sampled stage (this is for estimating theta and phi). 
        Then we calculate this_participant_ln_likelihood using average likelihood. 
        """
        all_participant_ln_likelihood += this_participant_ln_likelihood
    return all_participant_ln_likelihood


"""The version without reverting back to the max order
"""
def metropolis_hastings_with_conjugate_priors(
    data_we_have,
    iterations,
    log_folder_name,
    n_shuffle
):
    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_biomarkers = len(biomarkers)
    n_stages = n_biomarkers + 1
    diseased_stages = np.arange(start=1, stop=n_stages, step=1)

    non_diseased_participant_ids = data_we_have.loc[
        data_we_have.diseased == False].participant.unique()

    all_order_dicts = []
    all_current_accepted_likelihoods = []
    acceptance_count = 0
    all_current_acceptance_ratios = []
    all_current_accepted_order_dicts = []
    terminal_output_strings = []
    all_participant_stages_at_the_end_of_each_iteration = []
    hashmap_of_estimated_theta_phi_dicts = {}

    # initialize an ordering and likelihood
    # note that it should be a random permutation of numbers 1-10
    current_accepted_order = np.random.permutation(np.arange(1, n_stages))
    current_accepted_order_dict = dict(zip(biomarkers, current_accepted_order))
    current_accepted_likelihood = -np.inf

    participant_stages = np.zeros(n_participants)
    for idx in range(n_participants):
        if idx not in non_diseased_participant_ids:
            # 1-len(diseased_stages), inclusive on both ends
            participant_stages[idx] = random.randint(1, len(diseased_stages))

    for _ in range(iterations):

        # print(f"should revert: {should_revert_to_max_likelihood_order}")
        # we are going to shuffle new_order below. So it's better to copy first.
        new_order = current_accepted_order.copy()
        # random.shuffle(new_order)

        shuffle_order(new_order, n_shuffle)

        current_order_dict = dict(zip(biomarkers, new_order))

        # copy the data to avoid modifying the original
        data = data_we_have.copy()
        data['S_n'] = data.apply(
            lambda row: current_order_dict[row['biomarker']], axis=1)
        # add kj and affected for the whole dataset based on participant_stages
        # also modify diseased col (because it will be useful for the new theta_phi_kmeans)
        data = add_kj_and_affected_and_modify_diseased(
            data, participant_stages, n_participants)
        # obtain the iniial theta and phi estimates
        theta_phi_kmeans = get_theta_phi_estimates(
            data_we_have,
            biomarkers,
        )
        estimated_theta_phi = get_theta_phi_conjugate_priors(
            biomarkers, data, theta_phi_kmeans)

        # update theta_phi_dic
        hashmap_of_estimated_theta_phi_dicts[_] = estimated_theta_phi

        all_participant_ln_likelihood = compute_all_participant_ln_likelihood_and_update_participant_stages(
            n_participants,
            data,
            non_diseased_participant_ids,
            estimated_theta_phi,
            diseased_stages,
            participant_stages,
        )

        # ratio = likelihood/best_likelihood
        # because we are using np.log(likelihood) and np.log(best_likelihood)
        # np.exp(a)/np.exp(b) = np.exp(a - b)
        # if a > b, then np.exp(a - b) > 1

        # Log-Sum-Exp Trick
        max_likelihood = max(all_participant_ln_likelihood,
                             current_accepted_likelihood)
        prob_of_accepting_new_order = np.exp(
            (all_participant_ln_likelihood - max_likelihood) -
            (current_accepted_likelihood - max_likelihood)
        )
        # prob_of_accepting_new_order = np.exp(
        #     all_participant_ln_likelihood - current_accepted_likelihood)

        # it will definitly update at the first iteration
        if np.random.rand() < prob_of_accepting_new_order:
            acceptance_count += 1
            current_accepted_order = new_order
            current_accepted_likelihood = all_participant_ln_likelihood
            current_accepted_order_dict = current_order_dict

        all_participant_stages_at_the_end_of_each_iteration.append(
            participant_stages)
        all_current_accepted_likelihoods.append(current_accepted_likelihood)
        acceptance_ratio = acceptance_count*100/(_+1)
        all_current_acceptance_ratios.append(acceptance_ratio)
        all_order_dicts.append(current_order_dict)
        all_current_accepted_order_dicts.append(current_accepted_order_dict)

        # if _ >= burn_in and _ % thining == 0:
        if (_+1) % 10 == 0:
            formatted_string = (
                f"iteration {_ + 1} done, "
                f"current accepted likelihood: {current_accepted_likelihood}, "
                f"current acceptance ratio is {acceptance_ratio:.2f} %, "
                f"current accepted order is {current_accepted_order_dict}, "
            )
            terminal_output_strings.append(formatted_string)
            print(formatted_string)
    final_acceptance_ratio = acceptance_count/iterations

    save_output_strings(log_folder_name, terminal_output_strings)

    save_all_dicts(all_order_dicts, log_folder_name, "all_order")
    save_all_dicts(
        all_current_accepted_order_dicts,
        log_folder_name,
        "all_current_accepted_order_dicts")
    save_all_current_accepted(
        all_current_accepted_likelihoods,
        "all_current_accepted_likelihoods",
        log_folder_name)
    save_all_current_accepted(
        all_current_acceptance_ratios,
        "all_current_acceptance_ratios",
        log_folder_name)
    save_all_current_participant_stages(
        all_participant_stages_at_the_end_of_each_iteration,
        "participant_stages_at_the_end_of_each_iteartion",
        log_folder_name)
    # save hashmap_of_estimated_theta_and_phi_dicts
    with open(f'{log_folder_name}/hashmap_of_estimated_theta_phi_dicts.json', 'w') as fp:
        json.dump(hashmap_of_estimated_theta_phi_dicts, fp)
    print("done!")
    return (
        current_accepted_order_dict,
        participant_stages,
        all_order_dicts,
        all_participant_stages_at_the_end_of_each_iteration,
        all_current_accepted_order_dicts,
        all_current_accepted_likelihoods,
        all_current_acceptance_ratios,
        final_acceptance_ratio,
        hashmap_of_estimated_theta_phi_dicts,
    )


def get_biomarker_stage_probability(df, burn_in, thining):
    """filter through all_dicts using burn_in and thining 
    and for each biomarker, get probability of being in each possible stage

    Input:
        - df: all_order_dicts or all_current_accepted_order_dicts, but after turning into
            dataframes.
        - burn_in
        - thinning
    Output:
        - dff: a pandas dataframe where index is biomarker name, each col is each stage
        and each cell is the probability of that biomarker indicating that stage

        Note that in dff, its index follows the asending order. That is the say, 
        index[0] is the the biomarker with order 1
    """
    df = df[(df.index > burn_in) & (df.index % thining == 0)]
    # Create an empty list to hold dictionaries
    dict_list = []

    # assume biomarker names are in a very messy order
    biomarker_names = np.array(df.columns)
    biomarker_order_dic = {}
    for biomarker_name in biomarker_names:
        biomarker_order = int(re.findall(r'\d+', biomarker_name)[0])
        biomarker_order_dic[biomarker_name] = biomarker_order
    # print(biomarker_order_dic)
    ascending_order_biomarkers = [None]*len(biomarker_names)
    for biomarker, order in biomarker_order_dic.items():
        ascending_order_biomarkers[order - 1] = biomarker

    # iterate through biomarkers
    for biomarker in ascending_order_biomarkers:
        dic = {"biomarker": biomarker}
        # get the frequency of biomarkers
        # value_counts will generate a Series where index is each cell's value
        # and the value is the frequency of that value
        stage_counts = df[biomarker].value_counts()
        # for each stage
        # not that df.shape[1] should be equal to num_biomarkers
        for i in range(1, df.shape[1] + 1):
            # get stage:prabability
            dic[i] = stage_counts.get(i, 0)/len(df)
        dict_list.append(dic)

    dff = pd.DataFrame(dict_list)
    dff.set_index(dff.columns[0], inplace=True)
    return dff


def save_heatmap(all_dicts, burn_in, thining, folder_name, file_name, title):
    # Check if the directory exists
    if not os.path.exists(folder_name):
        # Create the directory if it does not exist
        os.makedirs(folder_name)
    df = pd.DataFrame(all_dicts)
    biomarker_stage_probability_df = get_biomarker_stage_probability(
        df, burn_in, thining)
    sns.heatmap(biomarker_stage_probability_df,
                annot=True, cmap="Greys", linewidths=.5,
                cbar_kws={'label': 'Probability'},
                fmt=".1f",
                # vmin=0, vmax=1,
                )
    plt.xlabel('Stage')
    plt.ylabel('Biomarker')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{folder_name}/{file_name}.png")
    # plt.savefig(f'{file_name}.pdf')
    plt.close()

def sampled_row_based_on_column_frequencies(a):
    """for ndarray, sample one element in each col based on elements' frequencies
    input:
        a: a numpy ndarray 
    output:
        a 1d array 
    """
    sampled_row = []
    for col in range(a.shape[1]):
        col_arr = a[:, col]
        unique_elements, counts = np.unique(col_arr, return_counts=True)
        probs = counts/counts.sum()
        sampled_element = np.random.choice(unique_elements, p=probs)
        sampled_row.append(sampled_element)
    return np.array(sampled_row)


def save_trace_plot(burn_in, all_current_likelihoods, folder_name, file_name, title):
    # Check if the directory exists
    if not os.path.exists(folder_name):
        # Create the directory if it does not exist
        os.makedirs(folder_name)
    current_likelihoods_to_plot = all_current_likelihoods[burn_in:]
    x = np.arange(
        start=burn_in + 1, stop=len(all_current_likelihoods) + 1, step=1)
    plt.scatter(x, current_likelihoods_to_plot, alpha=0.5)
    plt.xlabel('Iteration #')
    plt.ylabel('Current Likelihood')
    plt.title(title)
    plt.savefig(f'{folder_name}/{file_name}.png')
    plt.close()

# def process_chen_data(file):
#     """Prepare data for analysis below
#     """
#     df = pd.read_excel(file)
#     biomarker_name_change_dic = dict(zip(['FCI(HIP)', 'GMI(HIP)', 'FCI(Fusi)', 'FCI(PCC)', 'GMI(FUS)'],
#                                          [1, 3, 5, 2, 4]))
#     df.rename(
#         columns={df.columns[0]:
#                  'participant_category', df.columns[1]:
#                  'participant'},
#                  inplace=True)
#     # df = df[df.participant_category.isin(['CN', 'AD '])]
#     df['diseased'] = df.apply(lambda row: row.participant_category != 'CN', axis = 1)
#     df = pd.melt(df, id_vars=['participant_category', "participant", "timestamp", 'diseased'],
#                         value_vars=["FCI(HIP)", "GMI(HIP)", "FCI(Fusi)", "FCI(PCC)", "GMI(FUS)"],
#                         var_name='biomarker', value_name='measurement')
#     # convert participant id
#     n_participant = len(df.participant.unique())
#     participant_ids = [_ for _ in range(n_participant)]
#     participant_string_id_dic = dict(zip(df.participant.unique(), participant_ids))
#     df['participant'] = df.apply(lambda row: participant_string_id_dic[row.participant], axis = 1 )
#     df['biomarker'] = df.apply(lambda row: f"{row.biomarker}-{biomarker_name_change_dic[row.biomarker]}",
#                                axis = 1)
#     return df


def process_chen_data(file, chen_real_order, participant_size, seed=None):
    """Prepare data for analysis below
    """
    if seed is not None:
        # Set the seed for numpy's random number generator
        np.random.seed(seed)
    df = pd.read_excel(file)
    biomarker_name_change_dic = dict(zip(['FCI(HIP)', 'GMI(HIP)', 'FCI(Fusi)', 'FCI(PCC)', 'GMI(FUS)'],
                                         chen_real_order))
    df.rename(
        columns={df.columns[0]:
                 'participant_category', df.columns[1]:
                 'participant'},
        inplace=True)
    # df = df[df.participant_category.isin(['CN', 'AD '])]
    df['diseased'] = df.apply(
        lambda row: row.participant_category != 'CN', axis=1)
    df = pd.melt(df, id_vars=['participant_category', "participant", "timestamp", 'diseased'],
                 value_vars=["FCI(HIP)", "GMI(HIP)",
                             "FCI(Fusi)", "FCI(PCC)", "GMI(FUS)"],
                 var_name='biomarker', value_name='measurement')
    # convert participant id
    n_participant = len(df.participant.unique())
    participant_ids = [_ for _ in range(n_participant)]
    participant_string_id_dic = dict(
        zip(df.participant.unique(), participant_ids))
    df['participant'] = df.apply(
        lambda row: participant_string_id_dic[row.participant], axis=1)
    df['biomarker'] = df.apply(lambda row: f"{row.biomarker}-{biomarker_name_change_dic[row.biomarker]}",
                               axis=1)
    if participant_size > n_participant:
        df = expand_and_save_data(df, participant_size)
    return df


def expand_and_save_data(df, participant_size):
    if not os.path.exists("data/chen_data"):
        # Create the directory if it does not exist
        os.makedirs("data/chen_data")
    participant_ids = df.participant.unique()
    sampled_participant_ids = np.random.choice(
        participant_ids, participant_size)
    dff = df[df.participant == sampled_participant_ids[0]]
    dff = dff.assign(participant=[0]*len(dff))
    # Parameters for the normal distribution
    mean = 0
    variance = 10**(-4)
    std_deviation = np.sqrt(variance)

    for idx, old_participant_id in enumerate(sampled_participant_ids[1:]):
        subset_df = df[df.participant == old_participant_id]
        # update participant id
        subset_df = subset_df.assign(participant=[idx + 1]*len(subset_df))
        delta = np.random.normal(mean, std_deviation)
        new_measurement = [(x + delta) for x in subset_df.measurement]
        subset_df = subset_df.assign(measurement=new_measurement)
        dff = pd.concat([dff, subset_df])
    dff.reset_index(drop=True, inplace=True)
    dff.to_csv("data/chen_data/expanded.csv", index=False)
    return dff


def run_conjugate_priors(
    data_we_have,
    data_source,
    iterations,
    log_folder_name,
    img_folder_name,
    n_shuffle,
    burn_in,
    thining,
    tau_dic,
    ln_ll_dic,
    time_dic
):
    n_biomarkers = len(data_we_have.biomarker.unique())
    n = len(data_we_have.participant.unique())
    healthy_participants = len(
        data_we_have[data_we_have.diseased == False].participant.unique())
    r = healthy_participants/n
    print(
        f"Now begins with {data_source} with conjugate priors, {healthy_participants}|{n}")
    start_time = time.time()
    biomarker_best_order_dic, \
        participant_stages, \
        all_dicts, \
        all_current_participant_stages, \
        all_current_order_dicts, \
        all_current_likelihoods, \
        all_current_acceptance_ratios, \
        final_acceptance_ratio, \
        hashmap_of_estimated_theta_phi_dic = metropolis_hastings_with_conjugate_priors(
            data_we_have, iterations, log_folder_name, n_shuffle,
        )
    save_heatmap(
        all_dicts, burn_in, thining,
        folder_name=img_folder_name,
        file_name="heatmap_all_orderings",
        title=f"{data_source} with Conjugate Priors, All Orderings"
    )
    save_heatmap(
        all_current_order_dicts,
        burn_in=0, thining=1,
        folder_name=img_folder_name,
        file_name="heatmap_all_current_accepted",
        title=f"{data_source} with Conjugate Priors, All Current Best Orderings"
    )
    save_trace_plot(
        burn_in,
        all_current_likelihoods,
        folder_name=img_folder_name,
        file_name="trace_plot",
        title=f"Trace Plot, {data_source} with Conjugate Priors"
    )
    most_likely_order_dic = obtain_most_likely_order_dic(
        all_current_order_dicts, burn_in, thining)
    most_likely_order = list(most_likely_order_dic.values())
    # the index with the most likely order and also the highest likelihood
    idx, estimated_theta_phi_dic = obtain_estimated_theta_phi(
        all_current_order_dicts,
        all_current_likelihoods,
        hashmap_of_estimated_theta_phi_dic,
        most_likely_order_dic,
    )
    real_order_ln_ll, most_likely_ln_ll = compare_ll(
        data_we_have,
        estimated_theta_phi_dic,
        most_likely_order_dic,
    )
    if set(most_likely_order) != set(range(1, n_biomarkers + 1)):
        raise ValueError(
            "This most likely order has repeated stages or different stages than expected.")
    tau, p_value = kendalltau(most_likely_order, range(1, n_biomarkers + 1))
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    if data_source == "Synthetic Data":
        tau_dic[f"{int(r*n)}/{n}"] = tau
        time_dic[f"{int(r*n)}/{n}"] = {}
        time_dic[f"{int(r*n)}/{n}"]['iterations'] = iterations 
        time_dic[f"{int(r*n)}/{n}"]['duration'] = execution_time
        ln_ll_dic[f"{int(r*n)}/{n}"] = {}
        ln_ll_dic[f"{int(r*n)}/{n}"]['most_likely_order_dic'] = most_likely_order_dic
        ln_ll_dic[f"{int(r*n)}/{n}"]['index'] = idx 
        ln_ll_dic[f"{int(r*n)}/{n}"]['estimated_theta_phi_dic'] = estimated_theta_phi_dic
        ln_ll_dic[f"{int(r*n)}/{n}"]['real_order_ln_ll'] = real_order_ln_ll
        ln_ll_dic[f"{int(r*n)}/{n}"]['most_likely_ln_ll'] = most_likely_ln_ll
    else:
        tau_dic[f"{n}"] = tau
        time_dic[f"{n}"] = {}
        time_dic[f"{n}"]['iterations'] = iterations 
        time_dic[f"{n}"]['duration'] = execution_time
        ln_ll_dic[f"{n}"] = {}
        ln_ll_dic[f"{n}"]['most_likely_order_dic'] = most_likely_order_dic
        ln_ll_dic[f"{n}"]['index'] = idx 
        ln_ll_dic[f"{n}"]['estimated_theta_phi_dic'] = estimated_theta_phi_dic
        ln_ll_dic[f"{n}"]['real_order_ln_ll'] = real_order_ln_ll
        ln_ll_dic[f"{n}"]['most_likely_ln_ll'] = most_likely_ln_ll
    print(
        f"Execution time: {execution_time} min for {data_source} using conjugate priors.")
    print("---------------------------------------------")


def obtain_estimated_theta_phi(
        all_current_accepted_order_dicts,
        all_current_accepted_likelihoods,
        hashmap_of_estimated_theta_phi_dic,
        most_likely_order_dic,
):
    """
    most_likely_order_dic, iteration (from 0): estimated_theta_phi_dic

    The purpose of this function is to find the idx with the highest
    likelihoods and with the most likely order

    Then return the estimated_theta_phi
    """
    indices = np.where(
        np.array(all_current_accepted_order_dicts) == most_likely_order_dic)[0]
    max_ll = np.max(np.array(all_current_accepted_likelihoods)[indices])
    winner = np.argwhere(np.array(all_current_accepted_likelihoods) == max_ll)
    winner = winner.flatten().tolist()
    idx = np.random.choice(winner)
    return int(idx), hashmap_of_estimated_theta_phi_dic[idx]

def calculate_all_participant_ll_likelihood_based_on_order_dic(
        data, order_dic, estimated_theta_phi_dic):
    """Compute all_participant_ll_likelihood based on a certain order dict 
    and a certain theta phi dict
    """
    data['S_n'] = data.apply(lambda row: order_dic[row.biomarker], axis=1)
    n_participants = len(data.participant.unique())
    non_diseased_participant_ids = data[
        data.diseased == False].participant.unique()
    disease_stages = np.array(list(order_dic.values()))
    all_participant_ln_likelihood = 0
    for p in range(n_participants):
        pdata = data[data.participant == p].reset_index(drop=True)
        if p in non_diseased_participant_ids:
            this_participant_likelihood = compute_likelihood(
                pdata, k_j=0, theta_phi=estimated_theta_phi_dic)
            this_participant_ln_likelihood = np.log(
                this_participant_likelihood + 1e-10)
        else:
            stage_likelihood_dict = {}
            for k_j in disease_stages:
                kj_likelihood = compute_likelihood(
                    pdata, k_j, estimated_theta_phi_dic)
                # update each stage likelihood for this participant
                stage_likelihood_dict[k_j] = kj_likelihood
            # Add a small epsilon to avoid division by zero
            likelihood_sum = sum(stage_likelihood_dict.values())
            this_participant_ln_likelihood = np.log(np.mean(likelihood_sum))

        all_participant_ln_likelihood += this_participant_ln_likelihood
    return all_participant_ln_likelihood


def compare_ll(
    data,
    estimated_theta_phi_dic,
    most_likely_order_dic,
):
    real_order_dic = dict(
        zip(most_likely_order_dic.keys(), range(1, len(
            most_likely_order_dic) + 1)))
    real_order_ln_ll = calculate_all_participant_ll_likelihood_based_on_order_dic(
        data, real_order_dic, estimated_theta_phi_dic)
    most_likely_ln_ll = calculate_all_participant_ll_likelihood_based_on_order_dic(
        data, most_likely_order_dic, estimated_theta_phi_dic)
    return real_order_ln_ll, most_likely_ln_ll


def run_soft_kmeans(
    data_we_have,
    data_source,
    iterations,
    n_shuffle,
    log_folder_name,
    img_folder_name,
    burn_in,
    thining,
    tau_dic,
    ln_ll_dic,
    time_dic,
):
    n_biomarkers = len(data_we_have.biomarker.unique())
    n = len(data_we_have.participant.unique())
    healthy_participants = len(
        data_we_have[data_we_have.diseased == False].participant.unique())
    r = healthy_participants/n
    # theta_phi_estimates = pd.read_csv('data/means_stds.csv')

    print(
        f"Now begins with {data_source} with soft kmeans, {healthy_participants}|{n}")
    start_time = time.time()
    current_accepted_order_dict, \
        all_order_dicts, \
        all_current_accepted_order_dicts, \
        all_current_accepted_likelihoods, \
        all_current_acceptance_ratios, \
        final_acceptance_ratio, \
        hashmap_of_estimated_theta_phi_dic = metropolis_hastings_soft_kmeans(
            data_we_have, iterations, n_shuffle, log_folder_name,
        )

    save_heatmap(
        all_order_dicts,
        burn_in, thining,
        folder_name=img_folder_name,
        file_name="heatmap_all_orderings",
        title=f"{data_source} with Soft KMeans, All Orderings"
    )
    save_heatmap(
        all_current_accepted_order_dicts,
        burn_in=0, thining=1,
        folder_name=img_folder_name,
        file_name="heatmap_all_current_accepted",
        title=f"{data_source} with Soft KMeans, All Current Accepted Orderings"
    )
    save_trace_plot(
        burn_in,
        all_current_accepted_likelihoods,
        folder_name=img_folder_name,
        file_name="trace_plot",
        title=f"Trace Plot, {data_source} with Soft KMeans"
    )
    most_likely_order_dic = obtain_most_likely_order_dic(
        all_current_accepted_order_dicts,  burn_in, thining)
    most_likely_order = list(most_likely_order_dic.values())
    idx, estimated_theta_phi_dic = obtain_estimated_theta_phi(
        all_current_accepted_order_dicts,
        all_current_accepted_likelihoods,
        hashmap_of_estimated_theta_phi_dic,
        most_likely_order_dic,
    )
    real_order_ln_ll, most_likely_ln_ll = compare_ll(
        data_we_have,
        estimated_theta_phi_dic,
        most_likely_order_dic,
    )
    if set(most_likely_order) != set(range(1, n_biomarkers + 1)):
        raise ValueError(
            "This most likely order has repeated stages or different stages than expected.")
    tau, p_value = kendalltau(most_likely_order, range(1, n_biomarkers + 1))
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    if data_source == "Synthetic Data":
        tau_dic[f"{int(r*n)}/{n}"] = tau
        time_dic[f"{int(r*n)}/{n}"] = {}
        time_dic[f"{int(r*n)}/{n}"]['iterations'] = iterations 
        time_dic[f"{int(r*n)}/{n}"]['duration'] = execution_time
        ln_ll_dic[f"{int(r*n)}/{n}"] = {}
        ln_ll_dic[f"{int(r*n)}/{n}"]['most_likely_order_dic'] = most_likely_order_dic
        ln_ll_dic[f"{int(r*n)}/{n}"]['index'] = idx 
        ln_ll_dic[f"{int(r*n)}/{n}"]['estimated_theta_phi_dic'] = estimated_theta_phi_dic
        ln_ll_dic[f"{int(r*n)}/{n}"]['real_order_ln_ll'] = real_order_ln_ll
        ln_ll_dic[f"{int(r*n)}/{n}"]['most_likely_ln_ll'] = most_likely_ln_ll
    else:
        tau_dic[f"{n}"] = tau
        time_dic[f"{n}"] = {}
        time_dic[f"{n}"]['iterations'] = iterations 
        time_dic[f"{n}"]['duration'] = execution_time
        ln_ll_dic[f"{n}"] = {}
        ln_ll_dic[f"{n}"]['most_likely_order_dic'] = most_likely_order_dic
        ln_ll_dic[f"{n}"]['index'] = idx 
        ln_ll_dic[f"{n}"]['estimated_theta_phi_dic'] = estimated_theta_phi_dic
        ln_ll_dic[f"{n}"]['real_order_ln_ll'] = real_order_ln_ll
        ln_ll_dic[f"{n}"]['most_likely_ln_ll'] = most_likely_ln_ll
    print(
        f"Execution time: {execution_time} mins for {data_source} using soft kmeans.")
    print("---------------------------------------------")


def run_kmeans(
    data_we_have,
    data_source,
    iterations,
    n_shuffle,
    log_folder_name,
    img_folder_name,
    burn_in,
    thining,
    tau_dic,
    ln_ll_dic,
    time_dic,
):
    n_biomarkers = len(data_we_have.biomarker.unique())
    n = len(data_we_have.participant.unique())
    healthy_participants = len(
        data_we_have[data_we_have.diseased == False].participant.unique())
    r = healthy_participants/n

    print(
        f"Now begins with {data_source} with kmeans, {healthy_participants}|{n}")
    start_time = time.time()
    current_accepted_order_dict, \
        all_order_dicts, \
        all_current_accepted_order_dicts, \
        all_current_accepted_likelihoods, \
        all_current_acceptance_ratios, \
        final_acceptance_ratio, \
        theta_phi_kmeans = metropolis_hastings_kmeans(
            data_we_have, iterations, n_shuffle, log_folder_name
        )

    save_heatmap(
        all_order_dicts,
        burn_in, thining,
        folder_name=img_folder_name,
        file_name="heatmap_all_orderings",
        title=f"{data_source} with KMeans, All Orderings"
    )
    save_heatmap(
        all_current_accepted_order_dicts,
        burn_in=0, thining=1,
        folder_name=img_folder_name,
        file_name="heatmap_all_current_accepted",
        title=f"{data_source} with KMeans, All Current Accepted Orderings"
    )
    save_trace_plot(
        burn_in,
        all_current_accepted_likelihoods,
        folder_name=img_folder_name,
        file_name="trace_plot",
        title=f"Trace Plot, {data_source} with KMeans"
    )
    most_likely_order_dic = obtain_most_likely_order_dic(
        all_current_accepted_order_dicts,  burn_in, thining)
    most_likely_order = list(most_likely_order_dic.values())
    real_order_ln_ll, most_likely_ln_ll = compare_ll(
        data_we_have,
        theta_phi_kmeans,
        most_likely_order_dic,
    )
    if set(most_likely_order) != set(range(1, n_biomarkers + 1)):
        raise ValueError(
            "This most likely order has repeated stages or different stages than expected.")
    tau, p_value = kendalltau(most_likely_order, range(1, n_biomarkers + 1))
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    if data_source == "Synthetic Data":
        tau_dic[f"{int(r*n)}/{n}"] = tau
        time_dic[f"{int(r*n)}/{n}"] = {}
        time_dic[f"{int(r*n)}/{n}"]['iterations'] = iterations 
        time_dic[f"{int(r*n)}/{n}"]['duration'] = execution_time
        ln_ll_dic[f"{int(r*n)}/{n}"] = {}
        ln_ll_dic[f"{int(r*n)}/{n}"]['most_likely_order_dic'] = most_likely_order_dic
        ln_ll_dic[f"{int(r*n)}/{n}"]['estimated_theta_phi_dic'] = theta_phi_kmeans
        ln_ll_dic[f"{int(r*n)}/{n}"]['real_order_ln_ll'] = real_order_ln_ll
        ln_ll_dic[f"{int(r*n)}/{n}"]['most_likely_ln_ll'] = most_likely_ln_ll
    else:
        tau_dic[f"{n}"] = tau
        time_dic[f"{n}"] = {}
        time_dic[f"{n}"]['iterations'] = iterations 
        time_dic[f"{n}"]['duration'] = execution_time
        ln_ll_dic[f"{n}"] = {}
        ln_ll_dic[f"{n}"]['most_likely_order_dic'] = most_likely_order_dic
        ln_ll_dic[f"{n}"]['estimated_theta_phi_dic'] = theta_phi_kmeans
        ln_ll_dic[f"{n}"]['real_order_ln_ll'] = real_order_ln_ll
        ln_ll_dic[f"{n}"]['most_likely_ln_ll'] = most_likely_ln_ll
    print(
        f"Execution time: {execution_time} mins for {data_source} using kmeans.")
    print("---------------------------------------------")

def plot_tau_synthetic(
        tau_file,
        algorithm,
        ns,
        rs,
):
    with open(tau_file) as f:
        tau = json.load(f)
    data = tau['synthetic'][algorithm]
    formatted_algorithm = algorithm.replace('_', ' ')
    dict_list = []
    for n in ns:
        for r in rs:
            dic = {}
            dic["J"] = f"$J={n}$"  # Use LaTeX formatting for italic
            dic['r'] = f"{int(r*100)}%"
            key = f"{int(n*r)}/{n}"
            dic['tau'] = data[key]
            dict_list.append(dic)
    df = pd.DataFrame(dict_list)

    with sns.axes_style("whitegrid"):  # Temporarily set style
        g = sns.catplot(
            data=df, kind="bar",
            x="J", y="tau", hue="r",
            palette="bright", alpha=.8,
            height=6, aspect=1.2
        )
        g.despine(left=True)

        # Set the overall title using suptitle
        g.fig.suptitle(
            f"Kendall's tau across different combinations ({formatted_algorithm})",
            fontsize=20,
            y=1.1)

        g.legend.set_bbox_to_anchor((1.15, 0.5))  # Adjust position as needed
        # Set the axis labels
        g.set_axis_labels("", "Kendall's Tau", fontsize=18)
        g.legend.set_title("Healthy Ratio", prop={'size': 16})
        # Adjust the font size for the legend text
        for text in g.legend.texts:
            text.set_fontsize(14)  # Adjust as needed

        # Increase font size for x-axis and y-axis tick labels
        g.set_xticklabels(fontsize=14)
        g.set_yticklabels(fontsize=14)

        g.savefig(f'img/{"synthetic"}/{algorithm}_tau.png', dpi=200)

def plot_tau_chen_data(
        tau_file,
        chen_participant_size,
        algorithms,
):
    with open(tau_file) as f:
        tau = json.load(f)

    dict_list = []
    for algorithm in algorithms:
        data = tau['chen_data'][algorithm]
        for n in chen_participant_size:
            dic = {}
            dic["J"] = f"$J={n}$"  # Use LaTeX formatting for italic
            formatted_algorithm = algorithm.replace('_', ' ')
            dic['algorithm'] = formatted_algorithm
            dic['tau'] = data[str(n)]
            dict_list.append(dic)
    df = pd.DataFrame(dict_list)

    with sns.axes_style("whitegrid"):  # Temporarily set style
        g = sns.catplot(
            data=df, kind="bar",
            x="J", y="tau", hue="algorithm",
            palette="bright", alpha=.8,
            height=6, aspect=1.2
        )
        g.despine(left=True)

        # Set the overall title using suptitle
        g.fig.suptitle(
            f"Kendall's tau across different algorithms (Chen's data)",
            fontsize=20,
            y=1.1)

        g.legend.set_bbox_to_anchor((1.15, 0.5))  # Adjust position as needed
        # Set the axis labels
        g.set_axis_labels("", "Kendall's Tau", fontsize=18)
        g.legend.set_title("Algorithm", prop={'size': 16})
        # Adjust the font size for the legend text
        for text in g.legend.texts:
            text.set_fontsize(14)  # Adjust as needed

        # Increase font size for x-axis and y-axis tick labels
        g.set_xticklabels(fontsize=14)
        g.set_yticklabels(fontsize=14)

        g.savefig(f'img/{"chen_data"}/tau.png', dpi=200)

def plot_ln_ll_synthetic(
    ln_ll_file,
    algorithm,
    ns,
    rs
):
    with open(ln_ll_file) as f:
        tau = json.load(f)
    data = tau['synthetic'][algorithm]
    formatted_algorithm = algorithm.replace('_', ' ')
    dict_list = []
    for n in ns:
        for r in rs:
            dic = {}
            dic["J"] = f"$J={n}$"  # Use LaTeX formatting for italic
            dic['r'] = f"{int(r*100)}%"
            key = f"{int(n*r)}/{n}"
            dic['ln_diff'] = data[key]['most_likely_ln_ll'] - \
                data[key]['real_order_ln_ll']
            dict_list.append(dic)
    df = pd.DataFrame(dict_list)
    with sns.axes_style("whitegrid"):  # Temporarily set style
        g = sns.catplot(
            data=df, kind="bar",
            x="J", y="ln_diff", hue="r",
            palette="bright", alpha=.8,
            height=6, aspect=1.2
        )
        g.despine(left=True)

        # Set the overall title using suptitle
        g.fig.suptitle(
            f"Log likelihood differences with different combinations ({formatted_algorithm})",
            fontsize=20,
            y=1.1)

        g.legend.set_bbox_to_anchor((1.4, 0.5))  # Adjust position as needed
        # Set the axis labels
        g.set_axis_labels(
            "", "ln(most likely order) - ln(real order)", fontsize=18)
        g.legend.set_title("Healthy Ratio", prop={'size': 16})
        # Adjust the font size for the legend text
        for text in g.legend.texts:
            text.set_fontsize(14)  # Adjust as needed

        # Increase font size for x-axis and y-axis tick labels
        g.set_xticklabels(fontsize=14)
        g.set_yticklabels(fontsize=14)
        # plt.yscale('log')

        g.savefig(f'img/{"synthetic"}/{algorithm}_ln_ll.png', dpi=200)

def plot_ln_ll_chen_data(
        ln_ll_file,
        chen_participant_size,
        algorithms,
):
    with open(ln_ll_file) as f:
        tau = json.load(f)

    dict_list = []
    for algorithm in algorithms:
        data = tau['chen_data'][algorithm]
        for n in chen_participant_size:
            dic = {}
            dic["J"] = f"$J={n}$"  # Use LaTeX formatting for italic
            formatted_algorithm = algorithm.replace('_', ' ')
            dic['algorithm'] = formatted_algorithm
            dic['ln_diff'] = data[str(n)]['most_likely_ln_ll'] - data[str(n)]['real_order_ln_ll']
            dict_list.append(dic)
    df = pd.DataFrame(dict_list)

    with sns.axes_style("whitegrid"):  # Temporarily set style
        g = sns.catplot(
            data=df, kind="bar",
            x="J", y="ln_diff", hue="algorithm",
            palette="bright", alpha=.8, 
            height=6, aspect=1.2
        )
        g.despine(left=True)
        
        # Set the overall title using suptitle
        g.fig.suptitle(
            f"Comparing log likelihood differences across different algorithms (Chen's data)", 
            fontsize=20, 
            y=1.1)
        
        g.legend.set_bbox_to_anchor((1.15, 0.5))  # Adjust position as needed
        # Set the axis labels
        g.set_axis_labels("", "ln(most likely order) - ln(real order)", fontsize=18)
        g.legend.set_title("Algorithm", prop={'size': 16})
        # Adjust the font size for the legend text
        for text in g.legend.texts:
            text.set_fontsize(14)  # Adjust as needed
        # plt.yscale('log')
        # Increase font size for x-axis and y-axis tick labels
        g.set_xticklabels(fontsize=14)
        g.set_yticklabels(fontsize=14)

        g.savefig(f'img/{"chen_data"}/ln_ll.png', dpi=200)


def plot_time_synthetic(
        time_file,
        algorithm,
        ns,
        rs,
):
    with open(time_file) as f:
        time = json.load(f)
    data = time['synthetic'][algorithm]
    formatted_algorithm = algorithm.replace('_', ' ')
    dict_list = []
    for n in ns:
        for r in rs:
            dic = {}
            dic["J"] = f"$J={n}$"  # Use LaTeX formatting for italic
            dic['r'] = f"{int(r*100)}%"
            key = f"{int(n*r)}/{n}"
            dic['time'] = data[key]['duration']
            dict_list.append(dic)
    df = pd.DataFrame(dict_list)

    with sns.axes_style("whitegrid"):  # Temporarily set style
        g = sns.catplot(
            data=df, kind="bar",
            x="J", y="time", hue="r",
            palette="bright", alpha=.8,
            height=6, aspect=1.2
        )
        g.despine(left=True)

        # Set the overall title using suptitle
        g.fig.suptitle(
            f"Execution time across different combinations ({formatted_algorithm})",
            fontsize=20,
            y=1.1)

        g.legend.set_bbox_to_anchor((1.15, 0.5))  # Adjust position as needed
        # Set the axis labels
        g.set_axis_labels("", "Execution time (in minutes)", fontsize=18)
        g.legend.set_title("Healthy Ratio", prop={'size': 16})
        # Adjust the font size for the legend text
        for text in g.legend.texts:
            text.set_fontsize(14)  # Adjust as needed

        # Increase font size for x-axis and y-axis tick labels
        g.set_xticklabels(fontsize=14)
        g.set_yticklabels(fontsize=14)

        g.savefig(f'img/{"synthetic"}/{algorithm}_time.png', dpi=200)

def plot_time_chen_data(
        time_file,
        chen_participant_size,
        algorithms,
):
    with open(time_file) as f:
        time = json.load(f)

    dict_list = []
    for algorithm in algorithms:
        data = time['chen_data'][algorithm]
        for n in chen_participant_size:
            dic = {}
            dic["J"] = f"$J={n}$"  # Use LaTeX formatting for italic
            formatted_algorithm = algorithm.replace('_', ' ')
            dic['algorithm'] = formatted_algorithm
            dic['time'] = data[str(n)]['duration']
            dict_list.append(dic)
    df = pd.DataFrame(dict_list)

    with sns.axes_style("whitegrid"):  # Temporarily set style
        g = sns.catplot(
            data=df, kind="bar",
            x="J", y="time", hue="algorithm",
            palette="bright", alpha=.8,
            height=6, aspect=1.2
        )
        g.despine(left=True)

        # Set the overall title using suptitle
        g.fig.suptitle(
            f"Execution time across different algorithms (Chen's data)",
            fontsize=20,
            y=1.1)

        g.legend.set_bbox_to_anchor((1.15, 0.5))  # Adjust position as needed
        # Set the axis labels
        g.set_axis_labels("", "Execution time (in minutes)", fontsize=18)
        g.legend.set_title("Algorithm", prop={'size': 16})
        # Adjust the font size for the legend text
        for text in g.legend.texts:
            text.set_fontsize(14)  # Adjust as needed

        # Increase font size for x-axis and y-axis tick labels
        g.set_xticklabels(fontsize=14)
        g.set_yticklabels(fontsize=14)
        
        g.savefig(f'img/{"chen_data"}/time.png', dpi=200)
