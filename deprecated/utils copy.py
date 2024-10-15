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
from typing import List, Optional

def generate_data_from_ebm(
    n_participants: int,
    S_ordering: List[str],
    real_theta_phi_file: str,
    healthy_ratio: float,
    output_dir: str,
    seed: Optional[int] = None
):
    """
    Simulate an Event-Based Model (EBM) for disease progression.

    Args:
    n_participants (int): Number of participants.
    S_ordering (List[str]): Biomarker names ordered according to the order 
        in which each of them get affected by the disease.
    real_theta_phi_file (str): Directory of a JSON file which contains 
        theta and phi values for all biomarkers.
        See real_theta_phi.json for example format.
    output_dir (str): Directory where output files will be saved.
    healthy_ratio (float): Proportion of healthy participants out of n_participants.
    seed (Optional[int]): Seed for the random number generator for reproducibility.

    Returns:
    pd.DataFrame: A DataFrame with columns 'participant', "biomarker", 'measurement', 
        'diseased'.
    """
    # Parameter validation
    assert n_participants > 0, "Number of participants must be greater than 0."
    assert 0 <= healthy_ratio <= 1, "Healthy ratio must be between 0 and 1."

    # Set the seed for numpy's random number generator
    rng = np.random.default_rng(seed)

    # Load theta and phi values from the JSON file
    try:
        with open(real_theta_phi_file) as f:
            real_theta_phi = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {real_theta_phi} not fount")
    except json.JSONDecodeError:
        raise ValueError(f"File {real_theta_phi_file} is not a valid JSON file.")

    n_biomarkers = len(S_ordering)
    n_stages = n_biomarkers + 1

    n_healthy = int(n_participants * healthy_ratio)
    n_diseased = int(n_participants - n_healthy)

    # Generate disease stages
    kjs = np.concatenate((np.zeros(n_healthy, dtype=int), rng.integers(1, n_stages, n_diseased)))
    # shuffle so that it's not 0s first and then disease stages bur all random
    rng.shuffle(kjs)

    # Initiate biomarker measurement matrix (J participants x N biomarkers) with None
    X = np.full((n_participants, n_biomarkers), None, dtype=object)

    # Create distributions for each biomarker
    theta_dist = {biomarker: stats.norm(
        real_theta_phi[biomarker]['theta_mean'],
        real_theta_phi[biomarker]['theta_std']
    ) for biomarker in S_ordering}

    phi_dist = {biomarker: stats.norm(
        real_theta_phi[biomarker]['phi_mean'],
        real_theta_phi[biomarker]['phi_std']
    ) for biomarker in S_ordering}

    # Populate the matrix with biomarker measurements
    for j in range(n_participants):
        for n, biomarker in enumerate(S_ordering):
            k_j = kjs[j]
            S_n = n + 1

            # Assign biomarker values based on the participant's disease stage
            # affected, or not_affected, is regarding the biomarker, not the participant
            if k_j >= 1:
                if k_j >= S_n:
                    # rvs() is affected by np.random()
                    X[j, n] = (
                        j, biomarker, theta_dist[biomarker].rvs(random_state=rng), k_j, S_n, 'affected')
                else:
                    X[j, n] = (j, biomarker, phi_dist[biomarker].rvs(random_state=rng),
                               k_j, S_n, 'not_affected')
            # if the participant is healthy
            else:
                X[j, n] = (j, biomarker, phi_dist[biomarker].rvs(random_state=rng),
                           k_j, S_n, 'not_affected')

    df = pd.DataFrame(X, columns=S_ordering)
    # make this dataframe wide to long
    df_long = df.melt(var_name="Biomarker", value_name="Value")
    data = df_long['Value'].apply(pd.Series)
    data.columns = ['participant', "biomarker", 'measurement', 'k_j', 'S_n', 'affected_or_not']

    biomarker_name_change_dic = dict(zip(S_ordering, range(1, n_biomarkers + 1)))
    data['diseased'] = data.apply(lambda row: row.k_j > 0, axis=1)
    data.drop(['k_j', 'S_n', 'affected_or_not'], axis=1, inplace=True)
    data['biomarker'] = data.apply(
        lambda row: f"{row.biomarker} ({biomarker_name_change_dic[row.biomarker]})", axis=1)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combination_str = f"{int(healthy_ratio*n_participants)}|{n_participants}"
    data.to_csv(f'{output_dir}/{combination_str}.csv', index=False)
    print("Data generation done! Output saved to:", combination_str)
    return data

def get_theta_phi_for_single_biomarker(data, biomarker, clustering_setup):
    """To get theta and phi parametesr for a single biomarker 
    Input:
        - data: data we have right now, without access to S_n and kj
        - biomarker: a string of biomarker name
        - clustering_setup: kmeans_only, hierarchical_clustering, or both
    Output:
        mean and std of theta and phi
    """
    # two empty clusters to strore measurements
    clusters = [[] for _ in range(2)]

    # dataframe for this biomarker
    biomarker_df = data[data['biomarker'] == biomarker].reset_index(drop=True)
    # reshape to satisfy sklearn requirements
    measurements = np.array(biomarker_df['measurement']).reshape(-1, 1)

    # dataframe for non-diseased participants
    healthy_df = biomarker_df[biomarker_df['diseased'] == False]
    healthy_measurements = np.array(healthy_df['measurement']).reshape(-1, 1)

    # Fit clustering method
    clustering_result = clustering_setup.fit(measurements)

    if isinstance(clustering_setup, KMeans):
        predictions = clustering_result.predict(measurements)
    else:
        predictions = clustering_result.labels_

    # to store measurements into their cluster
    for i, prediction in enumerate(predictions):
        clusters[prediction].append(measurements[i][0])

    # which cluster are healthy participants in
    healthy_predictions = predictions[healthy_df.index]

    # the mode of the above predictions will be the phi cluster index
    phi_cluster_idx = mode(healthy_predictions, keepdims=False).mode
    theta_cluster_idx = 1 - phi_cluster_idx
    theta_mean, theta_std = np.mean(
        clusters[theta_cluster_idx]), np.std(clusters[theta_cluster_idx])
    phi_mean, phi_std = np.mean(clusters[phi_cluster_idx]), np.std(
        clusters[phi_cluster_idx])
    return theta_mean, theta_std, phi_mean, phi_std


def get_theta_phi_for_single_biomarker_using_kmeans_and_hierarchical_clustering(
        data, biomarker):
    """
    To get theta and phi parameters for a single biomarker using the K-means algorithm.
    Input:
        - data: DataFrame containing the data.
        - biomarker: A string representing the biomarker name.
        - kmeans_setup: An instance of KMeans from scikit-learn.
    Output:
        - Mean and standard deviation of theta and phi.
    """
    kmeans_setup = KMeans(n_clusters=2, random_state=0, n_init="auto")

    # two empty clusters to strore measurements
    clusters = [[] for _ in range(2)]

    # dataframe for this biomarker
    biomarker_df = data[data['biomarker'] == biomarker].reset_index(drop=True)
    # you need to make sure each measurment is a np.array before putting it into "fit"
    measurements = np.array(biomarker_df['measurement']).reshape(-1, 1)

    # dataframe for non-diseased participants
    healthy_df = biomarker_df[biomarker_df['diseased'] == False]
    healthy_measurements = np.array(healthy_df['measurement']).reshape(-1, 1)

    # Fit k-means
    kmeans = kmeans_setup.fit(measurements)
    predictions = kmeans.predict(measurements)

    # Verify that all healthy participants are in one cluster
    # which clusters are healthy participants in:
    healthy_predictions = kmeans.predict(healthy_measurements)
    # Identify healthy cluster index
    phi_cluster_idx = mode(healthy_predictions, keepdims=False).mode

    if len(set(healthy_predictions)) > 1:
        # Reassign clusters using Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=2).fit(healthy_measurements)

        # Find the dominant cluster for healthy participants
        phi_cluster_idx = mode(clustering.labels_, keepdims=False).mode

        # Update predictions to ensure all healthy participants are in the dominant cluster
        updated_predictions = predictions.copy()
        for i in healthy_df.index:
            updated_predictions[i] = phi_cluster_idx
    else:
        updated_predictions = predictions

    # Identify diseased cluster index
    theta_cluster_idx = 1 - phi_cluster_idx

    # Store measurements into their cluster
    for i, prediction in enumerate(updated_predictions):
        clusters[prediction].append(measurements[i][0])

    # Calculate means and standard deviations
    theta_mean, theta_std = np.mean(
        clusters[theta_cluster_idx]), np.std(clusters[theta_cluster_idx])
    phi_mean, phi_std = np.mean(clusters[phi_cluster_idx]), np.std(
        clusters[phi_cluster_idx])
    return theta_mean, theta_std, phi_mean, phi_std


def get_theta_phi_estimates(
    data_we_have,
    biomarkers,
    n_clusters,
    method="kmeans_and_hierarchical_clustering"
):
    """
    Get the DataFrame of theta and phi using the K-means and/or hierarchical clustering 
    algorithm for all biomarkers.
    Input:
        - data_we_have: DataFrame containing the data.
        - biomarkers: List of biomarkers in string.
        - n_clusters: Number of clusters (should be 2).
    Output:
        - a dictionary where key is biomarker and 
        value is the means and standard deviations for theta and phi for that biomarker.
    """
    kmeans_setup = KMeans(n_clusters, random_state=0, n_init="auto")
    hierarchical_clustering_setup = AgglomerativeClustering(n_clusters=2)
    # empty list of dictionaries to store the estimates
    hashmap_of_means_stds_estimate_dicts = {}
    for idx, biomarker in enumerate(biomarkers):
        dic = {'biomarker': biomarker}
        if method == "kmeans_only":
            theta_mean, theta_std, phi_mean, phi_std = get_theta_phi_for_single_biomarker(
                data_we_have, biomarker, kmeans_setup)
        elif method == "hierarchical_clustering_only":
            theta_mean, theta_std, phi_mean, phi_std = get_theta_phi_for_single_biomarker(
                data_we_have, biomarker, hierarchical_clustering_setup)
        elif method == "kmeans_and_hierarchical_clustering":
            theta_mean, theta_std, phi_mean, phi_std = get_theta_phi_for_single_biomarker_using_kmeans_and_hierarchical_clustering(
                data_we_have, biomarker)
        dic['theta_mean'] = theta_mean
        dic['theta_std'] = theta_std
        dic['phi_mean'] = phi_mean
        dic['phi_std'] = phi_std
        hashmap_of_means_stds_estimate_dicts[biomarker] = dic
    return hashmap_of_means_stds_estimate_dicts


def fill_up_pdata(pdata, k_j):
    '''Fill up a single participant's data using k_j; basically add two columns: 
    k_j and affected
    Note that this function assumes that pdata already has the S_n column

    Input:
    - pdata: a dataframe of ten biomarker values for a specific participant 
    - k_j: a scalar
    '''
    data = pdata.copy()
    data['k_j'] = k_j
    data['affected'] = data.apply(lambda row: row.k_j >= row.S_n, axis=1)
    return data


def compute_single_measurement_likelihood(theta_phi, biomarker, affected, measurement):
    '''Computes the likelihood of the measurement value of a single biomarker
    We know the normal distribution defined by either theta or phi
    and we know the measurement. This will give us the probability
    of this given measurement value. 

    input:
    - theta_phi: the dictionary containing theta and phi values for each biomarker
    - biomarker: an integer between 0 and 9 
    - affected: boolean 
    - measurement: the observed value for a biomarker in a specific participant

    output: a scalar
    '''
    # print(biomarker)
    # print(theta_phi)
    biomarker_dict = theta_phi[biomarker]
    mu = biomarker_dict['theta_mean'] if affected else biomarker_dict['phi_mean']
    std = biomarker_dict['theta_std'] if affected else biomarker_dict['phi_std']
    var = std**2
    likelihood = np.exp(-(measurement - mu)**2/(2*var))/np.sqrt(2*np.pi*var)
    return likelihood


def compute_likelihood(pdata, k_j, theta_phi):
    '''This implementes the formula of https://ebm-book2.vercel.app/distributions.html#known-k-j
    This function computes the likelihood of seeing this sequence of biomarker values 
    for a specific participant, assuming that this participant is at stage k_j
    '''
    data = fill_up_pdata(pdata, k_j)
    likelihood = 1
    for i, row in data.iterrows():
        biomarker = row['biomarker']
        measurement = row['measurement']
        affected = row['affected']
        likelihood *= compute_single_measurement_likelihood(
            theta_phi, biomarker, affected, measurement)
    return likelihood


def calculate_soft_kmeans_for_biomarker(
        data,
        biomarker,
        order_dict,
        n_participants,
        non_diseased_participants,
        hashmap_of_normalized_stage_likelihood_dicts,
        diseased_stages,
        seed=None
):
    """
    Process soft K-means clustering for a single biomarker.

    Parameters:
        data (pd.DataFrame): The data containing measurements.
        biomarker (str): The biomarker to process.
        order_dict (dict): Dictionary mapping biomarkers to their order.
        n_participants (int): Number of participants in the study.
        non_diseased_participants (list): List of non-diseased participants.
        hashmap_of_normalized_stage_likelihood_dicts (dict): Hash map of 
            dictionaries containing stage likelihoods for each participant.
        diseased_stages (list): List of diseased stages.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: Means and standard deviations for affected and non-affected clusters.
    """
    if seed is not None:
        np.random.seed(seed)  # Set random seed for reproducibility

    # DataFrame for this biomarker
    biomarker_df = data[data['biomarker'] == biomarker].reset_index(drop=True)
    # Extract measurements
    measurements = np.array(biomarker_df['measurement'])

    this_biomarker_order = order_dict[biomarker]

    affected_cluster = []
    non_affected_cluster = []

    for p in range(n_participants):
        if p in non_diseased_participants:
            non_affected_cluster.append(measurements[p])
        else:
            if this_biomarker_order == 1:
                affected_cluster.append(measurements[p])
            else:
                normalized_stage_likelihood_dict = hashmap_of_normalized_stage_likelihood_dicts[
                    p]
                # Calculate probabilities for affected and non-affected states
                affected_prob = sum(
                    normalized_stage_likelihood_dict[s] for s in diseased_stages if s >= this_biomarker_order
                )
                non_affected_prob = sum(
                    normalized_stage_likelihood_dict[s] for s in diseased_stages if s < this_biomarker_order
                )
                if affected_prob > non_affected_prob:
                    affected_cluster.append(measurements[p])
                elif affected_prob < non_affected_prob:
                    non_affected_cluster.append(measurements[p])
                else:
                    # Assign to either cluster randomly if probabilities are equal
                    if np.random.rand() > 0.5:
                        affected_cluster.append(measurements[p])
                    else:
                        non_affected_cluster.append(measurements[p])

    # Compute means and standard deviations
    theta_mean = np.mean(affected_cluster) if affected_cluster else np.nan
    theta_std = np.std(affected_cluster) if affected_cluster else np.nan
    phi_mean = np.mean(
        non_affected_cluster) if non_affected_cluster else np.nan
    phi_std = np.std(non_affected_cluster) if non_affected_cluster else np.nan

    return theta_mean, theta_std, phi_mean, phi_std

def soft_kmeans_theta_phi_estimates(
        iteration,
        prior_theta_phi_estimates,
        data_we_have,
        biomarkers,
        order_dict,
        n_participants,
        non_diseased_participants,
        hashmap_of_normalized_stage_likelihood_dicts,
        diseased_stages,
        seed=None):
    """
    Get the DataFrame of theta and phi using the soft K-means algorithm for all biomarkers.

    Parameters:
        data_we_have (pd.DataFrame): DataFrame containing the data.
        biomarkers (list): List of biomarkers in string.
        order_dict (dict): Dictionary mapping biomarkers to their order.
        n_participants (int): Number of participants in the study.
        non_diseased_participants (list): List of non-diseased participants.
        hashmap_of_normalized_stage_likelihood_dicts (dict): Hash map of dictionaries containing stage likelihoods for each participant.
        diseased_stages (list): List of diseased stages.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        a dictionary containing the means and standard deviations for theta and phi for each biomarker.
    """
    # List to store the estimates
    hashmap_of_means_stds_estimate_dicts = {}
    for biomarker in biomarkers:
        dic = {'biomarker': biomarker}
        prior_theta_phi_estimates_biomarker = prior_theta_phi_estimates[biomarker]
        theta_mean, theta_std, phi_mean, phi_std = calculate_soft_kmeans_for_biomarker(
            data_we_have,
            biomarker,
            order_dict,
            n_participants,
            non_diseased_participants,
            hashmap_of_normalized_stage_likelihood_dicts,
            diseased_stages,
            seed
        )
        if theta_std == 0 or math.isnan(theta_std):
            theta_mean = prior_theta_phi_estimates_biomarker['theta_mean']
            theta_std = prior_theta_phi_estimates_biomarker['theta_std']
        if phi_std == 0 or math.isnan(phi_std):
            phi_mean = prior_theta_phi_estimates_biomarker['phi_mean']
            phi_std = prior_theta_phi_estimates_biomarker['phi_std']
        dic['theta_mean'] = theta_mean
        dic['theta_std'] = theta_std
        dic['phi_mean'] = phi_mean
        dic['phi_std'] = phi_std
        hashmap_of_means_stds_estimate_dicts[biomarker] = dic
    return hashmap_of_means_stds_estimate_dicts


"""
If soft kmeans, no matter uniform prior on kjs or not, I always need to update hashmap of dicts
    This is because, even if when we do not have uniform prior, we don't need normalized_stage_likelihood_dict
    to calculate the weighted average, we still need it to calculate soft kmeans
If kmeans only, if with uniform prior, we don't need normalized_stage_likelihood_dict to calculate 
    weighted average;
    but we do need to calculate normalized_stage_likelihood_dict when without uniform prior 
"""


def calculate_all_participant_ln_likelihood_and_update_hashmap(
        iteration,
        data_we_have,
        current_order_dict,
        n_participants,
        non_diseased_participant_ids,
        theta_phi_estimates,
        diseased_stages,
):
    data = data_we_have.copy()
    data['S_n'] = data.apply(
        lambda row: current_order_dict[row['biomarker']], axis=1)
    all_participant_ln_likelihood = 0
    # key is participant id
    # value is normalized_stage_likelihood_dict
    hashmap_of_normalized_stage_likelihood_dicts = {}
    for p in range(n_participants):
        pdata = data[data.participant == p].reset_index(drop=True)
        if p in non_diseased_participant_ids:
            this_participant_likelihood = compute_likelihood(
                pdata, k_j=0, theta_phi=theta_phi_estimates)
            this_participant_ln_likelihood = np.log(
                this_participant_likelihood + 1e-10)
        else:
            normalized_stage_likelihood_dict = None
            # initiaze stage_likelihood
            stage_likelihood_dict = {}
            for k_j in diseased_stages:
                kj_likelihood = compute_likelihood(
                    pdata, k_j, theta_phi_estimates)
                # update each stage likelihood for this participant
                stage_likelihood_dict[k_j] = kj_likelihood
            # Add a small epsilon to avoid division by zero
            likelihood_sum = sum(stage_likelihood_dict.values())
            epsilon = 1e-10
            if likelihood_sum == 0:
                print("Invalid likelihood_sum: zero encountered.")
                likelihood_sum = epsilon  # Handle the case accordingly
            normalized_stage_likelihood = [
                l/likelihood_sum for l in stage_likelihood_dict.values()]
            normalized_stage_likelihood_dict = dict(
                zip(diseased_stages, normalized_stage_likelihood))
            hashmap_of_normalized_stage_likelihood_dicts[p] = normalized_stage_likelihood_dict

            # calculate weighted average
            this_participant_likelihood = weighted_average_likelihood(
                likelihood_sum)
            this_participant_ln_likelihood = np.log(
                this_participant_likelihood + 1e-10)
        all_participant_ln_likelihood += this_participant_ln_likelihood
    return all_participant_ln_likelihood, hashmap_of_normalized_stage_likelihood_dicts


def weighted_average_likelihood(
    likelihood_sum,
):
    """using weighted average likelihood
    https://ebm-book2.vercel.app/distributions.html#unknown-k-j
    Note that we have uniform prior on kj
    """
    return np.mean(likelihood_sum)


def metropolis_hastings_soft_kmeans(
    data_we_have,
    iterations,
    n_shuffle,
    log_folder_name,
):
    '''Implement the metropolis-hastings algorithm
    Inputs: 
        - data: data_we_have
        - iterations: number of iterations
        - log_folder_name: the folder where log files locate

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
    prior_theta_phi_estimates = get_theta_phi_estimates(
        data_we_have,
        biomarkers,
        n_clusters=2,
        method="kmeans_only"
    )
    theta_phi_estimates = prior_theta_phi_estimates.copy()

    # initialize empty lists
    all_order_dicts = []
    all_current_accepted_likelihoods = []
    acceptance_count = 0
    all_current_acceptance_ratios = []
    all_current_accepted_order_dicts = []
    terminal_output_strings = []
    hashmaps_of_theta_phi_estimates = {}
    hashmap_of_estimated_theta_phi_dicts = {}

    current_accepted_order = np.random.permutation(np.arange(1, n_stages))
    current_accepted_order_dict = dict(zip(biomarkers, current_accepted_order))
    current_accepted_likelihood = -np.inf

    for _ in range(iterations):
        new_order = current_accepted_order.copy()
        # random.shuffle(new_order)
        shuffle_order(new_order, n_shuffle)
        current_order_dict = dict(zip(biomarkers, new_order))
        all_participant_ln_likelihood, \
            hashmap_of_normalized_stage_likelihood_dicts = calculate_all_participant_ln_likelihood_and_update_hashmap(
                _,
                data_we_have,
                current_order_dict,
                n_participants,
                non_diseased_participant_ids,
                theta_phi_estimates,
                diseased_stages,
            )

        # Now, update theta_phi_estimates using soft kmeans
        # based on the updated hashmap of normalized stage likelihood dicts
        theta_phi_estimates = soft_kmeans_theta_phi_estimates(
            _,
            prior_theta_phi_estimates,
            data_we_have,
            biomarkers,
            current_order_dict,
            n_participants,
            non_diseased_participant_ids,
            hashmap_of_normalized_stage_likelihood_dicts,
            diseased_stages,
            seed=None,
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
        hashmaps_of_theta_phi_estimates[_] = theta_phi_estimates
        # update theta_phi_dic
        hashmap_of_estimated_theta_phi_dicts[_] = theta_phi_estimates

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
    with open(f'{log_folder_name}/hashmap_of_estimated_theta_phi_dicts.json', 'w') as fp:
        json.dump(hashmap_of_estimated_theta_phi_dicts, fp)
    print("done!")
    return (
        current_accepted_order_dict,
        all_order_dicts,
        all_current_accepted_order_dicts,
        all_current_accepted_likelihoods,
        all_current_acceptance_ratios,
        final_acceptance_ratio,
        hashmap_of_estimated_theta_phi_dicts,
    )

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
        n_clusters=2,
        method="kmeans_only"
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

# def output_likelihood_comparison(
#         most_likely_order_dic,
#         data_we_have,
#         n_participants,
#         non_diseased_participant_ids,
#         theta_phi_kmeans,
#         diseased_stages,
#         log_folder_name,
#         doc_strings,
#         ):
#     """This is to output a text file comparing the likelihood of the most likely ordering
#     and the real ordering
#     """
#     real_order_dic = dict(zip(most_likely_order_dic.keys(), range(1, len(most_likely_order_dic) + 1)))
#     output_filename = f"{log_folder_name}/compare_most_likely_and_true_ordering.txt"
#     with open(output_filename, 'w') as file:
#         if most_likely_order_dic == real_order_dic:
#             file.write("The most likely ordering is the true ordering")
#         else:
#             file.write("The most likely ordering is different from the true ordering. \n")
#             most_likely_ln_likelihood, hashmap = calculate_all_participant_ln_likelihood_and_update_hashmap(
#                 "iteration",
#                 data_we_have,
#                 most_likely_order_dic,
#                 n_participants,
#                 non_diseased_participant_ids,
#                 theta_phi_kmeans,
#                 diseased_stages,
#             )
#             real_order_ln_likelihood, hashmap = calculate_all_participant_ln_likelihood_and_update_hashmap(
#                 "iteration",
#                 data_we_have,
#                 real_order_dic,
#                 n_participants,
#                 non_diseased_participant_ids,
#                 theta_phi_kmeans,
#                 diseased_stages,
#             )
#             file.write(f"Likelihood of the most likely ordering ({most_likely_order_dic.values()}): {most_likely_ln_likelihood}. \n")
#             file.write(f"Likelihood of the true ordering ({real_order}): {real_order_ln_likelihood}.")

#         if doc_strings:
#             for string in doc_strings:
#                 file.write(string)


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

    Output: 
    - a hashmap of dictionaries. Key is biomarker name and value is a dictionary. 
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


def shuffle_order(arr, n_shuffle):
    # randomly choose three indices
    indices = random.sample(range(len(arr)), n_shuffle)
    # obtain the elements represented by these three random indices and shuffle these elements
    selected_elements = [arr[i] for i in indices]
    random.shuffle(selected_elements)
    # shuffle the original arr
    for i, index in enumerate(indices):
        arr[index] = selected_elements[i]


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
            n_clusters=2,
            method="kmeans_only"
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


def save_output_strings(
        log_folder_name,
        terminal_output_strings
):
    # Check if the directory exists
    if not os.path.exists(log_folder_name):
        # Create the directory if it does not exist
        os.makedirs(log_folder_name)
    terminal_output_filename = f"{log_folder_name}/terminal_output.txt"
    with open(terminal_output_filename, 'w') as file:
        for result in terminal_output_strings:
            file.write(result + '\n')


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


def save_all_dicts(all_dicts, log_folder_name, file_name):
    """Save all_dicts into a CSV file within a specified directory.

    If the directory does not exist, it will be created.
    """
    # Check if the directory exists
    if not os.path.exists(log_folder_name):
        # Create the directory if it does not exist
        os.makedirs(log_folder_name)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_dicts)

    # Add an 'iteration' column
    df['iteration'] = np.arange(start=1, stop=len(df) + 1, step=1)

    # Set 'iteration' as the index
    df.set_index("iteration", inplace=True)

    # Save the DataFrame to a CSV file
    df.to_csv(f"{log_folder_name}/{file_name}.csv", index=True)


def save_all_current_accepted(var, var_name, log_folder_name):
    """save all_current_order_dicts, all_current_ikelihoods, 
    and all_current_acceptance_ratios
    """
    # Check if the directory exists
    if not os.path.exists(log_folder_name):
        # Create the directory if it does not exist
        os.makedirs(log_folder_name)
    x = np.arange(start=1, stop=len(var) + 1, step=1)
    df = pd.DataFrame({"iteration": x, var_name: var})
    df = df.set_index('iteration')
    df.to_csv(f"{log_folder_name}/{var_name}.csv", index=True)


def save_all_current_participant_stages(var, var_name, log_folder_name):
    # Check if the directory exists
    if not os.path.exists(log_folder_name):
        # Create the directory if it does not exist
        os.makedirs(log_folder_name)
    df = pd.DataFrame(var)
    df.index.name = 'iteration'
    df.index = df.index + 1
    df.to_csv(f"{log_folder_name}/{var_name}.csv", index=False)


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
