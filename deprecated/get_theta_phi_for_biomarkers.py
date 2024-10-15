def compute_theta_phi_for_biomarker_(
    biomarker_df: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """
    Calculate the mean and standard deviation of theta and phi parameters 
    for a specified biomarker using clustering techniques.

    Use KMeans first as default and then reassign clusters using Agglomerative 
    Clustering if healthy participants are not in a single cluster after KMeans

    Args:
    biomarker_df (pd.DataFrame): DataFrame containing participant data for a specific biomarker 
        with columns 'participant', 'biomarker', 'measurement', and 'diseased'.

    Returns:
    tuple: A tuple containing the mean and standard deviation of theta and phi:
        - theta_mean (float): Mean of the measurements in the theta cluster.
        - theta_std (float): Standard deviation of the measurements in the theta cluster.
        - phi_mean (float): Mean of the measurements in the phi cluster.
        - phi_std (float): Standard deviation of the measurements in the phi cluster.
    """
    # Ensure n_init is set properly (use "auto" or an integer)
    n_init_value = int(10)
    clustering_setup = KMeans(n_clusters=2, n_init=n_init_value)

    # you need to make sure each measurment is a np.array before putting it into "fit"
    measurements = np.array(biomarker_df['measurement']).reshape(-1, 1)

    # Fit clustering method
    clustering_result = clustering_setup.fit(measurements)
    # in our case, since n_cluster = 2, the labels_ will be a numpy array containing 0s and 1s
    predictions = clustering_result.labels_

    # dataframe for non-diseased participants
    healthy_df = biomarker_df[biomarker_df['diseased'] == False]
    diseased_df = biomarker_df[biomarker_df['diseased'] == True]
    # healthy_measurements = np.array(healthy_df['measurement']).reshape(-1, 1)
    # which cluster are healthy participants in
    healthy_predictions = predictions[healthy_df.index]
    diseased_predictions = predictions[diseased_df.index]

    # the mode of the healthy predictions will be the phi cluster index
    phi_cluster_idx = mode(healthy_predictions, keepdims=False).mode
    theta_cluster_idx = 1 - phi_cluster_idx





    # if len(set(healthy_predictions)) <= int(1) or len(set(diseased_predictions)) <= int(1):
    #     clustering = AgglomerativeClustering(n_clusters=2).fit(
    #         measurements)
    #     updated_predictions = clustering.labels_
    # else:
    #     updated_predictions = predictions.copy()

    # if len(set(healthy_predictions)) > 1:
    #     # Reassign clusters using Agglomerative Clustering
    #     clustering = AgglomerativeClustering(
    #         n_clusters=2).fit(healthy_measurements)

    #     # Find the dominant cluster for healthy participants
    #     phi_cluster_idx = mode(clustering.labels_, keepdims=False).mode

    #     # Update predictions to ensure all healthy participants are in the dominant cluster
    #     updated_predictions = predictions.copy()
    #     for i in healthy_df.index:
    #         updated_predictions[i] = phi_cluster_idx
    # else:
    #     updated_predictions = predictions


    # two empty clusters to strore measurements
    clusters = [[] for _ in range(2)]
    # Store measurements into their cluster
    for i, prediction in enumerate(updated_predictions):
        clusters[prediction].append(measurements[i][0])

    # Calculate means and standard deviations
    theta_mean, theta_std = np.mean(
        clusters[theta_cluster_idx]), np.std(clusters[theta_cluster_idx])
    phi_mean, phi_std = np.mean(clusters[phi_cluster_idx]), np.std(
        clusters[phi_cluster_idx])

    # check whether the prior_theta_phi contain 0s or nan
    if theta_std == 0 or math.isnan(theta_std):
        print(f"In prior_theta_phi, theta_std is {theta_std}")
    if phi_std == 0 or math.isnan(phi_std):
        print(f"In prior_theta_phi, phi_std is {phi_std}")

    # check whether the prior_theta_phi contain 0s or nan
    if theta_mean == 0 or math.isnan(theta_mean):
        print(f"In prior_theta_phi, theta_mean is {theta_mean}")
    if phi_mean == 0 or math.isnan(phi_mean):
        print(f"In prior_theta_phi, phi_mean is {phi_mean}")

    return theta_mean, theta_std, phi_mean, phi_std


def compute_theta_phi_for_biomarker(biomarker_df, max_attempt = 50):
    x = 0
    n_clusters = 2
    measurements = np.array(biomarker_df['measurement']).reshape(-1, 1)
    healthy_df = biomarker_df[biomarker_df['diseased'] == False]
    predictions = KMeans(n_clusters=n_clusters, n_init=10, random_state=x).fit(measurements).labels_
    cluster_counts = np.bincount(predictions) # array([25, 25])
    # we want to make sure two clusters exist and each cluster has more than 1 element
    if not all(c > 1 for c in cluster_counts) or len(cluster_counts) != n_clusters:
        x += 1
        while x < max_attempt:
            predictions = KMeans(n_clusters=2, n_init=10, random_state=x).fit(measurements).labels_
            cluster_counts = np.bincount(predictions) # array([25, 25])
            if all(c > 1 for c in cluster_counts) and len(cluster_counts) == n_clusters:
                break 
            x += 1
    
    healthy_predictions = predictions[healthy_df.index]

    # we want to make sure as much as possible that all healthy participants belong to one group
    # however, we cannot gaurantee this
    if len(set(healthy_predictions)) > 1:
        temp_predictions = AgglomerativeClustering(
            n_clusters=n_clusters).fit(measurements).labels_
        temp_cluster_counts = np.bincount(temp_predictions) # array([25, 25])
        temp_healthy_predictions = temp_predictions[healthy_df.index]
        if all(
            c > 1 for c in temp_cluster_counts) and len(
                temp_cluster_counts) == n_clusters and len(
                    set(temp_healthy_predictions)) == 1:
            predictions = temp_predictions
    
    healthy_predictions = predictions[healthy_df.index]

    phi_cluster_idx = mode(healthy_predictions, keepdims=False).mode
    theta_cluster_idx = 1 - phi_cluster_idx

    # two empty clusters to strore measurements
    clusters = [[] for _ in range(2)]
    # Store measurements into their cluster
    for i, prediction in enumerate(predictions):
        clusters[prediction].append(measurements[i][0])

    # Calculate means and standard deviations
    theta_mean, theta_std = np.mean(
        clusters[theta_cluster_idx]), np.std(clusters[theta_cluster_idx])
    phi_mean, phi_std = np.mean(clusters[phi_cluster_idx]), np.std(
        clusters[phi_cluster_idx])
    
    # check whether the prior_theta_phi contain 0s or nan
    if math.isnan(theta_std) or theta_std == 0:
        raise ValueError(f"Invalid theta_std: {theta_std}")
    if math.isnan(phi_std) or phi_std == 0:
        raise ValueError(f"Invalid phi_std: {phi_std}")
    if theta_mean == 0 or math.isnan(theta_mean):
        raise ValueError(f"Invalid theta_mean: {theta_mean}")
    if phi_mean == 0 or math.isnan(phi_mean):
        raise ValueError(f"Invalid phi_mean: {phi_mean}")

    return theta_mean, theta_std, phi_mean, phi_std