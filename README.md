# Event-Based Models

This repository contains our implementation of Event-Based Models for disease progression. For more details about this technique, refer to [our online book](https://ebmbook.vercel.app/).

Below is a breakdown of the repository structure.

This repository is tailored toward use on [CHTC at UW-Madison](https://chtc.cs.wisc.edu/), a high-throughput computing platform. However, the algorithm can also be applied in general settings.

In the future, we plan to develop this implementation into a Python package. Stay tuned!

## Data

The `data` folder contains the 750 synthetic datasets used in our experiments, generated through `generate_data.py`.

## json_files

This folder includes `real_theta_phi.json`, which contains the actual theta and phi parameters for the ten biomarkers.

## Algorithm

This is not a folder.

- `new_utils.py`
- `hard_kmeans_alg.py`
- `soft_kmeans_alg.py`
- `conjugate_priors_alg.py`

The above files form the core of our algorithms.

Additionally, `save_res.py` is used for saving and plotting results.

## hard_kmeans, soft_kmeans, conjugate_priors

These folders contain results for each corresponding algorithm.

## requirements.txt

This file lists all packages used in the repository.
