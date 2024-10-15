#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p conjugate_priors
mkdir -p conjugate_priors/logs
mkdir -p conjugate_priors/img
# Create temp results directory if it doesn't exist
mkdir -p conjugate_priors/temp_json_results

echo "eval_cp.sh started at $(date)"
echo "Running in directory: $(pwd)" 
echo "Running with arguments: $@"

# Unpack the environment if not done already
if [ ! -d "./bayes" ]; then
    echo "Unpacking virtual environment..."
    tar -xzf bayes.tar.gz || { echo "Failed to extract environment"; exit 1; }
fi

# Activate the environment
source ./bayes/bin/activate || { echo "Failed to activate environment"; exit 1; }

# Verify if essential packages are installed after activating the environment
if ! python -c "import pandas; import matplotlib; import numpy; import seaborn; import scipy; import sklear; from copkmeans.cop_kmeans import cop_kmeans
n" &> /dev/null; then
    echo "Some packages not found, attempting to install dependencies with force reinstall and no cache..."
    pip install --force-reinstall --no-cache-dir -r requirements.txt || { echo "Failed to install requirements"; exit 1; }
fi

tar -xzf data.tar.gz 

echo "Files present:"
ls -l

# pip install -r requirements.txt

# Run the Python script with arguments and log errors
python ./eval_cp.py "$@"

