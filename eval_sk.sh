#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p soft_kmeans
mkdir -p soft_kmeans/logs
mkdir -p soft_kmeans/img
# Create temp results directory if it doesn't exist
mkdir -p soft_kmeans/temp_json_results

echo "eval_sk.sh started at $(date)"
echo "Running in directory: $(pwd)" 
echo "Running with arguments: $@"

MAX_ATTEMPTS=50
attempt=1

while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "Attempting to unpack and activate virtual environment (Attempt $attempt/$MAX_ATTEMPTS)..."

    # Unpack the environment if not done already
    if [ ! -d "./ebm" ]; then
        echo "Unpacking virtual environment..."
        tar -xzf ebm.tar.gz || { echo "Failed to extract environment"; exit 1; }
    fi

    # Activate the environment
    source ./ebm/bin/activate

    # Check if the virtual environment is correctly activated
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "Virtual environment activated successfully!"
        break
    fi

    echo "Activation failed, retrying..."
    attempt=$((attempt + 1))
    sleep 2  # Wait 2 seconds before retrying
done

# Clean up previous installations and pip cache
echo "Cleaning up previous installations and pip cache..."
rm -rf ~/.local/lib/python3.9/site-packages/*
rm -rf ~/.cache/pip


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
python ./eval_sk.py "$@"

