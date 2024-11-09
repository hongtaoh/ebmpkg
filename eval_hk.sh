#!/bin/bash

mkdir -p hard_kmeans
mkdir -p hard_kmeans/logs
mkdir -p hard_kmeans/img
mkdir -p hard_kmeans/temp_json_results

echo "eval_sc.sh started at $(date)"
echo "Running in directory: $(pwd)" 
echo "Running with arguments: $@"

if [ -f "wheels.tar.gz" ]; then
    echo "Unpacking wheels..."
    tar -xzf wheels.tar.gz || { echo "Failed to extract wheels"; exit 1; }
fi

if [ -d "./wheels" ]; then
    echo "Installing dependencies from wheel files..."
    pip install --no-index --find-links=./wheels -r requirements.txt --target ./packages --no-cache-dir || { echo "Failed to install requirements from wheels"; exit 1; }
    export PYTHONPATH=$(pwd)/packages:$PYTHONPATH
else
    echo "Wheels directory not found, cannot install dependencies."; exit 1;
fi

tar -xzf data.tar.gz 

echo "Files present:"
ls -l

python ./eval_hk.py "$@"

echo "Script completed at $(date)"

