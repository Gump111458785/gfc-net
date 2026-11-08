#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Acquiring Graph Benchmark Datasets (Cora, Citeseer, Pubmed)"
echo "=========================================================="

DATA_DIR=${1:-"./data/cache"}
mkdir -p "$DATA_DIR"

PYTHON_BIN="python3"
if ! command -v python3 &> /dev/null; then
    if command -v python &> /dev/null; then
        PYTHON_BIN="python"
    fi
fi

$PYTHON_BIN data/download_data.py --dataset all --data_dir "$DATA_DIR"

echo "=========================================================="
echo " Graph datasets successfully prepared!"
echo "=========================================================="
