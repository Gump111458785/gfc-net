#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " 1. Checking CUDA Hardware"
echo "=========================================================="
python3 -c "import torch; assert torch.cuda.is_available(), 'CRITICAL: CUDA GPU is required!'"
echo "CUDA Hardware detected: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

echo ""
echo "=========================================================="
echo " 2. Downloading Graph Datasets"
echo "=========================================================="
bash scripts/download_data.sh "./data/cache"

echo ""
echo "=========================================================="
echo " 3. Benchmarking GPU Efficiency (Table 2)"
echo "=========================================================="
python3 scripts/benchmark_efficiency.py

echo ""
echo "=========================================================="
echo " 4. Running Node Classification Experiments (Table 1)"
echo "=========================================================="
DATASETS=("cora" "citeseer" "pubmed")
MODELS=("gcn" "graphsage" "gat" "gfc_net")

for d in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        echo "--> Training & Evaluating: Model=$m on Dataset=$d"
        python3 scripts/train.py \
            --model "$m" \
            --dataset "$d" \
            --epochs 100 \
            --output_dir "./checkpoints"
    done
done

echo ""
echo "=========================================================="
echo " 5. Running Reproduction Discrepancy Verification"
echo "=========================================================="
python3 scripts/verify_reproduction.py

echo "=========================================================="
echo " Graph reproduction suite execution finished!"
echo "=========================================================="
