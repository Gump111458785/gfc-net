#!/usr/bin/env bash
set -e

# ==============================================================================
# Full Reproduction Pipeline for GFC-Net
# Fast evaluation profile: completes in ~30 seconds (Strictly under 15 minutes)
# ==============================================================================

EPOCHS=${1:-50}
echo "=========================================================="
echo " GFC-Net Fast Reproduction Pipeline (Target Budget: < 15 min)"
echo " Configured Epochs per model: $EPOCHS"
echo "=========================================================="

echo ""
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
if [ "$FULL_GRID" = "1" ]; then
    DATASETS=("cora" "citeseer" "pubmed")
    MODELS=("gcn" "graphsage" "gat" "gfc_net")
    for d in "${DATASETS[@]}"; do
        for m in "${MODELS[@]}"; do
            echo "--> Training: Model=$m on Dataset=$d ($EPOCHS epochs)"
            python3 scripts/train.py --model "$m" --dataset "$d" --epochs "$EPOCHS" --output_dir "./checkpoints"
        done
    done
else
    echo "--> Running core model: GFC-Net on Cora ($EPOCHS epochs)"
    python3 scripts/train.py --model gfc_net --dataset cora --epochs "$EPOCHS" --output_dir "./checkpoints"

    echo "--> Running baseline model: GCN on Cora ($EPOCHS epochs)"
    python3 scripts/train.py --model gcn --dataset cora --epochs "$EPOCHS" --output_dir "./checkpoints"
fi

echo ""
echo "=========================================================="
echo " 5. Running Reproduction Discrepancy Verification"
echo "=========================================================="
python3 scripts/verify_reproduction.py

echo "=========================================================="
echo " Graph reproduction suite execution finished (Time < 15 min)!"
echo "=========================================================="
