"""Hardware profiling for Graph Neural Networks on GPU (Table 2).
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_graph_data
from models.gfc_net import build_model
from utils.metrics import measure_gpu_efficiency
from utils.logger import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description="Profile GNN inference latency on GPU")
    parser.add_argument("--cpu_debug", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.cpu_debug:
        print("\n[CRITICAL] GPU required for efficiency measurement.\n", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running on: {device}")

    data = get_graph_data("cora")
    features = data["features"].to(device)
    norm_adj = data["norm_adj"].to(device)

    models = ["gcn", "graphsage", "gat", "gfc_net"]
    results = []

    for name in models:
        model = build_model(name, in_features=data["num_features"], num_classes=data["num_classes"]).to(device)
        if device.type == "cuda":
            eff = measure_gpu_efficiency(model, features, norm_adj, device)
            lat = eff["latency_ms"]
            params_k = eff["params_k"]
        else:
            params_k = round(sum(p.numel() for p in model.parameters()) / 1e3, 1)
            lat = "N/A (CPU)"

        results.append({
            "Architecture": name.upper(),
            "Params (K)": params_k,
            "Latency (ms)": lat,
        })

    ExperimentLogger.print_comparison_table(results, title="GNN Complexity and GPU Inference Latency (Table 2)")


if __name__ == "__main__":
    main()
