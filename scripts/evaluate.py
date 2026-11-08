"""Graph evaluation script for testing node classification performance.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_graph_data, SUPPORTED_DATASETS
from models.gfc_net import build_model
from utils.metrics import evaluate_node_classification
from utils.logger import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description="Evaluate Graph model on test nodes")
    parser.add_argument("--model", type=str, default="gfc_net", choices=["gfc_net", "gcn", "graphsage", "gat"])
    parser.add_argument("--dataset", type=str, default="cora", choices=SUPPORTED_DATASETS)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--cpu_debug", action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.cpu_debug:
        print("\n[CRITICAL] CUDA GPU required for evaluation.\n", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = get_graph_data(args.dataset, data_dir=args.data_dir)

    features = data["features"].to(device)
    norm_adj = data["norm_adj"].to(device)
    labels = data["labels"].to(device)
    test_mask = data["test_mask"].to(device)

    model = build_model(
        model_name=args.model,
        in_features=data["num_features"],
        num_classes=data["num_classes"],
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device)
    metrics = evaluate_node_classification(model, features, norm_adj, labels, test_mask)

    print("\n" + "=" * 55)
    print(f" ACTUAL EVALUATION SUMMARY: {args.model.upper()} on {args.dataset.upper()}")
    print("=" * 55)
    print(f"  • Accuracy: {metrics['accuracy']:.2f} %")
    print(f"  • Macro-F1: {metrics['macro_f1']:.2f} %")
    print(f"  • Micro-F1: {metrics['micro_f1']:.2f} %")
    print(f"  • ECE:      {metrics['ece']:.2f} %")
    print(f"  • Nodes:    {metrics['total_nodes']}")
    print("=" * 55 + "\n")

    logger = ExperimentLogger()
    run_name = f"eval_{args.model}_{args.dataset}"
    logger.log_metrics(run_name, metrics)


if __name__ == "__main__":
    main()
