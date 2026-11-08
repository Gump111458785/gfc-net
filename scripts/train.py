"""Graph model training script for node classification.
Requires CUDA GPU by default.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_graph_data, SUPPORTED_DATASETS
from models.gfc_net import build_model
from utils.loss import GraphCalibrationLoss
from utils.metrics import evaluate_node_classification
from utils.logger import ExperimentLogger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Train Graph Neural Networks on GPU")
    parser.add_argument("--model", type=str, default="gfc_net", choices=["gfc_net", "gcn", "graphsage", "gat"])
    parser.add_argument("--dataset", type=str, default="cora", choices=SUPPORTED_DATASETS)
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--use_gfc", action="store_true", default=True)
    parser.add_argument("--no_gfc", dest="use_gfc", action="store_false")
    parser.add_argument("--use_gcl", action="store_true", default=True)
    parser.add_argument("--no_gcl", dest="use_gcl", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_debug", action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.cpu_debug:
        print("\n" + "!" * 70, file=sys.stderr)
        print(" [CRITICAL HARDWARE REQUIREMENT] ", file=sys.stderr)
        print(" PyTorch CUDA acceleration is mandatory for this experiment.", file=sys.stderr)
        print(" No CUDA GPU detected. Please run on an NVIDIA GPU environment.", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running on: {device}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Data] Loading graph dataset '{args.dataset}'...")
    data = get_graph_data(args.dataset, data_dir=args.data_dir)

    features = data["features"].to(device)
    norm_adj = data["norm_adj"].to(device)
    labels = data["labels"].to(device)
    train_mask = data["train_mask"].to(device)
    val_mask = data["val_mask"].to(device)
    test_mask = data["test_mask"].to(device)

    model = build_model(
        model_name=args.model,
        in_features=data["num_features"],
        num_classes=data["num_classes"],
        hidden_dim=args.hidden_dim,
        use_gfc=args.use_gfc,
    ).to(device)

    criterion = GraphCalibrationLoss(smoothing=0.05, cal_weight=0.15, use_gcl=args.use_gcl).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = f"{args.model}_{args.dataset}_seed{args.seed}"
    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, f"{run_name}_best.pth")

    print(f"\n--- Starting Graph Training: {run_name} ({args.epochs} epochs) ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, norm_adj)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            val_metrics = evaluate_node_classification(model, features, norm_adj, labels, val_mask)
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] Loss: {loss.item():.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | Val ECE: {val_metrics['ece']:.2f}%")
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                torch.save({"model_state_dict": model.state_dict(), "val_metrics": val_metrics}, best_ckpt_path)

    print("\nTraining completed. Loading best checkpoint for test evaluation...")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate_node_classification(model, features, norm_adj, labels, test_mask)
    print("=" * 60)
    print(f" ACTUAL TEST PERFORMANCE (EXPERIMENT RESULTS): {run_name}")
    print(f"  • Test Accuracy: {test_metrics['accuracy']}%")
    print(f"  • Macro-F1:      {test_metrics['macro_f1']}%")
    print(f"  • Micro-F1:      {test_metrics['micro_f1']}%")
    print(f"  • ECE (15-bins): {test_metrics['ece']}%")
    print("=" * 60)

    logger = ExperimentLogger(log_dir="./results")
    logger.log_metrics(run_name, test_metrics)


if __name__ == "__main__":
    main()
