"""Evaluation metrics suite for graph node classification.
Computes Accuracy, Macro-F1, Micro-F1, and Expected Calibration Error (ECE).
"""

import time
from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece * 100.0)


def evaluate_node_classification(
    model: torch.nn.Module,
    features: torch.Tensor,
    norm_adj: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features, norm_adj)
        probs = F.softmax(logits, dim=1)

        mask_indices = mask.nonzero(as_tuple=True)[0]
        sub_probs = probs[mask_indices].cpu().numpy()
        sub_preds = torch.argmax(logits[mask_indices], dim=1).cpu().numpy()
        sub_targets = labels[mask_indices].cpu().numpy()

    acc = accuracy_score(sub_targets, sub_preds) * 100.0
    macro_f1 = f1_score(sub_targets, sub_preds, average="macro") * 100.0
    micro_f1 = f1_score(sub_targets, sub_preds, average="micro") * 100.0
    ece = compute_ece(sub_probs, sub_targets, n_bins=15)

    return {
        "accuracy": round(float(acc), 2),
        "macro_f1": round(float(macro_f1), 2),
        "micro_f1": round(float(micro_f1), 2),
        "ece": round(float(ece), 2),
        "total_nodes": len(sub_targets),
    }


def measure_gpu_efficiency(
    model: torch.nn.Module,
    features: torch.Tensor,
    norm_adj: torch.Tensor,
    device: torch.device,
    warmup_iters: int = 30,
    measure_iters: int = 100,
) -> Dict[str, float]:
    if device.type != "cuda":
        raise ValueError("CUDA GPU is required for efficiency measurement.")

    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(features, norm_adj)
        torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(measure_iters):
            _ = model(features, norm_adj)
    end_event.record()
    torch.cuda.synchronize()

    avg_latency_ms = start_event.elapsed_time(end_event) / measure_iters
    total_params = sum(p.numel() for p in model.parameters()) / 1e3  # in K

    return {
        "params_k": round(float(total_params), 1),
        "latency_ms": round(float(avg_latency_ms), 2),
    }
