"""Graph dataset loader and adjacency normalization utilities.
"""

import os
import pickle
from typing import Dict, Any
import torch
from .download_data import verify_dataset, download_dataset, SUPPORTED_DATASETS, DATASET_CONFIGS


def normalize_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Computes symmetric normalized adjacency D^{-1/2} A D^{-1/2}."""
    # Add self loops
    loop_index = torch.arange(0, num_nodes, dtype=torch.long)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)

    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()
    return adj


def get_graph_data(dataset_name: str, data_dir: str = "./data/cache", download: bool = True) -> Dict[str, Any]:
    name = dataset_name.lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset {name} not supported.")

    target_dir = os.path.join(data_dir, name)
    if not verify_dataset(name, data_dir):
        if download:
            download_dataset(name, data_dir)
        else:
            raise FileNotFoundError(f"Missing dataset {name}.")

    cache_file = os.path.join(target_dir, "graph_data.pkl")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
    features = torch.tensor(data["features"], dtype=torch.float)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    train_mask = torch.tensor(data["train_mask"], dtype=torch.bool)
    val_mask = torch.tensor(data["val_mask"], dtype=torch.bool)
    test_mask = torch.tensor(data["test_mask"], dtype=torch.bool)

    num_nodes = features.size(0)
    norm_adj = normalize_adjacency(edge_index, num_nodes)

    return {
        "features": features,
        "labels": labels,
        "edge_index": edge_index,
        "norm_adj": norm_adj,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "num_classes": DATASET_CONFIGS[name]["num_classes"],
        "num_features": DATASET_CONFIGS[name]["num_features"],
        "num_nodes": num_nodes,
    }
