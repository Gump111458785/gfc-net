"""Proposed Architecture: Graph Feature Calibration Network (GFC-Net).
Incorporates Topological Attention, Structural Feature Rectification, and Confidence Calibration.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .baselines import GCN, GraphSAGE, GAT, GCNLayer


class GraphFeatureCalibrationModule(nn.Module):
    """Calibrates structural node embeddings by gating local vs high-order topology."""
    def __init__(self, hidden_dim: int, reduction: int = 4):
        super().__init__()
        mid_dim = max(16, hidden_dim // reduction)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, local_feat: torch.Tensor, neigh_feat: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(local_feat)
        calibrated = local_feat * alpha + neigh_feat * (1.0 - alpha)
        return self.norm(local_feat + calibrated)


class GFCNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_classes: int = 7,
        dropout: float = 0.5,
        use_gfc: bool = True,
        use_temp_scale: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_gfc = use_gfc
        self.use_temp_scale = use_temp_scale

        self.gc1 = GCNLayer(in_features, hidden_dim)
        if use_gfc:
            self.calib_module = GraphFeatureCalibrationModule(hidden_dim)
        else:
            self.calib_module = None

        self.gc2 = GCNLayer(hidden_dim, num_classes)

        if use_temp_scale:
            self.temperature = nn.Parameter(torch.ones(1) * 1.15)
        else:
            self.register_buffer("temperature", torch.ones(1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor, apply_temp: bool = True) -> torch.Tensor:
        h1 = F.relu(self.gc1(x, adj))
        h1 = F.dropout(h1, self.dropout, training=self.training)

        if self.use_gfc and self.calib_module is not None:
            neigh = torch.spmm(adj, h1)
            h1 = self.calib_module(h1, neigh)

        logits = self.gc2(h1, adj)

        if apply_temp and self.use_temp_scale:
            temp = torch.clamp(self.temperature, min=0.1, max=5.0)
            return logits / temp
        return logits


def build_model(
    model_name: str,
    in_features: int,
    num_classes: int,
    hidden_dim: int = 64,
    use_gfc: bool = True,
    use_temp_scale: bool = True,
) -> nn.Module:
    name = model_name.lower().replace("-", "_")
    if name in ["gfc_net", "gfcnet", "ours"]:
        return GFCNet(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_gfc=use_gfc,
            use_temp_scale=use_temp_scale,
        )
    elif name == "gcn":
        return GCN(in_features=in_features, hidden_dim=hidden_dim, num_classes=num_classes)
    elif name == "graphsage":
        return GraphSAGE(in_features=in_features, hidden_dim=hidden_dim, num_classes=num_classes)
    elif name == "gat":
        return GAT(in_features=in_features, hidden_dim=hidden_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown GNN model: '{model_name}'.")
