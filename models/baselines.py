"""Baseline Graph Neural Network models for Node Classification.
Includes GCN, GAT, and GraphSAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # H' = A_norm * H * W
        support = self.linear(x)
        output = torch.spmm(adj, support) + self.bias
        return output


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.gc1 = GCNLayer(in_features, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_features * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Neighborhood mean aggregation
        neigh1 = torch.spmm(adj, x)
        h1 = F.relu(self.linear1(torch.cat([x, neigh1], dim=1)))
        h1 = F.dropout(h1, self.dropout, training=self.training)
        neigh2 = torch.spmm(adj, h1)
        h2 = self.linear2(torch.cat([h1, neigh2], dim=1))
        return h2


class GAT(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, num_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.gc1 = GCNLayer(in_features, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)
