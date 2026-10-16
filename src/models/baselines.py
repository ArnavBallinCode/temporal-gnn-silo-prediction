"""
Baseline models for comparison against the Temporal GNN.


Baselines Implemented:
----------------------
1. Persistence Baseline:
   Predicts future tension equals current tension: T_hat^(k+Delta)(C) = T^(k)(C).
2. Historical Mean Baseline:
   Predicts the historical average tension of the clique.
3. Feature-Only MLP:
   A neural network that uses the clique features directly, ignoring graph structure.
4. Static GCN:
   Spatial GCN without recurrent GRU temporal memory.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from src.graph.meta_graph import CliqueMetaGraph
from src.models.gnn import SpatialGCN


class PersistenceBaseline:
    """
    Predicts that future clique tension will equal current clique tension.
    """

    def predict(self, meta_graph: CliqueMetaGraph) -> torch.Tensor:
        """Returns current ground-truth tension as the prediction for future tension."""
        return meta_graph.clique_tensions.clone()


class HistoricalMeanBaseline:
    """
    Maintains a running mean of tension for each clique across observed snapshots.
    """

    def __init__(self):
        self.clique_history: Dict[tuple, List[float]] = {}

    def update(self, meta_graph: CliqueMetaGraph) -> None:
        """Records ground-truth tensions for observed cliques."""
        for c, t in zip(meta_graph.cliques, meta_graph.clique_tensions.tolist()):
            if c not in self.clique_history:
                self.clique_history[c] = []
            self.clique_history[c].append(t)

    def predict(self, meta_graph: CliqueMetaGraph, default_val: float = 0.5) -> torch.Tensor:
        """Predicts historical mean tension for each clique."""
        preds = []
        for c in meta_graph.cliques:
            hist = self.clique_history.get(c, [])
            if hist:
                preds.append(sum(hist) / len(hist))
            else:
                preds.append(default_val)
        return torch.tensor(preds, dtype=torch.float32)


class FeatureOnlyMLP(nn.Module):
    """
    Multi-Layer Perceptron using only clique features, ignoring graph structure.
    """

    def __init__(self, in_features: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 4 else hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 4 else hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32, device=features.device)
        preds = self.net(features).squeeze(-1)
        return torch.nn.functional.softplus(preds)


class StaticGCNModel(nn.Module):
    """
    Static GCN baseline: applies spatial graph convolution and prediction head
    without recurrent temporal memory.
    """

    def __init__(self, in_features: int = 5, hidden_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.spatial_gnn = SpatialGCN(
            in_features=in_features,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 4 else hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 4 else hidden_dim, 1),
        )

    def forward(self, meta_graph: CliqueMetaGraph) -> torch.Tensor:
        if meta_graph.num_cliques == 0:
            return torch.zeros((0,), dtype=torch.float32, device=meta_graph.features.device)
        emb = self.spatial_gnn(meta_graph.features, meta_graph.edge_index, meta_graph.edge_weights)
        preds = self.head(emb).squeeze(-1)
        return torch.nn.functional.softplus(preds)
