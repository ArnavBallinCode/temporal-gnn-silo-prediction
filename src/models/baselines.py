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

    def predict_two_stage(self, meta_graph: CliqueMetaGraph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-stage prediction for Persistence:
        - Assumes survival (P(persist) = 1.0)
        - Predicts current observed tension as conditional tension
        - Expected tension = 1.0 * current tension = current tension
        """
        cond = meta_graph.clique_tensions.clone()
        p_persist = torch.ones_like(cond)
        expected = cond.clone()
        return expected, p_persist, cond


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

    def predict_two_stage(self, meta_graph: CliqueMetaGraph, default_val: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-stage historical mean prediction.
        """
        cond = self.predict(meta_graph, default_val)
        p_persist = torch.ones_like(cond)
        expected = cond.clone()
        return expected, p_persist, cond


class FeatureOnlyMLP(nn.Module):
    """
    Multi-Layer Perceptron using only clique features, ignoring graph structure.
    Supports both single-stage regression and two-stage hurdle forecasting.
    """

    def __init__(self, in_features: int = 5, hidden_dim: int = 16):
        super().__init__()
        head_mid = hidden_dim // 2 if hidden_dim >= 4 else hidden_dim
        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
        )
        # Conditional tension regression head
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, head_mid),
            nn.ReLU(),
            nn.Linear(head_mid, 1),
        )
        # Persistence classification head (Stage A)
        self.persistence_head = nn.Sequential(
            nn.Linear(hidden_dim, head_mid),
            nn.ReLU(),
            nn.Linear(head_mid, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Single-stage forward pass (backward compatible)."""
        if features.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32, device=features.device)
        h = self.encoder(features)
        preds = self.net(h).squeeze(-1)
        return torch.nn.functional.softplus(preds)

    def forward_two_stage(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-stage forward pass:
        Returns (expected_tension, persistence_prob, conditional_tension).
        """
        if features.numel() == 0:
            device = features.device
            empty = torch.zeros((0,), dtype=torch.float32, device=device)
            return empty, empty, empty
        h = self.encoder(features)
        p_logits = self.persistence_head(h).squeeze(-1)
        p_prob = torch.sigmoid(p_logits)
        cond_t = torch.nn.functional.softplus(self.net(h).squeeze(-1))
        expected_t = p_prob * cond_t
        return expected_t, p_prob, cond_t


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
        head_mid = hidden_dim // 2 if hidden_dim >= 4 else hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_mid),
            nn.ReLU(),
            nn.Linear(head_mid, 1),
        )
        self.persistence_head = nn.Sequential(
            nn.Linear(hidden_dim, head_mid),
            nn.ReLU(),
            nn.Linear(head_mid, 1),
        )

    def forward(self, meta_graph: CliqueMetaGraph) -> torch.Tensor:
        if meta_graph.num_cliques == 0:
            return torch.zeros((0,), dtype=torch.float32, device=meta_graph.features.device)
        emb = self.spatial_gnn(meta_graph.features, meta_graph.edge_index, meta_graph.edge_weights)
        preds = self.head(emb).squeeze(-1)
        return torch.nn.functional.softplus(preds)

    def forward_two_stage(self, meta_graph: CliqueMetaGraph) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if meta_graph.num_cliques == 0:
            device = meta_graph.features.device
            empty = torch.zeros((0,), dtype=torch.float32, device=device)
            return empty, empty, empty
        emb = self.spatial_gnn(meta_graph.features, meta_graph.edge_index, meta_graph.edge_weights)
        p_logits = self.persistence_head(emb).squeeze(-1)
        p_prob = torch.sigmoid(p_logits)
        cond_t = torch.nn.functional.softplus(self.head(emb).squeeze(-1))
        expected_t = p_prob * cond_t
        return expected_t, p_prob, cond_t


class TemporalGNNAblationNoGRU(nn.Module):
    """
    Ablation model: Spatial GCN with MLP heads, completely bypassing the GRU.
    Isolates whether the GRU recurrent memory contributes to performance.
    """

    def __init__(self, in_features: int = 5, hidden_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.spatial_gnn = SpatialGCN(
            in_features=in_features,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )
        head_mid = hidden_dim // 2 if hidden_dim >= 4 else hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_mid),
            nn.ReLU(),
            nn.Linear(head_mid, 1),
        )
        self.persistence_head = nn.Sequential(
            nn.Linear(hidden_dim, head_mid),
            nn.ReLU(),
            nn.Linear(head_mid, 1),
        )

    def forward(self, meta_graph: CliqueMetaGraph) -> torch.Tensor:
        if meta_graph.num_cliques == 0:
            return torch.zeros((0,), dtype=torch.float32, device=meta_graph.features.device)
        emb = self.spatial_gnn(meta_graph.features, meta_graph.edge_index, meta_graph.edge_weights)
        preds = self.head(emb).squeeze(-1)
        return torch.nn.functional.softplus(preds)

    def forward_snapshot(self, meta_graph, prev_meta=None, prev_h=None):
        if meta_graph.num_cliques == 0:
            return torch.zeros((0,), dtype=torch.float32), torch.zeros((0, 16), dtype=torch.float32)
        preds = self.forward(meta_graph)
        return preds, None

    def forward_two_stage(self, meta_graph, prev_meta=None, prev_h=None):
        if meta_graph.num_cliques == 0:
            device = meta_graph.features.device
            empty = torch.zeros((0,), dtype=torch.float32, device=device)
            if prev_meta is not None or prev_h is not None:
                return empty, empty, empty, None
            return empty, empty, empty
        emb = self.spatial_gnn(meta_graph.features, meta_graph.edge_index, meta_graph.edge_weights)
        p_logits = self.persistence_head(emb).squeeze(-1)
        p_prob = torch.sigmoid(p_logits)
        cond_t = torch.nn.functional.softplus(self.head(emb).squeeze(-1))
        expected_t = p_prob * cond_t
        if prev_meta is not None or prev_h is not None:
            return expected_t, p_prob, cond_t, None
        return expected_t, p_prob, cond_t

    def predict_node_sri(self, predicted_tensions, meta_graph, all_node_ids=None):
        sri_dict = {nid: 0.0 for nid in (all_node_ids or [])}
        for nid, clique_indices in meta_graph.node_to_clique_map.items():
            total = sum(float(predicted_tensions[c].item()) for c in clique_indices if c < len(predicted_tensions))
            sri_dict[nid] = total
        return sri_dict

