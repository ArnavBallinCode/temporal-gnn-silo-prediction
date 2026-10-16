"""
Discrete-Time Recurrent Graph Neural Network (Temporal GNN).

Architecture Overview:
----------------------
At each snapshot time step k:
1. Spatial GNN:
       H^(k) = SpatialGCN( X^(k), A^(k) ) in R^(K_k x D)
   Encodes the current topological structure of inter-team conflict.

2. Temporal Tracking (Clique Alignment):
   Cliques can form, merge, persist, or dissolve across time windows.
   We align current cliques C^(k) with previous cliques C^(k-1) using Jaccard similarity:
       Jaccard(A, B) = |A intersect B| / |A union B|
   If Jaccard >= threshold (e.g. 0.5), the group persisted:
       s_i^(k) = GRUCell( h_i^(k), s_j^(k-1) )
   If Jaccard < threshold, it is a newly formed clique:
       s_i^(k) = h_i^(k)

3. Prediction Head:
       T_hat^(k+Delta)(C_i) = Linear( s_i^(k) )
   Predicts future structural tension for each clique.

4. Node-Level Silo Risk Index (SRI) Aggregation:
       SRI_hat^(k+Delta)(v) = sum_{C contains v} T_hat^(k+Delta)(C)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from src.graph.meta_graph import CliqueMetaGraph
from src.models.gnn import SpatialGCN


def match_cliques_jaccard(
    prev_cliques: List[Tuple[int, ...]],
    curr_cliques: List[Tuple[int, ...]],
    threshold: float = 0.5,
) -> Dict[int, int]:
    """
    Matches current cliques to previous cliques using Jaccard node similarity.

    Args:
        prev_cliques: List of cliques from snapshot k-1.
        curr_cliques: List of cliques from snapshot k.
        threshold: Minimum Jaccard similarity to consider them the same group.

    Returns:
        Dict mapping current_clique_idx -> previous_clique_idx for persisting cliques.
    """
    matches: Dict[int, int] = {}
    if not prev_cliques or not curr_cliques:
        return matches

    prev_sets = [set(c) for c in prev_cliques]

    for curr_idx, curr_c in enumerate(curr_cliques):
        curr_set = set(curr_c)
        best_jaccard = -1.0
        best_prev_idx = -1

        for prev_idx, prev_set in enumerate(prev_sets):
            intersection = len(curr_set.intersection(prev_set))
            union = len(curr_set.union(prev_set))
            if union > 0:
                jaccard = intersection / float(union)
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_prev_idx = prev_idx

        if best_jaccard >= threshold and best_prev_idx >= 0:
            matches[curr_idx] = best_prev_idx

    return matches


class TemporalGNN(nn.Module):
    """
    Spatial GCN + Recurrent GRU Temporal GNN.
    """

    def __init__(
        self,
        in_features: int = 5,
        hidden_dim: int = 16,
        dropout: float = 0.1,
        jaccard_threshold: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.jaccard_threshold = jaccard_threshold

        # 1. Spatial GNN layer: encodes meta-graph conflict topology
        self.spatial_gnn = SpatialGCN(
            in_features=in_features,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        # 2. Recurrent GRU Cell: tracks temporal evolution of clique embeddings
        # input size = hidden_dim, hidden size = hidden_dim
        self.gru_cell = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # 3. Prediction Head: regresses future tension score
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 4 else hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 4 else hidden_dim, 1),
        )

    def forward_snapshot(
        self,
        meta_graph: CliqueMetaGraph,
        prev_meta_graph: Optional[CliqueMetaGraph] = None,
        prev_hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a single snapshot meta-graph.

        Args:
            meta_graph: CliqueMetaGraph at current time k.
            prev_meta_graph: CliqueMetaGraph at previous time k-1 (if any).
            prev_hidden_state: Tensor of shape [K_prev, hidden_dim] (if any).

        Returns:
            (predicted_tensions, current_hidden_state)
            - predicted_tensions: shape [K_curr]
            - current_hidden_state: shape [K_curr, hidden_dim]
        """
        num_cliques = meta_graph.num_cliques
        if num_cliques == 0:
            empty_preds = torch.zeros((0,), dtype=torch.float32, device=meta_graph.features.device)
            empty_h = torch.zeros((0, self.hidden_dim), dtype=torch.float32, device=meta_graph.features.device)
            return empty_preds, empty_h

        # 1. Spatial GNN embedding: [K_curr, hidden_dim]
        spatial_emb = self.spatial_gnn(
            meta_graph.features,
            meta_graph.edge_index,
            meta_graph.edge_weights,
        )

        # 2. GRU recurrent update with clique alignment
        if prev_meta_graph is not None and prev_hidden_state is not None and prev_meta_graph.num_cliques > 0:
            matches = match_cliques_jaccard(
                prev_meta_graph.cliques,
                meta_graph.cliques,
                threshold=self.jaccard_threshold,
            )

            # Initialize current hidden state with spatial embedding
            curr_hidden = spatial_emb.clone()

            # For persisting cliques, update with GRUCell
            for curr_idx, prev_idx in matches.items():
                input_vec = spatial_emb[curr_idx].unsqueeze(0)  # [1, D]
                hx = prev_hidden_state[prev_idx].unsqueeze(0)   # [1, D]
                updated_vec = self.gru_cell(input_vec, hx)      # [1, D]
                curr_hidden[curr_idx] = updated_vec.squeeze(0)
        else:
            # First snapshot or previous had 0 cliques: initialize s_i = h_i
            curr_hidden = spatial_emb

        # Tension is non-negative: enforce with smooth Softplus (prevents dead-gradient issues)
        preds = torch.nn.functional.softplus(self.head(curr_hidden).squeeze(-1))

        return preds, curr_hidden

    def predict_node_sri(
        self,
        predicted_tensions: torch.Tensor,
        meta_graph: CliqueMetaGraph,
        all_node_ids: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """
        Projects predicted clique tensions back to employee nodes to produce node SRI.

        Equation (Section IV-A):
            SRI_hat(v) = sum_{C contains v} T_hat(C)
        """
        sri_dict: Dict[int, float] = {}
        if all_node_ids:
            for nid in all_node_ids:
                sri_dict[nid] = 0.0

        for nid, clique_indices in meta_graph.node_to_clique_map.items():
            total_risk = 0.0
            for c_idx in clique_indices:
                if c_idx < len(predicted_tensions):
                    total_risk += float(predicted_tensions[c_idx].item())
            sri_dict[nid] = total_risk

        return sri_dict
