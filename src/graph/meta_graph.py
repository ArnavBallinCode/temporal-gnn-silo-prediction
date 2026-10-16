"""
Clique-Conflict Meta-Graph Construction.


Conceptual Background for Beginners:
-------------------------------------
In standard graph neural networks, nodes are people and edges are emails.
In this research, we construct a META-GRAPH (G_C) at a higher level of abstraction:
- Each NODE in the meta-graph is an entire CLIQUE (a cohesive team).
- Each EDGE in the meta-graph represents STRUCTURAL CONFLICT (ICCS > 0) between two teams.
- Each meta-node has FEATURES describing the team (size, conflict degree, tension, etc.).

Why is this powerful?
Instead of predicting individual emails, the GNN learns how INTER-TEAM TENSION
propagates across the organizational network!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import networkx as nx
import torch

from src.data.snapshots import GraphSnapshot
from src.graph.scores import (
    compute_clique_tensions,
    compute_iccs,
)


@dataclass
class CliqueMetaGraph:
    """
    Represents the clique-conflict meta-graph G_C^(k) for snapshot k.

    Attributes:
        snapshot_id: Index k of the corresponding time snapshot.
        num_cliques: Number of meta-nodes (teams/cliques) K in this snapshot.
        cliques: Ordered list of cliques (each clique is a tuple of employee IDs).
        features: Node feature tensor X of shape [K, F].
        feature_names: Names of each feature column.
        edge_index: PyTorch edge index tensor of shape [2, E_meta] for GNN message passing.
        edge_weights: ICCS edge conflict weights of shape [E_meta].
        clique_tensions: Target ground-truth clique tension tensor T(C) of shape [K].
        node_to_clique_map: Maps each employee node ID to the list of clique indices containing them.
    """
    snapshot_id: int
    num_cliques: int
    cliques: List[Tuple[int, ...]]
    features: torch.Tensor
    feature_names: List[str]
    edge_index: torch.Tensor
    edge_weights: torch.Tensor
    clique_tensions: torch.Tensor
    node_to_clique_map: Dict[int, List[int]]


def build_clique_meta_graph(
    snapshot: GraphSnapshot,
    cliques: List[Tuple[int, ...]],
    conflict_threshold: float = 0.0,
) -> CliqueMetaGraph:
    """
    Constructs the clique-conflict meta-graph G_C for a given snapshot.

    Feature Definitions (F = 5 features per clique):
    1. `clique_size`: Number of employees in the clique |C|.
    2. `conflict_degree`: Number of other cliques with ICCS > conflict_threshold.
    3. `total_tension`: Sum of ICCS scores with all other cliques: T(C).
    4. `avg_internal_comm`: Average email interaction weight on edges within the clique.
    5. `external_comm_vol`: Total email communication weight to employees outside the clique.

    Args:
        snapshot: GraphSnapshot containing the communication graph G^(k).
        cliques: List of extracted maximal cliques for this snapshot.
        conflict_threshold: Minimum ICCS required to form an edge in the meta-graph (default: 0.0).

    Returns:
        CliqueMetaGraph dataclass with PyTorch tensors ready for GNN layers.
    """
    num_cliques = len(cliques)
    feature_names = [
        "clique_size",
        "conflict_degree",
        "total_tension",
        "avg_internal_comm",
        "external_comm_vol",
    ]

    # Map each employee node ID -> list of clique indices [0, ..., K-1]
    node_to_cliques: Dict[int, List[int]] = {}
    for c_idx, c in enumerate(cliques):
        for nid in c:
            if nid not in node_to_cliques:
                node_to_cliques[nid] = []
            node_to_cliques[nid].append(c_idx)

    # Handle edge case: no cliques in this snapshot
    if num_cliques == 0:
        return CliqueMetaGraph(
            snapshot_id=snapshot.snapshot_id,
            num_cliques=0,
            cliques=[],
            features=torch.zeros((0, len(feature_names)), dtype=torch.float32),
            feature_names=feature_names,
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_weights=torch.zeros((0,), dtype=torch.float32),
            clique_tensions=torch.zeros((0,), dtype=torch.float32),
            node_to_clique_map=node_to_cliques,
        )

    G = snapshot.graph

    # 1. Compute ground-truth tensions and pairwise ICCS values
    tensions_dict = compute_clique_tensions(G, cliques)

    # 2. Build meta-graph edges (ICCS > threshold)
    src_list: List[int] = []
    dst_list: List[int] = []
    weights_list: List[float] = []
    conflict_degrees = [0] * num_cliques

    for i in range(num_cliques):
        c_i = cliques[i]
        for j in range(i + 1, num_cliques):
            c_j = cliques[j]
            iccs_val = compute_iccs(G, c_i, c_j)

            if iccs_val > conflict_threshold:
                # Add bidirectional edge (i -> j and j -> i) for message passing
                src_list.extend([i, j])
                dst_list.extend([j, i])
                weights_list.extend([iccs_val, iccs_val])
                conflict_degrees[i] += 1
                conflict_degrees[j] += 1

    # 3. Construct feature matrix X of shape [K, F]
    feature_rows: List[List[float]] = []
    for i in range(num_cliques):
        c = cliques[i]
        c_set = set(c)
        size = float(len(c))
        c_deg = float(conflict_degrees[i])
        tension = float(tensions_dict.get(c, 0.0))

        # Internal communication weight within clique
        internal_weights = []
        for u in c:
            if G.has_node(u):
                for v in G.neighbors(u):
                    if u < v and v in c_set:
                        internal_weights.append(G[u][v].get("weight", 1))
        avg_internal = float(sum(internal_weights) / len(internal_weights)) if internal_weights else 1.0

        # External communication volume outside clique
        ext_vol = 0.0
        for u in c:
            if G.has_node(u):
                for v in G.neighbors(u):
                    if v not in c_set:
                        ext_vol += float(G[u][v].get("weight", 1))

        feature_rows.append([size, c_deg, tension, avg_internal, ext_vol])

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(feature_rows, dtype=torch.float32)
    edge_index_tensor = (
        torch.tensor([src_list, dst_list], dtype=torch.long)
        if src_list
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
    tensions_tensor = torch.tensor([tensions_dict[c] for c in cliques], dtype=torch.float32)

    return CliqueMetaGraph(
        snapshot_id=snapshot.snapshot_id,
        num_cliques=num_cliques,
        cliques=cliques,
        features=features_tensor,
        feature_names=feature_names,
        edge_index=edge_index_tensor,
        edge_weights=edge_weights_tensor,
        clique_tensions=tensions_tensor,
        node_to_clique_map=node_to_cliques,
    )
