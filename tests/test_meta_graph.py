"""
Automated unit tests for clique-conflict meta-graph construction.
"""

import pytest
import networkx as nx
import torch

from src.data.snapshots import GraphSnapshot
from src.graph.meta_graph import build_clique_meta_graph


def test_build_clique_meta_graph_tiny():
    """
    Tests meta-graph construction on two triangles sharing node 3:
    C_0 = (1, 2, 3), C_1 = (3, 4, 5)
    """
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (3, 5)])

    snap = GraphSnapshot(
        snapshot_id=0,
        start_time_ms=0,
        end_time_ms=1000,
        graph=G,
        num_events=2,
    )

    cliques = [(1, 2, 3), (3, 4, 5)]
    meta = build_clique_meta_graph(snap, cliques, conflict_threshold=0.0)

    # 2 cliques -> 2 meta-nodes
    assert meta.num_cliques == 2
    assert meta.features.shape == (2, 5)

    # Both cliques have size 3
    assert meta.features[0, 0].item() == 3.0
    assert meta.features[1, 0].item() == 3.0

    # Conflict degree = 1 (each conflicts with the other)
    assert meta.features[0, 1].item() == 1.0
    assert meta.features[1, 1].item() == 1.0

    # Tension = 2/3
    assert meta.features[0, 2].item() == pytest.approx(2.0 / 3.0)
    assert meta.features[1, 2].item() == pytest.approx(2.0 / 3.0)

    # Bidirectional edge between 0 and 1
    assert meta.edge_index.shape == (2, 2)
    assert set(zip(meta.edge_index[0].tolist(), meta.edge_index[1].tolist())) == {(0, 1), (1, 0)}

    # Edge weight = ICCS = 2/3
    assert meta.edge_weights[0].item() == pytest.approx(2.0 / 3.0)
    assert meta.edge_weights[1].item() == pytest.approx(2.0 / 3.0)

    # Node to clique map
    assert meta.node_to_clique_map[3] == [0, 1]  # Bridge node 3 belongs to both cliques
    assert meta.node_to_clique_map[1] == [0]
    assert meta.node_to_clique_map[4] == [1]


def test_build_clique_meta_graph_unconnected():
    """
    Two disjoint triangles with no conflict.
    Meta-graph should have 2 nodes but 0 edges.
    """
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6)])

    snap = GraphSnapshot(
        snapshot_id=0,
        start_time_ms=0,
        end_time_ms=1000,
        graph=G,
        num_events=2,
    )

    cliques = [(1, 2, 3), (4, 5, 6)]
    meta = build_clique_meta_graph(snap, cliques, conflict_threshold=0.0)

    assert meta.num_cliques == 2
    assert meta.edge_index.shape == (2, 0)
    assert meta.edge_weights.shape == (0,)
    assert meta.features[0, 1].item() == 0.0  # Conflict degree = 0
    assert meta.features[0, 2].item() == 0.0  # Tension = 0.0


def test_empty_cliques():
    """
    Snapshot with 0 cliques should produce empty tensors without errors.
    """
    G = nx.Graph()
    snap = GraphSnapshot(snapshot_id=0, start_time_ms=0, end_time_ms=1000, graph=G, num_events=0)
    meta = build_clique_meta_graph(snap, [], conflict_threshold=0.0)

    assert meta.num_cliques == 0
    assert meta.features.shape == (0, 5)
    assert meta.edge_index.shape == (2, 0)
    assert meta.edge_weights.shape == (0,)
    assert meta.clique_tensions.shape == (0,)
