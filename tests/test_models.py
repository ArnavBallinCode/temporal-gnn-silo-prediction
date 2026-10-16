"""
Automated unit tests for GCN message passing, Temporal GNN, and training steps.
"""

import pytest
import torch
import torch.nn as nn
import networkx as nx

from src.data.snapshots import GraphSnapshot
from src.graph.meta_graph import build_clique_meta_graph
from src.models.gnn import SimpleGCNLayer, build_dense_normalized_adjacency
from src.models.temporal_gnn import TemporalGNN, match_cliques_jaccard
from src.models.baselines import PersistenceBaseline, HistoricalMeanBaseline, FeatureOnlyMLP
from src.training.trainer import build_forecasting_targets


def test_simple_gcn_layer_numerical_check():
    """
    Step-by-step hand-calculated numerical message passing test:
    2-node graph with 1 edge between node 0 and node 1.

    Mathematical Derivation:
    Adjacency with self-loops:
        A_tilde = [[1, 1],
                   [1, 1]]
    Degree vector:
        deg = [2, 2]
    Normalized adjacency:
        A_hat = D^(-1/2) @ A_tilde @ D^(-1/2)
              = [[0.5, 0.5],
                 [0.5, 0.5]]

    Let input features X = [[1.0, 0.0],
                            [0.0, 2.0]]
    Let weight W = [[1.0],
                    [1.0]], bias = 0.0

    Then:
        X @ W = [[1.0],
                 [2.0]]
        A_hat @ (X @ W) = [[0.5 * 1.0 + 0.5 * 2.0],
                           [0.5 * 1.0 + 0.5 * 2.0]]
                        = [[1.5],
                           [1.5]]
    """
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    num_nodes = 2

    A_hat = build_dense_normalized_adjacency(num_nodes, edge_index)
    assert A_hat[0, 0].item() == pytest.approx(0.5)
    assert A_hat[0, 1].item() == pytest.approx(0.5)
    assert A_hat[1, 0].item() == pytest.approx(0.5)
    assert A_hat[1, 1].item() == pytest.approx(0.5)

    # Initialize GCN layer with exact known weights
    layer = SimpleGCNLayer(in_features=2, out_features=1, bias=False)
    layer.linear.weight.data = torch.tensor([[1.0, 1.0]])

    x = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    out = layer(x, edge_index)

    assert out.shape == (2, 1)
    assert out[0, 0].item() == pytest.approx(1.5)
    assert out[1, 0].item() == pytest.approx(1.5)


def test_temporal_gnn_forward_and_gru_update():
    """
    Verifies that the Temporal GNN forward pass produces expected shapes,
    and updates hidden states via GRU for persisting cliques.
    """
    G0 = nx.Graph()
    G0.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (3, 5)])
    snap0 = GraphSnapshot(snapshot_id=0, start_time_ms=0, end_time_ms=1000, graph=G0, num_events=2)
    meta0 = build_clique_meta_graph(snap0, [(1, 2, 3), (3, 4, 5)])

    # Snapshot 1 has persisting clique (1, 2, 3) and new clique (6, 7, 8)
    G1 = nx.Graph()
    G1.add_edges_from([(1, 2), (2, 3), (1, 3), (6, 7), (7, 8), (6, 8)])
    snap1 = GraphSnapshot(snapshot_id=1, start_time_ms=1000, end_time_ms=2000, graph=G1, num_events=2)
    meta1 = build_clique_meta_graph(snap1, [(1, 2, 3), (6, 7, 8)])

    model = TemporalGNN(in_features=5, hidden_dim=8, dropout=0.0)

    # Step 0: First snapshot
    preds0, h0 = model.forward_snapshot(meta0)
    assert preds0.shape == (2,)
    assert h0.shape == (2, 8)
    assert not torch.isnan(preds0).any()

    # Step 1: Second snapshot (temporal update)
    preds1, h1 = model.forward_snapshot(meta1, prev_meta_graph=meta0, prev_hidden_state=h0)
    assert preds1.shape == (2,)
    assert h1.shape == (2, 8)
    assert not torch.isnan(preds1).any()


def test_training_step_updates_parameters():
    """
    Verifies that a backward step computes finite gradients and updates parameters.
    """
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (3, 5)])
    snap = GraphSnapshot(snapshot_id=0, start_time_ms=0, end_time_ms=1000, graph=G, num_events=2)
    meta = build_clique_meta_graph(snap, [(1, 2, 3), (3, 4, 5)])

    model = TemporalGNN(in_features=5, hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    target = torch.tensor([1.0, 0.5], dtype=torch.float32)

    # Record weight before update
    initial_weight = model.head[0].weight.clone()

    optimizer.zero_grad()
    preds, _ = model.forward_snapshot(meta)
    loss = criterion(preds, target)
    loss.backward()

    # Check finite gradients
    for p in model.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()
            assert not torch.isinf(p.grad).any()

    optimizer.step()
    updated_weight = model.head[0].weight.clone()

    # Parameters must have changed
    assert not torch.equal(initial_weight, updated_weight)


def test_forecasting_targets_construction():
    """
    Verifies that build_forecasting_targets correctly aligns future tensions.
    """
    G0 = nx.Graph()
    G0.add_edges_from([(1, 2), (2, 3), (1, 3)])
    snap0 = GraphSnapshot(snapshot_id=0, start_time_ms=0, end_time_ms=1000, graph=G0, num_events=1)
    meta0 = build_clique_meta_graph(snap0, [(1, 2, 3)])

    G1 = nx.Graph()
    G1.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (3, 5)])
    snap1 = GraphSnapshot(snapshot_id=1, start_time_ms=1000, end_time_ms=2000, graph=G1, num_events=2)
    meta1 = build_clique_meta_graph(snap1, [(1, 2, 3), (3, 4, 5)])

    targets = build_forecasting_targets([meta0, meta1], delta=1)
    assert len(targets) == 2
    assert targets[0] is not None
    # Clique (1, 2, 3) persisted into snapshot 1, where its tension is 2/3
    assert targets[0][0].item() == pytest.approx(2.0 / 3.0)
    # Final snapshot has target None (no future snapshot)
    assert targets[1] is None
