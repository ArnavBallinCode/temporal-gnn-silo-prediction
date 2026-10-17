"""
Unit tests for Temporal State Handling, Warm-Up History, and Leakage Prevention.
"""

import pytest
import torch
import networkx as nx

from src.data.snapshots import GraphSnapshot
from src.graph.meta_graph import build_clique_meta_graph
from src.models.temporal_gnn import TemporalGNN


def test_forward_only_hidden_state_propagation():
    """
    Verify that hidden states only propagate forward in chronological order,
    and that snapshot k predictions never depend on snapshot k+1.
    """
    torch.manual_seed(42)
    model = TemporalGNN(in_features=5, hidden_dim=8)
    model.eval()

    G0 = nx.Graph([(1, 2), (2, 3), (1, 3)])
    G1 = nx.Graph([(1, 2), (2, 3), (1, 3), (3, 4)])
    snap0 = GraphSnapshot(0, 0, 1000, G0, 3)
    snap1 = GraphSnapshot(1, 1000, 2000, G1, 4)
    m0 = build_clique_meta_graph(snap0, [(1, 2, 3)])
    m1 = build_clique_meta_graph(snap1, [(1, 2, 3)])

    # Pass 1: compute step 0 normally
    pred0_a, h0_a = model.forward_snapshot(m0)

    # Pass 2: compute step 1
    pred1, h1 = model.forward_snapshot(m1, prev_meta_graph=m0, prev_hidden_state=h0_a)

    # Re-evaluate step 0: must be 100% identical, totally independent of step 1
    pred0_b, h0_b = model.forward_snapshot(m0)
    assert torch.allclose(pred0_a, pred0_b)
    assert torch.allclose(h0_a, h0_b)


def test_warm_up_vs_cold_start_distinction():
    """
    Verify that warm-up maintains historical memory while cold-start resets state to None.
    """
    torch.manual_seed(42)
    model = TemporalGNN(in_features=5, hidden_dim=8)
    model.eval()

    G0 = nx.Graph([(1, 2), (2, 3), (1, 3)])
    G1 = nx.Graph([(1, 2), (2, 3), (1, 3), (3, 4)])
    snap0 = GraphSnapshot(0, 0, 1000, G0, 3)
    snap1 = GraphSnapshot(1, 1000, 2000, G1, 4)
    m0 = build_clique_meta_graph(snap0, [(1, 2, 3)])
    m1 = build_clique_meta_graph(snap1, [(1, 2, 3)])

    # Compute step 0
    _, h0 = model.forward_snapshot(m0)

    # Cold start on step 1 (prev_hidden_state = None)
    pred_cold, h_cold = model.forward_snapshot(m1, prev_meta_graph=None, prev_hidden_state=None)

    # Warm start on step 1 (prev_hidden_state = h0)
    pred_warm, h_warm = model.forward_snapshot(m1, prev_meta_graph=m0, prev_hidden_state=h0)

    # Recurrent state update should make warm start distinct from cold start
    assert not torch.allclose(h_cold, h_warm)


def test_state_detachment_prevents_unintended_backprop():
    """
    Verify that detaching hidden states prevents backpropagation through earlier history.
    """
    model = TemporalGNN(in_features=5, hidden_dim=8)
    model.train()

    G0 = nx.Graph([(1, 2), (2, 3), (1, 3)])
    G1 = nx.Graph([(1, 2), (2, 3), (1, 3), (3, 4)])
    snap0 = GraphSnapshot(0, 0, 1000, G0, 3)
    snap1 = GraphSnapshot(1, 1000, 2000, G1, 4)
    m0 = build_clique_meta_graph(snap0, [(1, 2, 3)])
    m1 = build_clique_meta_graph(snap1, [(1, 2, 3)])

    _, h0 = model.forward_snapshot(m0)
    h0.retain_grad()
    detached_h0 = h0.detach()

    assert not detached_h0.requires_grad

    pred1, _ = model.forward_snapshot(m1, prev_meta_graph=m0, prev_hidden_state=detached_h0)
    loss = pred1.sum()
    loss.backward()

    # h0 should not have received gradients because detached_h0 was passed to m1
    assert h0.grad is None
