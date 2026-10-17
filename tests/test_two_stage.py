"""
Unit tests for Two-Stage Target Construction and Hurdle Forecasting.
"""

import pytest
import torch
import networkx as nx

from src.data.snapshots import GraphSnapshot
from src.graph.cliques import extract_maximal_cliques
from src.graph.meta_graph import build_clique_meta_graph
from src.models.temporal_gnn import TemporalGNN
from src.training.trainer import build_two_stage_targets, train_temporal_gnn


def test_two_stage_targets_disentanglement():
    """
    Verify that Stage A (persistence 0/1) and Stage B (conditional tension)
    are separated cleanly and do not conflate dissolution with zero tension.
    """
    G0 = nx.Graph()
    # Clique 0: (1, 2, 3), Clique 1: (4, 5, 6)
    G0.add_edges_from([(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6)])
    snap0 = GraphSnapshot(0, 0, 1000, G0, 6)
    m0 = build_clique_meta_graph(snap0, [(1, 2, 3), (4, 5, 6)])

    # Snapshot 1: only (1, 2, 3) survives; (4, 5, 6) completely dissolves
    G1 = nx.Graph()
    G1.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 7)])  # (1, 2, 3) has boundary tension with 7
    snap1 = GraphSnapshot(1, 1000, 2000, G1, 4)
    m1 = build_clique_meta_graph(snap1, [(1, 2, 3)])

    targets = build_two_stage_targets([m0, m1], delta=1, jaccard_threshold=0.5)

    p_targets = targets.persistence_targets[0]
    c_targets = targets.conditional_targets[0]
    z_targets = targets.zero_fill_targets[0]
    p_masks = targets.persisting_masks[0]

    assert p_targets is not None
    assert len(p_targets) == 2

    # Clique (1, 2, 3) survived
    assert p_targets[0].item() == 1.0
    assert p_masks[0].item() is True
    assert c_targets[0].item() == pytest.approx(m1.clique_tensions[0].item())
    assert z_targets[0].item() == pytest.approx(m1.clique_tensions[0].item())

    # Clique (4, 5, 6) dissolved
    assert p_targets[1].item() == 0.0
    assert p_masks[1].item() is False
    # Zero-fill baseline sets it to 0.0
    assert z_targets[1].item() == 0.0


def test_two_stage_temporal_gnn_forward_and_expected_tension():
    """
    Verify that the two-stage forward pass computes:
        E[T] = P(persist) * T_cond
    """
    model = TemporalGNN(in_features=5, hidden_dim=8)
    model.eval()

    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4)])
    snap = GraphSnapshot(0, 0, 1000, G, 4)
    meta = build_clique_meta_graph(snap, [(1, 2, 3)])

    with torch.no_grad():
        exp_t, p_prob, cond_t, curr_h = model.forward_two_stage(meta)

    assert exp_t.shape == (1,)
    assert p_prob.shape == (1,)
    assert cond_t.shape == (1,)

    # Probability in [0, 1]
    assert 0.0 <= p_prob[0].item() <= 1.0
    # Conditional tension >= 0
    assert cond_t[0].item() >= 0.0
    # Expected tension is exact product
    expected_product = p_prob[0].item() * cond_t[0].item()
    assert exp_t[0].item() == pytest.approx(expected_product, rel=1e-5)
