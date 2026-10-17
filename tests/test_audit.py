"""
Regression and Invariant Tests for the Scientific Audit.

Validates core mathematical and temporal invariants:
1. Target-Feature Temporal Alignment Invariant: target[t] maps to snapshot t+1.
2. Protocol A vs Protocol B Partition Invariant: every clique is either persisting (J >= tau) or dissolved (J < tau).
3. SRI Conservation Invariant: sum_v SRI(v) == sum_C |C| * T(C).
4. Hidden State Isolation Invariant: resetting hidden states prevents leak across train/val/test splits.
5. Fragility Monotonicity Invariant: Giant Connected Component size non-increases with node removal.
"""

import pytest
import numpy as np
import networkx as nx
import torch

from src.data.snapshots import GraphSnapshot
from src.graph.cliques import extract_maximal_cliques
from src.graph.scores import compute_clique_tensions, compute_silo_risk_index
from src.graph.meta_graph import build_clique_meta_graph
from src.models.temporal_gnn import TemporalGNN, match_cliques_jaccard
from src.training.trainer import build_forecasting_targets
from src.training.audit_fragility import simulate_node_removal


def test_target_temporal_alignment():
    """Verify that forecasting targets map snapshot t to snapshot t+1."""
    # Create two synthetic snapshots with a persisting clique
    G0 = nx.Graph()
    G0.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4)])
    snap0 = GraphSnapshot(0, 0, 1000, G0, 4)

    G1 = nx.Graph()
    G1.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 5), (5, 6)])
    snap1 = GraphSnapshot(1, 1000, 2000, G1, 5)

    c0 = extract_maximal_cliques(G0, min_size=3)
    c1 = extract_maximal_cliques(G1, min_size=3)
    m0 = build_clique_meta_graph(snap0, c0)
    m1 = build_clique_meta_graph(snap1, c1)

    targets = build_forecasting_targets([m0, m1], delta=1, jaccard_threshold=0.5)

    assert len(targets) == 2
    assert targets[0] is not None
    assert targets[1] is None  # Last snapshot has no future target

    # Clique (1, 2, 3) persisted exactly (Jaccard = 1.0)
    # Its target should equal the tension of (1, 2, 3) in snapshot 1
    assert targets[0].numel() == 1
    assert targets[0][0].item() == m1.clique_tensions[0].item()


def test_protocol_a_vs_b_partition():
    """Verify that every test clique is partitioned into persisting or dissolved with no overlap."""
    cliques_prev = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    cliques_curr = [(1, 2, 3), (4, 5, 10)]  # (1, 2, 3) J=1.0; (4, 5, 6) vs (4, 5, 10) J = 2/4 = 0.5; (7, 8, 9) J=0

    matches = match_cliques_jaccard(cliques_prev, cliques_curr, threshold=0.5)

    # (1, 2, 3) and (4, 5, 10) matched
    assert len(matches) == 2
    assert 0 in matches  # curr clique 0 matched prev clique 0
    assert 1 in matches  # curr clique 1 matched prev clique 1

    # In Protocol B, only matched cliques are evaluated
    # In Protocol A, all 3 prev cliques get targets (1 unmatched -> 0.0)
    persisting_indices = [prev_idx for curr_idx, prev_idx in matches.items()]
    assert set(persisting_indices) == {0, 1}
    assert 2 not in persisting_indices


def test_sri_conservation_invariant():
    """Verify mathematical conservation: sum_v SRI(v) == sum_C |C| * T(C)."""
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (3, 5)])
    cliques = extract_maximal_cliques(G, min_size=3)
    tensions = compute_clique_tensions(G, cliques)
    all_nodes = list(G.nodes())

    sri_dict = compute_silo_risk_index(cliques, tensions, all_nodes)

    sum_sri = sum(sri_dict.values())
    sum_clique_weighted_tension = sum(len(c) * tensions[c] for c in cliques)

    assert np.isclose(sum_sri, sum_clique_weighted_tension, atol=1e-5)


def test_hidden_state_isolation():
    """Verify that resetting prev_h = None isolates the recurrent state."""
    model = TemporalGNN(in_features=5, hidden_dim=16)
    model.eval()

    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3)])
    snap = GraphSnapshot(0, 0, 1000, G, 3)
    m = build_clique_meta_graph(snap, extract_maximal_cliques(G, min_size=3))

    preds1, h1 = model.forward_snapshot(m, prev_meta_graph=None, prev_hidden_state=None)

    # If isolated, a fresh call with None must yield identical predictions to preds1
    preds2, h2 = model.forward_snapshot(m, prev_meta_graph=None, prev_hidden_state=None)

    assert torch.allclose(preds1, preds2)
    assert torch.allclose(h1, h2)


def test_fragility_monotonicity():
    """Verify that GCC is monotonically non-increasing under node removal."""
    G = nx.erdos_renyi_graph(50, 0.15, seed=42)
    nodes = list(G.nodes())
    fractions = np.linspace(0.0, 0.5, 10)

    gcc_fractions = simulate_node_removal(G, nodes, fractions)

    # GCC should never increase as nodes are removed
    for i in range(len(gcc_fractions) - 1):
        assert gcc_fractions[i] >= gcc_fractions[i + 1] - 1e-6
