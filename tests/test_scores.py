"""
Automated unit tests for structural conflict scores: ICCS, Clique Tension, and SRI.
"""

import pytest
import networkx as nx

from src.graph.scores import (
    compute_external_neighbors,
    compute_boundary_overlap,
    compute_iccs,
    compute_clique_tensions,
    compute_silo_risk_index,
    compute_temporal_smoothed_sri,
)


def test_hand_calculated_shared_node_bridge():
    """
    Hand-calculated Example 1:
    Two triangles sharing node 3:
      Clique A = {1, 2, 3}
      Clique B = {3, 4, 5}
    Edges: (1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (3, 5).

    Mathematical Derivation:
    - N_ext(A) = {4, 5} (neighbors of 3 outside A)
    - N_ext(B) = {1, 2} (neighbors of 3 outside B)
    - N_ext(A) intersect B = {4, 5}
    - N_ext(B) intersect A = {1, 2}
    - overlap(A, B) = {1, 2, 4, 5} -> size 4
    - |A| + |B| = 3 + 3 = 6
    - ICCS(A, B) = 4 / 6 = 2/3 approx 0.6667
    - T(A) = 2/3, T(B) = 2/3
    - SRI:
        Node 1: in A only -> SRI = 2/3 approx 0.6667
        Node 2: in A only -> SRI = 2/3 approx 0.6667
        Node 3: in BOTH A and B -> SRI = 2/3 + 2/3 = 4/3 approx 1.3333 (highest risk!)
        Node 4: in B only -> SRI = 2/3 approx 0.6667
        Node 5: in B only -> SRI = 2/3 approx 0.6667
    """
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (3, 5)])

    clique_a = (1, 2, 3)
    clique_b = (3, 4, 5)
    cliques = [clique_a, clique_b]

    # External neighbors
    n_ext_a = compute_external_neighbors(G, clique_a)
    assert n_ext_a == {4, 5}
    n_ext_b = compute_external_neighbors(G, clique_b)
    assert n_ext_b == {1, 2}

    # Boundary overlap
    overlap = compute_boundary_overlap(G, clique_a, clique_b)
    assert overlap == {1, 2, 4, 5}

    # ICCS
    iccs = compute_iccs(G, clique_a, clique_b)
    assert iccs == pytest.approx(4.0 / 6.0)

    # Tensions
    tensions = compute_clique_tensions(G, cliques)
    assert tensions[clique_a] == pytest.approx(2.0 / 3.0)
    assert tensions[clique_b] == pytest.approx(2.0 / 3.0)

    # SRI
    sri = compute_silo_risk_index(cliques, tensions)
    assert sri[1] == pytest.approx(2.0 / 3.0)
    assert sri[2] == pytest.approx(2.0 / 3.0)
    assert sri[4] == pytest.approx(2.0 / 3.0)
    assert sri[5] == pytest.approx(2.0 / 3.0)
    # The bridge node has the highest SRI!
    assert sri[3] == pytest.approx(4.0 / 3.0)
    assert sri[3] > sri[1]


def test_hand_calculated_disjoint_unconnected():
    """
    Hand-calculated Example 2:
    Two disconnected triangles:
      Clique A = {1, 2, 3}
      Clique B = {4, 5, 6}
    Zero cross-edges.
    - overlap(A, B) = empty
    - ICCS(A, B) = 0.0
    - Tension = 0.0, SRI = 0.0
    """
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6)])

    clique_a = (1, 2, 3)
    clique_b = (4, 5, 6)
    cliques = [clique_a, clique_b]

    assert compute_boundary_overlap(G, clique_a, clique_b) == set()
    assert compute_iccs(G, clique_a, clique_b) == 0.0

    tensions = compute_clique_tensions(G, cliques)
    assert tensions[clique_a] == 0.0
    assert tensions[clique_b] == 0.0

    sri = compute_silo_risk_index(cliques, tensions)
    assert all(val == 0.0 for val in sri.values())


def test_hand_calculated_bridging_edge():
    """
    Hand-calculated Example 3:
    Two triangles connected by a single bridging edge (3, 4):
      Clique A = {1, 2, 3}
      Clique B = {4, 5, 6}
      Edge: (3, 4)

    Mathematical Derivation:
    - N_ext(A) = {4}
    - N_ext(B) = {3}
    - overlap(A, B) = {3, 4} (size 2)
    - ICCS(A, B) = 2 / (3 + 3) = 2/6 = 1/3 approx 0.3333
    """
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6), (3, 4)])

    clique_a = (1, 2, 3)
    clique_b = (4, 5, 6)

    overlap = compute_boundary_overlap(G, clique_a, clique_b)
    assert overlap == {3, 4}

    iccs = compute_iccs(G, clique_a, clique_b)
    assert iccs == pytest.approx(1.0 / 3.0)


def test_temporal_smoothing():
    """
    Verifies exponential windowed smoothing:
    Snapshot 0: SRI = 1.0
    Snapshot 1: SRI = 2.0
    decay = 0.5, L = 2
    w_0 = 0.5, w_1 = 1.0, Z = 1.5
    smoothed = (0.5 * 1.0 + 1.0 * 2.0) / 1.5 = 2.5 / 1.5 = 5/3 approx 1.6667
    """
    history = [
        {1: 1.0, 2: 10.0},
        {1: 2.0, 2: 20.0},
    ]
    smoothed = compute_temporal_smoothed_sri(history, decay_rate=0.5, window_length=2)
    assert smoothed[1] == pytest.approx(5.0 / 3.0)
    assert smoothed[2] == pytest.approx(50.0 / 3.0)
