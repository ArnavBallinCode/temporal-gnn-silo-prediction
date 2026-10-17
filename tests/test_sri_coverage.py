"""
Unit tests for SRI coverage, zero-participation breakdown, and complementary structural metrics.
"""

import pytest
import networkx as nx

from src.graph.scores import (
    compute_silo_risk_index,
    compute_clique_tensions,
    audit_sri_coverage,
    compute_boundary_bridging_index,
)
from src.graph.cliques import extract_maximal_cliques


def test_sri_coverage_audit_breakdown():
    """
    Verify exact breakdown of employee nodes across:
    - Inactive (degree 0)
    - Dyad-only (degree > 0, no clique >= 3)
    - Truncation-omitted
    - Retained clique members
    """
    G = nx.Graph()
    # Clique 1 (retained): (1, 2, 3)
    G.add_edges_from([(1, 2), (2, 3), (1, 3)])
    # Dyad-only: (4, 5) with degree 1 each, no triangles
    G.add_edge(4, 5)
    # Bridge between (1, 2, 3) and dyad
    G.add_edge(3, 4)

    all_nodes = [1, 2, 3, 4, 5, 6]  # node 6 is inactive
    retained_cliques = [(1, 2, 3)]
    all_max_cliques = [(1, 2, 3)]  # (3, 4) and (4, 5) are size 2, so no size >= 3 cliques omitted

    audit = audit_sri_coverage(G, retained_cliques, all_max_cliques, all_nodes)

    assert audit["total_nodes"] == 6
    assert audit["num_graph_communicators"] == 5  # 1, 2, 3, 4, 5
    assert audit["num_inactive_nodes"] == 1       # node 6
    assert audit["num_dyad_only_nodes"] == 2      # 4, 5 are communicating but in no size >= 3 clique
    assert audit["dyad_only_nodes"] == [4, 5]
    assert audit["num_retained_clique_nodes"] == 3 # 1, 2, 3


def test_boundary_bridging_index_captures_non_clique_brokers():
    """
    Verify that Boundary Bridging Index (BBI) produces positive scores for non-clique bridging nodes,
    providing a complementary measure without altering original SRI.
    """
    G = nx.Graph()
    # Two separate cliques bridged by node 4
    # Clique A: (1, 2, 3)
    G.add_edges_from([(1, 2), (2, 3), (1, 3)])
    # Clique B: (5, 6, 7)
    G.add_edges_from([(5, 6), (6, 7), (5, 7)])
    # Node 4 bridges between node 3 and node 5 (node 4 is not in any triangle!)
    G.add_edges_from([(3, 4), (4, 5)])

    all_nodes = [1, 2, 3, 4, 5, 6, 7, 8]  # node 8 inactive

    bbi = compute_boundary_bridging_index(G, all_nodes)
    sri = compute_silo_risk_index([(1, 2, 3), (5, 6, 7)], {(1, 2, 3): 1.0, (5, 6, 7): 1.0}, all_nodes)

    # Node 4 is NOT in any clique, so original SRI must be 0.0
    assert sri[4] == 0.0

    # Node 4 has 2 neighbors (3 and 5) that do not talk to each other (clustering = 0)
    # BBI should be positive!
    assert bbi[4] > 0.0

    # Inactive node 8 has 0.0 for both
    assert sri[8] == 0.0
    assert bbi[8] == 0.0
