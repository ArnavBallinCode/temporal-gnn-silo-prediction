"""
Automated unit tests for maximal clique extraction.


"""

import pytest
import networkx as nx

from src.graph.cliques import extract_maximal_cliques, format_clique_str


def test_maximal_vs_maximum_clique():
    """
    Key educational test verifying maximal vs maximum cliques:
    Consider a graph with:
    - A 4-clique: {1, 2, 3, 4} (all pairs connected)
    - A 3-clique (triangle): {4, 5, 6} sharing node 4 with the 4-clique.

    Properties:
    - Global Maximum Clique: {1, 2, 3, 4} (size 4).
    - Maximal Cliques: BOTH {1, 2, 3, 4} (size 4) AND {4, 5, 6} (size 3) are maximal!
      Neither can be enlarged by adding any other node.
    - Our implementation must return BOTH, ranked by size.
    """
    G = nx.Graph()
    # 4-clique on 1, 2, 3, 4
    for u in [1, 2, 3, 4]:
        for v in [1, 2, 3, 4]:
            if u < v:
                G.add_edge(u, v)

    # Triangle on 4, 5, 6
    G.add_edge(4, 5)
    G.add_edge(5, 6)
    G.add_edge(4, 6)

    cliques = extract_maximal_cliques(G, min_size=3, top_k=10)

    assert len(cliques) == 2
    # First is the size 4 clique
    assert cliques[0] == (1, 2, 3, 4)
    # Second is the size 3 clique
    assert cliques[1] == (4, 5, 6)


def test_min_size_filtering():
    """
    Verifies that cliques below min_size are excluded.
    A graph with a triangle {1, 2, 3} and an isolated edge {4, 5}.
    """
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (1, 3)])  # size 3
    G.add_edge(4, 5)  # size 2

    # min_size = 3: only triangle returned
    cliques_3 = extract_maximal_cliques(G, min_size=3)
    assert cliques_3 == [(1, 2, 3)]

    # min_size = 2: both triangle and edge returned
    cliques_2 = extract_maximal_cliques(G, min_size=2)
    assert len(cliques_2) == 2
    assert cliques_2[0] == (1, 2, 3)
    assert cliques_2[1] == (4, 5)


def test_top_k_truncation():
    """
    Verifies that only top_k largest cliques are returned if top_k is specified.
    """
    G = nx.Graph()
    # 4 independent triangles
    for base in [10, 20, 30, 40]:
        G.add_edges_from([(base, base + 1), (base + 1, base + 2), (base, base + 2)])

    # top_k = 2 should return only 2 cliques
    cliques = extract_maximal_cliques(G, min_size=3, top_k=2)
    assert len(cliques) == 2


def test_empty_and_sub_threshold_graph():
    """
    Graphs smaller than min_size should return empty list.
    """
    G = nx.Graph()
    G.add_edge(1, 2)
    assert extract_maximal_cliques(G, min_size=3) == []

    empty_G = nx.Graph()
    assert extract_maximal_cliques(empty_G, min_size=3) == []


def test_format_clique_str():
    """
    Checks human-readable formatting of cliques.
    """
    labels = {1: "alice@corp.com", 2: "bob@corp.com", 3: "carol@corp.com"}
    text = format_clique_str((1, 2, 3), labels)
    assert "alice" in text
    assert "bob" in text
    assert "carol" in text
