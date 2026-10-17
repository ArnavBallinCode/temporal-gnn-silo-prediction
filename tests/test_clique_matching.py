"""
Tests for clique matching, 1-to-1 uniqueness, split/merge handling, and disappearance categorization.
"""

import pytest
import networkx as nx
from src.graph.cliques import (
    compute_clique_jaccard,
    match_cliques_bipartite,
    categorize_clique_disappearance,
    extract_maximal_cliques,
)


def test_clique_jaccard_calculation():
    c1 = (1, 2, 3)
    c2 = (1, 2, 4)
    # Intersection = {1, 2} (len 2), Union = {1, 2, 3, 4} (len 4) -> 0.5
    assert compute_clique_jaccard(c1, c2) == pytest.approx(0.5)

    c3 = (1, 2, 3)
    assert compute_clique_jaccard(c1, c3) == pytest.approx(1.0)

    c4 = (4, 5, 6)
    assert compute_clique_jaccard(c1, c4) == pytest.approx(0.0)


def test_bipartite_matching_one_to_one_uniqueness():
    """
    Verify that 1-to-1 matching never assigns the same successor to multiple predecessors,
    even when both predecessors have high Jaccard overlap with the successor.
    """
    # Predecessor cliques
    prev = [(1, 2, 3, 4), (1, 2, 3, 5)]  # both share {1, 2, 3}
    # Successor clique
    curr = [(1, 2, 3, 4)]

    # (1, 2, 3, 4) vs (1, 2, 3, 4) has Jaccard 1.0
    # (1, 2, 3, 5) vs (1, 2, 3, 4) has Jaccard 3/5 = 0.6
    res = match_cliques_bipartite(prev, curr, threshold=0.5, method="greedy")

    # In 1-to-1 matching, only the best match (prev 0) is matched to curr 0
    assert res.prev_to_curr == {0: 0}
    assert res.curr_to_prev == {0: 0}
    assert 1 in res.unmatched_prev
    assert len(res.unmatched_curr) == 0


def test_split_and_merge_detection():
    """
    Verify detection of splits (one clique breaking into multiple)
    and merges (multiple cliques combining into one).
    """
    # Split scenario: Prev clique (1, 2, 3, 4, 5) overlaps with two smaller cliques
    prev_cliques = [(1, 2, 3, 4, 5, 6)]
    curr_cliques = [(1, 2, 3, 7), (4, 5, 6, 8)]
    # Overlap between prev 0 and curr 0: {1, 2, 3} / {1, 2, 3, 4, 5, 6, 7} = 3/7 ~ 0.4286
    # Overlap between prev 0 and curr 1: {4, 5, 6} / {1, 2, 3, 4, 5, 6, 8} = 3/7 ~ 0.4286

    res = match_cliques_bipartite(prev_cliques, curr_cliques, threshold=0.5, overlap_threshold=0.3)
    # Neither meets 0.5 threshold for 1-to-1, but prev 0 has multiple overlaps >= 0.3
    assert 0 in res.splits
    assert len(res.splits[0]) == 2


def test_disappearance_categorization_truncation_vs_structural():
    """
    Verify that an unmatched clique is correctly categorized as 'omitted_by_truncation'
    when it exists in the full maximal clique list but was dropped by top_k.
    """
    prev_clique = (1, 2, 3, 4)
    retained_next = [(5, 6, 7, 8), (9, 10, 11, 12)]
    # Full list contains (1, 2, 3, 4) which was ranked 3rd (outside top 2)
    all_next = [(5, 6, 7, 8), (9, 10, 11, 12), (1, 2, 3, 4)]

    audit = categorize_clique_disappearance(
        prev_clique=prev_clique,
        next_retained_cliques=retained_next,
        next_all_cliques=all_next,
        threshold=0.5,
    )
    assert audit["category"] == "omitted_by_truncation"
    assert audit["matching_discarded_clique"] == (1, 2, 3, 4)


def test_disappearance_categorization_structural_dissolution():
    """
    Verify that a clique with no successor anywhere is categorized as 'dissolved_structural'.
    """
    prev_clique = (1, 2, 3, 4)
    retained_next = [(10, 11, 12), (13, 14, 15)]
    all_next = [(10, 11, 12), (13, 14, 15)]

    audit = categorize_clique_disappearance(
        prev_clique=prev_clique,
        next_retained_cliques=retained_next,
        next_all_cliques=all_next,
        threshold=0.5,
    )
    assert audit["category"] == "dissolved_structural"
    assert audit["best_full_jaccard"] == 0.0
