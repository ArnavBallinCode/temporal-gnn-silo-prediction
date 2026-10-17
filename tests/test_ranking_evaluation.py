"""
Unit tests for Ranking Evaluation, active node partitioning, and tie handling.
"""

import pytest
import numpy as np

from src.training.metrics import compute_ranking_metrics, compute_comprehensive_ranking_metrics


def test_ranking_metrics_active_vs_all():
    """
    Verify that compute_ranking_metrics correctly separates all nodes from active nodes,
    preventing spurious high rank correlation caused by matching large pools of zero scores.
    """
    # 10 nodes: nodes 0..4 have positive scores, nodes 5..9 are inactive (0.0)
    true_sri = {0: 10.0, 1: 8.0, 2: 6.0, 3: 4.0, 4: 2.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}
    # Predicted scores: perfectly identifies 5..9 as 0.0, but inverts the order of 0..4!
    pred_sri = {0: 2.0, 1: 4.0, 2: 6.0, 3: 8.0, 4: 10.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}

    # When evaluated across all 10 nodes, separating non-zero from zero yields positive correlation
    m_all = compute_ranking_metrics(true_sri, pred_sri, top_k=5, active_only=False)
    # When evaluated strictly on active nodes (0..4), correlation is inverted (-1.0)
    m_act = compute_ranking_metrics(true_sri, pred_sri, top_k=5, active_only=True)

    assert m_all["spearman_rho"] > 0.0
    assert m_act["spearman_rho"] == pytest.approx(-1.0)
    assert m_act["num_nodes"] == 5


def test_ranking_tie_handling():
    """
    Verify that rank ties are handled deterministically without errors or NaNs.
    """
    true_sri = {0: 5.0, 1: 5.0, 2: 5.0, 3: 1.0}
    pred_sri = {0: 2.0, 1: 2.0, 2: 1.0, 3: 0.0}

    res = compute_ranking_metrics(true_sri, pred_sri, top_k=2)
    assert not np.isnan(res["spearman_rho"])
    assert not np.isnan(res["p_value"])
