"""
Unit tests for the statistics and target handling used in revision2_experiments.py.
"""

import math

import numpy as np
import pytest

from src.graph.cliques import match_cliques_bipartite
from src.training.revision2_experiments import dm_test, holm, mbb_ratio_ci, pair_equivalence


def test_dm_test_zero_mean_differential_is_not_significant():
    rng = np.random.default_rng(0)
    d = rng.normal(0.0, 1.0, 200)
    d -= d.mean()
    stat, p = dm_test(d)
    assert abs(stat) < 1e-8
    assert p > 0.99


def test_dm_test_detects_consistent_loss_difference():
    rng = np.random.default_rng(1)
    d = 1.0 + rng.normal(0.0, 0.2, 40)
    stat, p = dm_test(d)
    assert stat > 0
    assert p < 1e-6


def test_dm_test_constant_series_returns_nan():
    stat, p = dm_test(np.zeros(10))
    assert math.isnan(stat) and math.isnan(p)


def test_holm_adjustment_is_monotone_and_capped():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
    assert math.isnan(adj["d"])
    assert all(v <= 1.0 for v in adj.values() if not math.isnan(v))


def test_mbb_ratio_ci_brackets_one_for_identical_errors():
    rng = np.random.default_rng(2)
    se = [rng.exponential(1.0, 10) for _ in range(20)]
    lo, hi = mbb_ratio_ci(se, se)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_pair_equivalence_flags_clearly_better_model():
    rng = np.random.default_rng(3)
    per_origin = {"good": [], "bad": []}
    for _ in range(24):
        y = rng.normal(5, 1, 10)
        per_origin["good"].append((y, y + rng.normal(0, 0.5, 10)))
        per_origin["bad"].append((y, y + rng.normal(0, 2.0, 10)))
    res = pair_equivalence(per_origin, "good", "bad")
    assert res["verdict"] == "a better"
    assert res["ratio"] < 1


def test_one_to_one_matching_respects_threshold_and_uniqueness():
    prev = [(1, 2, 3, 4), (1, 2, 3, 5)]
    curr = [(1, 2, 3, 4)]
    res = match_cliques_bipartite(prev, curr, threshold=0.5, method="greedy")
    # both predecessors overlap the successor with J >= 0.5, but only one may claim it
    assert len(res.prev_to_curr) == 1
    assert res.prev_to_curr == {0: 0}
    # a four-member clique that swaps two members falls below tau = 0.5
    res2 = match_cliques_bipartite([(1, 2, 3, 4)], [(1, 2, 7, 8)], threshold=0.5, method="greedy")
    assert res2.prev_to_curr == {}
