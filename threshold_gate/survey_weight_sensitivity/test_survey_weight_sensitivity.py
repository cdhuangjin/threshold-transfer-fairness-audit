from __future__ import annotations

import numpy as np
import pytest

from survey_weight_sensitivity import select_weighted_threshold, weighted_binary_metrics


def test_weighted_threshold_uses_weighted_negative_fpr():
    negative_scores = np.array([0.1, 0.2, 0.3])
    weights = np.array([1.0, 1.0, 8.0])

    weighted = select_weighted_threshold(negative_scores, weights, target_fpr=0.85)
    unweighted = select_weighted_threshold(negative_scores, np.ones(3), target_fpr=0.85)

    assert weighted == pytest.approx(0.2)
    assert unweighted == pytest.approx(0.1)


def test_weighted_threshold_tie_is_smallest_threshold():
    negative_scores = np.array([0.1, 0.1, 0.2, 0.3])
    weights = np.ones(4)

    assert select_weighted_threshold(negative_scores, weights, target_fpr=0.5) == pytest.approx(0.1)


def test_weighted_metrics_are_finite_and_group_aligned():
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.9, 0.1, 0.7, 0.2, 0.8, 0.3])
    older = np.array([True, True, True, False, False, False])
    weights = np.array([9.0, 1.0, 1.0, 9.0, 1.0, 9.0])

    metric = weighted_binary_metrics(labels, scores, 0.5, older, weights)

    assert metric["global_fpr"] == pytest.approx(10.0 / 11.0)
    assert metric["recall"] == pytest.approx(1.0 / 19.0)
    assert metric["fpr_older"] == pytest.approx(0.9)
    assert metric["fpr_younger"] == pytest.approx(1.0)
    assert metric["pe_ratio"] == pytest.approx(0.9)
    assert metric["fpr_difference"] == pytest.approx(-0.1)
    assert metric["n"] == 6
    assert metric["n_older"] == 3
    assert metric["n_younger"] == 3
    assert all(np.isfinite(metric[key]) for key in ("brier", "logloss", "logloss_older", "logloss_younger"))


@pytest.mark.parametrize(
    "bad_weights",
    [np.array([1.0, 1.0]), np.array([1.0, 0.0, 1.0]), np.array([1.0, -1.0, 1.0]), np.array([1.0, np.nan, 1.0])],
)
def test_invalid_weights_are_rejected(bad_weights):
    with pytest.raises(ValueError):
        select_weighted_threshold(np.array([0.1, 0.2, 0.3]), bad_weights, target_fpr=0.5)

    labels = np.array([0, 0, 1])
    scores = np.array([0.1, 0.2, 0.8])
    older = np.array([True, False, True])
    with pytest.raises(ValueError):
        weighted_binary_metrics(labels, scores, 0.5, older, bad_weights)
