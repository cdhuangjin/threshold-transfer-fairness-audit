import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from threshold_gate.data import older_group, temporal_split
from threshold_gate.metrics import binary_metrics, score_diagnostics, select_threshold_from_negatives


def test_age_boundary_is_strictly_greater_than_50():
    frame = pd.DataFrame({"customer_age": [50, 51, 49]})
    assert older_group(frame).tolist() == [False, True, False]


def test_temporal_split_has_no_month_overlap():
    frame = pd.DataFrame({"month": [0, 4, 5, 6, 7], "value": range(5)})
    fit, calibration, test = temporal_split(frame)
    assert set(fit.month).isdisjoint(calibration.month)
    assert set(fit.month).isdisjoint(test.month)
    assert set(calibration.month).isdisjoint(test.month)
    assert fit.month.tolist() == [0, 4]
    assert calibration.month.tolist() == [5]
    assert test.month.tolist() == [6, 7]


def test_threshold_depends_only_on_negative_scores():
    negatives = np.array([0.1, 0.2, 0.3, 0.4])
    threshold_a = select_threshold_from_negatives(negatives, target_fpr=0.5)
    threshold_b = select_threshold_from_negatives(negatives, target_fpr=0.5)
    assert threshold_a == threshold_b == pytest.approx(0.2)


def test_binary_metrics_reports_group_fpr_and_predictive_equality():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.8, 0.9, 0.1])
    older = np.array([False, True, True, False])
    result = binary_metrics(labels, scores, threshold=0.5, older=older)
    assert result["fpr_younger"] == pytest.approx(0.0)
    assert result["fpr_older"] == pytest.approx(1.0)
    assert result["pe_ratio"] == pytest.approx(0.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["alert_rate"] == pytest.approx(0.5)


def test_score_diagnostics_reports_negative_tail_by_group():
    labels = np.array([0, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    older = np.array([False, False, True, True])
    result = score_diagnostics(labels, scores, older)
    assert result["negative_score_q95"] == pytest.approx(0.29)
    assert result["negative_score_q95_older"] == pytest.approx(0.3)
    assert result["negative_score_q95_younger"] == pytest.approx(0.195)
