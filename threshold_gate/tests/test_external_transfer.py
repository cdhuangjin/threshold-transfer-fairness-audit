import numpy as np
import pandas as pd

from threshold_gate.external_transfer import (
    DOMAIN_PAIRS,
    MODEL_PARAMETERS,
    external_policy_thresholds,
    source_calibration_split,
)


def test_domain_pairs_are_frozen_and_directed():
    assert DOMAIN_PAIRS == (("CA", "PR"), ("CA", "HI"), ("MS", "HI"), ("MA", "PR"))


def test_source_calibration_split_is_disjoint_and_deterministic():
    frame = pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 0, 1, 0, 1] * 4,
            "older": [False, False, True, True] * 8,
            "x": np.arange(32),
        }
    )
    train_a, calibration_a = source_calibration_split(frame, seed=42)
    train_b, calibration_b = source_calibration_split(frame, seed=42)
    assert set(train_a.index).isdisjoint(set(calibration_a.index))
    assert train_a.index.tolist() == train_b.index.tolist()
    assert calibration_a.index.tolist() == calibration_b.index.tolist()
    assert set(train_a.index) | set(calibration_a.index) == set(frame.index)


def test_external_thresholds_keep_oracle_and_source_calibration_separate():
    thresholds = external_policy_thresholds(
        calibration_scores=np.array([0.1, 0.2, 0.3, 0.4]),
        calibration_labels=np.array([0, 0, 1, 1]),
        target_scores=np.array([0.05, 0.15, 0.25, 0.9]),
        target_labels=np.array([0, 0, 1, 1]),
        target_fpr=0.05,
    )
    assert set(thresholds) == {"target_oracle", "source_calibrated"}
    assert thresholds["target_oracle"] != thresholds["source_calibrated"]


def test_model_parameter_capacities_are_fixed():
    assert tuple(MODEL_PARAMETERS) == ("small", "reference", "large")
    assert MODEL_PARAMETERS["small"]["num_leaves"] < MODEL_PARAMETERS["reference"]["num_leaves"] < MODEL_PARAMETERS["large"]["num_leaves"]
