import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd

from threshold_gate.full_runner import corrupt_numeric_features, expanded_policy_thresholds, fit_key, noise_seed
from run_full_experiment import load_resume_checkpoint, write_fit_checkpoint


def test_expanded_policy_thresholds_keep_clean_history_separate_from_test_oracle():
    scores = {
        5: np.array([0.1, 0.2, 0.3, 0.9]),
        6: np.array([0.2, 0.4, 0.5, 0.8]),
        7: np.array([0.05, 0.15, 0.6, 0.7]),
    }
    labels = {
        5: np.array([0, 0, 0, 1]),
        6: np.array([0, 0, 0, 1]),
        7: np.array([0, 0, 0, 1]),
    }
    result = expanded_policy_thresholds(scores, labels, target_fpr=0.5)
    assert result["fixed_m5"][6] == result["fixed_m5"][7]
    assert result["lag1"][6] == result["fixed_m5"][6]
    assert result["test_oracle"][6] == result["test_oracle"][7]


def test_noise_seed_is_deterministic_and_severity_specific():
    assert noise_seed(42, 0.25) == noise_seed(42, 0.25)
    assert noise_seed(42, 0.25) != noise_seed(42, 0.50)


def test_corrupt_numeric_features_accepts_integer_source_columns_without_schema_change():
    frame = pd.DataFrame({"integer_feature": [1, 2, 3], "month": [6, 6, 7], "customer_age": [40, 51, 52]})
    fit = frame.iloc[:2].copy()
    result = corrupt_numeric_features(frame, fit, severity=0.25, seed=42)
    assert list(result.columns) == list(frame.columns)
    assert result.shape == frame.shape
    assert result["integer_feature"].dtype.kind == "f"
    assert result["month"].tolist() == frame["month"].tolist()
    assert result["customer_age"].tolist() == frame["customer_age"].tolist()


def test_fit_key_is_stable_for_resume_and_unique_across_grid_cells():
    assert fit_key("Base", "reference", 42) == "Base|reference|42"
    assert fit_key("Base", "reference", 42) != fit_key("Base", "reference", 314)


def test_resume_checkpoint_round_trip_preserves_completed_fit_rows(tmp_path):
    rows = [{"variant": "Base", "model_config": "small", "seed": 42, "value": 1}]
    path = tmp_path / "checkpoint.json"
    write_fit_checkpoint(path, rows, {"Base|small|42"}, expected_fit_count=2)
    loaded_rows, completed = load_resume_checkpoint(path, expected_fit_count=2)
    assert loaded_rows == rows
    assert completed == {"Base|small|42"}
