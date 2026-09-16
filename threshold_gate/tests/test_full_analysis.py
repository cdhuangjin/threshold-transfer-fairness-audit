import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from analyze_full_experiment import confirmatory_rows, target_sensitivity


def _row(variant, model, seed, policy, target, scenario="clean", period="pooled_test", recall=0.5, pe=0.8):
    return {
        "variant": variant,
        "model_config": model,
        "seed": seed,
        "policy": policy,
        "target_fpr": target,
        "scenario": scenario,
        "period": period,
        "recall": recall,
        "pe_ratio": pe,
        "global_fpr": 0.05,
        "fpr_difference": 0.01,
        "alert_rate": 0.06,
    }


def test_confirmatory_rows_selects_clean_target_and_pooled_rows_across_capacities():
    frame = pd.DataFrame([
        _row("Base", "reference", 42, "fixed_m5", 0.05),
        _row("Base", "reference", 42, "fixed_m5", 0.10),
        _row("Base", "reference", 42, "fixed_m5", 0.05, period="month6"),
        _row("Base", "small", 42, "fixed_m5", 0.05),
        _row("Base", "reference", 42, "fixed_m5", 0.05, scenario="noise_0.25"),
    ])
    result = confirmatory_rows(frame)
    assert len(result) == 2
    assert set(result["period"]) == {"pooled_test", "pooled_test"}
    assert set(result["model_config"]) == {"reference", "small"}
    assert result["target_fpr"].unique().tolist() == [pytest.approx(0.05)]


def test_target_sensitivity_reports_each_frozen_target():
    rows = []
    for target in (0.01, 0.05):
        rows.extend([
            _row("Base", "small", 42, "test_oracle", target, recall=0.6, pe=0.9),
            _row("Base", "small", 42, "fixed_m5", target, recall=0.5, pe=0.8),
        ])
    result = target_sensitivity(pd.DataFrame(rows))
    assert result["target_fpr"].tolist() == [0.01, 0.05]
    assert result.loc[result["target_fpr"].eq(0.05), "median_abs_pe_gap"].iloc[0] == pytest.approx(0.1)
