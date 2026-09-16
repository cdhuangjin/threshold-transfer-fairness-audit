import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from threshold_gate.analysis import capacity_ranking_turnover, primary_effects, ranking_turnover_ci


def test_primary_effects_pair_oracle_and_fixed_rows():
    rows = []
    for config, oracle_pe, fixed_pe in (("small", 0.2, 0.7), ("reference", 0.4, 0.4)):
        for policy, pe in (("test_oracle", oracle_pe), ("fixed_m5", fixed_pe)):
            rows.append({
                "variant": "Base",
                "model_config": config,
                "seed": 42,
                "policy": policy,
                "period": "pooled_test",
                "pe_ratio": pe,
                "recall": 0.5,
            })
    effects = primary_effects(pd.DataFrame(rows))
    assert effects[["variant", "model_config", "seed"]].to_dict("records") == [
        {"variant": "Base", "model_config": "small", "seed": 42},
        {"variant": "Base", "model_config": "reference", "seed": 42},
    ]
    assert effects["abs_pe_gap"].tolist() == pytest.approx([0.5, 0.0])
    assert effects["signed_pe_gap"].tolist() == pytest.approx([-0.5, 0.0])


def test_capacity_ranking_turnover_counts_changed_order():
    rows = []
    for config, oracle_recall, fixed_recall in (("small", 0.6, 0.8), ("reference", 0.8, 0.6), ("large", 0.7, 0.7)):
        for policy, recall in (("test_oracle", oracle_recall), ("fixed_m5", fixed_recall)):
            rows.append({
                "variant": "Base",
                "seed": 42,
                "model_config": config,
                "policy": policy,
                "period": "pooled_test",
                "recall": recall,
            })
    result = capacity_ranking_turnover(pd.DataFrame(rows))
    assert result["units"] == 1
    assert result["changed_units"] == 1
    assert result["turnover"] == 1.0


def test_ranking_turnover_ci_is_bounded_and_deterministic():
    rows = []
    for seed, changed in ((42, True), (314, False), (2718, True), (1618, False)):
        values = ((0.8, 0.6), (0.6, 0.8), (0.8, 0.6)) if changed else ((0.8, 0.7), (0.6, 0.5), (0.7, 0.6))
        for config, (oracle, fixed) in zip(("small", "reference", "large"), values):
            for policy, recall in (("test_oracle", oracle), ("fixed_m5", fixed)):
                rows.append({
                    "variant": "Base",
                    "seed": seed,
                    "model_config": config,
                    "policy": policy,
                    "period": "pooled_test",
                    "recall": recall,
                })
    frame = pd.DataFrame(rows)
    first = ranking_turnover_ci(frame, replicates=500, seed=7)
    second = ranking_turnover_ci(frame, replicates=500, seed=7)
    assert first == second
    assert first["turnover"] == pytest.approx(0.5)
    assert 0.0 <= first["ci_low"] <= first["turnover"] <= first["ci_high"] <= 1.0
