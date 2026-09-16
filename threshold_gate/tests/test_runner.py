import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from threshold_gate.runner import policy_thresholds


def test_threshold_policies_use_only_their_declared_history():
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
    result = policy_thresholds(scores, labels, target_fpr=0.5)
    assert result["fixed_m5"][6] == result["fixed_m5"][7]
    assert result["lag1"][6] == result["fixed_m5"][6]
    assert result["lag1"][7] != result["lag1"][6]
    assert result["test_oracle"][6] == result["test_oracle"][7]
