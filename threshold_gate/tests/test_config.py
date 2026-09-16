import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from threshold_gate.config import GateConfig


def test_frozen_gate_grid_and_budget():
    cfg = GateConfig()
    assert cfg.datasets == ("Base", "Variant I", "Variant II", "Variant III", "Variant IV", "Variant V")
    assert cfg.policies == ("test_oracle", "fixed_m5", "lag1")
    assert cfg.seeds == (42, 314, 2718)
    assert tuple(cfg.model_configs) == ("small", "reference", "large")
    assert cfg.fit_months == (0, 1, 2, 3, 4)
    assert cfg.calibration_months == (5,)
    assert cfg.test_months == (6, 7)
    assert len(cfg.datasets) * len(cfg.model_configs) * len(cfg.seeds) == 54
    assert cfg.max_wall_hours == 4
    assert cfg.max_cpu_hours == 8
    assert cfg.max_ram_gb == 12
    assert cfg.max_artifact_gb == 5
