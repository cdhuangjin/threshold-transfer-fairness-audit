from dataclasses import dataclass


MODEL_PARAMETERS = {
    "small": {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "max_bin": 255,
    },
    "reference": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "max_bin": 255,
    },
    "large": {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "max_bin": 255,
    },
}


@dataclass(frozen=True, slots=True)
class GateConfig:
    datasets: tuple[str, ...] = (
        "Base",
        "Variant I",
        "Variant II",
        "Variant III",
        "Variant IV",
        "Variant V",
    )
    policies: tuple[str, ...] = ("test_oracle", "fixed_m5", "lag1")
    model_configs: tuple[str, ...] = ("small", "reference", "large")
    seeds: tuple[int, ...] = (42, 314, 2718)
    fit_months: tuple[int, ...] = (0, 1, 2, 3, 4)
    calibration_months: tuple[int, ...] = (5,)
    test_months: tuple[int, ...] = (6, 7)
    target_fpr: float = 0.05
    bootstrap_replicates: int = 2_000
    max_wall_hours: float = 4.0
    max_cpu_hours: float = 8.0
    max_ram_gb: float = 12.0
    max_artifact_gb: float = 5.0
