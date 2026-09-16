from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import MODEL_PARAMETERS
from .data import CATEGORICAL_COLUMNS, encode_like_training, older_group, temporal_split
from .metrics import binary_metrics, score_diagnostics, select_threshold_from_negatives


def expanded_policy_thresholds(
    scores_by_month: dict[int, np.ndarray],
    labels_by_month: dict[int, np.ndarray],
    target_fpr: float = 0.05,
) -> dict[str, dict[int, float]]:
    m5 = select_threshold_from_negatives(scores_by_month[5][labels_by_month[5] == 0], target_fpr)
    m6 = select_threshold_from_negatives(scores_by_month[6][labels_by_month[6] == 0], target_fpr)
    test_scores = np.concatenate([scores_by_month[6], scores_by_month[7]])
    test_labels = np.concatenate([labels_by_month[6], labels_by_month[7]])
    oracle = select_threshold_from_negatives(test_scores[test_labels == 0], target_fpr)
    return {
        "test_oracle": {6: oracle, 7: oracle},
        "fixed_m5": {6: m5, 7: m5},
        "lag1": {6: m5, 7: m6},
    }


def noise_seed(seed: int, severity: float) -> int:
    return int((seed * 1_000_003 + round(severity * 1_000) * 9_176) % (2**32 - 1))


def fit_key(variant: str, model_name: str, seed: int) -> str:
    return f"{variant}|{model_name}|{int(seed)}"


def corrupt_numeric_features(test_frame: pd.DataFrame, fit_frame: pd.DataFrame, severity: float, seed: int) -> pd.DataFrame:
    """Add deterministic fit-scale Gaussian noise without changing protected fields."""
    if severity < 0:
        raise ValueError("severity must be non-negative")
    protected = {"month", "customer_age", "fraud_bool"}
    columns = [
        column for column in test_frame.columns
        if column not in protected and pd.api.types.is_numeric_dtype(test_frame[column])
    ]
    result = test_frame.copy()
    if not columns or severity == 0:
        return result
    scales = fit_frame[columns].std(ddof=0).replace(0, 1.0).fillna(1.0).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    values = result[columns].to_numpy(dtype=float) + rng.normal(0.0, severity * scales, size=(len(result), len(columns)))
    result[columns] = result[columns].astype(float)
    result.loc[:, columns] = values
    return result


def _append_policy_rows(
    rows: list[dict],
    predictions: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    older: dict[int, np.ndarray],
    thresholds: dict[str, dict[int, float]],
    variant: str,
    model_name: str,
    seed: int,
    elapsed: float,
    target_fpr: float,
    scenario: str,
) -> None:
    source_months = {
        "test_oracle": [6, 7],
        "fixed_m5": [5],
        "lag1": [5],
    }
    for policy, by_month in thresholds.items():
        for period, months in (("month6", (6,)), ("month7", (7,)), ("pooled_test", (6, 7))):
            y = np.concatenate([labels[m] for m in months])
            score = np.concatenate([predictions[m] for m in months])
            group = np.concatenate([older[m] for m in months])
            threshold_values = np.concatenate([np.full(len(labels[m]), by_month[m]) for m in months])
            metric = binary_metrics(y, score, threshold=threshold_values, older=group)
            diag = score_diagnostics(y, score, group)
            if policy == "lag1":
                policy_sources = [5] if months == (6,) else ([6] if months == (7,) else [5, 6])
            else:
                policy_sources = source_months[policy]
            rows.append({
                "variant": variant,
                "model_config": model_name,
                "seed": int(seed),
                "policy": policy,
                "period": period,
                "scenario": scenario,
                "target_fpr": float(target_fpr),
                "threshold": float(by_month[months[0]]) if len(months) == 1 else (float(by_month[6]) if policy != "lag1" else None),
                "threshold_by_month": {str(month): float(by_month[month]) for month in months},
                "threshold_source_months": policy_sources,
                "elapsed_seconds": float(elapsed),
                **metric,
                **diag,
            })


def run_full_one_model(
    frame: pd.DataFrame,
    variant: str,
    model_name: str,
    seed: int,
    target_fprs: tuple[float, ...],
    noise_severities: tuple[float, ...],
) -> list[dict]:
    fit, calibration, test = temporal_split(frame)
    encoded_fit = encode_like_training(fit, fit).drop(columns=["fraud_bool"])
    encoded_calibration = encode_like_training(calibration, fit).drop(columns=["fraud_bool"])
    encoded_test = encode_like_training(test, fit).drop(columns=["fraud_bool"])
    model = lgb.LGBMClassifier(
        **MODEL_PARAMETERS[model_name],
        boosting_type="gbdt",
        enable_bundle=True,
        n_jobs=10,
        random_state=seed,
        verbosity=-1,
    )
    started = __import__("time").perf_counter()
    model.fit(encoded_fit, fit["fraud_bool"].to_numpy(dtype=np.int8), categorical_feature=list(CATEGORICAL_COLUMNS))
    predictions = {
        5: model.predict_proba(encoded_calibration)[:, 1],
        6: model.predict_proba(encoded_test.loc[test["month"].eq(6)])[:, 1],
        7: model.predict_proba(encoded_test.loc[test["month"].eq(7)])[:, 1],
    }
    labels = {
        5: calibration["fraud_bool"].to_numpy(dtype=np.int8),
        6: test.loc[test["month"].eq(6), "fraud_bool"].to_numpy(dtype=np.int8),
        7: test.loc[test["month"].eq(7), "fraud_bool"].to_numpy(dtype=np.int8),
    }
    older = {
        5: older_group(calibration).to_numpy(dtype=bool),
        6: older_group(test.loc[test["month"].eq(6)]).to_numpy(dtype=bool),
        7: older_group(test.loc[test["month"].eq(7)]).to_numpy(dtype=bool),
    }
    elapsed = __import__("time").perf_counter() - started
    rows: list[dict] = []
    for target_fpr in target_fprs:
        thresholds = expanded_policy_thresholds(predictions, labels, target_fpr)
        _append_policy_rows(rows, predictions, labels, older, thresholds, variant, model_name, seed, elapsed, target_fpr, "clean")

    if model_name == "reference":
        for severity in noise_severities:
            corrupted = corrupt_numeric_features(encoded_test, encoded_fit, severity, noise_seed(seed, severity))
            corrupt_predictions = {
                5: predictions[5],
                6: model.predict_proba(corrupted.loc[test["month"].eq(6)])[:, 1],
                7: model.predict_proba(corrupted.loc[test["month"].eq(7)])[:, 1],
            }
            thresholds = expanded_policy_thresholds(corrupt_predictions, labels, 0.05)
            _append_policy_rows(
                rows,
                corrupt_predictions,
                labels,
                older,
                thresholds,
                variant,
                model_name,
                seed,
                elapsed,
                0.05,
                f"noise_{severity:.2f}",
            )
    return rows
