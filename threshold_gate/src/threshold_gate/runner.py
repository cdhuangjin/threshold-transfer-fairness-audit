from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import MODEL_PARAMETERS
from .data import CATEGORICAL_COLUMNS, encode_like_training, load_dataset, older_group, temporal_split
from .metrics import binary_metrics, score_diagnostics, select_threshold_from_negatives


def policy_thresholds(
    scores_by_month: dict[int, np.ndarray],
    labels_by_month: dict[int, np.ndarray],
    target_fpr: float = 0.05,
) -> dict[str, dict[int, float]]:
    """Return thresholds and make each policy's label dependency explicit."""
    m5 = select_threshold_from_negatives(scores_by_month[5][labels_by_month[5] == 0], target_fpr)
    m6 = select_threshold_from_negatives(scores_by_month[6][labels_by_month[6] == 0], target_fpr)
    pooled_scores = np.concatenate([scores_by_month[6], scores_by_month[7]])
    pooled_labels = np.concatenate([labels_by_month[6], labels_by_month[7]])
    oracle = select_threshold_from_negatives(pooled_scores[pooled_labels == 0], target_fpr)
    return {
        "test_oracle": {6: oracle, 7: oracle},
        "fixed_m5": {6: m5, 7: m5},
        "lag1": {6: m5, 7: m6},
    }


def _hash_config(model_name: str, seed: int) -> str:
    payload = json.dumps({"model_name": model_name, "seed": seed, "params": MODEL_PARAMETERS[model_name]}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_frame(frame: pd.DataFrame, fit_frame: pd.DataFrame) -> pd.DataFrame:
    encoded = encode_like_training(frame, fit_frame)
    return encoded.drop(columns=["fraud_bool"])


def run_one_model(frame: pd.DataFrame, variant: str, model_name: str, seed: int, target_fpr: float) -> list[dict]:
    fit, calibration, test = temporal_split(frame)
    encoded_fit = _model_frame(fit, fit)
    encoded_calibration = _model_frame(calibration, fit)
    encoded_test = _model_frame(test, fit)
    fit_y = fit["fraud_bool"].to_numpy(dtype=np.int8)

    params = {
        **MODEL_PARAMETERS[model_name],
        "boosting_type": "gbdt",
        "enable_bundle": True,
        "n_jobs": 10,
        "random_state": seed,
        "verbosity": -1,
    }
    started = time.perf_counter()
    model = lgb.LGBMClassifier(**params)
    model.fit(encoded_fit, fit_y, categorical_feature=list(CATEGORICAL_COLUMNS))
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
    thresholds = policy_thresholds(predictions, labels, target_fpr)
    elapsed = time.perf_counter() - started
    rows: list[dict] = []
    for policy, by_month in thresholds.items():
        for period, months in (("month6", (6,)), ("month7", (7,)), ("pooled_test", (6, 7))):
            y = np.concatenate([labels[m] for m in months])
            score = np.concatenate([predictions[m] for m in months])
            group = np.concatenate([older[m] for m in months])
            threshold_values = np.concatenate([np.full(len(labels[m]), by_month[m]) for m in months])
            metric = binary_metrics(y, score, threshold=threshold_values, older=group)
            rows.append({
                "variant": variant,
                "model_config": model_name,
                "seed": int(seed),
                "policy": policy,
                "period": period,
                "threshold": float(by_month[months[0]]) if len(set(months)) == 1 else (float(by_month[6]) if policy != "lag1" else None),
                "threshold_by_month": {str(month): float(by_month[month]) for month in months},
                "threshold_source_months": {
                    "test_oracle": [6, 7],
                    "fixed_m5": [5],
                    "lag1": [5] if months == (6,) else ([6] if months == (7,) else [5, 6]),
                }[policy],
                "config_hash": _hash_config(model_name, seed),
                "elapsed_seconds": elapsed,
                **metric,
                **score_diagnostics(y, score, group),
            })
    return rows


def run_variant(path: Path, variant: str, model_configs: tuple[str, ...], seeds: tuple[int, ...], target_fpr: float) -> list[dict]:
    frame = load_dataset(path)
    rows: list[dict] = []
    for model_name in model_configs:
        for seed in seeds:
            rows.extend(run_one_model(frame, variant, model_name, seed, target_fpr))
    return rows
