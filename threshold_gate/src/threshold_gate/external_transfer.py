from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from folktables import ACSIncome
from sklearn.model_selection import StratifiedShuffleSplit

from .config import MODEL_PARAMETERS
from .metrics import binary_metrics, select_threshold_from_negatives


DOMAIN_PAIRS = (("CA", "PR"), ("CA", "HI"), ("MS", "HI"), ("MA", "PR"))
STATE_FILES = {
    "CA": "psam_p06.csv",
    "PR": "psam_p72.csv",
    "MS": "psam_p28.csv",
    "HI": "psam_p15.csv",
    "MA": "psam_p25.csv",
}
STATE_HASHES = {
    "CA": "DC2187FC90DF2C5F6B546EE89A2B41C9A97379C9E7136461B6A6C8DE871B43E0",
    "PR": "74436D4FFEE9E5974982AFF45622E49A192DE2F22DE528E2156C7EC13832E244",
    "MS": "2944DF2DBDC582720746F4A29E8FC47B409B06E65A53C952A5CB1E4691CB0A70",
    "HI": "57F821DD8C4190FAD8D677948028D75E200A4537C0A2944AFFD9CB50C88DCB8A",
    "MA": "47BE6D5A3636D3A9A02450239E93D8D25A6D86D25B108F6E6A5A786169F4DD21",
}
FEATURE_COLUMNS = ("AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P")
CATEGORICAL_COLUMNS = ("COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P")
RAW_COLUMNS = (*FEATURE_COLUMNS, "PINCP", "PWGTP")


def load_acs_state(path: Path) -> pd.DataFrame:
    """Load the public ACS state file with the official ACSIncome filter."""
    if tuple(ACSIncome.features) != FEATURE_COLUMNS or ACSIncome.target != "PINCP":
        raise RuntimeError("installed folktables ACSIncome definition does not match the frozen protocol")
    frame = pd.read_csv(path, usecols=list(RAW_COLUMNS))
    frame = frame.loc[
        frame["AGEP"].gt(16)
        & frame["PINCP"].gt(100)
        & frame["WKHP"].gt(0)
        & frame["PWGTP"].ge(1)
    ].copy()
    if frame.loc[:, list(FEATURE_COLUMNS + ("PINCP", "PWGTP"))].isna().any().any():
        raise ValueError(f"missing values after official filter: {path}")
    frame["y"] = (frame["PINCP"] > 50_000).astype(np.int8)
    frame["older"] = frame["AGEP"].gt(50)
    return frame.reset_index(drop=True)


def source_calibration_split(frame: pd.DataFrame, seed: int, calibration_fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic, disjoint source train/calibration split."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must lie in (0, 1)")
    strata = frame["y"].astype(str) + "|" + frame["older"].astype(int).astype(str)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=calibration_fraction, random_state=seed)
    train_idx, calibration_idx = next(splitter.split(frame, strata))
    return frame.iloc[train_idx].copy(), frame.iloc[calibration_idx].copy()


def _encode_like_training(frame: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    encoded = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_COLUMNS:
        categories = pd.Index(training[column].drop_duplicates())
        mapping = pd.Series(np.arange(len(categories), dtype=np.int32), index=categories)
        encoded[column] = encoded[column].map(mapping).fillna(-1).astype(np.int32)
    return encoded


def external_policy_thresholds(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    target_scores: np.ndarray,
    target_labels: np.ndarray,
    target_fpr: float,
) -> dict[str, float]:
    calibration_negative = calibration_scores[np.asarray(calibration_labels, dtype=int) == 0]
    target_negative = target_scores[np.asarray(target_labels, dtype=int) == 0]
    return {
        "target_oracle": select_threshold_from_negatives(target_negative, target_fpr),
        "source_calibrated": select_threshold_from_negatives(calibration_negative, target_fpr),
    }


def _fit_model(training: pd.DataFrame, model_name: str, seed: int) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        **MODEL_PARAMETERS[model_name],
        boosting_type="gbdt",
        enable_bundle=True,
        n_jobs=10,
        random_state=seed,
        verbosity=-1,
    )
    x_train = _encode_like_training(training, training)
    model.fit(x_train, training["y"].to_numpy(dtype=np.int8), categorical_feature=list(CATEGORICAL_COLUMNS))
    return model


def run_external_one_model(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_name: str,
    target_name: str,
    model_name: str,
    seed: int,
    target_fprs: tuple[float, ...],
) -> list[dict]:
    train, calibration = source_calibration_split(source, seed)
    model_started = time.perf_counter()
    model = _fit_model(train, model_name, seed)
    calibration_scores = model.predict_proba(_encode_like_training(calibration, train))[:, 1]
    target_scores = model.predict_proba(_encode_like_training(target, train))[:, 1]
    elapsed = time.perf_counter() - model_started
    target_labels = target["y"].to_numpy(dtype=np.int8)
    target_older = target["older"].to_numpy(dtype=bool)
    calibration_labels = calibration["y"].to_numpy(dtype=np.int8)
    rows: list[dict] = []
    for target_fpr in target_fprs:
        thresholds = external_policy_thresholds(
            calibration_scores,
            calibration_labels,
            target_scores,
            target_labels,
            target_fpr,
        )
        for policy, threshold in thresholds.items():
            metric = binary_metrics(target_labels, target_scores, threshold, target_older)
            rows.append(
                {
                    "pair": f"{source_name}->{target_name}",
                    "source": source_name,
                    "target": target_name,
                    "model_config": model_name,
                    "seed": int(seed),
                    "policy": policy,
                    "target_fpr": float(target_fpr),
                    "threshold": float(threshold),
                    "threshold_source": "target_labels" if policy == "target_oracle" else "source_calibration_labels",
                    "source_train_rows": int(len(train)),
                    "source_calibration_rows": int(len(calibration)),
                    "target_rows": int(len(target)),
                    "elapsed_seconds": float(elapsed),
                    **metric,
                }
            )
    return rows
