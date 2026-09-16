from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"fraud_bool", "customer_age", "month"}
CATEGORICAL_COLUMNS = (
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
)


def validate_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if frame["fraud_bool"].isna().any() or not set(frame["fraud_bool"].unique()).issubset({0, 1}):
        raise ValueError("fraud_bool must be a complete binary column")
    if frame["customer_age"].isna().any() or not np.isfinite(frame["customer_age"]).all():
        raise ValueError("customer_age must be finite")
    if frame["month"].isna().any() or not set(frame["month"].unique()).issubset(set(range(8))):
        raise ValueError("month must contain only integers from 0 to 7")
    missing_categoricals = set(CATEGORICAL_COLUMNS).difference(frame.columns)
    if missing_categoricals:
        raise ValueError(f"missing categorical columns: {sorted(missing_categoricals)}")


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_frame(frame)
    return frame


def older_group(frame: pd.DataFrame) -> pd.Series:
    return frame["customer_age"].gt(50)


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "month" not in frame.columns:
        raise ValueError("month is required for temporal splitting")
    if frame["month"].isna().any() or not set(frame["month"].unique()).issubset(set(range(8))):
        raise ValueError("month must contain only integers from 0 to 7")
    fit = frame.loc[frame["month"].isin((0, 1, 2, 3, 4))].copy()
    calibration = frame.loc[frame["month"].eq(5)].copy()
    test = frame.loc[frame["month"].isin((6, 7))].copy()
    return fit, calibration, test


def encode_like_training(frame: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals using maps learned only from the fitting period."""
    encoded = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        categories = pd.Index(training[column].drop_duplicates())
        mapping = pd.Series(np.arange(len(categories), dtype=np.int32), index=categories)
        values = encoded[column].map(mapping)
        if values.isna().any():
            unseen = sorted(set(encoded.loc[values.isna(), column].dropna().tolist()))
            raise ValueError(f"unseen {column} values outside fit period: {unseen}")
        encoded[column] = values.astype(np.int32)
    return encoded
