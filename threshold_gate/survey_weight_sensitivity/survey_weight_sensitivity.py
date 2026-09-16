from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss


def _validate_weights(weights: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) != size:
        raise ValueError("weights must be an aligned one-dimensional vector")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("weights must be finite and strictly positive")
    if values.sum() <= 0:
        raise ValueError("weights must have a positive sum")
    return values


def select_weighted_threshold(
    negative_scores: np.ndarray,
    negative_weights: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """Select the highest attainable weighted negative rate below target FPR."""
    scores = np.asarray(negative_scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("negative_scores must be a non-empty vector")
    if not np.isfinite(scores).all():
        raise ValueError("negative_scores must be finite")
    weights = _validate_weights(negative_weights, len(scores))
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must lie strictly between zero and one")

    candidates = np.unique(scores)
    denominator = float(weights.sum())
    weighted_fprs = np.array([weights[scores > threshold].sum() / denominator for threshold in candidates])
    eligible = weighted_fprs <= target_fpr + 1e-15
    if not eligible.any():
        raise ValueError("no finite threshold satisfies target_fpr")
    closest_fpr = weighted_fprs[eligible].max()
    tied = candidates[eligible][np.isclose(weighted_fprs[eligible], closest_fpr, rtol=0, atol=1e-15)]
    return float(tied.min())


def _weighted_rate(predicted: np.ndarray, weights: np.ndarray) -> float | None:
    if len(predicted) == 0:
        return None
    return float(weights[predicted].sum() / weights.sum())


def _weighted_group_logloss(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float | None:
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        return None
    return float(log_loss(labels, scores, labels=[0, 1], sample_weight=weights))


def weighted_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    older: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | None]:
    """Compute the primary binary metrics using positive finite case weights."""
    raw_labels = np.asarray(labels)
    raw_scores = np.asarray(scores, dtype=float)
    raw_older = np.asarray(older, dtype=bool)
    if not (raw_labels.ndim == raw_scores.ndim == raw_older.ndim == 1 and len(raw_labels) == len(raw_scores) == len(raw_older)):
        raise ValueError("labels, scores, and older must be aligned vectors")
    if not np.isfinite(raw_labels).all() or not np.isfinite(raw_scores).all() or not set(np.unique(raw_labels)).issubset({0, 1}):
        raise ValueError("labels must be binary and scores finite")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    weights_array = _validate_weights(weights, len(raw_labels))
    labels_array = raw_labels.astype(int)
    scores_array = np.clip(raw_scores, 1e-15, 1 - 1e-15)
    predicted = raw_scores > threshold
    positives = labels_array == 1
    negatives = labels_array == 0
    if not positives.any() or not negatives.any():
        raise ValueError("both classes are required")

    older_negative = raw_older & negatives
    younger_negative = (~raw_older) & negatives
    older_positive = raw_older & positives
    younger_positive = (~raw_older) & positives
    fpr_older = _weighted_rate(predicted[older_negative], weights_array[older_negative])
    fpr_younger = _weighted_rate(predicted[younger_negative], weights_array[younger_negative])
    if fpr_older is None or fpr_younger is None:
        raise ValueError("both age groups require at least one negative")
    high = max(fpr_older, fpr_younger)
    pe_ratio = float(min(fpr_older, fpr_younger) / high) if high > 0 else None
    positive_weights = weights_array[positives]
    negative_weights_array = weights_array[negatives]
    return {
        "global_fpr": float(weights_array[negatives & predicted].sum() / negative_weights_array.sum()),
        "recall": float(weights_array[positives & predicted].sum() / positive_weights.sum()),
        "alert_rate": float(weights_array[predicted].sum() / weights_array.sum()),
        "fpr_older": fpr_older,
        "fpr_younger": fpr_younger,
        "fpr_difference": float(fpr_older - fpr_younger),
        "pe_ratio": pe_ratio,
        "brier": float(np.average((raw_scores - labels_array) ** 2, weights=weights_array)),
        "logloss": float(log_loss(labels_array, scores_array, labels=[0, 1], sample_weight=weights_array)),
        "logloss_older": _weighted_group_logloss(labels_array[raw_older], scores_array[raw_older], weights_array[raw_older]),
        "logloss_younger": _weighted_group_logloss(labels_array[~raw_older], scores_array[~raw_older], weights_array[~raw_older]),
        "n": int(len(labels_array)),
        "positives": int(positives.sum()),
        "negatives": int(negatives.sum()),
        "n_older": int(raw_older.sum()),
        "n_younger": int((~raw_older).sum()),
        "n_older_positive": int(older_positive.sum()),
        "n_younger_positive": int(younger_positive.sum()),
        "weight_sum": float(weights_array.sum()),
        "positive_weight_sum": float(positive_weights.sum()),
        "negative_weight_sum": float(negative_weights_array.sum()),
    }


def weighted_policy_thresholds(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    calibration_weights: np.ndarray,
    target_scores: np.ndarray,
    target_labels: np.ndarray,
    target_weights: np.ndarray,
    target_fpr: float,
) -> dict[str, float]:
    """Return the source-calibrated and target-oracle weighted thresholds."""
    calibration_labels_array = np.asarray(calibration_labels, dtype=int)
    target_labels_array = np.asarray(target_labels, dtype=int)
    calibration_negative = calibration_labels_array == 0
    target_negative = target_labels_array == 0
    return {
        "target_oracle": select_weighted_threshold(
            np.asarray(target_scores, dtype=float)[target_negative],
            np.asarray(target_weights, dtype=float)[target_negative],
            target_fpr,
        ),
        "source_calibrated": select_weighted_threshold(
            np.asarray(calibration_scores, dtype=float)[calibration_negative],
            np.asarray(calibration_weights, dtype=float)[calibration_negative],
            target_fpr,
        ),
    }
