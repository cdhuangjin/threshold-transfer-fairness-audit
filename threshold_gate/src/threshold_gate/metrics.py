from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss


def select_threshold_from_negatives(negative_scores: np.ndarray, target_fpr: float = 0.05) -> float:
    """Select the closest finite score threshold whose FPR is below target.

    The selector uses only scores from known negative examples. It first chooses
    the largest attainable FPR not exceeding the target and then the smallest
    tied threshold, matching the source notebook's near-5% operating point.
    """
    negative_scores = np.asarray(negative_scores, dtype=float)
    if negative_scores.ndim != 1 or negative_scores.size == 0:
        raise ValueError("negative_scores must be a non-empty vector")
    if not np.isfinite(negative_scores).all():
        raise ValueError("negative_scores must be finite")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must lie strictly between zero and one")
    candidates = np.unique(negative_scores)
    fprs = np.array([(negative_scores > threshold).mean() for threshold in candidates])
    eligible = fprs <= target_fpr + 1e-15
    if not eligible.any():
        raise ValueError("no finite threshold satisfies target_fpr")
    closest_fpr = fprs[eligible].max()
    return float(candidates[eligible][np.isclose(fprs[eligible], closest_fpr)].min())


def _group_rate(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    if denominator.size == 0:
        return None
    return float(numerator.mean())


def score_diagnostics(labels: np.ndarray, scores: np.ndarray, older: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    older = np.asarray(older, dtype=bool)
    negatives = labels == 0

    def q95(values: np.ndarray) -> float | None:
        return float(np.quantile(values, 0.95)) if values.size else None

    return {
        "score_mean": float(scores.mean()),
        "score_q50": float(np.quantile(scores, 0.50)),
        "score_q90": float(np.quantile(scores, 0.90)),
        "negative_score_q95": q95(scores[negatives]),
        "negative_score_q95_older": q95(scores[negatives & older]),
        "negative_score_q95_younger": q95(scores[negatives & ~older]),
        "label_prevalence": float(labels.mean()),
        "older_share": float(older.mean()),
    }


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | np.ndarray, older: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    older = np.asarray(older, dtype=bool)
    if not (labels.ndim == scores.ndim == older.ndim == 1 and len(labels) == len(scores) == len(older)):
        raise ValueError("labels, scores, and older must be aligned vectors")
    if not np.isfinite(scores).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary and scores finite")
    predicted = scores > threshold
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        raise ValueError("both classes are required")

    older_negative = older & negatives
    younger_negative = (~older) & negatives
    older_positive = older & positives
    younger_positive = (~older) & positives
    fpr_older = _group_rate(predicted[older_negative], older_negative)
    fpr_younger = _group_rate(predicted[younger_negative], younger_negative)
    if fpr_older is None or fpr_younger is None:
        raise ValueError("both age groups require at least one negative")
    high = max(fpr_older, fpr_younger)
    pe_ratio = float(min(fpr_older, fpr_younger) / high) if high > 0 else None
    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    return {
        "global_fpr": float(predicted[negatives].mean()),
        "recall": float(predicted[positives].mean()),
        "alert_rate": float(predicted.mean()),
        "fpr_older": fpr_older,
        "fpr_younger": fpr_younger,
        "fpr_difference": float(fpr_older - fpr_younger),
        "pe_ratio": pe_ratio,
        "brier": float(np.mean((scores - labels) ** 2)),
        "logloss": float(log_loss(labels, clipped, labels=[0, 1])),
        "logloss_older": float(log_loss(labels[older], clipped[older], labels=[0, 1])) if older.any() and len(np.unique(labels[older])) == 2 else None,
        "logloss_younger": float(log_loss(labels[~older], clipped[~older], labels=[0, 1])) if (~older).any() and len(np.unique(labels[~older])) == 2 else None,
        "n": int(len(labels)),
        "positives": int(positives.sum()),
        "negatives": int(negatives.sum()),
        "n_older": int(older.sum()),
        "n_younger": int((~older).sum()),
        "n_older_positive": int(older_positive.sum()),
        "n_younger_positive": int(younger_positive.sum()),
    }
