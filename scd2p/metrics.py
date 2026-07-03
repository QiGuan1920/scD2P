"""Common metrics used in scD2P benchmarking."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation and return NaN when undefined."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(pearsonr(y_true, y_pred)[0])


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation and return NaN when undefined."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(spearmanr(y_true, y_pred)[0])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Return standard regression metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    key = f"{prefix}_" if prefix else ""
    mse = mean_squared_error(y_true.ravel(), y_pred.ravel())
    return {
        f"{key}pearson": safe_pearson(y_true, y_pred),
        f"{key}spearman": safe_spearman(y_true, y_pred),
        f"{key}r2": float(r2_score(y_true.ravel(), y_pred.ravel())),
        f"{key}mse": float(mse),
        f"{key}rmse": float(np.sqrt(mse)),
        f"{key}mae": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
    }


def expression_metrics(x_true: np.ndarray, x_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Return expression-level metrics for predicted expression matrices."""
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)
    key = f"{prefix}_" if prefix else ""
    mse = mean_squared_error(x_true.ravel(), x_pred.ravel())
    cos = cosine_similarity(x_true.reshape(1, -1), x_pred.reshape(1, -1))[0, 0]
    return {
        f"{key}pearson": safe_pearson(x_true, x_pred),
        f"{key}r2": float(r2_score(x_true.ravel(), x_pred.ravel())),
        f"{key}mse": float(mse),
        f"{key}rmse": float(np.sqrt(mse)),
        f"{key}euclidean": float(np.linalg.norm(x_true.ravel() - x_pred.ravel())),
        f"{key}cosine_similarity": float(cos),
        f"{key}cosine_distance": float(1.0 - cos),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """Return standard binary or multiclass classification metrics."""
    key = f"{prefix}_" if prefix else ""
    out = {
        f"{key}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{key}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{key}macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{key}macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{key}macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out[f"{key}auroc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out[f"{key}auroc"] = np.nan
        try:
            out[f"{key}aupr"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            out[f"{key}aupr"] = np.nan
    return out
