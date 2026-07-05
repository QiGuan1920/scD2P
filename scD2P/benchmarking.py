"""Lightweight benchmarking helpers for scD2P representations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from .metrics import classification_metrics, expression_metrics, regression_metrics


def aggregate_condition_scores(
    adata,
    group_cols: list[str],
    score_key: str = "X_zp",
    min_cells: int = 1,
) -> pd.DataFrame:
    """Aggregate cell-level program scores into condition-level mean features."""
    scores = np.asarray(adata.obsm[score_key], dtype=np.float32)
    obs = adata.obs.reset_index(drop=True)
    rows = []
    for keys, idx in obs.groupby(group_cols).indices.items():
        idx = np.asarray(idx)
        if len(idx) < min_cells:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["n_cells"] = len(idx)
        row["feature"] = scores[idx].mean(axis=0)
        rows.append(row)
    return pd.DataFrame(rows)


def kmeans_label_consistency(
    features: np.ndarray,
    labels: np.ndarray,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> dict[str, float]:
    """Cluster features and compare clusters with known labels."""
    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    pred = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20).fit_predict(features)
    return {
        "ari": float(adjusted_rand_score(labels, pred)),
        "nmi": float(normalized_mutual_info_score(labels, pred)),
        "ami": float(adjusted_mutual_info_score(labels, pred)),
    }


def leave_one_group_ridge(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    group_col: str,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Run leave-one-group-out ridge regression on condition-level features."""
    rows = []
    groups = sorted(df[group_col].unique())
    for group in groups:
        train = df[df[group_col] != group]
        test = df[df[group_col] == group]
        if len(train) == 0 or len(test) == 0:
            continue
        x_train = np.vstack(train[feature_col].values)
        x_test = np.vstack(test[feature_col].values)
        y_train = np.asarray(train[target_col].values, dtype=float)
        y_test = np.asarray(test[target_col].values, dtype=float)
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics = regression_metrics(y_test, y_pred)
        rows.append({"held_out_group": group, "n_test": len(test), **metrics})
    return pd.DataFrame(rows)


def leave_one_group_logistic(
    df: pd.DataFrame,
    feature_col: str,
    label_col: str,
    group_col: str,
    c: float = 1.0,
    max_iter: int = 2000,
) -> pd.DataFrame:
    """Run leave-one-group-out logistic regression for sensitivity classification."""
    rows = []
    groups = sorted(df[group_col].unique())
    for group in groups:
        train = df[df[group_col] != group]
        test = df[df[group_col] == group]
        if len(train) == 0 or len(test) == 0 or len(np.unique(train[label_col])) < 2:
            continue
        x_train = np.vstack(train[feature_col].values)
        x_test = np.vstack(test[feature_col].values)
        y_train = np.asarray(train[label_col].values)
        y_test = np.asarray(test[label_col].values)
        clf = LogisticRegression(C=c, max_iter=max_iter, class_weight="balanced")
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_score = clf.predict_proba(x_test)[:, 1] if len(clf.classes_) == 2 else None
        metrics = classification_metrics(y_test, y_pred, y_score)
        rows.append({"held_out_group": group, "n_test": len(test), **metrics})
    return pd.DataFrame(rows)


def reconstruct_expression_from_programs(
    z_basal: np.ndarray,
    z_pert: np.ndarray,
    ridge_coef: np.ndarray,
    ridge_intercept: np.ndarray,
    pert_components: np.ndarray,
) -> np.ndarray:
    """Reconstruct full-gene expression from basal and perturbation scores."""
    basal_expr = z_basal @ ridge_coef.T + ridge_intercept
    pert_expr = z_pert @ pert_components
    return basal_expr + pert_expr


def evaluate_expression_prediction(x_true: np.ndarray, x_pred: np.ndarray) -> dict[str, float]:
    """Evaluate predicted expression using matrix-level metrics."""
    return expression_metrics(x_true, x_pred)
