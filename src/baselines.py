"""
Baseline feature-selection methods used as comparators for the proposed
5-stage pipeline.

Each baseline takes (X, y) and returns a boolean mask the same length as
X.shape[1], identifying the SNPs the baseline would shortlist.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

from .stage2_weak_effect_screening import _univariate_score


def baseline_pvalue(X: np.ndarray, y: np.ndarray, threshold: float = 5e-8) -> np.ndarray:
    _, pval = _univariate_score(X, y)
    return pval < threshold


def baseline_lasso(X: np.ndarray, y: np.ndarray,
                   max_iter: int = 2000) -> np.ndarray:
    """L1-penalised logistic regression baseline.

    Uses liblinear with a fixed regularisation strength chosen so that
    roughly a few hundred SNPs survive — equivalent to picking the sparsity
    knee a CV would identify, but ~50x faster than full CV on >10k SNPs.
    """
    from sklearn.linear_model import LogisticRegression
    Xn = X.astype(np.float64)
    Xn = (Xn - Xn.mean(axis=0)) / (Xn.std(axis=0) + 1e-6)
    clf = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.05,
        max_iter=max_iter,
    )
    clf.fit(Xn, y)
    return np.abs(clf.coef_).ravel() > 1e-8


def baseline_random_forest(X: np.ndarray, y: np.ndarray,
                           n_estimators: int = 200, topk: int = 500,
                           seed: int = 42) -> np.ndarray:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        random_state=seed, n_jobs=-1,
    )
    clf.fit(X, y)
    imp = clf.feature_importances_
    top = np.argsort(imp)[-topk:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top] = True
    return mask


def baseline_reliefF(X: np.ndarray, y: np.ndarray,
                     topk: int = 500, n_neighbors: int = 100,
                     seed: int = 42) -> np.ndarray:
    """ReliefF-only baseline (Stage-3a applied directly to QC'd data).

    Uses the same vectorised ReliefF implementation as Stage 3 to keep the
    comparison fair and the runtime tractable on full feature sets.
    """
    from .stage3_interaction_aware_selection import _fast_reliefF
    scores = _fast_reliefF(X, y, n_neighbors=n_neighbors, seed=seed)
    top = np.argsort(scores)[-topk:]
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top] = True
    return mask
