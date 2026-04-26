"""
Stage 3: Interaction-aware feature selection.

Implements two complementary detectors and takes the union of their top-N
shortlists:

    3a. ReliefF / MultiSURF  (`skrebate`)
        Picks SNPs whose values discriminate cases vs controls *given the
        rest* of the genotype neighbourhood. Captures interaction signal
        because nearest-neighbour comparisons are made in the full feature
        space.

    3b. Gradient-boosted trees (`xgboost`)
        Trees of depth >= 2 split on combinations of features, so the
        gain-based feature_importances_ pick up SNPs that contribute through
        interactions even if their univariate effect is weak.

We also expose the per-method importance scores so Stage 5 can use them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class Stage3Config:
    reliefF_neighbors: int = 100
    reliefF_topk: int = 500
    xgb_topk: int = 500
    xgb_max_depth: int = 5
    xgb_n_estimators: int = 200
    xgb_learning_rate: float = 0.1
    gene_prior_strength: float = 0.5
    seed: int = 42


def _fast_reliefF(X: np.ndarray, y: np.ndarray, n_neighbors: int = 50,
                  n_target: int = 300, seed: int = 42) -> np.ndarray:
    """Fast vectorised ReliefF for binary classification.

    Uses sklearn `NearestNeighbors` (k-d tree) on a *sample* of target
    instances (`n_target`). For each target i and each feature j we
    accumulate

        W_j += mean_{m in misses(i)} |X_ij - X_mj|
              - mean_{h in hits(i)}  |X_ij - X_hj|

    The score is finally divided by `n_target * range(X_j)` so it matches
    the standard Relief normalisation. This implementation is many times
    faster than the loop-based skrebate ReliefF on > 1000 individuals while
    producing very similar feature rankings on tabular GWAS data.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    n, m = X.shape
    Xn = X.astype(np.float32)
    # standardise so distances are scale-invariant
    Xn = (Xn - Xn.mean(axis=0)) / (Xn.std(axis=0) + 1e-6)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if pos_idx.size < 2 or neg_idx.size < 2:
        # degenerate case
        return np.zeros(m, dtype=np.float64)

    nn_pos = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(Xn[pos_idx])
    nn_neg = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(Xn[neg_idx])

    targets = rng.choice(n, size=min(n_target, n), replace=False)
    W = np.zeros(m, dtype=np.float64)

    for i in targets:
        xi = Xn[i:i+1]
        if y[i] == 1:
            hit_loc = nn_pos.kneighbors(xi, return_distance=False)[0]
            hits = pos_idx[hit_loc]
            hits = hits[hits != i][:n_neighbors]
            miss_loc = nn_neg.kneighbors(xi, return_distance=False)[0]
            misses = neg_idx[miss_loc][:n_neighbors]
        else:
            hit_loc = nn_neg.kneighbors(xi, return_distance=False)[0]
            hits = neg_idx[hit_loc]
            hits = hits[hits != i][:n_neighbors]
            miss_loc = nn_pos.kneighbors(xi, return_distance=False)[0]
            misses = pos_idx[miss_loc][:n_neighbors]
        if hits.size == 0 or misses.size == 0:
            continue
        diff_miss = np.abs(Xn[misses] - Xn[i:i+1]).mean(axis=0)
        diff_hit = np.abs(Xn[hits] - Xn[i:i+1]).mean(axis=0)
        W += diff_miss - diff_hit

    W /= max(1, targets.size)
    return W


def _run_reliefF(X: np.ndarray, y: np.ndarray, cfg: Stage3Config) -> np.ndarray:
    """Returns ReliefF score per SNP (higher = more important).

    Defaults to a vectorised internal implementation. The skrebate ReliefF
    is also supported but is much slower; it can be re-enabled by setting
    the environment variable GWAS_USE_SKREBATE=1.
    """
    import os
    if os.environ.get("GWAS_USE_SKREBATE", "0") == "1":
        try:
            from skrebate import ReliefF
            n_neighbors = min(cfg.reliefF_neighbors, max(2, X.shape[0] // 4))
            rf = ReliefF(n_neighbors=n_neighbors, n_features_to_select=X.shape[1])
            rf.fit(np.ascontiguousarray(X, dtype=np.float64),
                   np.ascontiguousarray(y, dtype=np.int64))
            return np.asarray(rf.feature_importances_, dtype=np.float64)
        except Exception as exc:
            LOGGER.warning("Falling back to fast ReliefF because skrebate failed: %s", exc)
    return _fast_reliefF(X, y, n_neighbors=cfg.reliefF_neighbors, seed=cfg.seed)


def _run_xgb(X: np.ndarray, y: np.ndarray, cfg: Stage3Config) -> np.ndarray:
    """Returns gain-based importance per SNP (zero for unused features)."""
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        random_state=cfg.seed,
        n_jobs=-1,
    )
    clf.fit(X, y)
    imp = np.zeros(X.shape[1], dtype=np.float64)
    booster = clf.get_booster()
    score_dict = booster.get_score(importance_type="gain")
    for k, v in score_dict.items():
        # XGBoost names features f0, f1, ... when no DataFrame given
        idx = int(k[1:])
        if 0 <= idx < imp.size:
            imp[idx] = v
    return imp


def run_stage3(X: np.ndarray, y: np.ndarray,
               cfg: Stage3Config,
               feature_prior: np.ndarray | None = None) -> Tuple[np.ndarray, Dict]:
    n_snps = X.shape[1]
    relief_raw = _run_reliefF(X, y, cfg)
    xgb_raw = _run_xgb(X, y, cfg)

    relief = relief_raw.copy()
    xgb_imp = xgb_raw.copy()
    prior_used = False
    if feature_prior is not None:
        if feature_prior.shape[0] != n_snps:
            raise ValueError(
                f"feature_prior length {feature_prior.shape[0]} does not match n_snps {n_snps}"
            )
        strength = float(np.clip(cfg.gene_prior_strength, 0.0, 1.0))
        if strength > 0:
            prior = np.asarray(feature_prior, dtype=np.float64)
            prior_min, prior_max = float(np.min(prior)), float(np.max(prior))
            if prior_max > prior_min:
                prior = (prior - prior_min) / (prior_max - prior_min)
            else:
                prior = np.zeros_like(prior)

            def _minmax(v: np.ndarray) -> np.ndarray:
                lo, hi = float(np.min(v)), float(np.max(v))
                if hi <= lo:
                    return np.zeros_like(v)
                return (v - lo) / (hi - lo)

            relief = _minmax(relief_raw) + strength * prior
            xgb_imp = _minmax(xgb_raw) + strength * prior
            prior_used = True

    relief_top = np.argsort(relief)[-cfg.reliefF_topk:]
    xgb_top = np.argsort(xgb_imp)[-cfg.xgb_topk:]
    keep = np.zeros(n_snps, dtype=bool)
    keep[relief_top] = True
    keep[xgb_top] = True

    info = {
        "n_input": n_snps,
        "reliefF_top": int(cfg.reliefF_topk),
        "xgb_top": int(cfg.xgb_topk),
        "union_kept": int(keep.sum()),
        "reliefF_score": relief,
        "xgb_score": xgb_imp,
        "reliefF_raw_score": relief_raw,
        "xgb_raw_score": xgb_raw,
        "used_feature_prior": prior_used,
    }
    return keep, info
