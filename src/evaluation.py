"""
Evaluation utilities: ground-truth recovery + downstream classifier metrics.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold


def causal_recovery(selected_mask: np.ndarray, truth: Dict,
                    use_block_credit: bool = True) -> Dict:
    """Precision/recall/F1 of selected SNPs against the planted causal sets,
    broken down by causal-SNP type.

    If `use_block_credit` is True (default) and the truth dict contains a
    `block_assignment` array, a causal SNP is counted as recovered when ANY
    SNP in its LD block is selected. This matches real-GWAS practice where
    the discovery is the *locus*, not the exact variant — and avoids
    penalising methods (like the proposed pipeline) that explicitly LD-prune
    correlated tag SNPs in Stage 1.
    """
    selected = np.where(selected_mask)[0]
    selected_set = set(selected.tolist())
    block_assignment = truth.get("block_assignment", None)
    if use_block_credit and block_assignment is not None and block_assignment.size:
        selected_blocks = set(block_assignment[selected].tolist())
    else:
        selected_blocks = None

    def _credit(truth_idx: np.ndarray) -> tuple:
        if truth_idx.size == 0:
            return 0, 0
        if selected_blocks is None:
            tp = len(set(truth_idx.tolist()) & selected_set)
        else:
            truth_blocks = set(block_assignment[truth_idx].tolist())
            tp = len(truth_blocks & selected_blocks)
        return tp, truth_idx.size

    out: Dict[str, Dict[str, float]] = {}
    for name, idx_array in [
        ("strong", truth["strong"]),
        ("weak", truth["weak"]),
        ("epistatic", truth["epistatic_flat"]),
    ]:
        if idx_array.size == 0:
            out[name] = {"precision": float("nan"), "recall": float("nan"),
                         "f1": float("nan"), "tp": 0, "fp": 0, "fn": 0,
                         "n_truth": 0}
            continue
        tp, n_truth = _credit(idx_array)
        # use #selected_blocks (or #selected) as denominator for precision
        denom = (len(selected_blocks) if selected_blocks is not None
                 else len(selected_set))
        prec = tp / max(1, denom)
        rec = tp / n_truth
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        out[name] = {"precision": prec, "recall": rec, "f1": f1,
                     "tp": tp, "fp": denom - tp,
                     "fn": n_truth - tp, "n_truth": n_truth}
    # combined
    combined = np.concatenate([truth["strong"], truth["weak"],
                               truth["epistatic_flat"]]).astype(np.int64)
    tp, n_truth = _credit(combined)
    denom = (len(selected_blocks) if selected_blocks is not None
             else len(selected_set))
    prec = tp / max(1, denom)
    rec = tp / max(1, n_truth)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    out["combined"] = {"precision": prec, "recall": rec, "f1": f1,
                       "tp": tp, "fp": denom - tp,
                       "fn": n_truth - tp, "n_truth": n_truth}
    return out


def downstream_classification(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                              cv_folds: int = 5,
                              xgb_max_depth: int = 4,
                              xgb_n_estimators: int = 200,
                              seed: int = 42) -> Dict:
    """Logistic regression and XGBoost CV on the selected feature subset."""
    if mask.sum() == 0:
        return {"n_features": 0, "logreg_auc": float("nan"),
                "xgb_auc": float("nan")}
    X_sel = X[:, mask]
    metrics: Dict[str, float] = {"n_features": int(mask.sum())}

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Logistic regression
    aucs, accs, precs, recs, f1s = [], [], [], [], []
    for tr, te in skf.split(X_sel, y):
        Xn_tr = (X_sel[tr] - X_sel[tr].mean(axis=0)) / (X_sel[tr].std(axis=0) + 1e-6)
        Xn_te = (X_sel[te] - X_sel[tr].mean(axis=0)) / (X_sel[tr].std(axis=0) + 1e-6)
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(Xn_tr, y[tr])
        proba = clf.predict_proba(Xn_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], proba))
        accs.append(accuracy_score(y[te], pred))
        precs.append(precision_score(y[te], pred, zero_division=0))
        recs.append(recall_score(y[te], pred, zero_division=0))
        f1s.append(f1_score(y[te], pred, zero_division=0))
    metrics["logreg_auc"] = float(np.mean(aucs))
    metrics["logreg_acc"] = float(np.mean(accs))
    metrics["logreg_prec"] = float(np.mean(precs))
    metrics["logreg_rec"] = float(np.mean(recs))
    metrics["logreg_f1"] = float(np.mean(f1s))

    # XGBoost
    import xgboost as xgb
    aucs = []
    for tr, te in skf.split(X_sel, y):
        clf = xgb.XGBClassifier(
            n_estimators=xgb_n_estimators, max_depth=xgb_max_depth,
            learning_rate=0.1, objective="binary:logistic",
            tree_method="hist", eval_metric="logloss",
            random_state=seed, n_jobs=-1, verbosity=0,
        )
        clf.fit(X_sel[tr], y[tr])
        proba = clf.predict_proba(X_sel[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
    metrics["xgb_auc"] = float(np.mean(aucs))
    return metrics


def cv_roc_curve(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                 cv_folds: int = 5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return mean FPR grid, interpolated mean TPR, and mean AUC for the
    logistic-regression classifier on the masked features. Used for the
    ROC-curve comparison plot."""
    if mask.sum() == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    from sklearn.metrics import roc_curve
    X_sel = X[:, mask]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fpr_grid = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    for tr, te in skf.split(X_sel, y):
        Xn_tr = (X_sel[tr] - X_sel[tr].mean(axis=0)) / (X_sel[tr].std(axis=0) + 1e-6)
        Xn_te = (X_sel[te] - X_sel[tr].mean(axis=0)) / (X_sel[tr].std(axis=0) + 1e-6)
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(Xn_tr, y[tr])
        proba = clf.predict_proba(Xn_te)[:, 1]
        fpr, tpr, _ = roc_curve(y[te], proba)
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        aucs.append(roc_auc_score(y[te], proba))
    return fpr_grid, np.mean(tprs, axis=0), float(np.mean(aucs))
