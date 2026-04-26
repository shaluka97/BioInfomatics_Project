"""
All figure-generation utilities.

Each function saves to results/figures/ and returns the saved path.
Style is kept conservative (single-axis, default colormap) so the figures
embed cleanly into the LaTeX report.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def fig_snp_count_per_stage(stage_counts: Dict[str, int], out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    stages = list(stage_counts.keys())
    counts = [stage_counts[k] for k in stages]
    bars = ax.bar(stages, counts, color="#3b7dd8")
    ax.set_yscale("log")
    ax.set_ylabel("SNPs retained (log scale)")
    ax.set_title("Search-space reduction across pipeline stages")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(_ensure(out_path), dpi=150)
    plt.close(fig)
    return out_path


def fig_manhattan(pvalues: np.ndarray, kept_mask: np.ndarray, out_path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    log_p = -np.log10(np.clip(pvalues, 1e-30, 1.0))
    axes[0].scatter(np.arange(pvalues.size), log_p, s=2, color="#888")
    axes[0].set_title("Manhattan: before pipeline")
    axes[0].set_xlabel("SNP index")
    axes[0].set_ylabel("-log10(p)")
    keep_idx = np.where(kept_mask)[0]
    axes[1].scatter(keep_idx, log_p[keep_idx], s=4, color="#3b7dd8")
    axes[1].set_title(f"Manhattan: after pipeline (n={kept_mask.sum()})")
    axes[1].set_xlabel("SNP index")
    plt.tight_layout()
    fig.savefig(_ensure(out_path), dpi=150)
    plt.close(fig)
    return out_path


def fig_roc_curves(roc_data: Dict[str, Dict], out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, d in roc_data.items():
        ax.plot(d["fpr"], d["tpr"], label=f"{name} (AUC={d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC: proposed vs baselines (logistic regression on selected SNPs)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(_ensure(out_path), dpi=150)
    plt.close(fig)
    return out_path


def fig_recovery_breakdown(recovery: Dict[str, Dict], out_path: str) -> str:
    """recovery: { method_name: { 'strong':{precision,recall,f1}, 'weak':..., 'epistatic':..., 'combined':...}}"""
    methods = list(recovery.keys())
    types = ["strong", "weak", "epistatic"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    width = 0.18
    x = np.arange(len(types))
    for i, m in enumerate(methods):
        precs = [recovery[m][t]["precision"] for t in types]
        recs = [recovery[m][t]["recall"] for t in types]
        axes[0].bar(x + i * width, precs, width=width, label=m)
        axes[1].bar(x + i * width, recs, width=width, label=m)
    for ax, title in zip(axes, ["Precision", "Recall"]):
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(types)
        ax.set_title(f"Causal-SNP {title}")
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(_ensure(out_path), dpi=150)
    plt.close(fig)
    return out_path


def fig_runtime(runtime: Dict[str, float], out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = list(runtime.keys())
    vals = [runtime[k] for k in keys]
    ax.bar(keys, vals, color="#d8843b")
    ax.set_ylabel("Wall-clock seconds")
    ax.set_title("Runtime comparison")
    plt.xticks(rotation=15)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(_ensure(out_path), dpi=150)
    plt.close(fig)
    return out_path


def fig_top_snp_heatmap(X: np.ndarray, y: np.ndarray, snp_idx: np.ndarray,
                        out_path: str, n_samples: int = 80) -> str:
    """Heatmap of the top SNPs across a stratified random subset of samples."""
    rng = np.random.default_rng(0)
    cases = np.where(y == 1)[0]
    ctrls = np.where(y == 0)[0]
    rng.shuffle(cases); rng.shuffle(ctrls)
    sub = np.concatenate([cases[: n_samples // 2], ctrls[: n_samples // 2]])
    M = X[np.ix_(sub, snp_idx)]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(M.T, cmap="viridis", cbar_kws={"label": "Genotype (0/1/2)"},
                xticklabels=False, yticklabels=False, ax=ax)
    ax.set_title(f"Top {snp_idx.size} selected SNPs across {sub.size} samples "
                 "(left half = cases, right half = controls)")
    ax.axvline(n_samples // 2, color="white", lw=1)
    plt.tight_layout()
    fig.savefig(_ensure(out_path), dpi=150)
    plt.close(fig)
    return out_path


def fig_shap_summary(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                     out_path: str, max_display: int = 20, seed: int = 42) -> str:
    """SHAP summary on a final XGBoost classifier trained on the kept SNPs."""
    if mask.sum() == 0:
        # write an empty placeholder
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No SNPs selected — SHAP unavailable",
                ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(_ensure(out_path), dpi=150)
        plt.close(fig)
        return out_path
    import shap, xgboost as xgb
    X_sel = X[:, mask]
    feat_names = [f"snp_{i}" for i in np.where(mask)[0]]
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="binary:logistic", tree_method="hist",
        eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0,
        base_score=0.5,
    )
    clf.fit(X_sel, y)
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_sel)
    except Exception:
        # Some XGBoost / SHAP versions disagree on base_score parsing.
        # Fall back to a model-agnostic permutation explainer on a subsample.
        sub = np.random.default_rng(seed).choice(X_sel.shape[0],
                                                 min(200, X_sel.shape[0]),
                                                 replace=False)
        explainer = shap.Explainer(clf.predict_proba, X_sel[sub])
        sv_full = explainer(X_sel[sub])
        sv = sv_full.values[..., 1] if sv_full.values.ndim == 3 else sv_full.values
        X_sel = X_sel[sub]
    plt.figure(figsize=(7, 6))
    shap.summary_plot(sv, X_sel, feature_names=feat_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(_ensure(out_path), dpi=150)
    plt.close()
    return out_path
