"""
Stage 2: Weak-effect-preserving marginal screening.

For every SNP that survived Stage 1 we run a fast univariate score test
(equivalent to the Cochran-Armitage trend test under additive coding) and
record:
    * the absolute effect size  | beta_hat |
    * the p-value
We then keep the union of:
    1. SNPs whose Benjamini-Hochberg-adjusted p-value < q  (relaxed FDR)
    2. The top-k SNPs per LD block by |beta_hat|, regardless of FDR

The second clause is the *weak-effect-preserving* part: SNPs with weak but
non-zero marginal signal that would be missed by Bonferroni or even by a
strict FDR sometimes survive simply because they are the strongest in their
local block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm


@dataclass
class Stage2Config:
    fdr_q: float = 0.10
    topk_per_block: int = 5


def _univariate_score(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised score test for binary y vs additive SNP coding.

    Returns (beta_hat, pvalue) per SNP. This is much faster than fitting a
    separate logistic regression per SNP and gives equivalent ranking on the
    margin.
    """
    y = y.astype(np.float64)
    n = y.size
    p_y = y.mean()
    yc = y - p_y                     # mean-centred outcome
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    Xz = Xz.astype(np.float64, copy=False)

    # score = sum_i x_iz * (y_i - p_y); var ~ p_y(1-p_y) * sum x_iz^2 = p_y(1-p_y)*n
    score = Xz.T @ yc
    var = p_y * (1.0 - p_y) * n
    z = score / np.sqrt(var + 1e-12)
    pval = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    # rough beta estimate on standardised X
    beta = z / np.sqrt(n)
    return beta, pval


def _benjamini_hochberg(pvals: np.ndarray, q: float) -> np.ndarray:
    """Return a boolean mask of SNPs that pass BH FDR at level q."""
    m = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = q * np.arange(1, m + 1) / m
    passes = ranked <= thresh
    if not passes.any():
        return np.zeros(m, dtype=bool)
    k = np.where(passes)[0].max()
    mask = np.zeros(m, dtype=bool)
    mask[order[: k + 1]] = True
    return mask


def run_stage2(X: np.ndarray, y: np.ndarray, block_assignment: np.ndarray,
               cfg: Stage2Config) -> Tuple[np.ndarray, Dict]:
    beta, pval = _univariate_score(X, y)

    fdr_mask = _benjamini_hochberg(pval, cfg.fdr_q)

    # top-k by |beta| per block
    block_mask = np.zeros_like(fdr_mask)
    if block_assignment.size == X.shape[1]:
        for b in np.unique(block_assignment):
            members = np.where(block_assignment == b)[0]
            if members.size <= cfg.topk_per_block:
                block_mask[members] = True
            else:
                top = members[np.argsort(np.abs(beta[members]))[-cfg.topk_per_block:]]
                block_mask[top] = True

    keep = fdr_mask | block_mask
    info = {
        "n_input": int(X.shape[1]),
        "fdr_kept": int(fdr_mask.sum()),
        "block_topk_kept": int(block_mask.sum()),
        "union_kept": int(keep.sum()),
        "beta": beta,
        "pvalue": pval,
    }
    return keep, info
