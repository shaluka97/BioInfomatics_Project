"""
Stage 1: LD-based preprocessing.

Steps
-----
1. Quality control filters
   * Minor Allele Frequency (MAF) >= maf_min
   * Per-SNP missingness   < snp_missingness_max
   * Per-individual missingness < ind_missingness_max  (synthetic data has no
     missing calls, so this is a no-op there but is implemented for real data)
   * Hardy-Weinberg Equilibrium p-value > hwe_pvalue_min on controls only
2. LD pruning by sliding window: within each window of `ld_window_size`, drop
   any SNP that has r^2 >= `ld_r2_threshold` with an earlier retained SNP.
   The window slides forward by `ld_step_size` SNPs at a time.

Returns the kept-SNP boolean mask plus the filtered genotype matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.stats import chi2


@dataclass
class Stage1Config:
    maf_min: float = 0.01
    snp_missingness_max: float = 0.05
    ind_missingness_max: float = 0.05
    hwe_pvalue_min: float = 1e-6
    ld_window_size: int = 50
    ld_step_size: int = 5
    ld_r2_threshold: float = 0.5


def _maf(X: np.ndarray) -> np.ndarray:
    p = X.mean(axis=0) / 2.0
    return np.minimum(p, 1.0 - p)


def _hwe_pvalues(X: np.ndarray, controls_mask: np.ndarray) -> np.ndarray:
    """Chi-square HWE test on controls. Returns one p-value per SNP."""
    Xc = X[controls_mask]
    n = Xc.shape[0]
    if n == 0:
        return np.ones(X.shape[1])
    n_AA = (Xc == 0).sum(axis=0)
    n_Aa = (Xc == 1).sum(axis=0)
    n_aa = (Xc == 2).sum(axis=0)
    p = (2 * n_AA + n_Aa) / (2.0 * n)
    q = 1.0 - p
    exp_AA = n * p * p
    exp_Aa = 2 * n * p * q
    exp_aa = n * q * q
    chi2_stat = np.zeros(X.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        for obs, exp in ((n_AA, exp_AA), (n_Aa, exp_Aa), (n_aa, exp_aa)):
            valid = exp > 0
            chi2_stat[valid] += (obs[valid] - exp[valid]) ** 2 / exp[valid]
    return 1.0 - chi2.cdf(chi2_stat, df=1)


def _ld_prune(X: np.ndarray, window: int, step: int, r2_thr: float) -> np.ndarray:
    """Sliding-window LD prune. Returns boolean mask of kept SNPs."""
    n, m = X.shape
    keep = np.ones(m, dtype=bool)
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    Xz = Xz.astype(np.float32, copy=False)

    start = 0
    while start < m:
        end = min(start + window, m)
        idx = np.arange(start, end)
        idx = idx[keep[idx]]
        if idx.size > 1:
            sub = Xz[:, idx]
            corr = (sub.T @ sub) / n
            corr2 = corr ** 2
            np.fill_diagonal(corr2, 0.0)
            for k in range(1, idx.size):
                if not keep[idx[k]]:
                    continue
                # if any earlier kept SNP has r^2 >= threshold, drop this one
                earlier = idx[:k]
                still_kept = keep[earlier]
                if still_kept.any() and (corr2[k, :k][still_kept] >= r2_thr).any():
                    keep[idx[k]] = False
        start += step
    return keep


def run_stage1(X: np.ndarray, y: np.ndarray, cfg: Stage1Config) -> Tuple[np.ndarray, Dict]:
    """Apply QC + LD-pruning filters."""
    info: Dict[str, object] = {}
    n_ind, n_snps = X.shape

    # Per-individual missingness: synthetic = no missingness; this is a stub
    # that becomes meaningful when real data with -1 markers is supplied.
    if (X < 0).any():
        ind_miss = (X < 0).mean(axis=1)
        ind_keep = ind_miss < cfg.ind_missingness_max
        X = X[ind_keep]
        y = y[ind_keep]
        info["ind_dropped"] = int((~ind_keep).sum())
    else:
        info["ind_dropped"] = 0

    # Per-SNP missingness (same: no missing in synthetic)
    if (X < 0).any():
        snp_miss = (X < 0).mean(axis=0)
        miss_mask = snp_miss < cfg.snp_missingness_max
    else:
        miss_mask = np.ones(X.shape[1], dtype=bool)

    # MAF filter
    maf = _maf(X)
    maf_mask = maf >= cfg.maf_min

    # HWE on controls
    controls = (y == 0)
    hwe_p = _hwe_pvalues(X, controls)
    hwe_mask = hwe_p > cfg.hwe_pvalue_min

    qc_mask = miss_mask & maf_mask & hwe_mask
    info["qc_kept"] = int(qc_mask.sum())
    info["qc_total"] = int(n_snps)

    # LD pruning on the QC-passing subset
    X_qc = X[:, qc_mask]
    ld_keep_in_qc = _ld_prune(X_qc, cfg.ld_window_size, cfg.ld_step_size, cfg.ld_r2_threshold)

    # combine masks back to the full SNP space
    final_mask = np.zeros(n_snps, dtype=bool)
    qc_indices = np.where(qc_mask)[0]
    final_mask[qc_indices[ld_keep_in_qc]] = True

    info["ld_kept"] = int(final_mask.sum())
    info["dropped_total"] = int(n_snps - final_mask.sum())
    return final_mask, info
