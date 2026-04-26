"""
Gene-aware SNP prioritization utilities.

This module computes SNP-level prior weights from gene-level evidence so the
interaction-aware selector (Stage 3) can favor variants that already have
support from Stage-2 weak-effect screening aggregated at gene level.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class GenePrioritizationConfig:
    enabled: bool = True
    gene_table_path: str = "data/annotations/gene_table.tsv"
    beta_weight: float = 0.6
    pvalue_weight: float = 0.4


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _validate_gene_table(gene_table: pd.DataFrame) -> None:
    required = {"gene", "snp_start", "snp_end"}
    missing = required - set(gene_table.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"gene_table is missing required columns: {missing_cols}")


def _map_snps_to_gene_indices(snp_indices: np.ndarray,
                              starts: np.ndarray,
                              ends: np.ndarray) -> np.ndarray:
    mapped = np.full(snp_indices.shape[0], -1, dtype=np.int64)
    for i, snp_idx in enumerate(snp_indices):
        j = np.searchsorted(starts, snp_idx, side="right") - 1
        if 0 <= j < starts.size and starts[j] <= snp_idx <= ends[j]:
            mapped[i] = j
    return mapped


def build_feature_prior(
    snp_indices: Iterable[int],
    beta_full: np.ndarray,
    pvalue_full: np.ndarray,
    project_root: str,
    cfg: GenePrioritizationConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build per-SNP prior weights for a selected SNP subset.

    Args:
        snp_indices: SNP indices in the full genotype matrix for the subset
            that will be passed to Stage 3.
        beta_full: Stage-2 beta estimates in full SNP space.
        pvalue_full: Stage-2 p-values in full SNP space.
        project_root: Project root path.
        cfg: Gene-prioritization configuration.

    Returns:
        feature_prior: np.ndarray of shape (len(snp_indices),) in [0, 1].
        info: Dict with summary statistics and optional diagnostics.
    """
    snp_idx = np.asarray(list(snp_indices), dtype=np.int64)
    if not cfg.enabled or snp_idx.size == 0:
        return np.zeros(snp_idx.size, dtype=np.float64), {
            "enabled": False,
            "mapped_snps": 0,
            "mapped_genes": 0,
        }

    gene_table_path = os.path.join(project_root, cfg.gene_table_path)
    if not os.path.exists(gene_table_path):
        raise FileNotFoundError(
            f"Gene table not found for gene prioritization: {gene_table_path}"
        )

    gene_table = pd.read_csv(gene_table_path, sep="\t")
    _validate_gene_table(gene_table)
    gene_table = gene_table.sort_values("snp_start").reset_index(drop=True)

    starts = gene_table["snp_start"].to_numpy(dtype=np.int64)
    ends = gene_table["snp_end"].to_numpy(dtype=np.int64)
    genes = gene_table["gene"].astype(str).to_numpy()

    mapped_gene_idx = _map_snps_to_gene_indices(snp_idx, starts, ends)
    valid = mapped_gene_idx >= 0
    if not valid.any():
        return np.zeros(snp_idx.size, dtype=np.float64), {
            "enabled": True,
            "mapped_snps": 0,
            "mapped_genes": 0,
        }

    gene_beta_vals: Dict[int, List[float]] = {}
    gene_p_vals: Dict[int, List[float]] = {}
    for local_i, g_idx in enumerate(mapped_gene_idx):
        if g_idx < 0:
            continue
        full_i = snp_idx[local_i]
        gene_beta_vals.setdefault(int(g_idx), []).append(float(abs(beta_full[full_i])))
        gene_p_vals.setdefault(int(g_idx), []).append(float(pvalue_full[full_i]))

    gene_ids = np.array(sorted(gene_beta_vals.keys()), dtype=np.int64)
    gene_beta = np.array([np.mean(gene_beta_vals[g]) for g in gene_ids], dtype=np.float64)
    gene_sig = np.array(
        [
            np.mean(-np.log10(np.clip(gene_p_vals[g], 1e-300, 1.0)))
            for g in gene_ids
        ],
        dtype=np.float64,
    )

    beta_w = float(cfg.beta_weight)
    p_w = float(cfg.pvalue_weight)
    total = beta_w + p_w
    if total <= 0:
        beta_w, p_w = 0.5, 0.5
        total = 1.0
    beta_w /= total
    p_w /= total

    gene_score = beta_w * _minmax(gene_beta) + p_w * _minmax(gene_sig)
    score_lookup = {int(g): float(s) for g, s in zip(gene_ids, gene_score)}

    prior = np.zeros(snp_idx.size, dtype=np.float64)
    for i, g_idx in enumerate(mapped_gene_idx):
        if g_idx >= 0:
            prior[i] = score_lookup[int(g_idx)]
    prior = _minmax(prior)

    top_gene_rows = []
    order = np.argsort(gene_score)[::-1][:10]
    for rank_pos in order:
        g_idx = int(gene_ids[rank_pos])
        top_gene_rows.append(
            {
                "gene": genes[g_idx],
                "gene_score": float(gene_score[rank_pos]),
                "mean_abs_beta": float(gene_beta[rank_pos]),
                "mean_neglog10_p": float(gene_sig[rank_pos]),
            }
        )

    info: Dict[str, object] = {
        "enabled": True,
        "mapped_snps": int(valid.sum()),
        "mapped_genes": int(np.unique(mapped_gene_idx[valid]).size),
        "top_genes": top_gene_rows,
    }
    return prior, info
