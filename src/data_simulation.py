"""
Synthetic GWAS data simulator.

Generates a genotype matrix with realistic LD-block structure and a binary
phenotype that depends on a known set of causal SNPs:
    - n_causal_strong   : SNPs with large additive effects (easy to find)
    - n_causal_weak     : SNPs with weak additive effects (hard to find without
                          weak-effect-preserving screening)
    - n_causal_epistatic: pairs of SNPs whose product term drives the phenotype
                          (no marginal effect; only Stage-3 interaction-aware
                          methods should pick them up)

The simulator returns the genotype matrix `X` (n_individuals x n_snps), the
binary phenotype vector `y`, and a dictionary of ground-truth indices that the
evaluation code uses for precision/recall on causal recovery.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SimulationConfig:
    n_individuals: int = 2000
    n_snps: int = 20000
    n_causal_strong: int = 10
    n_causal_weak: int = 30
    n_causal_epistatic_pairs: int = 10
    ld_block_size: int = 50
    ld_block_corr: float = 0.85
    case_fraction: float = 0.5
    seed: int = 42


def _draw_block_genotype(rng: np.random.Generator, n_ind: int, n_snps: int,
                         maf: np.ndarray, corr: float) -> np.ndarray:
    """Sample a single LD block of genotypes preserving Hardy-Weinberg
    Equilibrium at the population level while creating within-block
    linkage disequilibrium.

    For each individual we draw **two independent haplotypes**. For each
    haplotype copy we sample a single latent factor `f` shared across the
    SNPs in this block, then sample each SNP's allele with frequency

        p = sigmoid(logit(maf) + scale * f)

    Drawing the two haplotypes independently guarantees that, conditional
    on the haplotype's latent factor, the two alleles within an individual
    are independent — so HWE holds within each individual and at the
    aggregated population level. LD across SNPs is created because they
    share `f` within a haplotype.
    """
    # Map the user-facing 'corr' parameter (target within-block LD strength)
    # to an internal scale on the latent logit. Empirically calibrated so
    # that corr~0.85 yields strong within-block r^2 (often > 0.5) and many
    # SNPs become candidates for pruning by the Stage-1 LD step.
    scale = 2.0 + 8.0 * corr
    eps = 1e-6
    logit_maf = np.log((maf + eps) / (1.0 - maf + eps))     # (n_snps,)
    G = np.zeros((n_ind, n_snps), dtype=np.int8)
    for _ in range(2):                                       # two haplotypes
        f = rng.standard_normal(n_ind)                       # (n_ind,)
        logit_p = logit_maf[None, :] + scale * f[:, None]    # (n_ind, n_snps)
        p = 1.0 / (1.0 + np.exp(-logit_p))
        G += (rng.random(p.shape) < p).astype(np.int8)
    return G


def simulate_dataset(cfg: SimulationConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
    rng = np.random.default_rng(cfg.seed)
    n_blocks = max(1, cfg.n_snps // cfg.ld_block_size)
    snps_remaining = cfg.n_snps

    blocks: List[np.ndarray] = []
    block_assignment = np.zeros(cfg.n_snps, dtype=np.int32)
    snp_idx = 0
    for b in range(n_blocks):
        size = cfg.ld_block_size if b < n_blocks - 1 else snps_remaining
        maf = rng.uniform(0.05, 0.45, size=size)
        block_geno = _draw_block_genotype(rng, cfg.n_individuals, size, maf, cfg.ld_block_corr)
        blocks.append(block_geno)
        block_assignment[snp_idx:snp_idx + size] = b
        snp_idx += size
        snps_remaining -= size

    X = np.concatenate(blocks, axis=1).astype(np.int8)
    n_snps_actual = X.shape[1]

    # --- Choose causal SNPs --------------------------------------------------
    all_idx = np.arange(n_snps_actual)
    rng.shuffle(all_idx)
    cur = 0
    strong_idx = sorted(all_idx[cur:cur + cfg.n_causal_strong].tolist())
    cur += cfg.n_causal_strong
    weak_idx = sorted(all_idx[cur:cur + cfg.n_causal_weak].tolist())
    cur += cfg.n_causal_weak
    epi_flat = all_idx[cur:cur + 2 * cfg.n_causal_epistatic_pairs].tolist()
    cur += 2 * cfg.n_causal_epistatic_pairs
    epi_pairs = [(epi_flat[2 * i], epi_flat[2 * i + 1]) for i in range(cfg.n_causal_epistatic_pairs)]

    # --- Build phenotype linear predictor -----------------------------------
    # standardize for stable logistic
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    eta = np.zeros(cfg.n_individuals, dtype=np.float64)
    # strong additive
    for j in strong_idx:
        beta = rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 0.9)
        eta += beta * Xz[:, j]
    # weak additive
    for j in weak_idx:
        beta = rng.choice([-1.0, 1.0]) * rng.uniform(0.10, 0.20)
        eta += beta * Xz[:, j]
    # pure interaction (no marginal terms)
    for (a, b) in epi_pairs:
        beta = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 0.9)
        eta += beta * Xz[:, a] * Xz[:, b]
    # noise
    eta += rng.normal(0, 0.5, size=cfg.n_individuals)

    # shift intercept so case fraction matches target
    target = cfg.case_fraction
    intercept = -np.quantile(eta, 1 - target)
    p = 1.0 / (1.0 + np.exp(-(eta + intercept)))
    y = (rng.random(cfg.n_individuals) < p).astype(np.int8)

    truth = {
        "strong": np.array(strong_idx, dtype=np.int64),
        "weak": np.array(weak_idx, dtype=np.int64),
        "epistatic_pairs": np.array(epi_pairs, dtype=np.int64) if epi_pairs else np.zeros((0, 2), dtype=np.int64),
        "epistatic_flat": np.array(sorted(epi_flat), dtype=np.int64),
        "block_assignment": block_assignment[:n_snps_actual],
    }
    return X, y, truth


def save_dataset(path: str, X: np.ndarray, y: np.ndarray, truth: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        X=X, y=y,
        truth_strong=truth["strong"],
        truth_weak=truth["weak"],
        truth_epistatic_pairs=truth["epistatic_pairs"],
        truth_epistatic_flat=truth["epistatic_flat"],
        block_assignment=truth["block_assignment"],
    )


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    d = np.load(path)
    truth = {
        "strong": d["truth_strong"],
        "weak": d["truth_weak"],
        "epistatic_pairs": d["truth_epistatic_pairs"],
        "epistatic_flat": d["truth_epistatic_flat"],
        "block_assignment": d["block_assignment"],
    }
    return d["X"], d["y"], truth


if __name__ == "__main__":
    cfg = SimulationConfig()
    X, y, truth = simulate_dataset(cfg)
    print("X:", X.shape, "y:", y.shape, "case-rate:", float(y.mean()))
    print("strong/weak/epi(flat):",
          truth["strong"].size, truth["weak"].size, truth["epistatic_flat"].size)
