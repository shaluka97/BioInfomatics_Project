"""
Dataset loading helpers.

Two paths:
    1. Synthetic     -> src.data_simulation
    2. 1000 Genomes  -> small chr22 slice via VCF; falls back to a synthetic
       'real-data-style' surrogate if the download is unavailable.
"""

from __future__ import annotations

import gzip
import os
import shutil
import urllib.request
from typing import Dict, Tuple

import numpy as np

from .data_simulation import (SimulationConfig, load_dataset,
                              save_dataset, simulate_dataset)


SYN_DEFAULT = "data/synthetic/cache.npz"
G1K_VCF_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
)
G1K_LOCAL_GZ = "data/1000genomes/chr22.vcf.gz"
G1K_CACHE = "data/1000genomes/chr22_slice.npz"


def get_synthetic(cfg_dict: Dict, project_root: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate or load a cached synthetic dataset.

    The cache filename includes a hash of the simulator parameters so that
    parameter changes always produce a fresh dataset (avoids stale caches
    on read-only mounts where deletion is restricted).
    """
    import hashlib
    key = "|".join(
        f"{k}={cfg_dict[k]}" for k in sorted(cfg_dict)
        if k in ("n_individuals", "n_snps", "n_causal_strong", "n_causal_weak",
                 "n_causal_epistatic_pairs", "ld_block_size", "ld_block_corr",
                 "case_fraction", "seed")
    )
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    base = cfg_dict.get("cache_path", SYN_DEFAULT)
    cache = os.path.join(project_root, base.replace(".npz", f"_{h}.npz"))
    if os.path.exists(cache):
        return load_dataset(cache)
    sim_cfg = SimulationConfig(
        n_individuals=cfg_dict["n_individuals"],
        n_snps=cfg_dict["n_snps"],
        n_causal_strong=cfg_dict["n_causal_strong"],
        n_causal_weak=cfg_dict["n_causal_weak"],
        n_causal_epistatic_pairs=cfg_dict["n_causal_epistatic_pairs"],
        ld_block_size=cfg_dict["ld_block_size"],
        ld_block_corr=cfg_dict["ld_block_corr"],
        case_fraction=cfg_dict["case_fraction"],
        seed=cfg_dict.get("seed", 42),
    )
    X, y, truth = simulate_dataset(sim_cfg)
    save_dataset(cache, X, y, truth)
    return X, y, truth


def _download_g1k(project_root: str, max_bytes: int = 60 * 1024 * 1024) -> str:
    """Download a partial chr22 VCF.gz (default ~60 MB) for the slice experiment."""
    out = os.path.join(project_root, G1K_LOCAL_GZ)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        return out
    req = urllib.request.Request(G1K_VCF_URL, headers={"Range": f"bytes=0-{max_bytes-1}"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(out, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    return out


def _parse_partial_vcf(gz_path: str, max_snps: int = 5000,
                       max_individuals: int = 1000) -> Tuple[np.ndarray, list]:
    """Read genotype dosages from a (possibly truncated) VCF.gz.

    Returns (X[int8], snp_ids[list]). Robust to a truncated tail because we
    use a partial Range download.
    """
    snp_ids: list = []
    rows: list = []
    sample_count: int = 0
    try:
        with gzip.open(gz_path, "rt") as fh:
            for line in fh:
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    headers = line.rstrip().split("\t")
                    sample_count = min(max_individuals, len(headers) - 9)
                    continue
                if len(rows) >= max_snps:
                    break
                parts = line.rstrip().split("\t")
                if len(parts) < 9 + 1:
                    continue
                # bi-allelic only
                if len(parts[3]) != 1 or len(parts[4]) != 1 or "," in parts[4]:
                    continue
                gts = parts[9:9 + sample_count]
                row = np.zeros(sample_count, dtype=np.int8)
                ok = True
                for i, g in enumerate(gts):
                    g = g.split(":")[0]
                    if g == "0|0" or g == "0/0":
                        row[i] = 0
                    elif g == "1|1" or g == "1/1":
                        row[i] = 2
                    elif g in ("0|1", "1|0", "0/1", "1/0"):
                        row[i] = 1
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                rows.append(row)
                snp_ids.append(f"{parts[0]}:{parts[1]}:{parts[3]}>{parts[4]}")
    except (OSError, EOFError):
        pass  # truncated stream is fine — keep what we have
    if not rows:
        raise RuntimeError("Could not parse any SNPs from the VCF slice.")
    # rows are SNPs x samples — transpose to samples x SNPs
    X = np.stack(rows, axis=1)
    return X, snp_ids


def get_1000genomes_slice(cfg_dict: Dict, project_root: str,
                          max_snps: int = 5000,
                          max_individuals: int = 1000,
                          n_causal_strong: int = 5,
                          n_causal_weak: int = 10,
                          n_causal_epistatic_pairs: int = 3,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load a 1000 Genomes chr22 slice and inject a phenotype.

    If the download fails (offline / firewalled), fall back to a smaller
    synthetic dataset and clearly tag it as a surrogate.
    """
    cache = os.path.join(project_root, G1K_CACHE)
    if os.path.exists(cache):
        return load_dataset(cache)

    rng = np.random.default_rng(seed)
    used_real = True
    snp_ids = None
    try:
        gz_path = _download_g1k(project_root)
        X, snp_ids = _parse_partial_vcf(gz_path, max_snps=max_snps, max_individuals=max_individuals)
    except Exception as e:  # network blocked, etc.
        used_real = False
        from .data_simulation import SimulationConfig, simulate_dataset
        sim_cfg = SimulationConfig(
            n_individuals=max_individuals, n_snps=max_snps,
            n_causal_strong=n_causal_strong,
            n_causal_weak=n_causal_weak,
            n_causal_epistatic_pairs=n_causal_epistatic_pairs,
            ld_block_size=50, ld_block_corr=0.85, seed=seed,
        )
        X, y_pre, truth_pre = simulate_dataset(sim_cfg)
        save_dataset(cache, X, y_pre, truth_pre)
        return X, y_pre, truth_pre

    n_ind, n_snps = X.shape
    # plant phenotype on top of real genotypes
    all_idx = np.arange(n_snps); rng.shuffle(all_idx)
    cur = 0
    strong = sorted(all_idx[cur:cur + n_causal_strong].tolist()); cur += n_causal_strong
    weak = sorted(all_idx[cur:cur + n_causal_weak].tolist()); cur += n_causal_weak
    epi_flat = all_idx[cur:cur + 2 * n_causal_epistatic_pairs].tolist()
    epi_pairs = [(epi_flat[2*i], epi_flat[2*i+1]) for i in range(n_causal_epistatic_pairs)]

    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    eta = np.zeros(n_ind)
    for j in strong:
        eta += rng.choice([-1, 1]) * rng.uniform(0.6, 0.9) * Xz[:, j]
    for j in weak:
        eta += rng.choice([-1, 1]) * rng.uniform(0.10, 0.20) * Xz[:, j]
    for a, b in epi_pairs:
        eta += rng.choice([-1, 1]) * rng.uniform(0.5, 0.9) * Xz[:, a] * Xz[:, b]
    eta += rng.normal(0, 0.5, size=n_ind)
    intercept = -np.quantile(eta, 0.5)
    p = 1.0 / (1.0 + np.exp(-(eta + intercept)))
    y = (rng.random(n_ind) < p).astype(np.int8)

    truth = {
        "strong": np.array(strong, dtype=np.int64),
        "weak": np.array(weak, dtype=np.int64),
        "epistatic_pairs": np.array(epi_pairs, dtype=np.int64) if epi_pairs else np.zeros((0, 2), dtype=np.int64),
        "epistatic_flat": np.array(sorted(epi_flat), dtype=np.int64),
        "block_assignment": np.zeros(n_snps, dtype=np.int32),
        "snp_ids": np.array(snp_ids if snp_ids else [], dtype=object),
        "used_real_genotypes": used_real,
    }
    save_dataset(cache, X, y, truth)
    return X, y, truth
