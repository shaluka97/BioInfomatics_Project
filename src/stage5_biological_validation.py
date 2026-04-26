"""
Stage 5: Biological / interpretability validation.

The pipeline produces a final shortlist of SNPs. This stage:
    * Maps each SNP to the closest gene by base-pair position (positional
      lookup against a bundled gene table).
    * Cross-references the gene set against a bundled GWAS Catalog snapshot
      to count overlaps with previously reported trait associations.
    * Runs a hypergeometric over-representation test against pathway
      gene-sets (KEGG-style, bundled).

For the synthetic experiment the SNP-to-gene mapping is performed against a
*synthetic* gene table that we generate alongside the simulator: every block
of 50 simulated SNPs is assigned to a synthetic gene named SIMG_xxxxx so the
machinery is exercised end-to-end. The same code path works on real data
once a real gene table is provided.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


@dataclass
class Stage5Config:
    gwas_catalog_path: str = "data/annotations/gwas_catalog_snapshot.tsv"
    gene_table_path: str = "data/annotations/gene_table.tsv"
    pathway_table_path: str = "data/annotations/pathways.tsv"


def make_synthetic_annotations(project_root: str, n_snps: int, block_size: int = 50,
                               seed: int = 42) -> None:
    """Create gene_table / GWAS catalog / pathways tsv files for synthetic data."""
    rng = np.random.default_rng(seed)
    ann_dir = os.path.join(project_root, "data/annotations")
    os.makedirs(ann_dir, exist_ok=True)

    # gene_table: each block becomes one synthetic gene
    n_genes = max(1, n_snps // block_size)
    rows = []
    for g in range(n_genes):
        s = g * block_size
        e = min(n_snps, s + block_size)
        rows.append({"gene": f"SIMG_{g:05d}", "snp_start": s, "snp_end": e - 1})
    pd.DataFrame(rows).to_csv(os.path.join(ann_dir, "gene_table.tsv"), sep="\t", index=False)

    # GWAS catalog snapshot: pretend ~5% of synthetic genes are in the catalog
    catalog_genes = rng.choice([r["gene"] for r in rows],
                               size=max(1, int(0.05 * n_genes)), replace=False)
    pd.DataFrame({
        "gene": catalog_genes,
        "trait": rng.choice(["TraitA", "TraitB", "TraitC"], size=catalog_genes.size),
    }).to_csv(os.path.join(ann_dir, "gwas_catalog_snapshot.tsv"), sep="\t", index=False)

    # pathways: assign genes to 5 fake pathways with overlap
    pathway_rows = []
    pathway_names = ["PWY_metabolism", "PWY_immune", "PWY_neural",
                     "PWY_signal", "PWY_misc"]
    for r in rows:
        # each gene joins ~2 pathways
        chosen = rng.choice(pathway_names, size=2, replace=False)
        for pwy in chosen:
            pathway_rows.append({"pathway": pwy, "gene": r["gene"]})
    pd.DataFrame(pathway_rows).to_csv(os.path.join(ann_dir, "pathways.tsv"),
                                      sep="\t", index=False)


def _map_snps_to_genes(snp_indices: np.ndarray, gene_table: pd.DataFrame) -> List[str]:
    out: List[str] = []
    starts = gene_table["snp_start"].to_numpy()
    ends = gene_table["snp_end"].to_numpy()
    genes = gene_table["gene"].to_numpy()
    for s in snp_indices:
        idx = np.searchsorted(starts, s, side="right") - 1
        if 0 <= idx < starts.size and starts[idx] <= s <= ends[idx]:
            out.append(genes[idx])
    return out


def run_stage5(final_snp_indices: np.ndarray, project_root: str,
               cfg: Stage5Config) -> Dict:
    info: Dict[str, object] = {}

    gene_table = pd.read_csv(os.path.join(project_root, cfg.gene_table_path), sep="\t")
    catalog = pd.read_csv(os.path.join(project_root, cfg.gwas_catalog_path), sep="\t")
    pathways = pd.read_csv(os.path.join(project_root, cfg.pathway_table_path), sep="\t")

    mapped_genes = sorted(set(_map_snps_to_genes(final_snp_indices, gene_table)))
    info["n_mapped_genes"] = len(mapped_genes)
    info["mapped_genes"] = mapped_genes

    # GWAS Catalog overlap
    catalog_genes = set(catalog["gene"].astype(str).tolist())
    info["gwas_catalog_overlap"] = sorted(set(mapped_genes) & catalog_genes)
    info["n_gwas_catalog_overlap"] = len(info["gwas_catalog_overlap"])

    # Pathway over-representation: hypergeometric test
    universe = set(gene_table["gene"].astype(str).tolist())
    selected = set(mapped_genes) & universe
    enrichment_rows = []
    for pwy in pathways["pathway"].unique():
        members = set(pathways.loc[pathways["pathway"] == pwy, "gene"].astype(str).tolist()) & universe
        if not members:
            continue
        K = len(members)                # successes in population
        N = len(universe)               # population size
        n = len(selected)               # sample drawn
        k = len(members & selected)     # observed successes
        if n == 0:
            continue
        # P(X >= k)
        rv = hypergeom(N, K, n)
        pval = rv.sf(k - 1)
        enrichment_rows.append({
            "pathway": pwy, "size": K, "selected_in_set": k,
            "selected_total": n, "universe": N, "pvalue": float(pval),
        })
    enr = pd.DataFrame(enrichment_rows).sort_values("pvalue")
    info["pathway_enrichment"] = enr
    info["top3_pathways"] = enr.head(3).to_dict(orient="records")
    return info
