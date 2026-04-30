# GWAS Search-Space Reduction with Machine Learning

A multi-stage hybrid pipeline that prunes a Genome-Wide Association Study
(GWAS) search space from millions of SNPs down to a small biologically
interpretable shortlist. The pipeline combines linkage-disequilibrium
pruning, weak-effect-preserving marginal screening, interaction-aware
ML selection (ReliefF + gradient-boosted trees), a sparse autoencoder for
representation learning, and gene-/pathway-level biological validation.

The repository ships everything needed to reproduce every figure and table
in the accompanying report (`report/report.pdf`).

## Group
- BioCode Lab

## Authors

- 258247B – Keirishan B
- 258284J – Rathnapala G.D.S.K

## Quick start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Reproduce all results on the synthetic dataset (the headline experiment).
#    This regenerates every CSV table and every figure used in the report.
python -m src.run_pipeline --dataset synthetic --config configs/default.yaml

# 3. (Optional) Run the smaller real-data experiment on a 1000 Genomes chr22 slice.
#    Downloads ~60 MB on first call; falls back to a synthetic surrogate if
#    the network is unavailable.
python -m src.run_pipeline --dataset 1000genomes --config configs/default.yaml --run-id g1k_chr22

# 4. Run the test suite
pytest tests/ -q
```

Outputs land in `results/<run_id>/`, with figures in `results/figures/<run_id>/`
and tables in `results/tables/<run_id>/`.

## Git hygiene

The project now includes a root `.gitignore` to keep local/derived files out of
version control. In particular, it ignores:

- virtual environments (`venv/`, `.venv/`)
- Python caches (`__pycache__/`, `.pytest_cache/`, `.mypy_cache/`)
- IDE metadata (`.idea/`, `.vscode/`)
- generated run outputs such as per-run logs and stage CSVs under `results/`
- local helper extraction files like `proposal_extracted.txt`

If you need to commit a specific generated artifact (for example, a frozen table
for the report), add it explicitly with `git add -f <path>`.

## Repository layout

```
BioInfomatics_Project/
├── README.md
├── requirements.txt
├── configs/default.yaml          # All hyperparameters live here
├── data/
│   ├── synthetic/                # cached simulator output
│   ├── 1000genomes/              # chr22 slice download + cache
│   └── annotations/              # gene table, GWAS Catalog snapshot, pathways
├── src/
│   ├── data_simulation.py        # planted-truth synthetic GWAS
│   ├── data_loaders.py           # synthetic + 1000G loader
│   ├── stage1_ld_preprocessing.py
│   ├── stage2_weak_effect_screening.py
│   ├── gwas_snp_gene_prioritization.py
│   ├── stage3_interaction_aware_selection.py
│   ├── stage4_representation_learning.py
│   ├── stage5_biological_validation.py
│   ├── baselines.py              # p-value / LASSO / RF / ReliefF
│   ├── evaluation.py
│   ├── visualization.py
│   └── run_pipeline.py           # orchestrator
├── notebooks/                    # exploration + walkthrough + analysis
├── results/                      # generated artefacts (per run-id)
│   ├── figures/<run_id>/         # PNG plots
│   ├── tables/<run_id>/          # CSV summary tables
│   └── <run_id>/                 # stage CSVs, logs, summary.json
├── tests/test_each_stage.py
└── report/report.pdf
```

## What each stage does

| Stage | Purpose | Core technique |
|-------|---------|----------------|
| 1 | Remove redundancy from correlated SNPs | QC (MAF, missingness, HWE) + sliding-window LD pruning |
| 2 | Retain SNPs with weak marginal effects | Univariate score test + BH FDR + top-k per LD block |
| 3 | Capture epistasis | ReliefF (skrebate) + XGBoost gain (depth ≥ 2), blended with SNP→gene prior weights |
| 4 | Compress further | Sparse autoencoder + gradient-based per-SNP contribution |
| 5 | Confirm biological relevance | Gene mapping + GWAS Catalog overlap + pathway hypergeometric |

Each stage is a standalone module with a `run_stageN(X, y, cfg)` entry-point;
they are composed end-to-end by `src/run_pipeline.py`.

## Reproducibility

- Every stochastic step takes a `seed` argument that flows from
  `configs/default.yaml::random_seed` (default: 42).
- Intermediate SNP shortlists are saved to `results/<run_id>/stageK_kept.csv`.
- Trained autoencoder + classifiers are saved under `results/<run_id>/` alongside the stage outputs.

## Limitations

This project is built for tractable demonstration on a single workstation.
The synthetic dataset is small (2,000 individuals × 20,000 SNPs by default,
configurable up to the proposal-spec 10k × 200k). The 1000 Genomes
secondary experiment uses a chr22 slice with an injected synthetic
phenotype, not a real disease cohort. These choices are stated honestly in
the report.

## Citing

If you use this pipeline, please cite the project report and the underlying
methods (LD pruning, ReliefF, XGBoost, sparse autoencoders) listed in the
references section of `report/report.pdf`.
