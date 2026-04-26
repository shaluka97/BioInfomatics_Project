"""
Top-level orchestrator for the proposed 5-stage pipeline + baseline matrix.

Usage:
    python -m src.run_pipeline --dataset synthetic --config configs/default.yaml
    python -m src.run_pipeline --dataset 1000genomes --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd
import yaml

# Allow running as both `python -m src.run_pipeline` and `python src/run_pipeline.py`
if __package__ in (None, ""):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(THIS_DIR))
    from src import (baselines, data_loaders, evaluation,
                     stage1_ld_preprocessing as s1,
                     stage2_weak_effect_screening as s2,
                     stage3_interaction_aware_selection as s3,
                     stage4_representation_learning as s4,
                     stage5_biological_validation as s5,
                     visualization as viz)
else:
    from . import baselines, data_loaders, evaluation
    from . import stage1_ld_preprocessing as s1
    from . import stage2_weak_effect_screening as s2
    from . import stage3_interaction_aware_selection as s3
    from . import stage4_representation_learning as s4
    from . import stage5_biological_validation as s5
    from . import visualization as viz


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_logging(run_dir: str) -> logging.Logger:
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")
    logger = logging.getLogger("gwas")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import xgboost  # noqa
    except Exception:
        pass


def run(config_path: str, dataset: str, run_id: str | None = None) -> Dict:
    root = project_root()
    cfg = yaml.safe_load(open(os.path.join(root, config_path)))
    if run_id is None:
        run_id = cfg.get("run_id", "default")
    cfg["run_id"] = run_id
    seed = cfg["random_seed"]
    seed_everything(seed)

    run_dir = os.path.join(root, "results", run_id)
    fig_dir = os.path.join(root, "results", "figures", run_id)
    tab_dir = os.path.join(root, "results", "tables", run_id)
    for d in (run_dir, fig_dir, tab_dir):
        os.makedirs(d, exist_ok=True)
    logger = setup_logging(run_dir)
    logger.info(f"Starting run_id={run_id}, dataset={dataset}")

    # ---------- 1. Load data ----------
    if dataset == "synthetic":
        d = cfg["dataset"]; d["seed"] = seed
        X, y, truth = data_loaders.get_synthetic(d, root)
    elif dataset == "1000genomes":
        d = cfg["dataset"]
        X, y, truth = data_loaders.get_1000genomes_slice(
            d, root, max_snps=5000, max_individuals=1000, seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    logger.info(f"Data: X={X.shape}, y mean={float(y.mean()):.3f}, "
                f"truth strong/weak/epi="
                f"{truth['strong'].size}/{truth['weak'].size}/{truth['epistatic_flat'].size}")

    # ensure annotations exist (synthetic helper)
    s5.make_synthetic_annotations(root, n_snps=X.shape[1],
                                  block_size=cfg["dataset"]["ld_block_size"], seed=seed)

    timings: Dict[str, float] = {}
    stage_counts: Dict[str, int] = {"input": X.shape[1]}

    # ---------- Stage 1 ----------
    t0 = time.time()
    s1_cfg = s1.Stage1Config(**cfg["stage1"])
    keep1, info1 = s1.run_stage1(X, y, s1_cfg)
    timings["stage1"] = time.time() - t0
    stage_counts["stage1"] = int(keep1.sum())
    logger.info(f"Stage1 done in {timings['stage1']:.2f}s, kept {keep1.sum()} / {X.shape[1]}")
    pd.DataFrame({"snp_idx": np.where(keep1)[0]}).to_csv(
        os.path.join(run_dir, "stage1_kept.csv"), index=False)

    X1 = X[:, keep1]
    blocks1 = truth["block_assignment"][keep1] if "block_assignment" in truth else np.zeros(X1.shape[1], dtype=np.int32)

    # ---------- Stage 2 ----------
    t0 = time.time()
    s2_cfg = s2.Stage2Config(**cfg["stage2"])
    keep2_local, info2 = s2.run_stage2(X1, y, blocks1, s2_cfg)
    timings["stage2"] = time.time() - t0
    stage_counts["stage2"] = int(keep2_local.sum())
    # lift mask back into full SNP space
    keep2 = np.zeros(X.shape[1], dtype=bool)
    keep2[np.where(keep1)[0][keep2_local]] = True
    logger.info(f"Stage2 done in {timings['stage2']:.2f}s, kept {keep2.sum()} "
                f"(FDR {info2['fdr_kept']}, top-k {info2['block_topk_kept']})")
    pd.DataFrame({"snp_idx": np.where(keep2)[0],
                  "beta": info2["beta"][keep2_local],
                  "pvalue": info2["pvalue"][keep2_local]}).to_csv(
        os.path.join(run_dir, "stage2_kept.csv"), index=False)
    full_pvalues_stage2 = np.ones(X.shape[1])
    full_pvalues_stage2[keep1] = info2["pvalue"]

    # ---------- Stage 3 ----------
    t0 = time.time()
    s3_cfg = s3.Stage3Config(**cfg["stage3"], seed=seed)
    X2 = X[:, keep2]
    keep3_local, info3 = s3.run_stage3(X2, y, s3_cfg)
    timings["stage3"] = time.time() - t0
    stage_counts["stage3"] = int(keep3_local.sum())
    keep3 = np.zeros(X.shape[1], dtype=bool)
    keep3[np.where(keep2)[0][keep3_local]] = True
    logger.info(f"Stage3 done in {timings['stage3']:.2f}s, kept {keep3.sum()}")
    pd.DataFrame({"snp_idx": np.where(keep3)[0],
                  "reliefF": info3["reliefF_score"][keep3_local],
                  "xgb_gain": info3["xgb_score"][keep3_local]}).to_csv(
        os.path.join(run_dir, "stage3_kept.csv"), index=False)

    # ---------- Stage 4 ----------
    t0 = time.time()
    s4_cfg = s4.Stage4Config(**cfg["stage4"], seed=seed)
    X3 = X[:, keep3]
    contrib4, embedding, info4 = s4.run_stage4(X3, s4_cfg, y=y)
    timings["stage4"] = time.time() - t0
    if s4_cfg.enabled and contrib4.size > 0:
        # Keep ~75% of Stage-3 SNPs by AE-traced supervised importance.
        # This is intentionally light because Stage 3 already produced a
        # focused candidate set; Stage 4's main job is to provide the
        # learned embedding, not to do another aggressive cull.
        n_keep = min(contrib4.size, max(50, int(0.75 * contrib4.size)))
        top_local = np.argsort(contrib4)[-n_keep:]
        keep4_local = np.zeros(contrib4.size, dtype=bool)
        keep4_local[top_local] = True
    else:
        keep4_local = np.ones(contrib4.size, dtype=bool)
    keep4 = np.zeros(X.shape[1], dtype=bool)
    keep4[np.where(keep3)[0][keep4_local]] = True
    stage_counts["stage4"] = int(keep4.sum())
    logger.info(f"Stage4 done in {timings['stage4']:.2f}s, kept {keep4.sum()}")
    pd.DataFrame({"snp_idx": np.where(keep4)[0],
                  "ae_contrib": contrib4[keep4_local]}).to_csv(
        os.path.join(run_dir, "stage4_kept.csv"), index=False)

    # ---------- Stage 5 ----------
    t0 = time.time()
    s5_cfg = s5.Stage5Config(**cfg["stage5"])
    final_idx = np.where(keep4)[0]
    info5 = s5.run_stage5(final_idx, root, s5_cfg)
    timings["stage5"] = time.time() - t0
    info5["pathway_enrichment"].to_csv(os.path.join(run_dir, "stage5_pathway_enrichment.csv"), index=False)
    pd.DataFrame({"gene": info5["mapped_genes"]}).to_csv(
        os.path.join(run_dir, "stage5_mapped_genes.csv"), index=False)
    logger.info(f"Stage5 done in {timings['stage5']:.2f}s, "
                f"genes={info5['n_mapped_genes']}, "
                f"GWAS Catalog overlap={info5['n_gwas_catalog_overlap']}")

    # ---------- Baselines ----------
    logger.info("Running baselines ...")
    bl_results: Dict[str, Dict] = {}
    bl_masks: Dict[str, np.ndarray] = {}

    t0 = time.time()
    bl_masks["pvalue"] = baselines.baseline_pvalue(X, y, threshold=cfg["baselines"]["pvalue_threshold"])
    timings["baseline_pvalue"] = time.time() - t0

    t0 = time.time()
    bl_masks["lasso"] = baselines.baseline_lasso(X, y, max_iter=cfg["baselines"]["lasso_max_iter"])
    timings["baseline_lasso"] = time.time() - t0

    t0 = time.time()
    bl_masks["random_forest"] = baselines.baseline_random_forest(
        X, y, n_estimators=cfg["baselines"]["rf_n_estimators"],
        topk=cfg["baselines"]["rf_topk"], seed=seed)
    timings["baseline_rf"] = time.time() - t0

    t0 = time.time()
    bl_masks["reliefF_only"] = baselines.baseline_reliefF(
        X, y, topk=cfg["baselines"]["reliefF_only_topk"])
    timings["baseline_reliefF"] = time.time() - t0

    bl_masks["proposed"] = keep4

    # ---------- Evaluation ----------
    eval_cfg = cfg["evaluation"]
    causal_table = {}
    perf_table = {}
    roc_data: Dict[str, Dict] = {}
    for name, mask in bl_masks.items():
        causal_table[name] = evaluation.causal_recovery(mask, truth)
        perf_table[name] = evaluation.downstream_classification(
            X, y, mask,
            cv_folds=eval_cfg["cv_folds"],
            xgb_max_depth=eval_cfg["downstream_xgb_max_depth"],
            xgb_n_estimators=eval_cfg["downstream_xgb_n_estimators"],
            seed=seed,
        )
        fpr, tpr, auc = evaluation.cv_roc_curve(X, y, mask, cv_folds=eval_cfg["cv_folds"], seed=seed)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
        logger.info(f"{name:>14s}: kept={mask.sum():>5d}  combined-recovery"
                    f" P/R/F1={causal_table[name]['combined']['precision']:.3f}/"
                    f"{causal_table[name]['combined']['recall']:.3f}/"
                    f"{causal_table[name]['combined']['f1']:.3f}  "
                    f"AUC(LR)={perf_table[name]['logreg_auc']:.3f}  "
                    f"AUC(XGB)={perf_table[name]['xgb_auc']:.3f}")

    # ---------- Save tables ----------
    rows = []
    for name in bl_masks:
        for ctype in ("strong", "weak", "epistatic", "combined"):
            r = causal_table[name][ctype]
            rows.append({"method": name, "type": ctype, **r})
    pd.DataFrame(rows).to_csv(os.path.join(tab_dir, "causal_recovery.csv"), index=False)

    perf_df = pd.DataFrame.from_dict(perf_table, orient="index")
    perf_df.index.name = "method"
    perf_df.to_csv(os.path.join(tab_dir, "downstream_performance.csv"))

    pd.DataFrame.from_dict(stage_counts, orient="index", columns=["snps"]).to_csv(
        os.path.join(tab_dir, "stage_counts.csv"))
    pd.DataFrame.from_dict(timings, orient="index", columns=["seconds"]).to_csv(
        os.path.join(tab_dir, "timings.csv"))

    # ---------- Figures ----------
    figs: Dict[str, str] = {}
    figs["snp_count"] = viz.fig_snp_count_per_stage(stage_counts, os.path.join(fig_dir, "01_snp_count.png"))
    figs["manhattan"] = viz.fig_manhattan(full_pvalues_stage2, keep4, os.path.join(fig_dir, "02_manhattan.png"))
    figs["roc"] = viz.fig_roc_curves(roc_data, os.path.join(fig_dir, "03_roc.png"))
    figs["recovery"] = viz.fig_recovery_breakdown(causal_table, os.path.join(fig_dir, "04_recovery.png"))
    figs["runtime"] = viz.fig_runtime(timings, os.path.join(fig_dir, "05_runtime.png"))
    if final_idx.size > 0:
        figs["heatmap"] = viz.fig_top_snp_heatmap(X, y, final_idx[:20],
                                                  os.path.join(fig_dir, "06_top_snp_heatmap.png"))
        try:
            figs["shap"] = viz.fig_shap_summary(X, y, keep4,
                                                os.path.join(fig_dir, "07_shap.png"))
        except Exception as e:
            logger.warning(f"SHAP figure skipped due to: {e}")
            figs["shap"] = None

    summary = {
        "run_id": run_id,
        "dataset": dataset,
        "X_shape": list(X.shape),
        "stage_counts": stage_counts,
        "timings_seconds": timings,
        "figures": figs,
        "stage5_pathway_top3": info5["top3_pathways"],
        "stage5_gwas_catalog_overlap_n": info5["n_gwas_catalog_overlap"],
        "stage5_mapped_genes_n": info5["n_mapped_genes"],
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Run complete: " + json.dumps({"run_id": run_id, "figures": list(figs)}))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "1000genomes"])
    p.add_argument("--run-id", default=None)
    args = p.parse_args()
    run(args.config, args.dataset, args.run_id)


if __name__ == "__main__":
    main()
