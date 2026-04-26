"""
Smoke tests for every stage of the pipeline.

These tests use a tiny synthetic dataset (200 individuals x 500 SNPs) so the
whole suite finishes in a few seconds. They verify *shape* and *non-trivial*
outputs rather than exact numerical values.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_simulation import SimulationConfig, simulate_dataset
from src import (stage1_ld_preprocessing as s1,
                 stage2_weak_effect_screening as s2,
                 stage3_interaction_aware_selection as s3,
                 stage4_representation_learning as s4,
                 stage5_biological_validation as s5,
                 baselines, evaluation)


@pytest.fixture(scope="module")
def tiny_data():
    cfg = SimulationConfig(n_individuals=200, n_snps=500,
                           n_causal_strong=3, n_causal_weak=5,
                           n_causal_epistatic_pairs=2, seed=0)
    return simulate_dataset(cfg)


def test_simulation_shapes(tiny_data):
    X, y, truth = tiny_data
    assert X.shape == (200, 500)
    assert y.shape == (200,)
    assert truth["strong"].size == 3
    assert truth["weak"].size == 5
    assert truth["epistatic_pairs"].shape == (2, 2)


def test_stage1_reduces_snps(tiny_data):
    X, y, truth = tiny_data
    cfg = s1.Stage1Config(maf_min=0.01, ld_window_size=20, ld_step_size=4, ld_r2_threshold=0.5)
    keep, info = s1.run_stage1(X, y, cfg)
    assert keep.dtype == bool
    assert keep.size == X.shape[1]
    assert keep.sum() <= X.shape[1]
    assert info["qc_kept"] >= keep.sum()


def test_stage2_returns_some(tiny_data):
    X, y, truth = tiny_data
    keep, info = s2.run_stage2(X, y, truth["block_assignment"], s2.Stage2Config())
    assert keep.size == X.shape[1]
    assert keep.sum() > 0


def test_stage3_union_topk(tiny_data):
    X, y, _ = tiny_data
    cfg = s3.Stage3Config(reliefF_neighbors=20, reliefF_topk=20, xgb_topk=20,
                          xgb_max_depth=3, xgb_n_estimators=50, seed=0)
    keep, info = s3.run_stage3(X, y, cfg)
    assert 1 <= keep.sum() <= 40


def test_stage4_runs(tiny_data):
    X, y, _ = tiny_data
    cfg = s4.Stage4Config(hidden_dim=32, bottleneck_dim=8, epochs=2,
                          batch_size=64, seed=0)
    contrib, Z, info = s4.run_stage4(X[:, :100], cfg)
    assert contrib.shape == (100,)
    assert Z.shape == (X.shape[0], 8)


def test_stage5_runs(tmp_path, tiny_data):
    X, y, _ = tiny_data
    project = tmp_path
    s5.make_synthetic_annotations(str(project), n_snps=X.shape[1], block_size=50, seed=0)
    out = s5.run_stage5(np.array([0, 50, 100, 200]), str(project), s5.Stage5Config())
    assert "n_mapped_genes" in out
    assert "pathway_enrichment" in out


def test_baselines(tiny_data):
    X, y, _ = tiny_data
    for fn in [baselines.baseline_pvalue,
               lambda X, y: baselines.baseline_random_forest(X, y, n_estimators=50, topk=20),
               lambda X, y: baselines.baseline_reliefF(X, y, topk=20, n_neighbors=20)]:
        m = fn(X, y)
        assert m.size == X.shape[1]


def test_evaluation_metrics(tiny_data):
    X, y, truth = tiny_data
    mask = np.zeros(X.shape[1], dtype=bool); mask[truth["strong"]] = True
    rec = evaluation.causal_recovery(mask, truth)
    assert rec["strong"]["recall"] == pytest.approx(1.0)
