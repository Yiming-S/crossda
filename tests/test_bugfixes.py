"""Regression tests for the bug-hunt batch."""

import numpy as np

from crossda.config import Config
from crossda.core.cross_cv import _outer_folds
from crossda.core.method_bank import generate_method_bank
from crossda.core import workers as W


def test_config_accepts_scalar_string_pipelines_and_datasets():
    cfg = Config(pipelines="MAP", datasets="ma2020")
    assert cfg.pipelines == ["MAP"]
    assert cfg.datasets == ["ma2020"]


def test_outer_folds_clamps_to_small_class_count():
    # 3 members per class with nfolds_out=5 used to raise; now adapts the fold count.
    folds = _outer_folds(np.array([0, 1] * 3), nfolds_out=5)
    assert 2 <= len(folds) <= 3


def test_checkpoint_different_column_count_does_not_crash(tmp_path):
    mb = generate_method_bank("smoke")
    partial = mb.drop(columns=["space_id"]).copy()  # fewer columns than current mb
    partial.loc[0, "cvMeanAcc"] = 0.9
    path = str(tmp_path / "c.pkl")
    W._save_checkpoint(path, partial, [], 0)
    loaded, _, _ = W._load_checkpoint(path, mb.copy())  # must not IndexError
    assert loaded.loc[0, "cvMeanAcc"] == 0.9


def test_checkpoint_reordered_value_columns_not_swapped(tmp_path):
    mb = generate_method_bank("smoke")
    cols = list(mb.columns)
    i, j = cols.index("cvMeanAcc"), cols.index("baseline")
    cols[i], cols[j] = cols[j], cols[i]          # same names, swapped order
    cached = mb[cols].copy()
    cached.loc[0, "cvMeanAcc"] = 0.71
    cached.loc[0, "baseline"] = 0.50
    path = str(tmp_path / "c.pkl")
    W._save_checkpoint(path, cached, [], 0)
    loaded, _, _ = W._load_checkpoint(path, mb.copy())
    assert loaded.loc[0, "cvMeanAcc"] == 0.71    # matched by label, not position
    assert loaded.loc[0, "baseline"] == 0.50


def test_distance_ci_estimate_finite_inside_finite_ci():
    from crossda.pipelines import pipeline_utils as pu
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 4))
    X[0, 0] = np.nan                              # full-data direct estimate fails
    Y = rng.standard_normal((10, 4))
    ci = pu.distance_ci(X, Y, "mahalanobis", B=200, est_method="direct",
                        rng=np.random.default_rng(1))
    if np.isfinite(ci["lwr"]) and np.isfinite(ci["upr"]):
        assert np.isfinite(ci["est"])            # est must not be inf within a finite CI


def test_mmp_moe_survives_single_class_source():
    from crossda.pipelines.mmp_pipeline import _combiner_moe
    from crossda.pipelines.pipeline_utils import get_classifier
    rng = np.random.default_rng(0)
    X_sel = [rng.standard_normal((20, 4)), rng.standard_normal((20, 4))]
    Y_sel = [np.array([1, 2] * 10), np.array([1] * 20)]   # source 2 is single-class
    X_tar = rng.standard_normal((14, 4))
    Y_tar = np.array([1, 2] * 7)
    out = _combiner_moe(X_sel, Y_sel, [0.5, 0.5], X_tar, Y_tar,
                        "none", {}, get_classifier("lda"), has_label=True)
    assert out["y_pred"] is not None              # no IndexError from the single-class fit
