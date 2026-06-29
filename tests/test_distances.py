"""Distances, CI table, feature extraction and classifier factory."""

import numpy as np
import pytest

from crossda.pipelines.pipeline_utils import (
    session_distance,
    distance_ci,
    scale_to_target,
    compute_distance_ci_table,
    get_classifier,
    extract_features_train,
    extract_features_test,
)


def _two_clouds(seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.standard_normal((30, 8))
    xt = rng.standard_normal((30, 8)) + 1.0  # shifted target
    return xs, xt


def test_session_distance_mmd_finite():
    xs, xt = _two_clouds()
    d = session_distance(xs, xt, "mmd")
    assert np.isfinite(d) and d >= 0


def test_session_distance_unknown_type_raises():
    xs, xt = _two_clouds()
    with pytest.raises(ValueError):
        session_distance(xs, xt, "not_a_distance")


def test_distance_ci_keys_and_order():
    xs, xt = _two_clouds()
    ci = distance_ci(xs, xt, "mmd", B=20, rng=np.random.default_rng(1))
    assert set(ci) == {"est", "lwr", "upr"}
    assert ci["lwr"] <= ci["upr"]


def test_scale_to_target_centres_on_target():
    xs, xt = _two_clouds()
    scaled_sources, scaled_target = scale_to_target([xs], xt)
    assert np.allclose(scaled_target.mean(axis=0), 0, atol=1e-6)
    assert scaled_sources[0].shape == xs.shape


def test_compute_distance_ci_table_one_row_per_source():
    rng = np.random.default_rng(2)
    sources = [rng.standard_normal((25, 8)) for _ in range(3)]
    target = rng.standard_normal((25, 8))
    tbl = compute_distance_ci_table(
        sources, target, "mmd", B=20,
        rng_for_index=lambda i: np.random.default_rng(100 + i),
    )
    assert [row["i"] for row in tbl] == [0, 1, 2]
    assert all({"est", "lwr", "upr"} <= set(row) for row in tbl)


def test_get_classifier_known_and_unknown():
    assert hasattr(get_classifier("lda"), "fit")
    assert hasattr(get_classifier("svm_radial"), "fit")
    with pytest.raises(ValueError):
        get_classifier("not_a_classifier")


@pytest.mark.parametrize("feat", ["logvar", "CSP", "TS"])
def test_feature_roundtrip_shapes(feat, synthetic_sessions):
    ses = synthetic_sessions[0]
    X, y = ses["x"], ses["y"]
    feats, obj = extract_features_train(X, y, feat)
    assert feats.shape[0] == X.shape[0]
    # Applying the fitted extractor to new data gives the same feature width.
    test_feats = extract_features_test(synthetic_sessions[1]["x"], obj, feat)
    assert test_feats.shape[1] == feats.shape[1]
