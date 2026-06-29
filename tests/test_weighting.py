"""Inverse-distance weighting and weighted oversampling."""

import numpy as np

from crossda.pipelines.weighting import _weight_from_D, _merge_weighted


def test_weights_sum_to_one_and_favour_near():
    w = _weight_from_D(np.array([1.0, 2.0, 4.0]))
    assert np.isclose(w.sum(), 1.0)
    # Smaller distance -> larger weight.
    assert w[0] > w[1] > w[2]


def test_weight_from_D_handles_nonfinite():
    w = _weight_from_D(np.array([1.0, np.inf, 2.0]))
    assert np.isclose(w.sum(), 1.0)
    assert np.all(np.isfinite(w))


def test_merge_weighted_oversamples_by_weight():
    X = [np.zeros((4, 3)), np.ones((4, 3))]
    y = [np.array([1, 1, 2, 2]), np.array([1, 1, 2, 2])]
    weights = np.array([0.9, 0.1])
    Xm, ym = _merge_weighted(X, y, weights, scale=50)
    assert Xm.shape[0] == ym.shape[0]
    # The heavily-weighted source contributes more rows.
    n_zero = int((Xm == 0).all(axis=1).sum())
    n_one = int((Xm == 1).all(axis=1).sum())
    assert n_zero > n_one
