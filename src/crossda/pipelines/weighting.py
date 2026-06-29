"""
weighting.py — inverse-distance weighting and weighted oversampling.

Neutral home for the distance -> weight -> merge helpers shared by the
distance-based pipelines (MMP, DWP). Previously these lived in mmp_pipeline.py
and were cross-imported by dwp_pipeline.py; centralizing them here removes that
inverted dependency. Behavior is identical to the original definitions.
"""

from __future__ import annotations

import numpy as np


def _weights_for_indices(ci_tbl, idx, epsilon, p_weight, w_max):
    est_map = {c["i"]: c["est"] for c in ci_tbl}
    ests = np.array([est_map[i] for i in idx], dtype=float)
    return _weight_from_D(ests, epsilon, p_weight, w_max)


def _weight_from_D(D, eps=1e-6, p=1.0, cap=1.0):
    D = np.array(D, dtype=float)  # copy — never mutate the caller's array in place
    D[~np.isfinite(D)] = np.nanmax(D[np.isfinite(D)]) if np.any(np.isfinite(D)) else 1.0
    w = (D + eps) ** (-p)
    w /= w.sum()
    if np.isfinite(cap) and cap < 1:
        w = np.minimum(w, cap)
        w /= w.sum()
    return w


def _merge_weighted(X_list, y_list, weights, scale=50):
    mult = np.maximum(1, np.round(weights * scale).astype(int))
    X_parts, y_parts = [], []
    for X, y, m in zip(X_list, y_list, mult):
        idx = np.tile(np.arange(X.shape[0]), m)[:m * X.shape[0]]
        X_parts.append(X[idx])
        y_parts.append(np.asarray(y)[idx])
    return np.vstack(X_parts), np.concatenate(y_parts)
