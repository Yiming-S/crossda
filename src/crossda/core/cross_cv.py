"""
cross_cv.py — Outer-fold CV helpers for MAP pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from ..pipelines.pipeline_utils import apply_da


def create_nested_folds(
    y: NDArray, nfolds_out: int = 5, nfolds_in: int = 3, random_state: int = 42,
) -> Dict[str, List]:
    """Create nested (outer + inner) stratified CV folds."""
    y = np.asarray(y)
    skf_out = StratifiedKFold(n_splits=nfolds_out, shuffle=True, random_state=random_state)
    outer_folds = [test_idx for _, test_idx in skf_out.split(np.zeros(len(y)), y)]

    inner_folds = []
    all_idx = np.arange(len(y))
    for i, test_idx in enumerate(outer_folds):
        train_idx = np.setdiff1d(all_idx, test_idx)
        y_train = y[train_idx]
        k_in = min(nfolds_in, len(np.unique(y_train)))
        if k_in < 2:
            inner_folds.append([train_idx])
            continue
        skf_in = StratifiedKFold(n_splits=k_in, shuffle=True, random_state=random_state + i + 1)
        inner_folds.append([
            train_idx[inner_test_rel]
            for _, inner_test_rel in skf_in.split(np.zeros(len(y_train)), y_train)
        ])

    return {"outer": outer_folds, "inner": inner_folds}


def map_cv(
    src_X: NDArray, src_y: NDArray,
    tgt_X: NDArray, tgt_y: NDArray,
    da_method: str, da_control: Optional[dict],
    clf: Any, nfolds_out: int = 5,
) -> float:
    """
    Outer-fold CV for cross-session adaptation (transductive DA setting).

    DA applied to ALL target data before splitting. Classifier trained on
    source + target-train fold, evaluated on target-test fold.
    """
    folds = create_nested_folds(tgt_y, nfolds_out=nfolds_out, nfolds_in=2)
    accs = np.zeros(len(folds["outer"]))

    for i, test_idx in enumerate(folds["outer"]):
        train_idx = np.setdiff1d(np.arange(len(tgt_y)), test_idx)

        x_src_da, x_tgt_da = apply_da(da_method, da_control, src_X, tgt_X)

        model = clone(clf)
        model.fit(
            np.vstack([x_src_da, x_tgt_da[train_idx]]),
            np.concatenate([src_y, tgt_y[train_idx]]),
        )
        accs[i] = np.mean(model.predict(x_tgt_da[test_idx]) == tgt_y[test_idx])

    return float(np.mean(accs))
