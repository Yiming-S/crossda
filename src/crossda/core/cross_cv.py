"""
cross_cv.py — outer-fold CV scorer for the MAP-style candidate selection.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from ..pipelines.pipeline_utils import apply_da


def _outer_folds(y: NDArray, nfolds_out: int = 5, random_state: int = 42) -> List[NDArray]:
    """Stratified outer-fold test-index lists.

    The split uses a fixed ``random_state`` so candidate scoring is deterministic.
    map_seed / fold_seed deliberately do NOT perturb this inner CV — the scoring
    of every (feature, DA, classifier) candidate is reproducible by design.
    """
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=nfolds_out, shuffle=True, random_state=random_state)
    return [test_idx for _, test_idx in skf.split(np.zeros(len(y)), y)]


def map_cv(
    src_X: NDArray, src_y: NDArray,
    tgt_X: NDArray, tgt_y: NDArray,
    da_method: str, da_control: Optional[dict],
    clf: Any, nfolds_out: int = 5,
) -> float:
    """
    Outer-fold CV for cross-session adaptation (transductive DA setting).

    Domain adaptation is applied once to ALL source/target data (it does not
    depend on the fold), then the classifier is trained on source + the
    target-train fold and evaluated on the target-test fold.
    """
    # DA is fold-invariant here — compute it once, not once per fold.
    x_src_da, x_tgt_da = apply_da(da_method, da_control, src_X, tgt_X)

    folds = _outer_folds(tgt_y, nfolds_out=nfolds_out)
    accs = np.zeros(len(folds))
    for i, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(tgt_y)), test_idx)
        model = clone(clf)
        model.fit(
            np.vstack([x_src_da, x_tgt_da[train_idx]]),
            np.concatenate([src_y, tgt_y[train_idx]]),
        )
        accs[i] = np.mean(model.predict(x_tgt_da[test_idx]) == tgt_y[test_idx])

    return float(np.mean(accs))
