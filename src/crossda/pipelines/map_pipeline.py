"""
MAP Pipeline — Merge & Adapt Pipeline (supervised selection)

  1) Score each feature × DA × classifier by cross-session CV
  2) Merge ALL training sessions, apply best DA, train classifier
  3) Return DA accuracy and no-DA baseline
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import clone

from ..pipelines.pipeline_utils import (
    extract_features_train,
    extract_features_test,
    apply_da,
)
from ..pipelines.scoring import _normalize_score_mode, run_grid_search
from ..pipelines.session_roles import EMPTY_DIST, target_rows

logger = logging.getLogger(__name__)


def MAP(
    train_sessions: List[dict],
    test_session: List[dict],
    params: dict,
    feature: Optional[str] = None,
    classifier: Optional[str] = None,
    da: Optional[str] = None,
    score: str = "kfold",
    k_sess: int = 4,
    n_repeats: int = 1,
    shuffle_sessions: bool = True,
    seed: int = 1,
    nfolds_out: int = 5,
    verbose: bool = False,
    **_ignored,
) -> Dict[str, Any]:
    if len(train_sessions) < 2:
        raise ValueError("MAP requires at least two training sessions.")

    score_mode = _normalize_score_mode(score)

    # ── Selection: best feature × DA × classifier by cross-session CV ─────
    best_setup = run_grid_search(
        params, feature, classifier, da, score_mode, train_sessions,
        k_sess=k_sess, n_repeats=n_repeats, shuffle_sessions=shuffle_sessions,
        seed=seed, nfolds_out=nfolds_out, log_tag="[MAP]", verbose=verbose,
    )

    if best_setup is None:
        raise RuntimeError("MAP could not identify a valid configuration.")

    # ── Final Training & Testing ─────────────────────────────────────────
    feat_obj = best_setup["feature_obj"]
    feat_nm = best_setup["feature"]["method"]

    # kfold mode: fit features on all training sessions for the final model
    if feat_obj is None:
        all_X = np.concatenate([ses["x"] for ses in train_sessions], axis=0)
        all_y = np.concatenate([ses["y"] for ses in train_sessions])
        _, feat_obj = extract_features_train(all_X, all_y, feat_nm, best_setup["feature"]["param"])
        best_setup["feature_obj"] = feat_obj

    X_train = np.concatenate([
        extract_features_test(ses["x"], feat_obj, feat_nm) for ses in train_sessions
    ], axis=0)
    y_train = np.concatenate([ses["y"] for ses in train_sessions])
    X_test = extract_features_test(test_session[0]["x"], feat_obj, feat_nm)
    y_test = test_session[0]["y"]

    da_method = best_setup["da"]["method"]
    da_ctrl = best_setup["da"]["control"]

    X_train_da, X_test_da = apply_da(da_method, da_ctrl, X_train, X_test)

    final_clf = clone(best_setup["clf_instance"])
    final_clf.fit(X_train_da, y_train)
    acc_da = float(np.mean(final_clf.predict(X_test_da) == y_test))

    if da_method == "none":
        acc_base = acc_da
    else:
        base_clf = clone(best_setup["clf_instance"])
        base_clf.fit(X_train, y_train)
        acc_base = float(np.mean(base_clf.predict(X_test) == y_test))

    # ── y_pred ───────────────────────────────────────────────────────
    y_pred = final_clf.predict(X_test_da)

    # ── session_roles ────────────────────────────────────────────────
    # NOTE: MAP records final-only provenance. Inner CV fold assignments
    # (which sessions were train/target within kfold/loso/pairwise scoring)
    # are NOT recorded. All train sessions participate equally in MAP.
    # If inner CV auditing is needed in the future, add a "cv" stage here.
    t_id = test_session[0].get("id")
    t_label = test_session[0].get("label", f"sess_{t_id}")

    session_roles = []
    for i in range(len(train_sessions)):
        sid = train_sessions[i].get("id", i)
        slabel = train_sessions[i].get("label", f"sess_{i}")
        session_roles.append({
            "stage": "final", "local_idx": i,
            "session_abs_idx": sid, "session_label": slabel,
            "role": "train", "is_best": False, "weight": None,
            **EMPTY_DIST,
        })

    session_roles += target_rows(("final",), t_id, t_label)

    return {
        "baseline": acc_base, "acc_DA": acc_da, "bestSetup": best_setup, "model": final_clf,
        "detail": {"pipeline": "MAP", "score_mode": best_setup["scoreMode"]},
        "target_id": t_id, "target_label": t_label,
        "session_roles": session_roles,
        "y_pred": y_pred, "y_true": y_test,
    }
