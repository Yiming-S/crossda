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
from numpy.typing import NDArray
from sklearn.base import clone

from da4bci import domain_adaptation

from ..core.cross_cv import map_cv
from ..pipelines.pipeline_utils import (
    extract_features_train,
    extract_features_test,
    get_classifier,
)

logger = logging.getLogger(__name__)


def MAP(
    train_sessions: List[dict],
    test_session: List[dict],
    params: dict,
    feature: Optional[str] = None,
    classifier: Optional[str] = None,
    da: Optional[str] = None,
    score: str = "kfold",
    k_sess: int = 2,
    n_repeats: int = 1,
    shuffle_sessions: bool = True,
    seed: int = 1,
    nfolds_out: int = 5,
    epsilon: float = 1e-6,
    verbose: bool = False,
    **_ignored,
) -> Dict[str, Any]:
    if len(train_sessions) < 2:
        raise ValueError("MAP requires at least two training sessions.")

    score_mode = _normalize_score_mode(score)

    feat_names = [feature] if feature else list(params.get("feature", {}).keys())
    da_names = [da] if da else list(params.get("da", {}).keys())
    clf_names = [classifier] if classifier else list(params.get("classifier", {}).keys())

    best_acc = -np.inf
    best_setup = None

    # ── Optimization Loop ────────────────────────────────────────────────
    for feat_name in feat_names:
        feat_param = params.get("feature", {}).get(feat_name, {})

        # For kfold: feature extraction done fold-wise inside scorer.
        # For loso/pairwise: fit once on all training data.
        feat_obj = None
        x_list = None
        y_list = None

        if score_mode != "kfold":
            all_X = np.concatenate([ses["x"] for ses in train_sessions], axis=0)
            all_y = np.concatenate([ses["y"] for ses in train_sessions])
            try:
                features, feat_obj = extract_features_train(all_X, all_y, feat_name, feat_param)
            except Exception as e:
                if verbose:
                    logger.warning(f"[MAP] feature fit failed: {feat_name} | {e}")
                continue

            # Per-session feature slices
            x_list, offset = [], 0
            for ses in train_sessions:
                n = ses["x"].shape[0]
                x_list.append(features[offset:offset + n])
                offset += n
            y_list = [ses["y"] for ses in train_sessions]

        for da_name in da_names:
            da_control = {} if da_name == "none" else params.get("da", {}).get(da_name, {})

            for clf_name in clf_names:
                clf_params_grid = params.get("classifier", {}).get(clf_name, {})
                clf_instance = get_classifier(clf_name, clf_params_grid)

                try:
                    if score_mode == "kfold":
                        acc = _score_kfold_session(
                            train_sessions, feat_name, feat_param,
                            da_name, da_control, clf_instance,
                            k_sess, n_repeats, shuffle_sessions, seed, nfolds_out,
                        )
                    elif score_mode == "loso":
                        acc = _score_loso(x_list, y_list, da_name, da_control, clf_instance, nfolds_out)
                    else:
                        acc = _score_pairwise(x_list, y_list, da_name, da_control, clf_instance, nfolds_out)
                except Exception as e:
                    if verbose:
                        logger.warning(f"[MAP][{score_mode}] {feat_name}/{da_name}/{clf_name} | {e}")
                    continue

                if verbose:
                    logger.info(f"[MAP][{score_mode}] {feat_name} | {da_name} | {clf_name} | cv={acc:.4f}")

                if np.isfinite(acc) and acc > best_acc:
                    best_acc = acc
                    best_setup = {
                        "feature_obj": feat_obj,
                        "feature": {"method": feat_name, "param": feat_param},
                        "da": {"method": da_name, "control": da_control},
                        "classifier": {"method": clf_name, "param": clf_params_grid},
                        "clf_instance": clf_instance,
                        "cvMeanAcc": acc,
                        "scoreMode": score_mode,
                    }

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

    if da_method == "none":
        X_train_da, X_test_da = X_train, X_test
    else:
        da_result = domain_adaptation(source_data=X_train, target_data=X_test,
                                      method=da_method, control=da_ctrl)
        X_train_da = da_result["weighted_source_data"]
        X_test_da = da_result["target_data"]

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
            "dist_to_session": None, "dist_est_method": None,
            "dist_est": None, "dist_lwr": None, "dist_upr": None,
        })

    session_roles.append({
        "stage": "final", "local_idx": None,
        "session_abs_idx": t_id, "session_label": t_label,
        "role": "target", "is_best": False, "weight": None,
        "dist_to_session": None, "dist_est_method": None,
        "dist_est": None, "dist_lwr": None, "dist_upr": None,
    })

    return {
        "baseline": acc_base, "acc_DA": acc_da, "bestSetup": best_setup, "model": final_clf,
        "detail": {"pipeline": "MAP", "score_mode": best_setup["scoreMode"]},
        "target_id": t_id, "target_label": t_label,
        "session_roles": session_roles,
        "y_pred": y_pred, "y_true": y_test,
    }


###############################################################################
# Score mode
###############################################################################

def _normalize_score_mode(score: str) -> str:
    s = score.lower().strip()
    aliases = {
        "cv": "pairwise", "pair": "pairwise", "pairwise_cv": "pairwise", "pairwise": "pairwise",
        "k-fold": "kfold", "kf": "kfold", "session_kfold": "kfold", "kfold": "kfold",
        "leave_one_session_out": "loso", "innercv": "loso", "inner_cv": "loso", "loso": "loso",
    }
    if s not in aliases:
        raise ValueError(f"MAP: `score` must be one of: 'loso', 'pairwise', 'kfold'. Got {s!r}")
    return aliases[s]


###############################################################################
# Scoring strategies
###############################################################################

def _score_pairwise(x_list, y_list, da_name, da_control, clf, nfolds_out):
    n = len(x_list)
    accs = [
        map_cv(x_list[j], y_list[j], x_list[i], y_list[i], da_name, da_control, clf, nfolds_out)
        for i in range(n) for j in range(n) if i != j
    ]
    return float(np.mean(accs)) if accs else np.nan


def _score_loso(x_list, y_list, da_name, da_control, clf, nfolds_out):
    n = len(x_list)
    accs = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        accs.append(map_cv(
            np.vstack([x_list[j] for j in tr]),
            np.concatenate([y_list[j] for j in tr]),
            x_list[i], y_list[i], da_name, da_control, clf, nfolds_out,
        ))
    return float(np.mean(accs))


def _score_kfold_session(
    train_sessions, feat_name, feat_param,
    da_name, da_control, clf,
    k_sess, n_repeats, shuffle_sessions, seed, nfolds_out,
):
    """Session-level k-fold CV with fold-wise feature fitting."""
    ntrain = len(train_sessions)
    k_eff = max(2, min(int(k_sess), ntrain))
    rep_scores = []

    for r in range(max(1, int(n_repeats))):
        rng = np.random.default_rng(seed + r if seed is not None else None)
        folds = _make_session_folds(ntrain, k_eff, shuffle_sessions, rng)
        fold_scores = []

        for val_idx in folds:
            tr_idx = [j for j in range(ntrain) if j not in val_idx]
            if not tr_idx:
                continue

            fold_X = np.concatenate([train_sessions[j]["x"] for j in tr_idx], axis=0)
            fold_y = np.concatenate([train_sessions[j]["y"] for j in tr_idx])

            try:
                _, feat_obj_fold = extract_features_train(fold_X, fold_y, feat_name, feat_param)
            except Exception:
                continue

            src_X = np.concatenate([
                extract_features_test(train_sessions[j]["x"], feat_obj_fold, feat_name) for j in tr_idx
            ], axis=0)

            acc_vals = []
            for j in val_idx:
                tgt_X = extract_features_test(train_sessions[j]["x"], feat_obj_fold, feat_name)
                try:
                    acc_vals.append(map_cv(
                        src_X, fold_y, tgt_X, train_sessions[j]["y"],
                        da_name, da_control, clf, nfolds_out,
                    ))
                except Exception:
                    pass

            if acc_vals:
                fold_scores.append(float(np.mean(acc_vals)))

        if fold_scores:
            rep_scores.append(float(np.mean(fold_scores)))

    return float(np.mean(rep_scores)) if rep_scores else np.nan


def _make_session_folds(n_sessions, k, shuffle, rng):
    idx = np.arange(n_sessions)
    if shuffle:
        rng.shuffle(idx)
    fold_id = np.arange(n_sessions) % k
    return [idx[fold_id == f].tolist() for f in range(k)]
