"""
scoring.py — cross-session CV scorers and score-mode dispatch.

Neutral home for the candidate-scoring strategies (kfold / loso / pairwise)
used by MAP, DWP and the BDP degrade path. These functions previously lived in
map_pipeline.py and were cross-imported by dwp_pipeline.py and bdp_pipeline.py;
centralizing them here removes that inverted dependency. The individual scorer
bodies are unchanged; ``score_session_cv`` factors out the 3-way dispatch that
each caller used to repeat inline.
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.cross_cv import map_cv
from ..pipelines.pipeline_utils import (
    extract_features_train,
    extract_features_test,
    get_classifier,
)

logger = logging.getLogger(__name__)


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
        raise ValueError(f"`score` must be one of: 'loso', 'pairwise', 'kfold'. Got {s!r}")
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


###############################################################################
# Dispatch
###############################################################################

def score_session_cv(
    score, train_sessions, x_list, y_list,
    feat_name, feat_param, da_name, da_control, clf,
    *, k_sess, n_repeats, shuffle_sessions, seed, nfolds_out,
):
    """Score one (feature, DA, classifier) candidate via the requested CV mode.

    kfold refits the feature per fold (no leakage). loso/pairwise reuse the
    per-session ``x_list`` / ``y_list`` built from a full-data feature fit.
    """
    mode = _normalize_score_mode(score)
    if mode == "kfold":
        return _score_kfold_session(
            train_sessions, feat_name, feat_param,
            da_name, da_control, clf,
            k_sess, n_repeats, shuffle_sessions, seed, nfolds_out,
        )
    if mode == "loso":
        return _score_loso(x_list, y_list, da_name, da_control, clf, nfolds_out)
    return _score_pairwise(x_list, y_list, da_name, da_control, clf, nfolds_out)


###############################################################################
# Candidate grid search
###############################################################################

def run_grid_search(
    params, feature, classifier, da, score_mode, train_sessions,
    *, k_sess, n_repeats, shuffle_sessions, seed, nfolds_out,
    log_tag="", verbose=False,
):
    """Resolve feature/DA/classifier candidates and pick the best by CV score.

    Returns the winning ``best_setup`` dict (feature_obj / feature / da /
    classifier / clf_instance / cvMeanAcc / scoreMode) or None if no candidate
    scored finitely. This is the supervised selection phase shared verbatim by
    MAP and DWP. For ``kfold`` the feature is fit fold-wise inside the scorer;
    for loso/pairwise it is fit once on all training data.
    """
    feat_names = [feature] if feature else list(params.get("feature", {}).keys())
    da_names = [da] if da else list(params.get("da", {}).keys())
    clf_names = [classifier] if classifier else list(params.get("classifier", {}).keys())

    best_acc = -np.inf
    best_setup = None

    for feat_name in feat_names:
        feat_param = params.get("feature", {}).get(feat_name, {})
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
                    logger.warning(f"{log_tag} feature fit failed: {feat_name} | {e}")
                continue

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
                    acc = score_session_cv(
                        score_mode, train_sessions, x_list, y_list,
                        feat_name, feat_param, da_name, da_control, clf_instance,
                        k_sess=k_sess, n_repeats=n_repeats,
                        shuffle_sessions=shuffle_sessions, seed=seed, nfolds_out=nfolds_out,
                    )
                except Exception as e:
                    if verbose:
                        logger.warning(f"{log_tag}[{score_mode}] {feat_name}/{da_name}/{clf_name} | {e}")
                    continue

                if verbose:
                    logger.info(f"{log_tag}[{score_mode}] {feat_name} | {da_name} | {clf_name} | cv={acc:.4f}")

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

    return best_setup
