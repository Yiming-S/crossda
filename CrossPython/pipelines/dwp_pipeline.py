"""
DWP Pipeline — Distance-Weighted Pooling

  Phase 1: MAP-identical CV to select best feature/DA/classifier
  Phase 2: Compute source-to-target distances in best feature space
  Phase 3: All-source continuous inverse-distance weighting
  Phase 4: Deterministic weighted merge (replication)
  Phase 5: DA once on merged pool + single classifier

  DWP is evaluated under the same MAP-selected configuration so that
  the comparison isolates the effect of replacing uniform full pooling
  with continuous distance-weighted full pooling at final training time.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone

from da4bci import domain_adaptation

from ..pipelines.pipeline_utils import (
    extract_features_train,
    extract_features_test,
    get_classifier,
    distance_ci,
    scale_to_target,
)
from ..pipelines.map_pipeline import (
    _normalize_score_mode,
    _score_kfold_session,
    _score_loso,
    _score_pairwise,
)
from ..pipelines.mmp_pipeline import _weight_from_D, _merge_weighted

logger = logging.getLogger(__name__)


def DWP(
    train_sessions: List[dict],
    test_session: List[dict],
    params: dict,
    feature: Optional[str] = None,
    classifier: Optional[str] = None,
    da: Optional[str] = None,
    dist_type: str = "mmd",
    dist_param: Optional[dict] = None,
    score: str = "kfold",
    k_sess: int = 2,
    n_repeats: int = 1,
    shuffle_sessions: bool = True,
    seed: int = 1,
    nfolds_out: int = 5,
    epsilon: float = 1e-6,
    scale_for_distance: bool = True,
    weight_power: float = 1.0,
    merge_scale: int = 50,
    # Bootstrap for phase-2 distance estimation.
    # Default off (B=1 + direct): deterministic single-shot distance, ~200x
    # faster than B=200/boot_mean. DWP only uses `est` (not CI), so bootstrap
    # smoothing is usually not worth its cost; set explicitly if needed.
    dist_bootstrap_B: int = 1,
    dist_est_method: str = "direct",
    verbose: bool = False,
    **_ignored,
) -> Dict[str, Any]:
    if len(train_sessions) < 2:
        raise ValueError("DWP requires at least two training sessions.")

    dist_param = dist_param or {}
    score_mode = _normalize_score_mode(score)

    feat_names = [feature] if feature else list(params.get("feature", {}).keys())
    da_names = [da] if da else list(params.get("da", {}).keys())
    clf_names = [classifier] if classifier else list(params.get("classifier", {}).keys())

    best_acc = -np.inf
    best_setup = None

    # ── Phase 1: MAP-style CV (identical to MAP, no weighting) ──────
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
                    logger.warning(f"[DWP] feature fit failed: {feat_name} | {e}")
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
                    if score_mode == "kfold":
                        acc = _score_kfold_session(
                            train_sessions, feat_name, feat_param,
                            da_name, da_control, clf_instance,
                            k_sess, n_repeats, shuffle_sessions, seed, nfolds_out,
                        )
                    elif score_mode == "loso":
                        acc = _score_loso(x_list, y_list, da_name, da_control,
                                          clf_instance, nfolds_out)
                    else:
                        acc = _score_pairwise(x_list, y_list, da_name, da_control,
                                              clf_instance, nfolds_out)
                except Exception as e:
                    if verbose:
                        logger.warning(f"[DWP][{score_mode}] {feat_name}/{da_name}/{clf_name} | {e}")
                    continue

                if verbose:
                    logger.info(f"[DWP][{score_mode}] {feat_name} | {da_name} | {clf_name} | cv={acc:.4f}")

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
        raise RuntimeError("DWP could not identify a valid configuration.")

    # ── Phase 2: Distance computation (in best feature space) ───────
    feat_obj = best_setup["feature_obj"]
    feat_nm = best_setup["feature"]["method"]

    # kfold mode: feature not fitted during CV, fit now on all sources
    if feat_obj is None:
        all_X = np.concatenate([ses["x"] for ses in train_sessions], axis=0)
        all_y = np.concatenate([ses["y"] for ses in train_sessions])
        _, feat_obj = extract_features_train(all_X, all_y, feat_nm,
                                              best_setup["feature"]["param"])
        best_setup["feature_obj"] = feat_obj

    X_list = [extract_features_test(ses["x"], feat_obj, feat_nm) for ses in train_sessions]
    y_list = [ses["y"] for ses in train_sessions]
    X_T = extract_features_test(test_session[0]["x"], feat_obj, feat_nm)
    y_T = test_session[0]["y"]

    # Source-to-target distances
    if scale_for_distance:
        x_pre, t_pre = scale_to_target(X_list, X_T)
    else:
        x_pre, t_pre = X_list, X_T

    dist_seed = seed if seed is not None else None
    dist_info = [
        {"i": i, **distance_ci(
            xs, t_pre, dist_type, dist_param,
            B=dist_bootstrap_B, alpha=0.05, est_method=dist_est_method,
            rng=np.random.default_rng(dist_seed + i) if dist_seed is not None else None,
        )}
        for i, xs in enumerate(x_pre)
    ]

    dists = np.array([d["est"] for d in dist_info])

    # ── Phase 3: All-source soft weights ────────────────────────────
    uniform_w = np.full(len(train_sessions), 1.0 / len(train_sessions))

    # Fallback: all distances non-finite → uniform weights
    distance_fallback = not np.any(np.isfinite(dists))
    if distance_fallback:
        if verbose:
            logger.warning("[DWP] all distances non-finite, falling back to uniform weights")
        weights = uniform_w.copy()
    else:
        # Replace individual non-finite distances with max finite distance
        finite_mask = np.isfinite(dists)
        dists[~finite_mask] = np.max(dists[finite_mask])
        weights = _weight_from_D(dists, eps=epsilon, p=weight_power, cap=1.0)

    # ── Phase 4: Weighted merge (deterministic replication) ─────────
    exact_map_bypass = distance_fallback or np.allclose(weights, uniform_w, atol=1e-8)

    if exact_map_bypass:
        # Exact MAP: no replication, use raw concatenated data
        X_merged = np.concatenate(X_list, axis=0)
        y_merged = np.concatenate(y_list)
        mult = np.ones(len(train_sessions), dtype=int)
        w_realized = uniform_w.copy()
    else:
        X_merged, y_merged = _merge_weighted(X_list, y_list, weights, scale=merge_scale)
        mult = np.maximum(1, np.round(weights * merge_scale).astype(int))
        w_realized = mult / mult.sum()

    # ── Phase 5a: MAP baseline (unweighted, no DA) ──────────────────
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list)

    clf_instance = best_setup["clf_instance"]
    map_base_clf = clone(clf_instance)
    map_base_clf.fit(X_all, y_all)
    acc_map_base = float(np.mean(map_base_clf.predict(X_T) == y_T))

    # ── Phase 5b: DWP baseline (weighted merge, no DA) ──────────────
    dwp_base_clf = clone(clf_instance)
    dwp_base_clf.fit(X_merged, y_merged)
    y_pred_base = dwp_base_clf.predict(X_T)
    acc_dwp_base = float(np.mean(y_pred_base == y_T))

    # ── Phase 5c: DWP with DA (weighted merge + DA once) ────────────
    da_method = best_setup["da"]["method"]
    da_ctrl = best_setup["da"]["control"]

    if da_method == "none":
        acc_dwp_da = acc_dwp_base
        final_model = dwp_base_clf
        y_pred = y_pred_base
    else:
        try:
            da_result = domain_adaptation(source_data=X_merged, target_data=X_T,
                                          method=da_method, control=da_ctrl)
            X_src_da = da_result.get("weighted_source_data")
            X_tgt_da = da_result.get("target_data")
            if X_src_da is None or X_tgt_da is None:
                raise KeyError("DA returned no recognized source/target keys")

            final_model = clone(clf_instance)
            final_model.fit(X_src_da, y_merged)
            y_pred = final_model.predict(X_tgt_da)
            acc_dwp_da = float(np.mean(y_pred == y_T))
        except Exception as e:
            if verbose:
                logger.warning(f"[DWP] DA failed, falling back to baseline: {e}")
            acc_dwp_da = acc_dwp_base
            final_model = dwp_base_clf
            y_pred = y_pred_base

    # ── Phase 6: Session roles + diagnostics ────────────────────────
    t_id = test_session[0].get("id")
    t_label = test_session[0].get("label", f"sess_{t_id}")

    neff = float(1.0 / np.sum(weights**2))
    w_entropy = float(-(weights * np.log(weights + 1e-12)).sum())
    rounding_err = float(np.abs(weights - w_realized).sum())

    session_roles = []
    for i in range(len(train_sessions)):
        sid = train_sessions[i].get("id", i)
        slabel = train_sessions[i].get("label", f"sess_{i}")
        ci = dist_info[i]
        dist_fields = {
            "dist_to_session": t_id,
            "dist_est_method": "boot_mean",
            "dist_est": ci.get("est"),
            "dist_lwr": ci.get("lwr"),
            "dist_upr": ci.get("upr"),
        }

        # weighting stage
        session_roles.append({
            "stage": "weighting", "local_idx": i,
            "session_abs_idx": sid, "session_label": slabel,
            "role": "weighted", "is_best": False,
            "weight": float(weights[i]), **dist_fields,
        })

        # final stage
        session_roles.append({
            "stage": "final", "local_idx": i,
            "session_abs_idx": sid, "session_label": slabel,
            "role": "train", "is_best": False,
            "weight": float(weights[i]), **dist_fields,
        })

    # Target rows
    for stg in ("weighting", "final"):
        session_roles.append({
            "stage": stg, "local_idx": None,
            "session_abs_idx": t_id, "session_label": t_label,
            "role": "target", "is_best": False, "weight": None,
            "dist_to_session": None, "dist_est_method": None,
            "dist_est": None, "dist_lwr": None, "dist_upr": None,
        })

    best_setup["cvMeanAcc"] = best_acc
    detail = {
        "pipeline": "DWP",
        "dist_type": dist_type,
        "score_mode": best_setup["scoreMode"],
        "weight_power": weight_power,
        "merge_scale": merge_scale,
        "dist_bootstrap_B": dist_bootstrap_B,
        "dist_est_method": dist_est_method,
        "distance_fallback": distance_fallback,
        "exact_map_bypass": exact_map_bypass,
        "weights": weights.tolist(),
        "multiplicities": mult.tolist(),
        "weights_realized": w_realized.tolist(),
        "neff": neff,
        "weight_entropy": w_entropy,
        "w_max": float(weights.max()),
        "w_min": float(weights.min()),
        "rounding_l1_error": rounding_err,
        "distance_table": [
            {"i": d["i"], "est": d["est"], "lwr": d["lwr"], "upr": d["upr"]}
            for d in dist_info
        ],
    }

    return {
        "baseline": acc_dwp_base,
        "acc_DA": acc_dwp_da,
        "acc_map_base": acc_map_base,
        "bestSetup": best_setup,
        "model": final_model,
        "detail": detail,
        "target_id": t_id,
        "target_label": t_label,
        "session_roles": session_roles,
        "y_pred": y_pred,
        "y_true": y_T,
    }
