"""
BDP Pipeline — Bridge-Domain Pipeline (Multi-Source)

  Step 1: CI-based partition into Bridge set B and Far set F
  Step 2: Proxy tuning: train on F_proxy, validate on B_proxy (with DA as part of phi)
  Step 3: Final training: ONLY B_final → target T (DA + classifier with best phi)

  Partition modes:
    normal           — F_raw non-empty, disjoint from B_raw
    fallback_nonbridge — F_raw empty, finite non-bridge session borrowed into F_proxy
    fallback_bridge  — F_raw empty AND all finite sessions in B_raw, farthest bridge borrowed
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

logger = logging.getLogger(__name__)


def BDP(
    train_sessions: List[dict],
    test_session: List[dict],
    params: dict,
    feature: Optional[str] = None,
    classifier: Optional[str] = None,
    da: Optional[str] = None,
    dist_type: str = "mmd",
    dist_param: Optional[dict] = None,
    epsilon: float = 1e-6,
    scale_for_distance: bool = True,
    verbose: bool = False,
    seed: Optional[int] = None,
    proxy_direction: str = "far_to_bridge",
    **_ignored,
) -> Dict[str, Any]:
    if len(train_sessions) < 2:
        raise ValueError("BDP requires at least two training sessions.")
    if proxy_direction not in ("far_to_bridge", "bridge_to_far"):
        raise ValueError(f"proxy_direction must be 'far_to_bridge' or 'bridge_to_far', got {proxy_direction!r}")

    dist_param = dist_param or {}
    feat_names = [feature] if feature else list(params.get("feature", {}).keys())
    da_names = [da] if da else list(params.get("da", {}).keys())
    clf_names = [classifier] if classifier else list(params.get("classifier", {}).keys())

    best_score = -np.inf
    best_setup = None

    for feat_idx, feat_name in enumerate(feat_names):
        feat_param = params.get("feature", {}).get(feat_name, {})
        feat_seed = (seed * 131 + feat_idx) % (2**31 - 1) if seed is not None else None

        try:
            all_X = np.concatenate([ses["x"] for ses in train_sessions], axis=0)
            all_y = np.concatenate([ses["y"] for ses in train_sessions])
            _, feat_obj = extract_features_train(all_X, all_y, feat_name, feat_param)
        except Exception as e:
            if verbose:
                logger.warning(f"[BDP] feature fit failed: {feat_name} | {e}")
            continue

        x_list = [extract_features_test(ses["x"], feat_obj, feat_name) for ses in train_sessions]
        y_list = [ses["y"] for ses in train_sessions]
        x_test = extract_features_test(test_session[0]["x"], feat_obj, feat_name)

        # ── Step 1: CI-based B/F partition ────────────────────────────
        if scale_for_distance:
            x_pre, t_pre = scale_to_target(x_list, x_test)
        else:
            x_pre, t_pre = x_list, x_test

        ci_tbl = [
            {"i": i, **distance_ci(
                xs, t_pre, dist_type, dist_param,
                B=200, alpha=0.05, est_method="boot_mean",
                rng=np.random.default_rng(feat_seed + i) if feat_seed is not None else None,
            )}
            for i, xs in enumerate(x_pre)
        ]

        try:
            part = _partition_bridge_far(ci_tbl)
        except RuntimeError as e:
            if verbose:
                logger.warning(f"[BDP] partition failed: {feat_name} | {e}")
            continue

        if verbose:
            logger.info(
                f"[BDP] {feat_name} | B_raw={part['B_raw']} | F_raw={part['F_raw']} | "
                f"B_proxy={part['B_proxy']} | F_proxy={part['F_proxy']} | "
                f"mode={part['partition_mode']}"
            )

        X_B_proxy = np.vstack([x_list[i] for i in part["B_proxy"]])
        y_B_proxy = np.concatenate([y_list[i] for i in part["B_proxy"]])
        X_F_proxy = np.vstack([x_list[i] for i in part["F_proxy"]])
        y_F_proxy = np.concatenate([y_list[i] for i in part["F_proxy"]])

        # Assign proxy source/target based on direction
        if proxy_direction == "far_to_bridge":
            X_src, y_src = X_F_proxy, y_F_proxy
            X_tgt, y_tgt = X_B_proxy, y_B_proxy
        else:  # bridge_to_far
            X_src, y_src = X_B_proxy, y_B_proxy
            X_tgt, y_tgt = X_F_proxy, y_F_proxy

        dir_label = "F->B" if proxy_direction == "far_to_bridge" else "B->F"

        # ── Step 2: Proxy tuning ─────────────────────────────────────────
        for da_name in da_names:
            da_control = {} if da_name == "none" else params.get("da", {}).get(da_name, {})

            for clf_name in clf_names:
                clf_params = params.get("classifier", {}).get(clf_name, {})
                clf_instance = get_classifier(clf_name, clf_params)

                try:
                    if da_name == "none":
                        X_src_A, X_tgt_A = X_src, X_tgt
                    else:
                        da_out = domain_adaptation(source_data=X_src, target_data=X_tgt,
                                                   method=da_name, control=da_control)
                        X_src_A = da_out.get("weighted_source_data", X_src)
                        X_tgt_A = da_out.get("target_data", X_tgt)

                    model_phi = clone(clf_instance)
                    model_phi.fit(X_src_A, y_src)
                    proxy_acc = float(np.mean(model_phi.predict(X_tgt_A) == y_tgt))
                except Exception as e:
                    if verbose:
                        logger.warning(f"[BDP-proxy] {feat_name}/{da_name}/{clf_name} | {e}")
                    continue

                if verbose:
                    logger.info(f"[BDP-proxy] {feat_name} | {da_name} | {clf_name} | {dir_label}={proxy_acc:.4f}")

                if np.isfinite(proxy_acc) and proxy_acc > best_score:
                    best_score = proxy_acc
                    best_setup = {
                        "feature_obj": feat_obj, "clf_instance": clf_instance,
                        "feature": {"method": feat_name, "param": feat_param},
                        "da": {"method": da_name, "control": da_control},
                        "classifier": {"method": clf_name, "param": clf_params},
                        "best_idx": part["best_i"],
                        "B_raw": part["B_raw"], "F_raw": part["F_raw"],
                        "B_proxy": part["B_proxy"], "F_proxy": part["F_proxy"],
                        "B_final": part["B_final"],
                        "partition_mode": part["partition_mode"],
                        "borrowed_idx": part["borrowed_idx"],
                        "invalid_ci_idx": part["invalid_ci_idx"],
                        "ci_table": ci_tbl, "threshold_lwr": part["thr_lwr"],
                        "proxyAcc": proxy_acc,
                        "proxy_direction": proxy_direction,
                    }

    if best_setup is None:
        raise RuntimeError("BDP could not identify a valid configuration.")

    # ── Step 3: Final training (B_final → target T) ──────────────────
    feat_obj = best_setup["feature_obj"]
    feat_nm = best_setup["feature"]["method"]
    B_final = best_setup["B_final"]

    X_B = np.vstack([extract_features_test(train_sessions[i]["x"], feat_obj, feat_nm) for i in B_final])
    y_B = np.concatenate([train_sessions[i]["y"] for i in B_final])
    X_T = extract_features_test(test_session[0]["x"], feat_obj, feat_nm)
    y_T = test_session[0]["y"]

    clf_instance = best_setup["clf_instance"]
    base_clf = clone(clf_instance)
    base_clf.fit(X_B, y_B)
    acc_base = float(np.mean(base_clf.predict(X_T) == y_T))

    da_method = best_setup["da"]["method"]
    da_ctrl = best_setup["da"]["control"]

    if da_method == "none":
        acc_DA = acc_base
        final_model = base_clf
        y_pred = base_clf.predict(X_T)
    else:
        try:
            da_result = domain_adaptation(source_data=X_B, target_data=X_T,
                                          method=da_method, control=da_ctrl)
            final_model = clone(clf_instance)
            final_model.fit(da_result.get("weighted_source_data", X_B), y_B)
            y_pred = final_model.predict(da_result.get("target_data", X_T))
            acc_DA = float(np.mean(y_pred == y_T))
        except Exception as e:
            if verbose:
                logger.warning(f"[BDP-final] DA failed, falling back to baseline: {e}")
            acc_DA = acc_base
            final_model = base_clf
            y_pred = base_clf.predict(X_T)

    best_setup["cvMeanAcc"] = best_score
    detail = {
        "pipeline": "BDP", "dist_type": dist_type, "score_mode": "bridge_proxy",
        "proxy_direction": proxy_direction,
        "best_idx": best_setup["best_idx"],
        "B_raw": best_setup["B_raw"], "F_raw": best_setup["F_raw"],
        "B_proxy": best_setup["B_proxy"], "F_proxy": best_setup["F_proxy"],
        "B_final": best_setup["B_final"],
        "partition_mode": best_setup["partition_mode"],
        "borrowed_idx": best_setup["borrowed_idx"],
        "borrowed_abs_idx": (
            train_sessions[best_setup["borrowed_idx"]].get("id")
            if best_setup["borrowed_idx"] is not None else None
        ),
        "invalid_ci_idx": best_setup["invalid_ci_idx"],
        "threshold_lwr": best_setup["threshold_lwr"],
        "ci_table": best_setup["ci_table"],
        "proxy_acc": best_score,
    }

    # ── session_roles (from best_setup, NOT from loop-local part) ────
    B_raw_set = set(best_setup["B_raw"])
    F_raw_set = set(best_setup["F_raw"])
    B_proxy_set = set(best_setup["B_proxy"])
    F_proxy_set = set(best_setup["F_proxy"])
    B_final_set = set(best_setup["B_final"])
    invalid_set = set(best_setup["invalid_ci_idx"])
    best_i = best_setup["best_idx"]
    ci_map = {c["i"]: c for c in best_setup["ci_table"]}

    t_id = test_session[0].get("id")
    t_label = test_session[0].get("label", f"sess_{t_id}")

    session_roles = []
    for i in range(len(train_sessions)):
        sid = train_sessions[i].get("id", i)
        slabel = train_sessions[i].get("label", f"sess_{i}")
        ci = ci_map.get(i, {})
        dist_fields = {
            "dist_to_session": t_id,
            "dist_est_method": "boot_mean",
            "dist_est": ci.get("est"),
            "dist_lwr": ci.get("lwr"),
            "dist_upr": ci.get("upr"),
        }

        # -- selection stage --
        if i in invalid_set:
            sel_role = "invalid_ci"
        elif i in B_raw_set:
            sel_role = "bridge"
        elif i in F_raw_set:
            sel_role = "far"
        else:
            sel_role = "dropped"
        session_roles.append({
            "stage": "selection", "local_idx": i,
            "session_abs_idx": sid, "session_label": slabel,
            "role": sel_role, "is_best": (i == best_i),
            "weight": None, **dist_fields,
        })

        # -- proxy stage (roles depend on proxy_direction) --
        if proxy_direction == "far_to_bridge":
            # F_proxy trains, B_proxy validates
            if i in F_proxy_set:
                proxy_role = "proxy_source"
            elif i in B_proxy_set:
                proxy_role = "proxy_target"
            else:
                proxy_role = "not_used"
        else:
            # bridge_to_far: B_proxy trains, F_proxy validates
            if i in B_proxy_set:
                proxy_role = "proxy_source"
            elif i in F_proxy_set:
                proxy_role = "proxy_target"
            else:
                proxy_role = "not_used"
        session_roles.append({
            "stage": "proxy", "local_idx": i,
            "session_abs_idx": sid, "session_label": slabel,
            "role": proxy_role, "is_best": False,
            "weight": None, **dist_fields,
        })

        # -- final stage (dist repeated for easy querying) --
        fin_role = "train" if i in B_final_set else "not_used"
        session_roles.append({
            "stage": "final", "local_idx": i,
            "session_abs_idx": sid, "session_label": slabel,
            "role": fin_role, "is_best": (i == best_i),
            "weight": None, **dist_fields,
        })

    # Target rows: only in selection and final (NOT proxy)
    for stg in ("selection", "final"):
        session_roles.append({
            "stage": stg, "local_idx": None,
            "session_abs_idx": t_id, "session_label": t_label,
            "role": "target", "is_best": False, "weight": None,
            "dist_to_session": None, "dist_est_method": None,
            "dist_est": None, "dist_lwr": None, "dist_upr": None,
        })

    return {
        "baseline": acc_base, "acc_DA": acc_DA, "bestSetup": best_setup,
        "model": final_model, "proxy_acc": best_score, "detail": detail,
        "target_id": t_id, "target_label": t_label,
        "session_roles": session_roles,
        "y_pred": y_pred, "y_true": y_T,
    }


def _partition_bridge_far(ci_tbl: list) -> dict:
    ests = np.array([c["est"] for c in ci_tbl])
    lwrs = np.array([c["lwr"] for c in ci_tbl])
    uprs = np.array([c["upr"] for c in ci_tbl])

    # Guard 1: need at least one fully finite CI
    finite_mask = np.isfinite(ests) & np.isfinite(lwrs) & np.isfinite(uprs)
    if not np.any(finite_mask):
        raise RuntimeError("BDP: all distance CIs are non-finite, cannot partition.")

    # Guard 2: need at least 2 finite sessions for proxy split
    if np.sum(finite_mask) < 2:
        raise RuntimeError(
            "BDP: fewer than 2 finite-CI train sessions, proxy split impossible."
        )

    finite_idx = set(np.where(finite_mask)[0])

    # Only finite sessions participate in ordering and partitioning
    ord_idx = [int(i) for i in np.argsort(ests) if i in finite_idx]

    best_i = ord_idx[0]
    base_l, base_u = lwrs[best_i], uprs[best_i]

    # B_raw: finite sessions whose CI overlaps with best
    B_raw = [i for i in range(len(ci_tbl))
             if i in finite_idx and lwrs[i] <= base_u and uprs[i] >= base_l]
    if not B_raw:
        B_raw = [best_i]

    finite_lwrs = lwrs[list(finite_idx)]
    thr_lwr = float(np.median(finite_lwrs))
    B_set = set(B_raw)

    # F_raw: finite far sessions, disjoint from B_raw
    F_raw = [i for i in range(len(ci_tbl))
             if i in finite_idx and lwrs[i] >= thr_lwr and i not in B_set]

    # Determine partition_mode and proxy sets
    if F_raw:
        partition_mode = "normal"
        F_proxy = list(F_raw)
        B_proxy = list(B_raw)
        borrowed_idx = None
    else:
        # cand: finite non-bridge sessions
        cand = [i for i in ord_idx if i not in B_set]
        if cand:
            # Non-bridge finite sessions exist but didn't meet far threshold
            partition_mode = "fallback_nonbridge"
            borrowed_idx = cand[-1]          # farthest non-bridge
            F_proxy = [borrowed_idx]
            B_proxy = list(B_raw)
        else:
            # ALL finite sessions are in B_raw, borrow farthest bridge
            # Guard 2 ensures len(B_raw) >= 2 here
            partition_mode = "fallback_bridge"
            b_sorted = sorted(B_raw, key=lambda i: ests[i])
            borrowed_idx = b_sorted[-1]
            F_proxy = [borrowed_idx]
            B_proxy = [i for i in B_raw if i != borrowed_idx]

    B_final = list(B_raw)

    # Invariant assertions
    assert not (set(F_proxy) & set(B_proxy)), \
        f"F_proxy ∩ B_proxy must be empty, got {set(F_proxy) & set(B_proxy)}"
    assert not (set(F_raw) & set(B_raw)), \
        f"F_raw ∩ B_raw must be empty, got {set(F_raw) & set(B_raw)}"
    assert set(B_final) == set(B_raw), \
        "B_final must equal B_raw"
    assert len(B_proxy) >= 1, "B_proxy must have at least 1 session"
    assert len(F_proxy) >= 1, "F_proxy must have at least 1 session"

    return {
        "B_raw": B_raw,
        "F_raw": F_raw,
        "B_proxy": B_proxy,
        "F_proxy": F_proxy,
        "B_final": B_final,
        "best_i": best_i,
        "thr_lwr": thr_lwr,
        "partition_mode": partition_mode,
        "borrowed_idx": borrowed_idx,
        "invalid_ci_idx": [i for i in range(len(ci_tbl)) if i not in finite_idx],
    }
