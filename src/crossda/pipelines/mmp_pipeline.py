"""
MMP Pipeline — Minimum-distance Multi-source Pipeline

  1) Distance-to-target with bootstrap CI; CI-aware source gating
  2) Optional CORAL harmonization to medoid
  3) Two combiners: merge_then_adapt / moe (weighted voting)
  4) Near/nearest proxy tuning within the target-defined near set
     - if |N(t)| >= 2: use N(t) without s* -> s*
     - if |N(t)| = 1: use second-nearest-to-t -> s*
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import clone

from ..pipelines.pipeline_utils import (
    extract_features_train,
    extract_features_test,
    get_classifier,
    session_distance,
    compute_distance_ci_table,
    apply_da,
)
from ..pipelines.weighting import (
    _weight_from_D,
    _weights_for_indices,
    _merge_weighted,
)
from ..pipelines.session_roles import distance_fields, target_rows

logger = logging.getLogger(__name__)


def MMP(
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
    ci_alpha: float = 0.05,
    B_boot: int = 200,
    k_candidates: int = 2,  # deprecated: retained for compatibility, no longer used to cap N(t)
    p_weight: float = 1.0,
    w_max: float = 1.0,
    harmonize: str = "coral",
    combiner: str = "merge_then_adapt",
    external_gate: Optional[dict] = None,
    bdp_compatible_final: bool = False,
    verbose: bool = False,
    seed: Optional[int] = None,
    **_ignored,
) -> Dict[str, Any]:
    if len(train_sessions) < 2:
        raise ValueError("MMP requires at least two training sessions.")
    for name, val in [("feature", feature), ("classifier", classifier), ("da", da)]:
        if val is None:
            raise ValueError(f"MMP (strict): specify exactly one {name}.")

    dist_param = dist_param or {}
    feat_name, clf_name, da_name = feature, classifier, da
    feat_param = params.get("feature", {}).get(feat_name, {})
    clf_params = params.get("classifier", {}).get(clf_name, {})
    da_control = {} if da_name == "none" else params.get("da", {}).get(da_name, {})

    test_ses = test_session[0] if isinstance(test_session, list) else test_session

    # ── Feature extraction ───────────────────────────────────────────
    all_X = np.concatenate([ses["x"] for ses in train_sessions], axis=0)
    all_y = np.concatenate([ses["y"] for ses in train_sessions])
    try:
        _, feat_obj = extract_features_train(all_X, all_y, feat_name, feat_param)
    except Exception as e:
        raise RuntimeError(f"MMP: feature extraction failed: {e}")

    X_list = [extract_features_test(ses["x"], feat_obj, feat_name) for ses in train_sessions]
    X_T = extract_features_test(test_ses["x"], feat_obj, feat_name)

    # ── Source selection (CI-gating) ─────────────────────────────────
    gate_kw = dict(dist_type=dist_type, dist_param=dist_param, scale_for_distance=scale_for_distance,
                   B_boot=B_boot, ci_alpha=ci_alpha, k_candidates=k_candidates,
                   epsilon=epsilon, p_weight=p_weight, w_max=w_max)
    if external_gate is None:
        main_rng = np.random.default_rng(seed) if seed is not None else None
        gate_main = _select_sources(X_list, X_T, **gate_kw, rng=main_rng)
        gate_source = "mmp_bootstrap"
    else:
        gate_main = _select_sources_from_external_gate(
            external_gate,
            epsilon=epsilon,
            p_weight=p_weight,
            w_max=w_max,
            uniform_weights=bdp_compatible_final,
        )
        gate_source = external_gate.get("source_pipeline", "external_bdp")

    # ── Near/nearest proxy tuning ────────────────────────────────────
    proxy_source_idx, proxy_target_idx, proxy_mode = _build_proxy_task_from_near_set(gate_main)
    proxy_weights = _weights_for_indices(
        gate_main["ci_tbl"], proxy_source_idx, epsilon, p_weight, w_max,
    )
    X_anchor_src = _maybe_harmonize(
        [X_list[i] for i in proxy_source_idx], harmonize, dist_type, dist_param,
    )
    Y_anchor_src = [train_sessions[i]["y"] for i in proxy_source_idx]

    clf_instance = get_classifier(clf_name, clf_params)
    anchor_eval = _run_combiner(
        X_anchor_src, Y_anchor_src, proxy_weights,
        X_list[proxy_target_idx], train_sessions[proxy_target_idx]["y"],
        da_name, da_control, clf_instance, combiner,
    )
    tune_acc = anchor_eval["acc_DA"] if np.isfinite(anchor_eval["acc_DA"]) else np.nan

    anchor_info = {
        "target_idx": proxy_target_idx,
        "source_idx": proxy_source_idx,
        "source_weights": proxy_weights.tolist(),
        "selection_mode": proxy_mode,
        "near_set_idx": list(gate_main["sel_idx"]),
        "ranked_idx": list(gate_main["ranked_idx"]),
        "second_nearest_idx": gate_main["ranked_idx"][1] if len(gate_main["ranked_idx"]) >= 2 else None,
        "ci_table": gate_main["ci_tbl"],
        "gate_source": gate_source,
    }

    # ── Final evaluation ─────────────────────────────────────────────
    Y_list = [ses["y"] for ses in train_sessions]
    # Final MMP_mta uses the target-defined near set directly; do not add an
    # extra source-to-source CORAL harmonization layer before the configured DA.
    final_harmonize = "none"
    X_sel = _maybe_harmonize(
        [X_list[i] for i in gate_main["sel_idx"]], final_harmonize, dist_type, dist_param,
    )
    Y_sel = [Y_list[i] for i in gate_main["sel_idx"]]

    eval_final = _run_combiner(
        X_sel, Y_sel, gate_main["weights"],
        X_T, test_ses.get("y"), da_name, da_control,
        get_classifier(clf_name, clf_params), combiner,
        weighted_merge=False,
    )

    if verbose:
        logger.info(
            f"[MMP({combiner})] {feat_name}/{clf_name}/{da_name} | "
            f"near={gate_main['sel_idx']} | s*={gate_main['best_i']} | tune={tune_acc:.4f} | "
            f"base={eval_final['baseline']:.4f} | da={eval_final['acc_DA']:.4f}"
        )

    best_setup = {
        "feature_obj": feat_obj,
        "feature": {"method": feat_name, "param": feat_param},
        "classifier": {"method": clf_name, "param": clf_params},
        "da": {"method": da_name, "control": da_control},
        "best_idx": gate_main["best_i"], "sel_idx": gate_main["sel_idx"],
        "weights": gate_main["weights"], "ci_table": gate_main["ci_tbl"],
        "overlap_set": gate_main["overlap_idx"], "ranked_idx": gate_main["ranked_idx"],
        "anchor": anchor_info,
        "combiner": combiner, "harmonize": final_harmonize,
        "requested_harmonize": harmonize,
        "gate_source": gate_source,
        "bdp_compatible_final": bool(bdp_compatible_final),
        "cvMeanAcc": tune_acc, "scoreMode": "near_nearest_proxy",
    }
    detail = {
        "pipeline": f"MMP_{combiner}", "dist_type": dist_type,
        "score_mode": "near_nearest_proxy", "selection_mode": proxy_mode,
        "combiner": combiner, "harmonize": final_harmonize,
        "requested_harmonize": harmonize,
        "gate_source": gate_source,
        "bdp_compatible_final": bool(bdp_compatible_final),
        "best_idx": gate_main["best_i"], "selected_idx": gate_main["sel_idx"],
        "overlap_set": gate_main["overlap_idx"], "ranked_idx": gate_main["ranked_idx"],
        "weights": gate_main["weights"],
        "ci_table": gate_main["ci_tbl"], "anchor": anchor_info,
    }

    # ── session_roles ────────────────────────────────────────────────
    sel_set = set(gate_main["sel_idx"])
    overlap_set = set(gate_main["overlap_idx"])
    ci_map_main = {c["i"]: c for c in gate_main["ci_tbl"]}

    t_id = test_ses.get("id")
    t_label = test_ses.get("label", f"sess_{t_id}")
    s_star = gate_main["best_i"]
    anchor_sel_set = set(anchor_info["source_idx"])

    session_roles = []
    for i in range(len(train_sessions)):
        sid = train_sessions[i].get("id", i)
        slabel = train_sessions[i].get("label", f"sess_{i}")
        ci_m = ci_map_main.get(i, {})

        # distance to real target
        dist_main = distance_fields(ci_m, t_id, "boot_mean")

        # ---- main stage ----
        if i == s_star:
            main_role = "nearest"
        elif i in overlap_set:
            main_role = "near"
        else:
            main_role = "far"

        sel_idx_list = list(gate_main["sel_idx"])
        if i in sel_set:
            w_pos = sel_idx_list.index(i)
            main_weight = float(gate_main["weights"][w_pos])
        else:
            main_weight = None

        session_roles.append({
            "stage": "main", "local_idx": i,
            "session_abs_idx": sid, "session_label": slabel,
            "role": main_role, "is_best": (i == gate_main["best_i"]),
            "weight": main_weight, **dist_main,
        })

        # ---- anchor stage ----
        if i == proxy_target_idx:
            anchor_role = "target"
            a_weight = None
        elif i in anchor_sel_set:
            if proxy_mode == "second_nearest_fallback":
                anchor_role = "second_nearest_fallback_source"
            else:
                anchor_role = "proxy_source"
            try:
                a_pos = list(anchor_info["source_idx"]).index(i)
                a_weight = float(anchor_info["source_weights"][a_pos])
            except (ValueError, IndexError):
                a_weight = None
        else:
            anchor_role = "not_used"
            a_weight = None

        session_roles.append({
            "stage": "anchor", "local_idx": i,
            "session_abs_idx": sid, "session_label": slabel,
            "role": anchor_role,
            "is_best": False,
            "weight": a_weight, **dist_main,
        })

        # ---- final stage (dist to real target repeated) ----
        fin_role = "train" if (i in sel_set or i == s_star) else "not_used"
        session_roles.append({
            "stage": "final", "local_idx": i,
            "session_abs_idx": sid, "session_label": slabel,
            "role": fin_role, "is_best": (i == gate_main["best_i"]),
            "weight": main_weight, **dist_main,
        })

    # Real target rows: only in main and final (NOT anchor)
    session_roles += target_rows(("main", "final"), t_id, t_label)

    return {
        "baseline": eval_final["baseline"], "acc_DA": eval_final["acc_DA"],
        "bestSetup": best_setup, "model": eval_final["model"], "detail": detail,
        "target_id": t_id, "target_label": t_label,
        "session_roles": session_roles,
        "y_pred": eval_final.get("y_pred"),
        "y_true": np.asarray(test_ses.get("y")),
    }


###############################################################################
# Source selection
###############################################################################

def _select_sources(
    X_sources, X_target, dist_type, dist_param,
    scale_for_distance, B_boot, ci_alpha,
    k_candidates, epsilon, p_weight, w_max,
    rng=None,
) -> dict:
    ci_tbl = compute_distance_ci_table(
        X_sources, X_target, dist_type, dist_param,
        scale=scale_for_distance, B=B_boot, alpha=ci_alpha, est_method="boot_mean",
        rng_for_index=lambda i: np.random.default_rng(rng.integers(2**31) + i) if rng is not None else None,
    )

    ok = [c for c in ci_tbl if all(np.isfinite(c[k]) for k in ("est", "lwr", "upr"))]
    if not ok:
        raise RuntimeError("MMP: all distance CIs are non-finite.")

    ok.sort(key=lambda c: c["est"])
    best_i = ok[0]["i"]
    bl, bu = ok[0]["lwr"], ok[0]["upr"]

    overlap_idx = [c["i"] for c in ok if not (c["upr"] < bl or bu < c["lwr"])]
    ranked_idx = [c["i"] for c in ok]
    overlap_sorted = sorted([c for c in ok if c["i"] in overlap_idx], key=lambda c: c["est"])
    sel_idx = [c["i"] for c in overlap_sorted]

    est_map = {c["i"]: c["est"] for c in ci_tbl}
    sel_ests = np.array([est_map[i] for i in sel_idx])
    weights = _weight_from_D(sel_ests, epsilon, p_weight, w_max)

    return {"sel_idx": sel_idx, "weights": weights, "best_i": best_i,
            "overlap_idx": overlap_idx, "ranked_idx": ranked_idx, "ci_tbl": ci_tbl}


def _select_sources_from_external_gate(
    external_gate: dict,
    epsilon: float,
    p_weight: float,
    w_max: float,
    uniform_weights: bool = False,
) -> dict:
    ci_tbl = list(external_gate.get("ci_table") or [])
    if external_gate.get("degraded", False):
        sel_idx = list(external_gate.get("effective_final_idx") or [])
    else:
        sel_idx = list(external_gate.get("B_final") or [])

    if not sel_idx:
        raise RuntimeError("MMP external gate: no selected final source indices.")

    ok = [c for c in ci_tbl if all(np.isfinite(c[k]) for k in ("est", "lwr", "upr"))]
    if not ok:
        raise RuntimeError("MMP external gate: all distance CIs are non-finite.")

    ok.sort(key=lambda c: c["est"])
    ranked_idx = [int(c["i"]) for c in ok]
    best_i = int(external_gate.get("best_idx", ranked_idx[0]))
    sel_idx = [int(i) for i in sel_idx]

    if uniform_weights:
        weights = np.ones(len(sel_idx), dtype=float) / len(sel_idx)
    else:
        est_map = {int(c["i"]): c["est"] for c in ci_tbl}
        try:
            sel_ests = np.array([est_map[i] for i in sel_idx], dtype=float)
        except KeyError as exc:
            raise RuntimeError(f"MMP external gate: selected index missing from CI table: {exc}") from exc
        weights = _weight_from_D(sel_ests, epsilon, p_weight, w_max)

    return {
        "sel_idx": sel_idx,
        "weights": weights,
        "best_i": best_i,
        "overlap_idx": list(sel_idx),
        "ranked_idx": ranked_idx,
        "ci_tbl": ci_tbl,
    }


def _build_proxy_task_from_near_set(gate_main: dict) -> tuple[list[int], int, str]:
    s_star = gate_main["best_i"]
    near_idx = list(gate_main["sel_idx"])
    ranked_idx = list(gate_main["ranked_idx"])

    if len(near_idx) >= 2:
        proxy_source_idx = [i for i in near_idx if i != s_star]
        proxy_mode = "near_minus_nearest"
    else:
        if len(ranked_idx) < 2:
            raise RuntimeError("MMP: cannot form proxy selection because no second-nearest session exists.")
        proxy_source_idx = [ranked_idx[1]]
        proxy_mode = "second_nearest_fallback"

    return proxy_source_idx, s_star, proxy_mode


###############################################################################
# Harmonization
###############################################################################

def _maybe_harmonize(X_sel, harmonize, dist_type, dist_param):
    if len(X_sel) <= 1 or harmonize != "coral":
        return X_sel

    S = len(X_sel)
    D = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            if i != j:
                try:
                    D[i, j] = session_distance(X_sel[i], X_sel[j], dist_type, dist_param)
                except Exception:
                    D[i, j] = np.inf

    medoid_scores = []
    for i in range(S):
        pos = D[i][(D[i] > 0) & np.isfinite(D[i])]
        medoid_scores.append(np.median(pos) if len(pos) > 0 else np.inf)
    ref = int(np.argmin(medoid_scores))

    X_ref = X_sel[ref]
    mu_r, S_r = X_ref.mean(axis=0), _cov_spd(X_ref)
    return [_coral_transform(Xi, Xi.mean(axis=0), _cov_spd(Xi), mu_r, S_r) for Xi in X_sel]


def _cov_spd(X, eps=1e-4):
    if X.shape[0] < 2:
        return np.eye(X.shape[1]) * eps
    S = np.cov(X, rowvar=False)
    S[~np.isfinite(S)] = 0
    return S + np.eye(S.shape[0]) * eps


def _spd_sqrt(S):
    vals, vecs = np.linalg.eigh(S)
    return vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T


def _spd_isqrt(S):
    vals, vecs = np.linalg.eigh(S)
    return vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T


def _coral_transform(Xs, mu_s, Sig_s, mu_t, Sig_t):
    return (Xs - mu_s) @ _spd_isqrt(Sig_s) @ _spd_sqrt(Sig_t) + mu_t


###############################################################################
# Combiners
###############################################################################

def _run_combiner(
    X_sel, Y_sel, w_sel, X_tar, Y_tar, da_name, da_ctl, clf, combiner,
    weighted_merge: bool = True,
):
    has_label = Y_tar is not None and len(Y_tar) > 0
    if combiner == "merge_then_adapt":
        return _combiner_merge(
            X_sel, Y_sel, w_sel, X_tar, Y_tar, da_name, da_ctl, clf, has_label,
            weighted_merge=weighted_merge,
        )
    return _combiner_moe(X_sel, Y_sel, w_sel, X_tar, Y_tar, da_name, da_ctl, clf, has_label)


def _combiner_merge(
    X_sel, Y_sel, w_sel, X_tar, Y_tar, da_name, da_ctl, clf, has_label,
    weighted_merge: bool = True,
):
    out = {"baseline": np.nan, "acc_DA": np.nan, "model": None, "y_pred": None}
    if weighted_merge:
        merged_X, merged_y = _merge_weighted(X_sel, Y_sel, w_sel)
    else:
        merged_X, merged_y = np.vstack(X_sel), np.concatenate(Y_sel)

    base_clf = clone(clf)
    base_clf.fit(merged_X, merged_y)
    base_pred = base_clf.predict(X_tar) if has_label else None
    if has_label:
        out["baseline"] = float(np.mean(base_pred == np.asarray(Y_tar)))

    if da_name == "none":
        out["acc_DA"] = out["baseline"]
        out["model"] = base_clf
        out["y_pred"] = base_pred               # path 1: no DA
        return out

    try:
        src_da, tgt_da = apply_da(da_name, da_ctl, merged_X, X_tar)
    except Exception:
        out["model"] = base_clf
        out["y_pred"] = base_pred               # path 2: DA failed
        return out

    da_clf = clone(clf)
    da_clf.fit(src_da, merged_y)
    out["model"] = da_clf
    if has_label:
        da_pred = da_clf.predict(tgt_da)
        out["acc_DA"] = float(np.mean(da_pred == np.asarray(Y_tar)))
        out["y_pred"] = da_pred                 # path 3: DA success
    return out


def _combiner_moe(X_sel, Y_sel, w_sel, X_tar, Y_tar, da_name, da_ctl, clf, has_label):
    out = {"baseline": np.nan, "acc_DA": np.nan, "model": None, "y_pred": None}
    models, preds_base, preds_da = [], [], []

    for Xi, yi in zip(X_sel, Y_sel):
        yi = np.asarray(yi)
        if len(np.unique(yi)) < 2:
            # A single-class source cannot train a discriminative classifier
            # (sklearn would crash on predict). It votes for its only class;
            # keep the per-source preds aligned with the MoE weights.
            const = np.full(X_tar.shape[0], yi[0])
            models.append(None)
            if has_label:
                preds_base.append(const)
                preds_da.append(const)
            continue
        base_mod = clone(clf)
        base_mod.fit(Xi, yi)
        models.append(base_mod)
        if has_label:
            preds_base.append(base_mod.predict(X_tar))

        if da_name == "none":
            if has_label:
                preds_da.append(preds_base[-1])
            continue

        try:
            src_da, tgt_da = apply_da(da_name, da_ctl, Xi, X_tar)
            da_mod = clone(clf)
            da_mod.fit(src_da, yi)
            models[-1] = da_mod
            if has_label:
                preds_da.append(da_mod.predict(tgt_da))
        except Exception:
            if has_label:
                preds_da.append(preds_base[-1] if preds_base else np.full(X_tar.shape[0], np.nan))

    out["model"] = models
    if not has_label:
        return out

    classes = sorted(set(np.concatenate([np.unique(Y_tar)] + [np.unique(y) for y in Y_sel])))
    w = np.asarray(w_sel) / np.sum(w_sel)

    def vote(preds_list):
        scores = np.zeros((len(classes), X_tar.shape[0]))
        for k, cl in enumerate(classes):
            for i, pred in enumerate(preds_list):
                scores[k] += w[i] * (np.asarray(pred) == cl).astype(float)
        return np.array([classes[j] for j in np.argmax(scores, axis=0)])

    base_voted = vote(preds_base)
    da_voted = vote(preds_da)
    out["baseline"] = float(np.mean(base_voted == np.asarray(Y_tar)))
    out["acc_DA"] = float(np.mean(da_voted == np.asarray(Y_tar)))
    out["y_pred"] = da_voted
    return out
