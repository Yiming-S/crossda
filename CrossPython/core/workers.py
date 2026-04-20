"""
workers.py — Subject-level worker functions for the cross-session pipeline.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import resolve_pipeline_spec, normalize_pipeline_labels

logger = logging.getLogger(__name__)


###############################################################################
# Parameter specification
###############################################################################

def make_params(dataset: str) -> dict:
    """Parameter specification for feature/classifier/DA grids."""
    return {
        "feature": {
            "logvar": {},
            "CSP": {"n_components": 8},
            "TS": {},
        },
        "classifier": {
            "lda": {"shrinkage": "auto"},
            "svm_linear": {"cost": 0.1},
            "svm_radial": {"cost": 0.1, "gamma": "scale"},
            "mdm": {},
            "el": {"alpha": 0.5},
            "lr": {},
            "lgbm": {"num_leaves": 31, "learning_rate": 0.05, "nrounds": 200},
            "elm": {"nhid": 500},
            "catboost": {"depth": 6, "lr": 0.05, "iters": 500},
        },
        "da": {
            "sa": {"k": 10},
            "tca": {"k": 10, "sigma": 1, "mu": 1},
            "pt": {},
            "coral": {},
            "none": {},
        },
    }


###############################################################################
# Data loading
###############################################################################

def _load_ma2020_sessions(subject_id: int, data_dir: str) -> List[dict]:
    """Load all sessions for one Ma2020 subject directly via MNE.

    Ma2020: 25 subjects, 15 sessions each, 65 EEG channels, 1000 Hz,
    binary MI (class 1 vs 2), 40 trials/session, .cnt format.
    """
    import mne
    mne.set_log_level("ERROR")

    subj_dir = os.path.join(data_dir, "MNE-ma2020-data", f"sub-{subject_id:03d}")
    if not os.path.isdir(subj_dir):
        raise FileNotFoundError(f"Ma2020 subject directory not found: {subj_dir}")

    cnt_files = sorted(
        f for f in os.listdir(subj_dir)
        if f.endswith(".cnt") and "motorimagery" in f.lower()
    )
    if not cnt_files:
        raise FileNotFoundError(f"No .cnt files in {subj_dir}")

    sessions = []
    for cnt_file in cnt_files:
        cnt_path = os.path.join(subj_dir, cnt_file)
        raw = mne.io.read_raw_cnt(cnt_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        raw.filter(8, 35, verbose=False)

        events, event_id = mne.events_from_annotations(raw, verbose=False)
        # Keep only class 1 and 2
        keep_ids = {k: v for k, v in event_id.items() if k in ("1", "2")}
        if len(keep_ids) < 2:
            logger.warning(f"[Ma2020 s{subject_id}] Skipping {cnt_file}: <2 classes")
            continue

        epochs = mne.Epochs(
            raw, events, event_id=keep_ids,
            tmin=0, tmax=3, baseline=None, preload=True, verbose=False,
        )
        if len(epochs) < 4:
            continue

        sessions.append({
            "x": epochs.get_data(),          # (n_trials, 65, 3001)
            "y": epochs.events[:, 2],        # class labels (1 or 2)
            "id": len(sessions),             # int, stable within subject
            "label": cnt_file,               # str, human-readable
        })

    if not sessions:
        raise RuntimeError(f"No valid sessions for Ma2020 subject {subject_id}")

    logger.info(f"[Ma2020 s{subject_id}] Loaded {len(sessions)} sessions, "
                f"{sum(s['x'].shape[0] for s in sessions)} trials total")
    return sessions


def _load_subject_sessions(
    subject_id: int, dataset_name: str, data_dir: str = "",
    cache_dir: Optional[str] = None,
) -> Optional[List[dict]]:
    """Load all sessions for one subject via MOABB. Returns [{"x": 3D, "y": 1D}, ...]."""
    if cache_dir:
        cache_file = os.path.join(cache_dir, f"{dataset_name}_{subject_id}_allsess.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                logger.warning(f"Cache unreadable, reloading: {cache_file}")
    else:
        cache_file = None

    try:
        if dataset_name == "ma2020":
            sessions = _load_ma2020_sessions(subject_id, data_dir)
        else:
            from moabb.datasets import BNCI2014_004, Stieger2021
            from moabb.paradigms import LeftRightImagery

            ds_map = {"bnci004": BNCI2014_004, "stieger2021": Stieger2021}
            if dataset_name not in ds_map:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            ds_obj = ds_map[dataset_name]()
            paradigm = LeftRightImagery(fmin=8, fmax=35)
            X, y, metadata = paradigm.get_data(dataset=ds_obj, subjects=[subject_id], return_epochs=True)

            sessions = []
            for idx, sess_name in enumerate(sorted(metadata["session"].unique())):
                mask = metadata["session"] == sess_name
                sess_X = X[mask]
                sess_data = sess_X.get_data() if hasattr(sess_X, "get_data") else np.array(sess_X)
                sessions.append({
                    "x": sess_data, "y": np.array(y[mask]),
                    "id": idx, "label": sess_name,
                })

        if not sessions:
            logger.warning(f"[Subject {subject_id}] No sessions found.")
            return None

        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(sessions, f)
            except Exception:
                pass

        # Backfill id/label for old caches that lack them
        for i, ses in enumerate(sessions):
            ses.setdefault("id", i)
            ses.setdefault("label", f"sess_{i}")
        return sessions
    except Exception as e:
        logger.warning(f"[Subject {subject_id}] Data load failed: {e}")
        return None


###############################################################################
# LOSO splits
###############################################################################

def _make_pairs_idx_loso(n_sess: int) -> List[List[int]]:
    if n_sess < 2:
        raise ValueError("Need at least 2 sessions for LOSO splits.")
    return [[j for j in range(n_sess) if j != i] for i in range(n_sess)]


###############################################################################
# Checkpoint
###############################################################################

def _ckpt_path(result_dir, dataset, subject_id, pipeline):
    return os.path.join(result_dir, f".ckpt_{dataset}_{subject_id}_{pipeline}.pkl")


def _save_checkpoint(path, mb, detail_records, m):
    try:
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump({"mb_partial": mb.iloc[:m + 1].copy(),
                         "detail_records": detail_records, "m_last": m}, f)
        os.replace(tmp, path)
    except Exception:
        pass


def _load_checkpoint(path, mb):
    if not os.path.exists(path):
        return mb, [], 0
    try:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
    except Exception:
        return mb, [], 0
    if not isinstance(ckpt, dict) or "m_last" not in ckpt:
        return mb, [], 0
    m_last = min(ckpt["m_last"], len(mb) - 1)
    partial = ckpt.get("mb_partial")
    if partial is not None and len(partial) <= len(mb):
        mb.iloc[:len(partial)] = partial.iloc[:len(partial)]
    logger.info(f"  [ckpt] Resuming from config {m_last + 1}/{len(mb)}")
    return mb, ckpt.get("detail_records", []), m_last + 1


###############################################################################
# Core processing loop
###############################################################################

def _process_subject_loaded(
    subject_id, data, session_ids, pipelines, result_dir, method_bank, pairs_idx,
    outer_eval_label, dataset,
    nfolds_out=5, nfolds_in=3,
    map_score="kfold", map_k_sess=2, map_n_repeats=1,
    map_shuffle_sessions=True, map_seed=1,
    epsilon=1e-6, default_dist_type="mmd", mmp_B_boot=10,
    dwp_dist_bootstrap_B=1, dwp_dist_est_method="direct",
    resume=True, verbose=True, seed=None,
    **_ignored,
):
    pipelines = normalize_pipeline_labels(pipelines)
    n_sess = len(data)
    params = make_params(dataset)
    os.makedirs(result_dir, exist_ok=True)

    from ..pipelines.map_pipeline import MAP as map_fn
    from ..pipelines.bdp_pipeline import BDP as bdp_fn
    from ..pipelines.mmp_pipeline import MMP as mmp_fn
    from ..pipelines.dwp_pipeline import DWP as dwp_fn
    pipeline_fn_map = {"MAP": map_fn, "BDP": bdp_fn, "MMP": mmp_fn, "DWP": dwp_fn}

    results_list = {}

    for this_pip in pipelines:
        pipe_spec = resolve_pipeline_spec(this_pip)
        mb = method_bank[method_bank["pipeline"] == this_pip].copy().reset_index(drop=True)
        if len(mb) == 0:
            continue

        outfile = os.path.join(result_dir, f"{dataset}_{subject_id}_{this_pip}.pkl")
        detail_outfile = os.path.join(result_dir, f"{dataset}_{subject_id}_{this_pip}_detail.pkl")
        ckpt_file = _ckpt_path(result_dir, dataset, subject_id, this_pip)

        if resume and os.path.exists(outfile):
            try:
                with open(outfile, "rb") as f:
                    cached = pickle.load(f)
                if isinstance(cached, pd.DataFrame) and len(cached) == len(mb):
                    if verbose:
                        logger.info(f"[Subject {subject_id} | {this_pip}] SKIP existing result")
                    results_list[this_pip] = cached
                    continue
            except Exception:
                pass

        for col, default in [("cvMeanAcc", np.nan), ("baseline", np.nan), ("score", None),
                             ("outer_eval", None), ("proxy_acc", np.nan), ("error", None),
                             ("n_valid_pairs", 0)]:
            mb[col] = default

        mb, detail_records, m_start = _load_checkpoint(ckpt_file, mb)

        algo_fn = pipeline_fn_map.get(pipe_spec["family"])
        if algo_fn is None:
            logger.warning(f"Pipeline '{pipe_spec['family']}' not implemented.")
            continue

        subj_hash = sum(ord(c) for c in str(subject_id)) % 99991
        pip_hash = sum(ord(c) for c in this_pip) % 100

        from tqdm import tqdm
        config_iter = range(m_start, len(mb))
        config_bar = tqdm(
            config_iter,
            desc=f"  S{subject_id} {this_pip}",
            unit="cfg",
            leave=False,
        )
        for m in config_bar:
            row = mb.iloc[m]
            config_bar.set_postfix_str(f"{row['feature']}/{row['da']}/{row['classifier']}")

            acc_vec = np.full(len(pairs_idx), np.nan)
            base_vec = np.full(len(pairs_idx), np.nan)
            error_msgs = []

            for p, train_idx in enumerate(pairs_idx):
                test_idx = [j for j in range(n_sess) if j not in train_idx]

                if seed is not None:
                    fold_seed = (seed + subj_hash*31 + pip_hash*7 + m*997 + p*31) % (2**31 - 1)
                    np.random.seed(fold_seed)
                else:
                    fold_seed = None

                call_args = dict(
                    train_sessions=[data[j] for j in train_idx],
                    test_session=[data[j] for j in test_idx],
                    params=params, feature=row["feature"],
                    classifier=row["classifier"], da=row["da"], epsilon=epsilon,
                )

                if pipe_spec["family"] == "MAP":
                    call_args.update(score=map_score, k_sess=map_k_sess, n_repeats=map_n_repeats,
                                     shuffle_sessions=map_shuffle_sessions, seed=map_seed,
                                     nfolds_out=nfolds_out)
                elif pipe_spec["family"] == "DWP":
                    call_args.update(score=map_score, k_sess=map_k_sess, n_repeats=map_n_repeats,
                                     shuffle_sessions=map_shuffle_sessions, seed=map_seed,
                                     nfolds_out=nfolds_out)
                    call_args["dist_type"] = default_dist_type
                    call_args["seed"] = fold_seed
                    call_args["dist_bootstrap_B"] = dwp_dist_bootstrap_B
                    call_args["dist_est_method"] = dwp_dist_est_method
                elif pipe_spec["family"] == "BDP":
                    call_args["dist_type"] = row.get("dist_type") or default_dist_type
                    call_args["seed"] = fold_seed
                    call_args["proxy_direction"] = pipe_spec.get("proxy_direction", "far_to_bridge")
                    # Forward MAP-style params (used only when BDP degrades).
                    # BDP reuses `seed` (fold_seed) for the degrade scorer.
                    call_args.update(
                        map_score=map_score,
                        map_k_sess=map_k_sess,
                        map_n_repeats=map_n_repeats,
                        map_shuffle_sessions=map_shuffle_sessions,
                        nfolds_out=nfolds_out,
                    )
                elif pipe_spec["family"] == "MMP":
                    call_args["dist_type"] = row.get("dist_type") or default_dist_type
                    call_args["B_boot"] = mmp_B_boot
                    call_args["combiner"] = pipe_spec.get("combiner", "merge_then_adapt")
                    call_args["seed"] = fold_seed

                try:
                    t0 = time.perf_counter()
                    res = algo_fn(**call_args, verbose=verbose)
                    elapsed = time.perf_counter() - t0

                    acc_vec[p] = res["acc_DA"]
                    base_vec[p] = res["baseline"]

                    # Map local_idx -> global session_abs_idx in session_roles
                    raw_roles = res.get("session_roles", [])
                    for r in raw_roles:
                        if r.get("local_idx") is not None:
                            r["session_abs_idx"] = train_idx[r["local_idx"]]

                    detail_records.append({
                        "subject": subject_id, "dataset": dataset, "pipeline": this_pip,
                        "feature": row["feature"], "classifier": row["classifier"], "da": row["da"],
                        "method_row": m, "pair_id": p,
                        "train_idx": train_idx, "test_idx": test_idx,
                        "baseline": res["baseline"], "acc_DA": res["acc_DA"],
                        "proxy_acc": res.get("proxy_acc", np.nan),
                        "detail": res.get("detail"),
                        "bestSetup": {k: v for k, v in res.get("bestSetup", {}).items()
                                      if k not in ("feature_obj", "clf_instance", "model")},
                        # New fields
                        "elapsed_sec": elapsed,
                        "seed_used": fold_seed,
                        "target_id": res.get("target_id"),
                        "test_label": res.get("target_label"),
                        "train_labels": [data[j].get("label") for j in train_idx],
                        "y_pred": res.get("y_pred"),
                        "y_true": res.get("y_true"),
                        "acc_map_base": res.get("acc_map_base", np.nan),
                        "session_roles": raw_roles,
                    })
                except Exception as e:
                    error_msgs.append(f"pair{p}: {e}")
                    logger.warning(f"[{subject_id}|{this_pip}|pair{p}] Error: {e}")

            mb.at[m, "cvMeanAcc"] = float(np.nanmean(acc_vec)) if np.any(np.isfinite(acc_vec)) else np.nan
            mb.at[m, "baseline"] = float(np.nanmean(base_vec)) if np.any(np.isfinite(base_vec)) else np.nan
            mb.at[m, "n_valid_pairs"] = int(np.sum(np.isfinite(acc_vec)))
            mb.at[m, "error"] = "; ".join(error_msgs) if error_msgs else None
            default_score = {
                "MAP": "kfold", "DWP": "kfold",
                "MMP": "near_nearest_proxy", "BDP": "bridge_proxy",
            }.get(pipe_spec["family"])
            row_modes = {
                (d.get("detail") or {}).get("score_mode")
                for d in detail_records if d.get("method_row") == m
            }
            row_modes.discard(None)
            if not row_modes:
                mb.at[m, "score"] = default_score
            elif len(row_modes) == 1:
                mb.at[m, "score"] = row_modes.pop()
            else:
                mb.at[m, "score"] = "mixed:" + "|".join(sorted(row_modes))
            mb.at[m, "outer_eval"] = outer_eval_label
            _save_checkpoint(ckpt_file, mb, detail_records, m)

        mb["subject"] = subject_id
        mb["dataset"] = dataset
        mb["n_session"] = n_sess
        mb["session_ids"] = ",".join(str(s) for s in session_ids)
        mb["session_labels"] = ",".join(
            s.get("label", str(i)) for i, s in enumerate(data)
        )
        mb["n_detail_records"] = len(detail_records)

        try:
            with open(detail_outfile, "wb") as f:
                pickle.dump(detail_records, f)
            with open(outfile, "wb") as f:
                pickle.dump(mb, f)
            if os.path.exists(ckpt_file):
                os.remove(ckpt_file)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

        # Flatten session_roles into a CSV for easy analysis
        role_rows = []
        for rec in detail_records:
            det = rec.get("detail") or {}
            for sr in rec.get("session_roles", []):
                role_rows.append({
                    "subject": rec["subject"],
                    "dataset": rec["dataset"],
                    "pipeline": rec["pipeline"],
                    "feature": rec["feature"],
                    "classifier": rec["classifier"],
                    "da": rec["da"],
                    "method_row": rec["method_row"],
                    "pair_id": rec["pair_id"],
                    "target_id": rec["target_id"],
                    "seed_used": rec.get("seed_used"),
                    "dist_type": det.get("dist_type"),
                    "partition_mode": det.get("partition_mode"),
                    "proxy_direction": det.get("proxy_direction"),
                    "borrowed_local_idx": det.get("borrowed_idx"),
                    "borrowed_abs_idx": det.get("borrowed_abs_idx"),
                    "combiner": det.get("combiner"),
                    "harmonize": det.get("harmonize"),
                    "score_mode": det.get("score_mode"),
                    **sr,
                })
        if role_rows:
            try:
                pd.DataFrame(role_rows).to_csv(
                    os.path.join(result_dir, f"{dataset}_{subject_id}_{this_pip}_roles.csv"),
                    index=False,
                )
            except Exception as e:
                logger.warning(f"Failed to save roles CSV: {e}")

        results_list[this_pip] = mb

    if not results_list:
        return None
    return pd.concat(results_list.values(), ignore_index=True)


###############################################################################
# Public worker
###############################################################################

def process_subject(
    subject_id: int, pipelines: List[str], result_dir: str,
    method_bank: pd.DataFrame, dataset: str, data_dir: str,
    cache_dir: Optional[str] = None, **kwargs,
) -> Optional[pd.DataFrame]:
    """Load sessions and run all pipelines for one subject."""
    sessions = _load_subject_sessions(subject_id, dataset, data_dir, cache_dir)
    if sessions is None:
        return None
    if len(sessions) < 2:
        logger.warning(f"[Subject {subject_id}] Need at least 2 sessions. Got {len(sessions)}.")
        return None

    return _process_subject_loaded(
        subject_id=subject_id, data=sessions,
        session_ids=list(range(len(sessions))),
        pipelines=pipelines, result_dir=result_dir,
        method_bank=method_bank,
        pairs_idx=_make_pairs_idx_loso(len(sessions)),
        outer_eval_label="session_loso_mean",
        dataset=dataset, **kwargs,
    )
