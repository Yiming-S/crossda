"""
workers.py — Subject-level worker functions for the cross-session pipeline.
"""

from __future__ import annotations

import logging
import gc
import os
import pickle
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import resolve_pipeline_spec, normalize_pipeline_labels

logger = logging.getLogger(__name__)


_MEMORY_ERROR_PATTERNS = (
    "unable to allocate",
    "cannot allocate memory",
    "out of memory",
    "memoryerror",
    "killed",
)


def _is_memory_allocation_error(exc: BaseException) -> bool:
    """Return True for allocation-like failures worth retrying."""
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return any(pattern in msg for pattern in _MEMORY_ERROR_PATTERNS)


def _run_with_memory_retries(
    fn,
    call_args: Dict[str, Any],
    verbose: bool,
    max_retries: int,
    retry_sleep: float,
    label: str,
):
    """Run one pipeline pair, retrying only transient memory-allocation errors."""
    attempts = max(1, int(max_retries) + 1)
    retry_sleep = max(0.0, float(retry_sleep))
    last_exc = None

    for attempt in range(attempts):
        t0 = time.perf_counter()
        try:
            res = fn(**call_args, verbose=verbose)
            return res, time.perf_counter() - t0, attempt
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts - 1 or not _is_memory_allocation_error(exc):
                raise

            gc.collect()
            sleep_s = retry_sleep * (attempt + 1)
            logger.warning(
                f"{label} memory allocation failed; retry "
                f"{attempt + 1}/{attempts - 1} after {sleep_s:.1f}s: {exc}"
            )
            if sleep_s > 0:
                time.sleep(sleep_s)

    raise last_exc


###############################################################################
# Parameter specification
###############################################################################

def make_params() -> dict:
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
            if data_dir:
                # MOABB/MNE reads the data root from MNE_DATA. Set it inside
                # each worker so CLI/YAML data_dir overrides stale shell config.
                os.environ["MNE_DATA"] = data_dir
                try:
                    import mne
                    mne.set_config("MNE_DATA", data_dir, set_env=True)
                except Exception:
                    logger.debug("Could not persist MNE_DATA config; using process environment only.")

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


def _external_gate_detail_path(base_dir: str, dataset: str, subject_id: int, pipeline: str) -> str:
    filename = f"{dataset}_{subject_id}_{pipeline}_detail.pkl"
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, dataset, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _load_external_bdp_gate_cache(
    gate_dir: str,
    dataset: str,
    subject_id: int,
    ref_pipeline: str = "BDP",
) -> dict:
    """Load saved BDP gates keyed by (method_row, pair_id)."""
    ref_pipeline = normalize_pipeline_labels([ref_pipeline])[0]
    detail_path = _external_gate_detail_path(gate_dir, dataset, subject_id, ref_pipeline)
    if not os.path.exists(detail_path):
        raise FileNotFoundError(f"External BDP gate detail not found: {detail_path}")

    with open(detail_path, "rb") as f:
        records = pickle.load(f)

    cache = {}
    for rec in records:
        det = rec.get("detail") or {}
        ci_table = det.get("ci_table")
        if ci_table is None:
            continue

        key = (int(rec["method_row"]), int(rec["pair_id"]))
        cache[key] = {
            "ci_table": ci_table,
            "B_final": det.get("B_final"),
            "effective_final_idx": det.get("effective_final_idx"),
            "best_idx": det.get("best_idx"),
            "degraded": bool(det.get("degraded", False)),
            "partition_mode": det.get("partition_mode"),
            "source_pipeline": ref_pipeline,
            "source_detail_path": detail_path,
        }

    if not cache:
        raise RuntimeError(f"No usable external BDP gates found in {detail_path}")
    return cache


###############################################################################
# Core processing loop
###############################################################################

def _process_subject_loaded(
    subject_id, data, session_ids, pipelines, result_dir, method_bank, pairs_idx,
    outer_eval_label, dataset,
    nfolds_out=5, nfolds_in=3,
    map_score="kfold", map_k_sess=2, map_n_repeats=1,
    map_shuffle_sessions=True, map_seed=1,
    epsilon=1e-6, default_dist_type="mmd", mmp_B_boot=200,
    mmp_use_external_bdp_gate=False, mmp_external_gate_pipeline="BDP",
    mmp_external_gate_dir=None, mmp_bdp_compatible_final=False,
    dwp_dist_bootstrap_B=1, dwp_dist_est_method="direct",
    memory_retry_attempts=0, memory_retry_sleep=5.0,
    resume=True, verbose=True, seed=None,
    **_ignored,
):
    pipelines = normalize_pipeline_labels(pipelines)
    n_sess = len(data)
    params = make_params()
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

        external_gate_cache = {}
        if pipe_spec["family"] == "MMP" and mmp_use_external_bdp_gate:
            gate_dir = mmp_external_gate_dir or result_dir
            external_gate_cache = _load_external_bdp_gate_cache(
                gate_dir, dataset, subject_id, mmp_external_gate_pipeline,
            )
            if verbose:
                logger.info(
                    f"[Subject {subject_id} | {this_pip}] loaded "
                    f"{len(external_gate_cache)} external BDP gates"
                )

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
                    # Keep MMP variants on the same bootstrap path so they share
                    # one CI-gated near set and differ only in the combiner.
                    mmp_selection_seed = (seed + subj_hash*31 + m*997 + p*31) % (2**31 - 1)
                    np.random.seed(fold_seed)
                else:
                    fold_seed = None
                    mmp_selection_seed = None

                call_args = dict(
                    train_sessions=[data[j] for j in train_idx],
                    test_session=[data[j] for j in test_idx],
                    params=params, feature=row["feature"],
                    classifier=row["classifier"], da=row["da"], epsilon=epsilon,
                )

                map_cv_kwargs = dict(
                    score=map_score, k_sess=map_k_sess, n_repeats=map_n_repeats,
                    shuffle_sessions=map_shuffle_sessions, seed=map_seed,
                    nfolds_out=nfolds_out,
                )

                if pipe_spec["family"] == "MAP":
                    call_args.update(map_cv_kwargs)
                elif pipe_spec["family"] == "DWP":
                    call_args.update(map_cv_kwargs)
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
                    call_args["seed"] = mmp_selection_seed
                    if mmp_use_external_bdp_gate:
                        gate_key = (m, p)
                        if gate_key not in external_gate_cache:
                            raise RuntimeError(
                                f"Missing external BDP gate for method_row={m}, pair_id={p}"
                            )
                        call_args["external_gate"] = external_gate_cache[gate_key]
                        call_args["bdp_compatible_final"] = mmp_bdp_compatible_final

                try:
                    res, elapsed, retry_count = _run_with_memory_retries(
                        algo_fn,
                        call_args,
                        verbose=verbose,
                        max_retries=memory_retry_attempts,
                        retry_sleep=memory_retry_sleep,
                        label=f"[{subject_id}|{this_pip}|cfg{m}|pair{p}]",
                    )

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
                        "memory_retry_count": retry_count,
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
                    "memory_retry_count": rec.get("memory_retry_count"),
                    "dist_type": det.get("dist_type"),
                    "partition_mode": det.get("partition_mode"),
                    "proxy_direction": det.get("proxy_direction"),
                    "borrowed_local_idx": det.get("borrowed_idx"),
                    "borrowed_abs_idx": det.get("borrowed_abs_idx"),
                    "combiner": det.get("combiner"),
                    "harmonize": det.get("harmonize"),
                    "gate_source": det.get("gate_source"),
                    "bdp_compatible_final": det.get("bdp_compatible_final"),
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
