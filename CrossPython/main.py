"""
main.py — Cross-session EEG classification pipeline (Python)

Usage:
  python -m CrossPython.main
  python -m CrossPython.main --config my_config.yaml
  python -m CrossPython.main --datasets bnci004 --mode smoke
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Allow running as `python main.py` (direct script) or `python -m CrossPython.main`
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "CrossPython"

import numpy as np

from .config import Config
from .core.method_bank import generate_method_bank
from .core.workers import process_subject

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

logger = logging.getLogger("CrossPython")


def _get_subjects(dataset: str, data_dir: str = "") -> list:
    if dataset == "bnci004":
        return list(range(1, 10))
    if dataset == "stieger2021":
        return _scan_local_stieger(data_dir)
    if dataset == "ma2020":
        return list(range(1, 26))
    raise ValueError(f"Unknown dataset: {dataset}")


def _scan_local_stieger(data_dir: str) -> list:
    """Scan local .mat files to find available Stieger2021 subjects."""
    import re
    stieger_dir = os.path.join(data_dir, "MNE-Stieger2021-data")
    if not os.path.isdir(stieger_dir):
        raise FileNotFoundError(f"Stieger2021 data not found at {stieger_dir}")
    subjects = set()
    for f in os.listdir(stieger_dir):
        m = re.match(r"S(\d+)_Session_\d+\.mat", f)
        if m:
            subjects.add(int(m.group(1)))
    # Filter: need at least 2 sessions for cross-session
    from collections import Counter
    sess_count = Counter()
    for f in os.listdir(stieger_dir):
        m = re.match(r"S(\d+)_Session_\d+\.mat", f)
        if m:
            sess_count[int(m.group(1))] += 1
    return sorted(s for s, n in sess_count.items() if n >= 2)


def run(cfg: Config):
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


    if not cfg.verbose:
        # Suppress noisy MNE / pyriemann / sklearn logs
        import mne
        mne.set_log_level("ERROR")
        for name in ("mne", "pyriemann", "sklearn", "moabb", "lightgbm", "catboost"):
            logging.getLogger(name).setLevel(logging.ERROR)

    logger.info(f"datasets: {cfg.datasets} | pipelines: {cfg.pipelines} | mode: {cfg.method_mode}")

    for dataset in cfg.datasets:
        logger.info(f"\n=== Dataset: {dataset} ===")
        out_dir = os.path.join(cfg.result_dir, dataset)
        os.makedirs(out_dir, exist_ok=True)

        subjects = _get_subjects(dataset, cfg.data_dir)
        method_bank = generate_method_bank(mode=cfg.method_mode, dist_policy=cfg.dist_policy)

        logger.info(f"{dataset}: {len(subjects)} subjects | bank: {method_bank.groupby('pipeline').size().to_dict()}")

        common_args = dict(
            pipelines=cfg.pipelines, result_dir=out_dir, method_bank=method_bank,
            data_dir=cfg.data_dir, cache_dir=cfg.prep_cache_dir, dataset=dataset,
            nfolds_out=cfg.nfolds_out, nfolds_in=cfg.nfolds_in,
            map_score=cfg.map_score, map_k_sess=cfg.map_k_sess,
            map_n_repeats=cfg.map_n_repeats, map_shuffle_sessions=cfg.map_shuffle_sessions,
            map_seed=cfg.map_seed, epsilon=cfg.epsilon, default_dist_type="mmd",
            mmp_B_boot=cfg.mmp_B_boot, resume=cfg.resume, seed=cfg.seed, verbose=True,
        )

        np.random.seed(cfg.seed)
        n_ok, n_fail, failures = 0, 0, []

        if cfg.n_cores > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            from concurrent.futures import ProcessPoolExecutor
            try:
                with ProcessPoolExecutor(max_workers=cfg.n_cores, mp_context=ctx) as pool:
                    futures = {pool.submit(process_subject, subject_id=sid, **common_args): sid
                               for sid in subjects}
                    for fut in futures:
                        sid = futures[fut]
                        try:
                            res = fut.result()
                        except Exception as e:
                            logger.error(f"[{dataset} | subject {sid}] Error: {e}")
                            res = None
                        if res is not None:
                            n_ok += 1
                        else:
                            n_fail += 1
                            failures.append(str(sid))
            except Exception as e:
                logger.warning(f"Parallel execution failed ({e}), falling back to sequential.")
                cfg.n_cores = 1

        if cfg.n_cores <= 1:
            from tqdm import tqdm
            for sid in tqdm(subjects, desc=f"{dataset} subjects", unit="sub"):
                try:
                    res = process_subject(subject_id=sid, **common_args)
                except Exception as e:
                    logger.error(f"[{dataset} | subject {sid}] Error: {e}")
                    res = None
                if res is not None:
                    n_ok += 1
                else:
                    n_fail += 1
                    failures.append(str(sid))

        logger.info(f"{dataset} done: {n_ok}/{len(subjects)} succeeded, {n_fail} failed.")
        if failures:
            logger.info(f"Failed subjects: {', '.join(failures)}")

    logger.info("\ncross_session pipeline complete.")


def main():
    parser = argparse.ArgumentParser(description="Cross-session EEG pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--pipelines", nargs="+", default=None)
    parser.add_argument("--mode", choices=["smoke", "mini", "fair", "practical"], default=None)
    parser.add_argument("--n-cores", type=int, default=None)
    args = parser.parse_args()

    # Load yaml as raw dict, merge CLI overrides, then construct Config
    default_yaml = Path(__file__).parent / "default.yaml"
    yaml_path = args.config or (default_yaml if default_yaml.exists() else None)
    if yaml_path:
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    if args.data_dir:
        data["data_dir"] = args.data_dir
    if args.result_dir:
        data["result_dir"] = args.result_dir
    if args.datasets:
        data["datasets"] = args.datasets
    if args.pipelines:
        data["pipelines"] = args.pipelines
    if args.mode:
        data["method_mode"] = args.mode
    if args.n_cores is not None:
        data["n_cores"] = args.n_cores

    cfg = Config(**data)

    run(cfg)


if __name__ == "__main__":
    main()
