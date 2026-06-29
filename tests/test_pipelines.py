"""End-to-end smoke + reproducibility for all pipeline families.

These exercise the full per-subject driver (feature extraction, domain adaptation
via da4bci, scoring, combiners and output writing) on synthetic sessions.
"""

import glob
import os

import pandas as pd
import pytest

pytest.importorskip("da4bci")  # the DA backend is required to run any pipeline

from crossda.core.method_bank import generate_method_bank
from crossda.core.workers import process_subject

RUN_KWARGS = dict(
    dataset="synthetic", data_dir="", cache_dir=None,
    nfolds_out=5,
    map_score="kfold", map_k_sess=4, map_n_repeats=1,
    map_shuffle_sessions=True, map_seed=1,
    epsilon=1e-6, default_dist_type="mmd",
    mmp_B_boot=50, mmp_use_external_bdp_gate=False,
    mmp_external_gate_pipeline="BDP", mmp_external_gate_dir=None,
    mmp_bdp_compatible_final=False,
    dwp_dist_bootstrap_B=1, dwp_dist_est_method="direct",
    memory_retry_attempts=0, memory_retry_sleep=5.0,
    resume=False, verbose=False,
)


def _run(result_dir, seed=2025):
    bank = generate_method_bank(mode="smoke", dist_policy="fixed_mmd")
    pipelines = sorted(bank["pipeline"].unique().tolist())
    return process_subject(
        subject_id=1, pipelines=pipelines, result_dir=str(result_dir),
        method_bank=bank, seed=seed, **RUN_KWARGS,
    )


def test_smoke_runs_all_families_with_finite_accuracy(patched_loader, tmp_path):
    res = _run(tmp_path)
    assert isinstance(res, pd.DataFrame)
    # One row per pipeline family in smoke mode.
    assert set(res["pipeline"]) == {
        "MAP", "DWP", "MMP_merge_then_adapt", "MMP_moe", "BDP", "BDP_bridge_to_far",
    }
    acc = pd.to_numeric(res["cvMeanAcc"], errors="coerce")
    base = pd.to_numeric(res["baseline"], errors="coerce")
    assert acc.notna().all() and base.notna().all()
    assert ((acc >= 0) & (acc <= 1)).all()


def test_smoke_writes_result_files(patched_loader, tmp_path):
    # process_subject writes straight into the result_dir it is given.
    _run(tmp_path)
    out = str(tmp_path)
    assert glob.glob(os.path.join(out, "*_MAP.pkl"))
    assert glob.glob(os.path.join(out, "*_detail.pkl"))
    assert glob.glob(os.path.join(out, "*_roles.csv"))


def test_unknown_kwarg_raises_not_silently_ignored(patched_loader, tmp_path):
    # A misspelled control parameter must raise, not land in a dropped catch-all
    # and silently run with defaults (which would produce a wrong-but-plausible number).
    bank = generate_method_bank(mode="smoke")
    with pytest.raises(TypeError):
        process_subject(
            subject_id=1, pipelines=["MAP"], result_dir=str(tmp_path),
            method_bank=bank, seed=2025, nfolds_outer=10, **RUN_KWARGS,
        )


def test_reproducible_with_fixed_seed(patched_loader, tmp_path):
    a = _run(tmp_path / "a", seed=2025)
    b = _run(tmp_path / "b", seed=2025)
    cols = ["pipeline", "feature", "classifier", "da", "cvMeanAcc", "baseline"]
    fa = a[cols].sort_values("pipeline").reset_index(drop=True)
    fb = b[cols].sort_values("pipeline").reset_index(drop=True)
    pd.testing.assert_frame_equal(fa, fb)
