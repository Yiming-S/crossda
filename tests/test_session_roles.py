"""Shared session-roles schema helpers."""

from crossda.pipelines.session_roles import EMPTY_DIST, distance_fields, target_rows

DIST_KEYS = {"dist_to_session", "dist_est_method", "dist_est", "dist_lwr", "dist_upr"}


def test_empty_dist_is_all_none():
    assert set(EMPTY_DIST) == DIST_KEYS
    assert all(v is None for v in EMPTY_DIST.values())


def test_distance_fields_reads_ci_and_threads_method():
    ci = {"est": 0.5, "lwr": 0.4, "upr": 0.6}
    df = distance_fields(ci, target_abs_idx=7, est_method="direct")
    assert set(df) == DIST_KEYS
    assert df["dist_to_session"] == 7
    assert df["dist_est_method"] == "direct"  # threaded, not hardcoded
    assert df["dist_est"] == 0.5 and df["dist_lwr"] == 0.4 and df["dist_upr"] == 0.6


def test_target_rows_one_per_stage_no_distance():
    rows = target_rows(("main", "final"), target_abs_idx=3, target_label="t")
    assert [r["stage"] for r in rows] == ["main", "final"]
    for r in rows:
        assert r["role"] == "target"
        assert r["session_abs_idx"] == 3
        assert r["weight"] is None
        assert all(r[k] is None for k in DIST_KEYS)
