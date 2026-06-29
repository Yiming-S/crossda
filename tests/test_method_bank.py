"""Method-bank construction."""

import pytest

from crossda.core.method_bank import generate_method_bank

REQUIRED_COLUMNS = {
    "pipeline", "feature", "classifier", "da", "dist_type",
    "combiner", "mode", "space_id",
}


@pytest.mark.parametrize("mode", ["smoke", "mini", "svm_radial_add", "fair", "practical"])
def test_modes_build_nonempty_banks(mode):
    mb = generate_method_bank(mode=mode, dist_policy="fixed_mmd")
    assert len(mb) > 0
    assert REQUIRED_COLUMNS.issubset(mb.columns)
    assert (mb["mode"] == mode).all()
    # No duplicate (pipeline, feature, classifier, da, dist_type) rows.
    key = ["pipeline", "feature", "classifier", "da", "dist_type"]
    assert not mb.duplicated(subset=key).any()


def test_smoke_has_one_row_per_pipeline():
    mb = generate_method_bank(mode="smoke")
    assert len(mb) == mb["pipeline"].nunique() == 6


def test_multi_dist_policy_adds_mahalanobis():
    mb = generate_method_bank(mode="practical", dist_policy="multi")
    geom = mb[mb["pipeline"].str.startswith(("MMP", "BDP"))]
    assert set(geom["dist_type"].dropna()) >= {"mmd", "mahalanobis"}
