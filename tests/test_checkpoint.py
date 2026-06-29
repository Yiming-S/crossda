"""Checkpoint resume must not graft cached accuracies onto the wrong rows."""

import numpy as np

from crossda.core.method_bank import generate_method_bank
from crossda.core import workers as W


def test_checkpoint_resumes_when_method_bank_matches(tmp_path):
    mb = generate_method_bank("smoke")
    cached = mb.copy()
    cached.loc[0, "cvMeanAcc"] = 0.77        # pretend row 0 was computed
    path = str(tmp_path / "ckpt.pkl")
    W._save_checkpoint(path, cached, [{"x": 1}], 0)

    loaded, details, resume = W._load_checkpoint(path, mb.copy())
    assert resume == 1
    assert loaded.loc[0, "cvMeanAcc"] == 0.77
    assert details == [{"x": 1}]


def test_checkpoint_discarded_when_method_bank_reordered(tmp_path):
    mb = generate_method_bank("smoke")
    cached = mb.copy()
    cached.loc[0, "cvMeanAcc"] = 0.77
    path = str(tmp_path / "ckpt.pkl")
    W._save_checkpoint(path, cached, [{"x": 1}], 0)

    # Row identities no longer line up positionally -> stale cache must be dropped.
    reordered = mb.iloc[::-1].reset_index(drop=True)
    loaded, details, resume = W._load_checkpoint(path, reordered.copy())
    assert resume == 0
    assert details == []
    assert np.isnan(loaded.loc[0, "cvMeanAcc"])  # not grafted onto the wrong row
