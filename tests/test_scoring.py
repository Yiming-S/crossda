"""Cross-session CV scorers and score-mode dispatch (kfold / loso / pairwise)."""

import numpy as np
import pytest

from crossda.pipelines.scoring import score_session_cv, _normalize_score_mode
from crossda.pipelines.pipeline_utils import (
    extract_features_train,
    extract_features_test,
    get_classifier,
)


def _feature_lists(sessions, feat="CSP"):
    allX = np.concatenate([s["x"] for s in sessions], axis=0)
    allY = np.concatenate([s["y"] for s in sessions])
    _, obj = extract_features_train(allX, allY, feat)
    x_list = [extract_features_test(s["x"], obj, feat) for s in sessions]
    y_list = [s["y"] for s in sessions]
    return x_list, y_list


@pytest.mark.parametrize("mode", ["kfold", "loso", "pairwise"])
def test_score_session_cv_finite(mode, synthetic_sessions):
    # da='none' keeps this independent of the da4bci backend while still
    # exercising the relocated _score_kfold_session / _score_loso / _score_pairwise.
    x_list, y_list = _feature_lists(synthetic_sessions)
    acc = score_session_cv(
        mode, synthetic_sessions, x_list, y_list, "CSP", {}, "none", {}, get_classifier("lda"),
        k_sess=4, n_repeats=1, shuffle_sessions=True, seed=1, nfolds_out=3,
    )
    assert np.isfinite(acc) and 0.0 <= acc <= 1.0


def test_normalize_score_mode_aliases_and_error():
    assert _normalize_score_mode("kf") == "kfold"
    assert _normalize_score_mode("Pairwise") == "pairwise"
    assert _normalize_score_mode("leave_one_session_out") == "loso"
    with pytest.raises(ValueError):
        _normalize_score_mode("bogus_mode")
