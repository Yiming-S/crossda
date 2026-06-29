"""Config validation and pipeline-label helpers."""

import pytest

from crossda.config import (
    Config,
    normalize_pipeline_labels,
    resolve_pipeline_spec,
)


def test_default_config_constructs():
    cfg = Config()
    assert cfg.method_mode in ("smoke", "mini", "svm_radial_add", "fair", "practical")
    assert cfg.nfolds_out >= 2
    # prep_cache_dir is derived from result_dir in __post_init__
    assert cfg.prep_cache_dir is not None


@pytest.mark.parametrize("kwargs", [
    {"method_mode": "nonsense"},
    {"dist_policy": "nonsense"},
    {"map_score": "nonsense"},
    {"default_dist_type": "nonsense"},
    {"datasets": ["not_a_dataset"]},
    {"nfolds_out": 1},
])
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_from_dict_rejects_unknown_keys():
    # A typo'd config key is caught with a clear ValueError, not silently ignored.
    with pytest.raises(ValueError):
        Config.from_dict({"nfolds_outer": 10})
    # A valid mapping still constructs.
    assert Config.from_dict({"method_mode": "smoke"}).method_mode == "smoke"


def test_normalize_pipeline_labels_canonicalises_and_dedups():
    # WMAP is an alias for DWP; duplicates collapse; order is preserved.
    out = normalize_pipeline_labels(["WMAP", "MAP", "map", "MMP_MOE"])
    assert out == ["DWP", "MAP", "MMP_moe"]


def test_normalize_pipeline_labels_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_pipeline_labels(["NOT_A_PIPELINE"])


def test_resolve_pipeline_spec():
    assert resolve_pipeline_spec("MMP_moe")["family"] == "MMP"
    assert resolve_pipeline_spec("MMP_moe")["combiner"] == "moe"
    assert resolve_pipeline_spec("BDP_bridge_to_far")["proxy_direction"] == "bridge_to_far"
    assert resolve_pipeline_spec("MAP")["family"] == "MAP"
