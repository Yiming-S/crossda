"""
config.py — Configuration for the cross-session pipeline.

All runtime settings are loaded from default.yaml (or a user-supplied YAML).
The Config dataclass defaults below are fallbacks only — edit default.yaml
to change behaviour.

Quick reference (see default.yaml for full documentation):

  Datasets:    bnci004 | stieger2021 | ma2020
  Pipelines:   MAP | MMP_merge_then_adapt | MMP_moe | BDP | BDP_bridge_to_far
  Features:    logvar | CSP | TS
  Classifiers: lda | svm_linear | svm_radial | el | lr | lgbm | elm | catboost | mdm
  DA methods:  none | sa | tca | pt | coral
  Distances:   mmd | wasserstein | energy | geodesic | mahalanobis
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

# ── Valid values ────────────────────────────────────────────────────────────
VALID_PIPELINES = (
    "MAP", "DWP",
    "MMP_merge_then_adapt", "MMP_moe",
    "BDP", "BDP_bridge_to_far",
)
VALID_DATASETS = ("bnci004", "stieger2021", "ma2020")
VALID_MODES = ("smoke", "mini", "svm_radial_add", "fair", "practical")
VALID_DIST_POLICIES = ("fixed_mmd", "multi")
VALID_MAP_SCORES = ("kfold", "loso", "pairwise")


# ── Configuration dataclass ────────────────────────────────────────────────
@dataclass
class Config:
    """Runtime configuration. Defaults match default.yaml."""

    # -- Paths --
    data_dir: str = ""                          # set in default.yaml per platform
    result_dir: str = ""                        # set in default.yaml per platform

    # -- Datasets --
    datasets: List[str] = field(default_factory=lambda: ["ma2020"])

    # -- Pipelines --
    # MAP                  : Merge & Adapt (supervised grid search)
    # MMP_merge_then_adapt : Minimum-distance multi-source, weighted merge
    # MMP_moe              : Minimum-distance multi-source, mixture-of-experts
    # DWP                  : Distance-Weighted Pooling (soft-weighted full pool)
    # BDP                  : Bridge-Domain proxy tuning (far → bridge)
    # BDP_bridge_to_far    : Bridge-Domain proxy tuning (bridge → far)
    pipelines: List[str] = field(
        default_factory=lambda: [
            "MAP", "DWP", "MMP_merge_then_adapt", "MMP_moe",
            "BDP", "BDP_bridge_to_far",
        ]
    )

    # -- Method bank --
    # smoke     : 1 config per pipeline (sanity test, ~minutes)
    # mini      : 2 features × 4 DA × 2 classifiers (16/pipeline, ~hours)
    # fair      : full balanced grid across pipelines (~days)
    # practical : larger grid with pipeline-specific pruning (~days)
    method_mode: str = "mini"

    # fixed_mmd : MMD only
    # multi     : MMD + Mahalanobis
    dist_policy: str = "fixed_mmd"

    # -- Cross-validation --
    nfolds_out: int = 5
    nfolds_in: int = 3

    # -- MAP settings --
    # kfold    : session-level k-fold (feature fitted per fold, strictest)
    # loso     : leave-one-session-out
    # pairwise : all source→target session pairs
    map_score: str = "kfold"
    map_k_sess: int = 4             # number of session folds (kfold only)
    map_n_repeats: int = 1          # repeated k-fold (kfold only)
    map_shuffle_sessions: bool = True
    map_seed: int = 1

    # -- MMP settings --
    mmp_B_boot: int = 10            # bootstrap replicates for distance CI

    # -- DWP settings --
    # DWP discards the phase-2 distance CI (only uses `est`), so bootstrap
    # smoothing is rarely worth its ~200x cost. Defaults turn it off.
    dwp_dist_bootstrap_B: int = 1
    dwp_dist_est_method: str = "direct"

    # -- Execution --
    n_cores: int = 5               # subject-level parallelism (1 = sequential)
    resume: bool = True             # skip subjects with existing results
    epsilon: float = 1e-6
    seed: int = 2025
    prep_cache_dir: Optional[str] = None
    verbose: bool = False           # True = full logs; False = progress bars only

    def __post_init__(self):
        if self.method_mode not in VALID_MODES:
            raise ValueError(f"method_mode must be one of {VALID_MODES}")
        if self.dist_policy not in VALID_DIST_POLICIES:
            raise ValueError(f"dist_policy must be one of {VALID_DIST_POLICIES}")
        if self.map_score not in VALID_MAP_SCORES:
            raise ValueError(f"map_score must be one of {VALID_MAP_SCORES}")
        for ds in self.datasets:
            if ds not in VALID_DATASETS:
                raise ValueError(f"Unknown dataset: {ds!r}. Valid: {VALID_DATASETS}")
        self.pipelines = normalize_pipeline_labels(self.pipelines)
        if self.nfolds_out < 2:
            raise ValueError("nfolds_out must be >= 2")
        if self.nfolds_in < 2:
            raise ValueError("nfolds_in must be >= 2")
        if self.prep_cache_dir is None:
            self.prep_cache_dir = os.path.join(self.result_dir, "prep_cache")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


# ── Pipeline label helpers ─────────────────────────────────────────────────

_PIPELINE_ALIASES = {
    "MAP": "MAP",
    "DWP": "DWP",
    "WMAP": "DWP",
    "BDP": "BDP",
    "BDP_BRIDGE_TO_FAR": "BDP_bridge_to_far",
    "BDP_BF": "BDP_bridge_to_far",
    "MMP": "MMP_merge_then_adapt",
    "MMP_MERGE": "MMP_merge_then_adapt",
    "MMP_MERGE_THEN_ADAPT": "MMP_merge_then_adapt",
    "MMP_MOE": "MMP_moe",
}


def normalize_pipeline_labels(pipelines: List[str]) -> List[str]:
    result = []
    for p in pipelines:
        key = p.upper().replace("-", "_").replace(" ", "_")
        canonical = _PIPELINE_ALIASES.get(key)
        if canonical is None:
            raise ValueError(f"Unknown pipeline label: {p!r}. Valid: {VALID_PIPELINES}")
        if canonical not in result:
            result.append(canonical)
    return result


_PIPELINE_SPECS = {
    "MAP":                  {"label": "MAP",                  "family": "MAP", "combiner": None},
    "DWP":                  {"label": "DWP",                  "family": "DWP", "combiner": None},
    "MMP_merge_then_adapt": {"label": "MMP_merge_then_adapt", "family": "MMP", "combiner": "merge_then_adapt"},
    "MMP_moe":              {"label": "MMP_moe",              "family": "MMP", "combiner": "moe"},
    "BDP":                  {"label": "BDP",                  "family": "BDP", "combiner": None, "proxy_direction": "far_to_bridge"},
    "BDP_bridge_to_far":    {"label": "BDP_bridge_to_far",    "family": "BDP", "combiner": None, "proxy_direction": "bridge_to_far"},
}


def resolve_pipeline_spec(pipeline_label: str) -> dict:
    lbl = normalize_pipeline_labels([pipeline_label])[0]
    return _PIPELINE_SPECS[lbl]
