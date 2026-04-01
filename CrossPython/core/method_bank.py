"""
method_bank.py
--------------
Build the candidate method space (feature × classifier × DA × dist_type)
for one or more pipelines.

Mirrors generate_methodBank() from workers.R.
"""

from __future__ import annotations

import itertools
from typing import List, Optional

import pandas as pd


def generate_method_bank(
    mode: str = "fair",
    dist_policy: str = "fixed_mmd",
) -> pd.DataFrame:
    """
    Build the candidate method space.

    Parameters
    ----------
    mode : {"smoke", "mini", "fair", "practical"}
        smoke     – 1 config per pipeline (sanity test)
        mini      – focused comparison: 2 features × 4 DA × 2 classifiers (16/pipeline)
        fair      – full balanced space across pipelines
        practical – larger space with pipeline-specific pruning
    dist_policy : {"fixed_mmd", "multi"}
        fixed_mmd – single MMD distance
        multi     – MMD + Mahalanobis

    Returns
    -------
    pd.DataFrame with columns:
        pipeline, feature, classifier, da, dist_type, combiner,
        mode, prune_tag, space_id, score, cvMeanAcc, baseline
    """
    mmp_variants = ["MMP_merge_then_adapt", "MMP_moe"]
    bdp_variants = ["BDP", "BDP_bridge_to_far"]
    features = ["logvar", "CSP", "TS"]
    classifiers = ["lda", "svm_linear", "el"]
    da_methods = ["sa", "tca", "pt", "none"]
    dist_types = ["mmd"] if dist_policy == "fixed_mmd" else ["mmd", "mahalanobis"]

    def combiner_from_pipeline(p: str) -> Optional[str]:
        if p == "MMP_merge_then_adapt":
            return "merge_then_adapt"
        if p == "MMP_moe":
            return "moe"
        return None

    # ── Build raw grid ───────────────────────────────────────────────────

    if mode == "smoke":
        rows = [
            dict(pipeline="MAP", feature="CSP", classifier="lda", da="sa", dist_type=None),
            dict(pipeline="DWP", feature="CSP", classifier="lda", da="sa", dist_type=None),
            dict(pipeline="MMP_merge_then_adapt", feature="CSP", classifier="lda", da="sa", dist_type="mmd"),
            dict(pipeline="MMP_moe", feature="CSP", classifier="lda", da="sa", dist_type="mmd"),
            dict(pipeline="BDP", feature="CSP", classifier="lda", da="sa", dist_type="mmd"),
            dict(pipeline="BDP_bridge_to_far", feature="CSP", classifier="lda", da="sa", dist_type="mmd"),
        ]
        mb = pd.DataFrame(rows)

    elif mode == "mini":
        # Focused pipeline comparison: 2 features × 4 DA × 2 classifiers = 16/pipeline
        mini_features = ["CSP", "logvar"]
        mini_da = ["none", "sa", "pt", "coral"]
        mini_clf = ["lda", "svm_linear"]
        all_pips = ["MAP", "DWP"] + mmp_variants + bdp_variants
        combos = list(itertools.product(all_pips, mini_features, mini_clf, mini_da))
        mb = pd.DataFrame(combos, columns=["pipeline", "feature", "classifier", "da"])
        mb["dist_type"] = mb["pipeline"].apply(
            lambda p: None if p in ("MAP", "DWP") else dist_types[0]
        )

    elif mode == "fair":
        all_pips = ["MAP", "DWP"] + mmp_variants + bdp_variants
        combos = list(itertools.product(all_pips, features, classifiers, da_methods))
        mb = pd.DataFrame(combos, columns=["pipeline", "feature", "classifier", "da"])
        mb["dist_type"] = mb["pipeline"].apply(
            lambda p: None if p in ("MAP", "DWP") else dist_types[0]
        )

    else:  # practical
        # Geometric pipelines: full dist_type expansion
        geom_pips = mmp_variants + bdp_variants
        geom_combos = list(itertools.product(
            geom_pips, features, classifiers, da_methods, dist_types
        ))
        grid_geom = pd.DataFrame(
            geom_combos,
            columns=["pipeline", "feature", "classifier", "da", "dist_type"],
        )
        # MAP and DWP: no dist_type
        map_combos = list(itertools.product(
            ["MAP", "DWP"], features, classifiers, da_methods
        ))
        grid_map = pd.DataFrame(
            map_combos, columns=["pipeline", "feature", "classifier", "da"]
        )
        grid_map["dist_type"] = None
        mb = pd.concat([grid_geom, grid_map], ignore_index=True)

    # ── Pruning ──────────────────────────────────────────────────────────

    if mode not in ("smoke", "mini"):
        mb = _prune_common(mb)
        if mode == "practical":
            mb = _prune_pipeline_specific(mb)

    # ── Metadata columns ─────────────────────────────────────────────────

    mb["combiner"] = mb["pipeline"].apply(combiner_from_pipeline)
    mb["mode"] = mode
    mb["prune_tag"] = {
        "smoke": "none",
        "mini": "none",
        "fair": "common",
        "practical": "common+pipeline_specific",
    }[mode]
    mb["space_id"] = f"{mode}_{dist_policy}"
    mb["score"] = None
    mb["cvMeanAcc"] = float("nan")
    mb["baseline"] = float("nan")

    mb = mb.drop_duplicates().reset_index(drop=True)
    return mb


# ── Pruning helpers (match R logic exactly) ──────────────────────────────────

def _prune_common(df: pd.DataFrame) -> pd.DataFrame:
    """Remove known-bad combinations across all pipelines."""
    mask = (
        # logvar + el: incompatible
        ~((df["feature"] == "logvar") & (df["classifier"] == "el"))
        # TS + sa/tca: not meaningful (tangent space already aligned)
        & ~((df["feature"] == "TS") & (df["da"].isin(["sa", "tca"])))
        # logvar + tca: poor performance
        & ~((df["feature"] == "logvar") & (df["da"] == "tca"))
    )
    return df[mask].reset_index(drop=True)


def _prune_pipeline_specific(df: pd.DataFrame) -> pd.DataFrame:
    """Additional pruning for 'practical' mode."""
    # el classifier not used with MMP pipelines
    mask = ~(
        (df["classifier"] == "el") & df["pipeline"].str.startswith("MMP")
    )
    return df[mask].reset_index(drop=True)
