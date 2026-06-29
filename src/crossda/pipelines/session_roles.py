"""
session_roles.py — shared schema for the per-session provenance table.

Each pipeline emits a flat list of role rows with a fixed schema: stage,
local_idx, session_abs_idx, session_label, role, is_best, weight, plus five
distance fields. The per-stage role assignment is genuinely family-specific and
stays in each pipeline; this module owns only the parts that were copy-pasted
across all four families — the distance sub-fields, the empty-distance fill, and
the trailing target-row appender.
"""

# Distance sub-fields, all empty — for rows/stages with no source-to-target distance.
EMPTY_DIST = {
    "dist_to_session": None, "dist_est_method": None,
    "dist_est": None, "dist_lwr": None, "dist_upr": None,
}


def distance_fields(ci, target_abs_idx, est_method):
    """Distance sub-fields for one source, read from its CI row ``ci``.

    ``est_method`` must be the method actually used to produce ``ci`` — pass the
    pipeline's real estimator, not a literal, so the provenance label matches.
    """
    return {
        "dist_to_session": target_abs_idx,
        "dist_est_method": est_method,
        "dist_est": ci.get("est"),
        "dist_lwr": ci.get("lwr"),
        "dist_upr": ci.get("upr"),
    }


def target_rows(stages, target_abs_idx, target_label):
    """Trailing 'target' provenance rows, one per stage (no distance)."""
    return [{
        "stage": stg, "local_idx": None,
        "session_abs_idx": target_abs_idx, "session_label": target_label,
        "role": "target", "is_best": False, "weight": None,
        **EMPTY_DIST,
    } for stg in stages]
