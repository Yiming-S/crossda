# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Installable package.** `pyproject.toml` (setuptools, src layout), a `crossda`
  console entry point (`crossda` / `python -m crossda`), a top-level public API
  (`Config`, `run`, `process_subject`, `generate_method_bank`, `MAP/DWP/MMP/BDP`),
  `LICENSE` (MIT), and a real `README.md`. A blanked `default.yaml` now ships
  inside the package. The domain-adaptation backend `da4bci` is declared as the
  `da` optional extra (git dependency).

### Removed
- Dead code: unused imports across the package and the ignored `dataset`
  parameter of `make_params()`.

### Repository (structure)
- Collapsed the doubled `CrossPython/CrossPython/` nesting into a `src/crossda/`
  layout. `main.py` → `cli.py`; run configs moved to `configs/examples/`; the
  machine-specific config is preserved as the git-ignored `configs/default.local.yaml`.

### Fixed
- **DWP provenance label.** The per-session `session_roles` table recorded
  `dist_est_method = "boot_mean"` for DWP regardless of the estimator actually
  used. DWP's `dist_est_method` defaults to `"direct"`, so the label
  contradicted the real method (and the matching field in the detail record).
  The label is now threaded from the real estimator. **Accuracies are
  unaffected**; only the `dist_est_method` column of DWP role CSVs changes
  (`boot_mean` → `direct`). MMP and BDP are unchanged — they genuinely use
  `boot_mean`. Re-runs of DWP will show `direct`; archived results carry the old
  mislabel.
- **Domain-adaptation result access.** `core/cross_cv.py` (and the MAP final
  step) read `da_result["weighted_source_data"]` / `["target_data"]` with
  unguarded subscripting, which hard-crashes if the DA returns unexpected keys,
  while the other pipelines degraded gracefully via `.get(..., raw)`. All call
  sites now share one `apply_da` helper that falls back to the raw inputs, so a
  DA failure degrades to no-DA everywhere. Behavior on normal runs is identical.

### Changed
- **De-duplicated the four pipeline families** (MAP/DWP/MMP/BDP) into shared
  neutral modules without changing scientific behavior: `pipelines/scoring.py`
  (CV scorers + `score_session_cv`), `pipelines/weighting.py`,
  `pipelines/session_roles.py`, and `pipeline_utils.compute_distance_ci_table` /
  `apply_da`. This removed the cross-pipeline imports of underscore-"private"
  helpers (dwp→map, bdp→map, dwp→mmp).

### Repository
- Added `.gitignore`; stopped tracking `__pycache__/`, `*.pyc`, and `.DS_Store`.
- Removed empty `analysis/` and `literature/` directories.
