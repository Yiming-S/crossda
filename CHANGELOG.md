# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed (bug hunt)
- **Checkpoint resume now merges by column label.** The identity-checked merge
  still used a positional `iloc` graft, which crashed (`IndexError`) when the
  cached method bank had a different column count and silently swapped values
  (e.g. cvMeanAcc ↔ baseline) when the column order differed. It now copies the
  cached values by column name.
- **MMP `moe` combiner no longer crashes on a single-class source.** Such a source
  votes for its only class instead of crashing the whole run in `predict`.
- **`distance_ci` never reports `est=inf` inside a finite CI.** When the direct
  full-data estimate fails but the bootstrap resamples are finite, the estimate
  falls back to the bootstrap mean (previously a finite-CI source could be ranked
  as the worst).
- **Small sessions no longer fail with an opaque error.** `_outer_folds` clamps
  the CV fold count to the smallest class instead of raising when a session has
  fewer members per class than `nfolds_out`.
- **Scalar-string config values are accepted.** `pipelines: MAP` / `datasets: ma2020`
  written as a YAML scalar are wrapped in a list instead of being iterated
  character-by-character.
- **Parallelism is no longer permanently downgraded.** A parallel-execution
  failure on one dataset used to mutate `cfg.n_cores=1`, forcing every later
  dataset to run sequentially; the fallback is now per-dataset.
- Note: fixing the da4bci CORAL bug changes the results of any `coral`-DA
  configuration (the previous CORAL output was mathematically wrong).

### Fixed (review batch)
- **Checkpoint resume is now identity-checked.** `_load_checkpoint` verifies the
  cached rows still match the current method bank by identity and discards a stale
  cache (recomputing) instead of grafting.
- **`process_subject` no longer silently swallows unknown keyword arguments.** The
  `**_ignored` catch-all on the per-subject driver was removed, so a misspelled
  control parameter (`nfolds_outer=10`, a typo'd `score`) now raises `TypeError`
  instead of being dropped and running with defaults. `Config.from_dict` likewise
  rejects unknown YAML keys with a clear `ValueError`.

### Changed (review batch, behavior-preserving)
- Hoisted the fold-invariant domain adaptation out of `map_cv`'s outer-fold loop
  (~5× fewer DA computations in the hottest scoring primitive; numerically identical).
- Removed the dead inner-fold machinery from `cross_cv` and the inert `nfolds_in`
  knob end-to-end (config, default.yaml, worker signature).
- `_weight_from_D` now copies its input instead of mutating it in place;
  `_normalize_score_mode`'s error message is no longer MAP-specific; dropped MAP's
  unused `epsilon` parameter; aligned the direct-API `MAP`/`DWP` `k_sess` default to
  4 (matching the CLI/Config default). Golden smoke/broad digests are unchanged.

### Added (review batch)
- `default_dist_type` Config field (validated against the five distance names) so
  the multi-distance support is reachable from config instead of a hardcoded literal.
- Tests for the loso/pairwise scorers, config validation, and the unknown-kwarg guard.

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
