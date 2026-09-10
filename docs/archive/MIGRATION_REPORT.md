# Migration report - Field Station / Virtual Station split

## Purpose

Separate the former mixed ET Fundacion repository/workspace into independent
Field Station and Virtual Station operating states without changing scientific
results.

## Source baseline

- Source Git commit: `726b914c54d4055ab9bb795179a08cd4db12824d`.
- Field Station repository: branch `field-station-stable`.
- Virtual Station repository: branch `main`.
- The original mixed repository/workspace and the migration backup were retained
  during validation.

## Field Station preservation

The following source-to-destination copies were checked by file count and total
bytes and matched exactly:

- `current`: 2994 files.
- `final`: 8 files.
- `_audit_package`: 159 files.

Field Station scientific reference remains the Stable5 design and run
`20260907T162048Z_2020_2024`, with 833 training rows and AOA threshold
`0.6089885022306`.

## Virtual Station preservation

The V5 workspace was reorganized from the former experiment layout into:

- `current`
- `training`
- `evaluation`
- `final`

Directory integrity checks matched file counts and total bytes for all nine
migrated components:

- production/current: 1584 files.
- selection: 17 files.
- training raw: 263 files.
- training master: 2 files.
- QA: 2 files.
- results: 16 files.
- field comparison: 178 files.
- basin coverage: 11 files.
- summary: 5 files.

SHA-256 matched source versus migrated destination for:

- `selected_supports.csv`
- `virtual_training_master.csv`
- `virtual10_training_population.csv`
- `virtual10_vs_stable_metrics.csv`
- `reciprocal_AOA_by_support.csv`
- `virtual10_vs_stable_field_metrics.csv`
- `coverage_comparison.csv`

Frozen V5 state validated after migration:

- seed: 42.
- supports: 10.
- distinct fixed UTM 10 km blocks: 10.
- training rows: 1454.
- AOA threshold: `0.8007328515330622`.
- frozen GE90 period total: 1454.
- extracted training row total: 1454.
- support-period sets identical: 10/10.
- selected MODIS pixel equals extracted MODIS pixel: 10/10.
- scientific rasters present for 2020-03-13, 2021-11-25 and 2022-03-30.

## Code migration policy

The Astra staging work was not copied wholesale. Only changes required for
Virtual Station workspace isolation and the canonical V5 workflow were adopted.

The operational dispatcher is non-destructive by default. Support selection,
training reproduction, field evaluation and raster production require explicit
subcommands.

Stable5 is never searched implicitly inside the Virtual Station workspace.
Comparisons require an explicit Field Station reference workspace.

Historical frozen result files were not edited solely to rename provenance
labels; display/summary code normalizes the known inherited `v4_basin10` label
to `v5_basin10` without changing the frozen source CSV.

## Tests before final commit

Before final commit, rerun:

```powershell
python -m pytest -q
git diff --check
python scripts/run_pipeline.py validate
python scripts/run_pipeline.py audit
```

The finalized Virtual Station suite passed 127 tests and 10 subtests with 20 warnings. The lower count reflects removal of Field-specific pipeline contracts after the repository split; four still-applicable scientific contracts and the package-root import guard were retained in Virtual-specific tests.
Warnings were the previously observed OpenMP runtime warning and rasterio/Affine
pending-deprecation warnings.

## Not performed during migration

- No support resampling.
- No change to seed or GE90 rule.
- No new model selection.
- No target/predictor change.
- No AOA methodological change.
- No new field interpretation.
- No Google Drive use.
- No push, merge, tag or commit before approval.

## Final Virtual Station integrity confirmation

The four regenerated basin-coverage CSV products were checked with SHA-256
against the original V5 experiment workspace and were byte-identical:

- `coverage_by_model.csv`
- `coverage_comparison.csv`
- `publication_transitions.csv`
- `aoa_transitions.csv`

Final Virtual Station validation:

- pytest: 127 passed.
- subtests: 10 passed.
- warnings: 20 known warnings.
- `git diff --check`: clean.
- frozen selection/training audit: PASS.
- frozen raster-state validation: PASS.
