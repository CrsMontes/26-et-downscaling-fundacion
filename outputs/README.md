# Local outputs (not tracked by Git)

Every generated file from the main RF workflow is written under this directory.
Only this README is versioned; all generated contents are ignored by Git.

Main layout after a fresh run:

- `training/selection/` — Virtual10 support selection and GE90 checks.
- `training/raw/` — raw training extractions.
- `training/master/` — reconstructed training master tables.
- `evaluation/results/` — RF spatial OOF, LOYO, fold metrics and training population.
- `current/models/` — fitted RF-25 model, weighted DI/AOA parameters and metadata.
- `current/raw/candidates/` — optional comprehensive candidate-predictor archive.
- `current/rasters/` — final 20 m ET products and temporary raw-support tiles.
- `current/rasters_modis/` — native-grid MODIS ET used by exact reconciliation.
- `current/diagnostics/` — QC, AOA/DI and sensitivity diagnostics.
- `current/figures/` — figures generated from final tables/rasters.
- `current/logs/` — run logs and manifests.

Do not add generated files below `outputs/` to GitHub.
