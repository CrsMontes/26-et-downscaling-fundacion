# Script classification

## Current production

- `scripts/run_pipeline.py`
- `scripts/select_v5_basin_random_sequential_ge90.py`
- `scripts/run_v5_basin_experiment.py` (`--extract-only` in final workflow)
- `scripts/train_rf25.py`
- `scripts/produce_rf25_rasters.py`
- `scripts/export_meteorology_data.py`
- `scripts/export_satellite_data.py`
- `scripts/build_training_dataset.py`

## Comprehensive candidate-predictor reconstruction

Orchestrated by `scripts/download_all_candidate_predictors.py`:

- `reproducibility/scripts/export_availability_diagnostic.py`
- `reproducibility/scripts/export_optical_source_experiment.py`
- `reproducibility/scripts/export_s2_rich_optical.py`
- `reproducibility/scripts/export_s1_geometry_predictors.py`
- `reproducibility/scripts/export_hls_albedo_fvc.py`
- `reproducibility/scripts/export_thermal_availability.py`
- `reproducibility/scripts/export_landsat_lst_predictor.py`
- `reproducibility/scripts/build_meteorology_experiment_table.py`
- `reproducibility/scripts/build_optical_source_populations.py`
- `reproducibility/scripts/build_experimental_feature_store.py`
- `reproducibility/scripts/build_candidate_master.py`

These reconstruct candidate/QA data on the canonical five field-station MODIS
footprints. They do not redefine the Virtual10 final training population.

## Historical evidence

All other scripts below `reproducibility/scripts/` are retained as historical
experiments or audits. They are not invoked by RF-25 production unless listed
explicitly above. The single narrative interpretation of those experiments is
`docs/EXPERIMENT_HISTORY.md`.
