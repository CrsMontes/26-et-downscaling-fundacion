# Script classification

## Current production
- `scripts/run_pipeline.py`
- `scripts/export_meteorology_data.py`
- `scripts/export_satellite_data.py`
- `scripts/build_training_dataset.py`

## Current Virtual Station field-proxy evaluation
- `scripts/evaluate_v5_field_proxy.py`
- `scripts/evaluate_field_ridge25.py`
- `scripts/compare_field_with_without_aoa.py`

The removed Field-only orchestration and exploratory station scripts remain
preserved in the `field-station-stable` branch.

## Full-study reconstruction dependencies

These scripts are used only when rebuilding the historical five-year candidate
universe and selection evidence. They are not dependencies of the accepted
Ridge-25 operational pipeline.
- `reproducibility/scripts/build_candidate_master.py`
- `reproducibility/scripts/build_experimental_feature_store.py`
- `reproducibility/scripts/build_meteorology_experiment_table.py`
- `reproducibility/scripts/build_optical_source_populations.py`
- `reproducibility/scripts/export_availability_diagnostic.py`
- `reproducibility/scripts/export_hls_albedo_fvc.py`
- `reproducibility/scripts/export_landsat_lst_predictor.py`
- `reproducibility/scripts/export_optical_source_experiment.py`
- `reproducibility/scripts/export_s1_geometry_predictors.py`
- `reproducibility/scripts/export_s2_rich_optical.py`
- `reproducibility/scripts/export_thermal_availability.py`
- `reproducibility/scripts/run_predictor_availability_ladder.py`

## Reproducibility / historical evidence

- `reproducibility/scripts/analyze_recalibrated_fvc_predictors.py`
- `reproducibility/scripts/audit_modis_grid_alignment_20220407.py`
- `reproducibility/scripts/audit_reconciliation_exact_overlap_20220407.py`
- `reproducibility/scripts/audit_ridge_fine_information.py`
- `reproducibility/scripts/evaluate_coverage_threshold_sensitivity.py`
- `reproducibility/scripts/evaluate_optical_source_experiment.py`
- `reproducibility/scripts/preflight_fvc_recalibration.py`
- `reproducibility/scripts/recheck_s2_fvc_albedo.py`
- `reproducibility/scripts/screen_feature_families.py`
- `reproducibility/scripts/screen_optical_algorithms.py`
- `reproducibility/scripts/test_overlap_reconciliation_20220407.py`
- `reproducibility/scripts/run_closure_diagnostics.py`
- `reproducibility/scripts/run_aoa_map_sensitivity.py`

Routine map production must not execute unused candidate experiments.
A clean full-study reconstruction may execute the reconstruction
dependencies listed above.

## Final audit closure diagnostics

`run_closure_diagnostics.py` is an offline audit only. It reproduces the
equal-weight versus absolute-standardized-Ridge-coefficient DI sensitivity,
three explicit persistence definitions, and the 5/8 versus 8/8 field-valid-day
sensitivity. `run_aoa_map_sensitivity.py` is a potentially expensive direct
Earth Engine map diagnostic used to quantify spatial disagreement between the
two DI variants. Neither script is called by production.
