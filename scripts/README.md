# Scripts

The repository contains both the operational ET downscaling workflow and
scripts retained to reproduce methodological experiments.

## Operational workflow

The normal entry point is:

    python scripts/run_pipeline.py

The operational pipeline orchestrates:

- `export_meteorology_data.py`
- `export_satellite_data.py`
- `build_training_dataset.py`
- final Ridge-25 fitting, AOA construction and local 20 m production.

These scripts define the current production workflow.

## Field evaluation

The following scripts are retained for the independent field-evaluation phase:

- `validate_field_downscaling.py`
- `analyze_field_diagnostics.py`

They are not part of routine raster production.

## Reproducibility scripts

All remaining scripts reproduce diagnostics, screening experiments or
superseded methodological alternatives used to reach the final method.

Examples include:

- S2 versus HLS experiments;
- Ridge versus Random Forest comparisons;
- predictor-family screening and ablations;
- FVC and albedo experiments;
- Sentinel-1 and LST availability experiments;
- coverage-threshold sensitivity;
- AOA and fine-scale information diagnostics;
- historical and alternative reconciliation tests;
- production smoke tests.

These scripts are intentionally retained as scientific provenance.

They are NOT executed by `run_pipeline.py` unless explicitly called by a user.

See:

    reproducibility/script_manifest.md
    docs/decisions/

for the complete classification and the scientific decisions supported by
these experiments.

## Reproducibility policy

A rejected predictor, model or workflow is not deleted solely because it was
not selected. Negative results and superseded alternatives are preserved when
they provide evidence for a methodological decision.

The Git history provides additional provenance but is not used as a substitute
for executable scientific evidence.
