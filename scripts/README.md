# Operational scripts

This directory contains only the current user-facing workflow.

## Production

Normal entry point:

    python scripts/run_pipeline.py --project <earth-engine-project>

The production pipeline orchestrates:

- `export_meteorology_data.py`
- `export_satellite_data.py`
- `build_training_dataset.py`
- Ridge-25 fitting and blocked validation
- Ridge-25 AOA reconstruction
- exact-overlap 20 m ET production when a raster date is requested

## Field evaluation

The current spatial-OOF field comparison is:

    python scripts/evaluate_field_ridge25.py --project <earth-engine-project>

Field observations are used only for the separate comparison phase and do not
constitute independent validation of the full 20 m raster domain.

## Historical and experimental scripts

Scripts used to reach methodological decisions are not mixed with operational
commands. They are retained under:

    reproducibility/scripts/

See `reproducibility/script_manifest.md` and `docs/decisions/README.md`.
