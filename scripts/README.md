# Operational scripts

This directory contains only the current user-facing workflow.

## Production

Normal entry point:

    python scripts/run_pipeline.py --project <earth-engine-project>

The production pipeline orchestrates:

- `export_meteorology_data.py --ridge25-only` (ERA5-Land + station support)
- `export_satellite_data.py --optical-source S2 --ridge25-only` (MODIS + S2; no S1 query)
- `build_training_dataset.py --optical-source S2 --ridge25-only` (no CHIRPS requirement)
- Ridge-25 fitting and blocked validation
- Ridge-25 AOA reconstruction
- exact-overlap 20 m ET production when a raster date is requested

The accepted operational model does not require HLS, Sentinel-1, CHIRPS, FVC
or albedo. Those sources/variables remain only in reproducibility diagnostics.

## Field evaluation

The current complete field-comparison entry point is:

    python scripts/run_field_evaluation.py --project <earth-engine-project>

It orchestrates the in-basin spatial-OOF exact-overlap evaluation, external
ST04 handling, the AOA-only sensitivity, and the final scenario tables.

Field observations are used only for the separate comparison phase and do not
constitute independent validation of the full 20 m raster domain.

## Historical and experimental scripts

Scripts used to reach methodological decisions are not mixed with operational
commands. They are retained under:

    reproducibility/scripts/

See `reproducibility/script_manifest.md` and `docs/decisions/README.md`.
