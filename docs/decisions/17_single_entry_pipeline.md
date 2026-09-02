# Decision 17 - Single user-facing parsimonious pipeline

## Objective

The final repository must be runnable from the three canonical local inputs
without requiring fitted-model files or historical `outputs/` directories.

## Entry point

The user-facing command is:

`python scripts/run_pipeline.py --project <earth-engine-project>`

The canonical five-year period is 2020-01-01 to 2025-01-01 (exclusive).

## Execution order

1. Verify the three canonical inputs.
2. Reuse or intentionally refresh complete raw satellite and meteorological
   extractions in the external workspace.
3. Rebuild the complete local master.
4. Derive the GE90 Ridge-25 population locally.
5. Refit and validate Ridge-25 from scratch:
   - leave-one-spatial-block-out;
   - leave-one-year-out;
   - mean baselines.
6. Save run-specific tables, model parameters, statistics and diagnostic
   figures.
7. Ask whether a raster should be generated unless the choice was provided as
   a command-line option.
8. If requested, use the model fitted in that same process for local adaptive
   tiled 20 m production and three-pass MODIS reconciliation.

## Reuse policy

Raw extraction files may be reused because they are expensive, deterministic
source data caches. The master-derived Ridge population, OOF predictions and
fitted Ridge model are rebuilt every run.

A `.joblib` or other serialized fitted model is not a pipeline input.

## Spatial production

Google Drive is not used. Persistent Earth Engine assets are not used.
Fine products are downloaded directly in small grid-aligned chunks and mosaiced
locally in the external workspace.

## Interpretation

The spatial OOF metrics evaluate transfer among the available station/block
supports at MODIS training scale. The resulting 20 m raster is a constrained
downscaled estimate and does not constitute independent validation at 20 m.
