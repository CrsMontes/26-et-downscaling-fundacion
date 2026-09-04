# ET downscaling - Fundación River basin

Reproducible evapotranspiration downscaling workflow for the Fundación River
basin, Colombia.

## Current method

The primary model estimates the MODIS-scale crop coefficient

`Kc = ET_MODIS / ETo`

using standardized Ridge regression (`alpha = 1`) with 25 predictors:

- 16 Sentinel-2 optical variables;
- 5 ERA5-Land atmospheric/context variables;
- 4 seasonal harmonics.

Training uses station × native MODIS footprint × MODIS period observations.
The final training population contains 799 observations.

Model performance:

| Validation | n | R2 | RMSE | MAE | Bias | KGE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial block OOF | 799 | 0.383175 | 0.252810 | 0.188848 | -0.016077 | 0.489509 |
| Leave-one-year-out | 799 | 0.525778 | 0.221669 | 0.163870 | -0.005585 | 0.632719 |

Fine-resolution ET is generated on a common 20 m grid.

Publication requires:

- complete Ridge-25 predictor stack;
- inside the Ridge-25 area of applicability (AOA);
- `Kc_raw >= 0`;
- usable support fraction >= 0.90 within the native MODIS parent.

For eligible MODIS parents, the final product uses one global exact-overlap
reconciliation after the raw 20 m support mosaic has been assembled. Real
intersection areas between the 20 m UTM grid and the native MODIS sinusoidal
grid define the coarse-support constraints. No coarse-to-fine nearest-neighbour
correction and no arbitrary reconciliation iterations are used.

Small negative ET values produced by the unconstrained global projection are
floored once to zero. The date is accepted only if the exact-overlap
conservation error remains <= 0.01 mm per MODIS period. Production uses fixed
4000 m raw-support tiles plus an external halo, followed by one global
reconciliation and final basin clipping.

The accepted production version is
`ridge25_exact_overlap_support90_tol001_v2`.

Field comparison is a separate evaluation, not a validation of the complete
20 m raster domain. In the current fixed-Kc main subset (ST01-ST03, n=10),
MODIS and the downscaled product show similar overall error: RMSE 9.034 versus
8.746 mm per period, respectively. The downscaled product does not show a
consistent accuracy improvement across metrics or stations. No independent
20 m validation claim is made.
## Run

Create the environment and install the repository in editable mode, then:

    python scripts/run_pipeline.py --project <earth-engine-project>

The default pipeline:

1. reuses or downloads the required raw source data;
2. rebuilds the local master dataset;
3. fits and validates Ridge-25 from scratch;
4. rebuilds the Ridge-25 AOA from the current training population;
5. saves current-run statistics and diagnostics;
6. optionally generates the final 20 m ET product for a requested MODIS period.

Use --refresh-raw only for an intentional complete re-extraction.
Use --no-raster for training and validation without raster production.
Use --raster-date YYYY-MM-DD for non-interactive raster production.

All generated files are written outside the Git repository under the external
ET_fundacion_workspace. Google Drive is not used for outputs and no persistent
Earth Engine asset is required.

## Repository inputs

Only three portable scientific inputs are kept locally in the repository:

- Fundación basin boundary;
- station geometries;
- field ETgage table.

## Reproducibility

The operational workflow has a single normal entry point:

    python scripts/run_pipeline.py

Scripts used to evaluate alternative predictors, models and methodological choices are retained for scientific reproducibility but are not executed by the default pipeline.

Rejected or superseded experiments are retained when they provide evidence for a methodological decision.

See reproducibility/README.md, reproducibility/script_manifest.md, docs/decisions/ and docs/METHODOLOGY_EVOLUTION.md.

## Field evaluation

Independent comparison with field observations is treated as a separate scientific phase. Field observations do not provide independent validation of the complete 20 m raster domain, so no 20 m validation claim is made.

## Authors

Cristian Montes
Manuel Coy Pertuz




