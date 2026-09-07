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
Eligibility is rebuilt from the current source query using the accepted QC
rules, including Sentinel-2 coverage >= 90%. The resulting training-row count,
spatial-block OOF metrics, LOYO metrics and numerical AOA threshold are run
outputs rather than hard-coded algorithm constants. Final manuscript values are
reported from the designated final scientific run.

The operational path queries only the accepted external dependencies: MODIS,
Sentinel-2 SR Harmonized + Cloud Score+, and ERA5-Land. Sentinel-1, CHIRPS, HLS,
FVC and albedo are not operational Ridge-25 dependencies.

Fine-resolution ET is generated on a common 20 m grid.

Publication requires:

- complete Ridge-25 predictor stack;
- inside the equal-weight multivariate applicability domain, implemented as a
  standardized Euclidean dissimilarity index adapted from the
  Meyer-Pebesma AOA framework and using the spatial validation blocks;
- `Kc_raw >= 0`;
- usable support fraction >= 0.90 within the native MODIS parent.

The applicability domain deliberately gives equal weight to all 25 standardized
predictors. A final sensitivity using absolute standardized Ridge coefficients
as weights changed 17.68% of valid basin pixels on 2020-03-13 and was strongly
dominated by the correlated optical predictor block, without evidence of
better separation of spatial OOF errors. It was therefore not adopted.

For eligible MODIS parents, the final product uses one global exact-overlap
reconciliation after the raw 20 m support mosaic has been assembled. Real
intersection areas between the 20 m UTM grid and the native MODIS sinusoidal
grid define the coarse-support constraints. No coarse-to-fine nearest-neighbour
correction and no arbitrary reconciliation iterations are used.

Small negative ET values produced by the unconstrained global projection are
floored once to zero. The date is accepted only if the exact-overlap
conservation error remains <= 0.01 mm per MODIS period. Production uses fixed
4000 m raw-support tiles plus an external halo, followed by one global
reconciliation and final basin clipping. Exact MODIS conservation applies to
the complete reconciled fine support before the final publication mask. The
masked published subset is not expected to reaggregate exactly to MODIS.

The accepted production version is
`ridge25_cs050_ge90_exact_overlap_support90_tol001_v3`.

The designated audited run `20260906T192839Z_2020_2024` contains 833 training
rows. Spatial-block OOF performance was R2=0.393749, RMSE=0.249689 and
MAE=0.185496 Kc units; LOYO R2 was 0.527433. These values are run outputs, not
hard-coded algorithm constants.

Field comparison is a separate evaluation, not a validation of the complete
20 m raster domain. With the accepted AOA, the all-cover field-proxy scenario
contains n=17 observations (Ridge RMSE 11.767 versus MODIS 12.397 mm per
period). The pre-specified fixed-Kc ST01-ST03 subset contains n=9 (Ridge RMSE
8.985 versus MODIS 9.420 mm per period). Results are mixed across metrics and
do not demonstrate a consistent accuracy improvement over MODIS. No
independent 20 m validation claim is made.
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
Use repeated --raster-date YYYY-MM-DD arguments for multi-period production.

The frozen three-period production command is:

    python scripts/run_pipeline.py --project <earth-engine-project> \
        --raster-date 2020-03-13 \
        --raster-date 2021-11-25 \
        --raster-date 2022-03-30

After a complete run passes QA, create the minimal local final view with:

    python scripts/build_final_outputs.py

This creates only three one-band ET GeoTIFFs, `raster_summary.csv`, and a
read-only visualization notebook under `ET_fundacion_workspace/final/`. The
scientific multiband rasters, diagnostics, tables, metadata and SHA-256
provenance remain in `ET_fundacion_workspace/current/` and are not duplicated.
Generated products are not versioned in Git.

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

Each new run records the Git commit and dirty state, SHA-256 hashes of the
three canonical repository inputs, the raw training-source caches, the rebuilt
master, `environment-lock.yml`, and the run-generated tables/AOA/figures in
`run_metadata.json`.

See reproducibility/README.md, reproducibility/script_manifest.md,
docs/decisions/ and docs/METHODOLOGY_EVOLUTION.md.

## Field evaluation

Independent comparison with field observations is treated as a separate scientific phase. Field observations do not provide independent validation of the complete 20 m raster domain, so no 20 m validation claim is made.

## Authors

Cristian C. Montes-Chaura,
Manuel Coy Pertuz




