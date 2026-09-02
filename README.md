# ET downscaling - Fundación River basin

Reproducible evapotranspiration downscaling workflow for the Fundación River
basin, Colombia.

## Current method

The primary model estimates MODIS-scale crop coefficient
`Kc = ET_MODIS / ETo` using a standardized Ridge regression (`alpha=1`) with
25 predictors:

- 16 Sentinel-2 optical variables;
- 5 ERA5-Land atmospheric/context variables;
- 4 seasonal harmonics.

Training uses station x native MODIS footprint x MODIS period observations.
The accepted Sentinel-2 coverage gate is >=90%. Primary validation is
leave-one-spatial-block-out, complemented by leave-one-year-out validation.

For the canonical 2020-2024 gate, the reference population is 799 observations
with spatial OOF R2 approximately 0.380 and RMSE approximately 0.253 Kc units.

Fine prediction is produced on a common 20 m UTM grid. Three proportional
reconciliation passes conserve native MODIS ET support. No independent
validation at 20 m is claimed.

## Run

Create the environment and install the repository in editable mode, then:

```powershell
python scripts/run_pipeline.py --project <earth-engine-project>
```

The pipeline:

1. reuses or downloads complete raw source data;
2. rebuilds the local master;
3. rebuilds and validates Ridge-25 from scratch;
4. saves current-run statistics and diagnostic figures;
5. optionally asks for a MODIS period to generate a 20 m ET raster.

Use `--refresh-raw` only for an intentional complete re-extraction.
Use `--no-raster` for training/validation only.
Use `--raster-date YYYY-MM-DD` for non-interactive raster production.

All generated files are written outside the Git repository under the external
`ET_fundacion_workspace`. Google Drive is not used for outputs.

## Repository inputs

Only three portable scientific inputs are kept locally in the repository:

- Fundación basin boundary;
- station geometries;
- field ETgage table.

## Reproducibility and limitations

A fitted model is never required as an input. Spatial and temporal OOF
statistics are regenerated on every run. Raw caches may be reused to avoid
unnecessary Earth Engine extraction.

The complete master retains variables that were evaluated and later excluded
from the primary model, including Sentinel-1 and CHIRPS-derived information.
Methodological decisions and their evolution are documented under
`docs/decisions/` and `docs/METHODOLOGY_EVOLUTION.md`.

## Authors

Cristian Montes
Manuel Coy Pertuz
