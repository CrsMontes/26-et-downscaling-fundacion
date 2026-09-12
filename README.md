# ET downscaling - Fundación River basin

Clean operational repository for 20 m evapotranspiration downscaling in the
Fundación River basin, Colombia.

`main` contains the final RF-25 workflow. Historical development states remain
available as Git branches:

- `virtual`: historical Virtual Station closure immediately before RF promotion.
- `field`: historical Field Station workflow.
- `diagnostic`: original diagnostic-reproduction state.

Repository remote: `origin` is configured for GitHub synchronization.

## Final scientific configuration

- Target: `Kc_target = MODIS_ET / ETo`.
- Training design: 10 whole-basin Virtual10 MODIS supports in 10 fixed 10 km
  UTM blocks, selected with the frozen sequential-random GE90 rule (seed 42).
- Final estimator: `RandomForestRegressor`, 300 trees, `max_features=0.33`,
  `min_samples_leaf=3`, bootstrap enabled, random state 42.
- Final model predictors: 25 frozen variables: 16 Sentinel-2 optical
  predictors, 5 ERA5-Land meteorological predictors and 4 DOY harmonics.
- No feature selection or hyperparameter tuning is performed during the final
  run.
- AOA/DI: standardized predictor space weighted by RF permutation importance;
  Euclidean DI normalized by the mean pairwise training distance; training DI
  respects the frozen spatial-CV blocks; threshold is
  `min(max(training_DI), Q3 + 1.5*IQR)`.
- AOA is a hard publication-support mask. Local Point Density (LPD) is retained
  as a diagnostic layer, not a second mask.
- Coarse MODIS ET is preserved by exact-overlap reconciliation on the full
  reconciled support before the final publication mask.
- Illustrative cartographic/QC dates: 2020-03-13, 2024-07-11 and 2022-03-30.
  These examples are not a sampling design for climatological inference.

Virtual supports and MODIS-derived targets are not independent 20 m ET
validation. Field comparisons use a derived field ET proxy and are reported as
such.

## Inputs and outputs

The two portable inputs required by RF-25 are tracked in Git:

```text
data/boundaries/fundacion_basin.geojson
data/stations/fundacion_stations.geojson
```

The field series is also tracked but is used only by the separate field-proxy
comparison and is not required by the canonical RF-25 pipeline:

```text
data/field/field_etgage.csv
```

All downloaded data, intermediate tables, models, rasters, diagnostics and
figures are written under `outputs/`. Everything there is ignored by Git except
`outputs/README.md`.

The final RF always uses the frozen 25 variables. Before model fitting, `fresh`
materializes only the required Sentinel-2, MODIS-target and ERA5-Land sources on
the same 10 Virtual10 supports. RF-25 then applies its frozen 25-column and GE90
support-period contract. HLS, Sentinel-1, FVC, albedo, LST, CHIRPS and other
historical candidates remain available through the separate
`download-candidates` command and are not RF-25 dependencies.

## Fresh run on Windows

From the new repository:

```powershell
conda activate et-fundacion
python -m pip install -e .
python scripts\run_pipeline.py preflight
python scripts\run_pipeline.py fresh --project <earth-engine-project> --yes
```

`fresh` deletes generated `outputs/` contents, reconstructs the Virtual10
selection, materializes the sources required by RF-25, derives the frozen GE90
population, trains and validates RF-25, derives weighted AOA/DI and LPD, produces
the three illustrative rasters and writes a provenance manifest.

To resume without deleting completed outputs:

```powershell
python scripts\run_pipeline.py run --project <earth-engine-project>
```

Useful individual stages:

```powershell
python scripts\run_pipeline.py select --project <earth-engine-project>
python scripts\run_pipeline.py extract --project <earth-engine-project>
python scripts\run_pipeline.py train
python scripts\run_pipeline.py produce --project <earth-engine-project>
python scripts\run_pipeline.py provenance
python scripts\run_pipeline.py download-candidates --project <earth-engine-project>
```

No Google Drive export and no persistent Earth Engine asset are part of the
workflow.

## Documentation

- `docs/METHODOLOGY.md`: current method and scientific guardrails.
- `docs/PREDICTOR_CATALOG.md`: final 25 predictors and implemented candidate
  universe.
- `docs/EXPERIMENT_HISTORY.md`: single narrative record of the experiments,
  negative results and methodological decisions that led to `main`.

## Validation

```powershell
python -m pytest -q
git diff --check
```

Authors: Cristian C. Montes-Chaura; Manuel Coy Pertuz.
