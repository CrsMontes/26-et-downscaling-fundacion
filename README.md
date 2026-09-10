# ET downscaling - Fundación River basin

Clean operational repository for 20 m evapotranspiration downscaling in the
Fundación River basin, Colombia.

`main` contains the final RF-25 workflow. Historical development states remain
available as Git branches:

- `virtual`: historical Virtual Station closure immediately before RF promotion.
- `field`: historical Field Station workflow.
- `diagnostic`: original diagnostic-reproduction state.

The repository was intentionally disconnected from the old Git remote. Add a
new `origin` only after creating the new GitHub repository.

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
- Final map dates: 2020-03-13, 2021-11-25 and 2022-03-30.

Virtual supports and MODIS-derived targets are not independent 20 m ET
validation. Field comparisons use a derived field ET proxy and are reported as
such.

## Inputs and outputs

Only the three portable scientific inputs are tracked in Git:

```text
data/boundaries/fundacion_basin.geojson
data/stations/fundacion_stations.geojson
data/field/field_etgage.csv
```

All downloaded data, intermediate tables, models, rasters, diagnostics and
figures are written under `outputs/`. Everything there is ignored by Git except
`outputs/README.md`.

The final RF always uses the frozen 25 variables. A separate candidate archive
can download/materialize every predictor family already implemented in this
repository (S2/HLS, S1 R077/R142, ERA5-Land, CHIRPS, Landsat LST, albedo/FVC,
seasonality and associated QA/provenance). Candidate availability never changes
the final RF feature set or silently removes Virtual10 training rows.

## Fresh run on Windows

From the new repository:

```powershell
conda activate et-fundacion
python -m pip install -e .
python scripts\run_pipeline.py preflight
python scripts\run_pipeline.py fresh --project ee-sneiderquintero --yes
```

`fresh` deletes generated `outputs/` contents, reconstructs the Virtual10
selection and 25-feature training population from source data, trains and
validates RF-25, derives the weighted AOA/DI and LPD, produces the three final
rasters, and then materializes the complete implemented candidate-predictor
archive. Candidate downloading is deliberately last so it cannot delay the
scientific core if an unused source is slow.

To resume without deleting completed outputs:

```powershell
python scripts\run_pipeline.py run --project ee-sneiderquintero
```

Useful individual stages:

```powershell
python scripts\run_pipeline.py select --project ee-sneiderquintero
python scripts\run_pipeline.py extract --project ee-sneiderquintero
python scripts\run_pipeline.py train
python scripts\run_pipeline.py produce --project ee-sneiderquintero
python scripts\run_pipeline.py download-candidates --project ee-sneiderquintero
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
