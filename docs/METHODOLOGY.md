# Current scientific methodology — RF-25 main

## Scientific objective

Estimate the spatial pattern of evapotranspiration (ET) within the Fundación River basin at a 20 m prediction grid while preserving the MOD16A2GF v6.1 coarse-scale ET signal. The 20 m product is a spatial downscaling of MODIS ET; it is not an independent 20 m ET observation.

## Training support and target

Training uses 10 virtual MODIS footprints distributed across the basin. Candidate footprints are evaluated sequentially from a fixed random seed (`42`) and accepted only when they satisfy the frozen Sentinel-2 GE90 availability design and occupy distinct fixed 10 km UTM blocks (EPSG:32618). The real field stations are not used as training supports.

The response is

`Kc_target = MODIS_ET_mm_period / ETo_mm_period`

where `ETo` is the short-reference evapotranspiration reconstructed from ERA5-Land meteorology. The model therefore learns a MODIS-derived relative ET signal. This distinction is retained throughout validation and interpretation.

## Final model

The main estimator is `RandomForestRegressor` with a frozen 25-predictor feature set and fixed hyperparameters:

- `n_estimators = 300`
- `max_features = 0.33`
- `min_samples_leaf = 3`
- `max_depth = None`
- `bootstrap = True`
- `random_state = 42`

No hyperparameter tuning or feature search occurs in the final training command. The canonical extraction materializes Sentinel-2, MODIS target support and ERA5-Land on the same 10 Virtual10 supports. RF-25 then applies only the frozen 25-column and GE90 population contract. Historical candidate families can be materialized separately and are never RF-25 eligibility gates.

The 25 model predictors are 16 Sentinel-2 optical variables, five ERA5-Land variables and four seasonal harmonics. See `docs/PREDICTOR_CATALOG.md`.

## Validation design

Two complementary out-of-fold evaluations are retained:

1. Spatial OOF: each of the 10 fixed spatial blocks is held out once. With one Virtual10 support per block, this is effectively leave-one-virtual-support-out transfer.
2. LOYO: one calendar year is held out at a time. This evaluates an unseen year at spatial supports represented in other years; it is not joint unseen-location/unseen-year validation.

R², RMSE, MAE, bias and KGE are reported together. Fold-level results are retained because pooled metrics can hide spatial heterogeneity.

## DI, AOA and LPD

The final applicability assessment follows the predictor-space framework of Meyer & Pebesma (2021; DOI `10.1111/2041-210X.13650`) and the current CAST `trainDI` threshold logic.

1. Each of the 25 predictors is centered and divided by its sample standard deviation in the training population.
2. Standardized predictors are multiplied by RF-derived importance weights. The Python implementation uses permutation importance with negative values clipped to zero; raw weights are saved for audit because correlated predictors can share importance.
3. Weighted Euclidean distance is calculated in predictor space.
4. DI for a prediction point is its distance to the nearest training point divided by the mean of all pairwise weighted distances among training observations.
5. For training DI used to define the threshold, the nearest reference observation must be outside the same spatial-CV block as the held-out observation.
6. The threshold is `min(max(training_DI), Q3 + 1.5 * IQR)`, matching current CAST source behavior.
7. Pixels with DI above the threshold are outside the AOA and are excluded from publication support.
8. Local Point Density (LPD) is also written as a diagnostic band. It counts training observations within the AOA-distance radius in predictor space. LPD is not an additional hard mask in the main workflow.

LPD follows the recent uncertainty-support concept described by Schumacher et al. (2025; DOI `10.5194/gmd-18-10185-2025`).

## Fine-scale prediction and MODIS reconciliation

RF predicts `Kc_raw` on the 20 m grid. Only pixels with a complete predictor stack, inside the AOA, finite non-negative Kc and valid support are publication candidates. A MODIS parent is eligible only when at least 90% of its represented fine support is usable.

Predicted Kc supplies the relative subpixel pattern; MODIS supplies the coarse ET magnitude. The algorithm then performs one global exact-overlap reconciliation using real overlap areas between the transformed native MODIS sinusoidal pixels and the 20 m grid. The reconciled full support must satisfy an absolute MODIS conservation tolerance of 0.01 mm per period after the one-time non-negative ET floor.

Conservation is audited on the full reconciled support before the final basin/publication mask. It must not be described as exact conservation after the final publication mask.

## Field comparison

Field-derived ET is reserved for external comparison and is not used to fit Virtual10. The field series remains a derived proxy based on station reference ET and land-cover Kc assumptions, not an independent 20 m ET measurement. Results are therefore described as field-proxy comparison rather than pixel-scale validation.

Run the separate workflow with:

```bash
python scripts/run_field_validation.py --project ee-sneiderquintero
```

The command reads the versioned ST01–ST05 observations and metadata, ignoring
Virtual10 station overrides. Raw daily ETgage readings are in cm (confirmed by
the project owner) and are multiplied by 10. Historical QC retains positive,
nonmissing readings within the recorded installation window and 0.05–12 mm/day.
ST01, ST03, ST04 and ST05 measure ETr: each daily reading is divided by the modeled station-day
ETr/ETo ratio before Kc is applied. ST02 already measures ETo. Period reference
ET is the mean of valid ETo-equivalent days multiplied by the actual MODIS
period length, requiring at least five valid days; incomplete periods are thus
expanded, not simply summed. Daily reference ET uses the current local
ASCE-EWRI implementation and field-specific ERA5-Land supports. A Virtual10-only
reference cache cannot substitute for real-station meteorology.

| Scenario | ST02 | ST03 | ST04 | ST01/ST05 |
|---|---|---|---|---|
| `fixed_kc_main` | 0.85 | 0.95 | 1.10 | Excluded (principal conservative set) |
| `historical_all_stations` | 0.85 | 0.95 | 1.10 | Local 20 m NDVI proxy |
| `fao_sensitivity` | 0.75 | 1.00 | 1.10 | Local 20 m NDVI proxy |
| `ndvi20_all` | Local 20 m NDVI proxy | Local 20 m NDVI proxy | Local 20 m NDVI proxy | Local 20 m NDVI proxy |

The NDVI proxy is `Kc = 1.457 * NDVI - 0.1725`; values outside 0.10–1.50 are
missing, not clipped. Field period ET proxy equals period ETo-equivalent times
Kc. Historical fixed coefficients and conversion rules come from `480f50b`
(`origin/diagnostic-reproduction`, `857adcc`) and `46b7d30`. Local NDVI follows
the later stable comparison at `726b914`; the earlier diagnostic instead used
MODIS-footprint NDVI. The new workflow samples the current RF25
`build_s2_rf25_predictors()` medoid on its 20 m grid. It does not import Ridge25
production. This restores calculation rules, not a claim to reproduce earlier
frozen metrics using different products or spatial supports.

The name `fao_sensitivity` denotes the requested alternative assumptions, not
verified FAO table citations. History labels the original fixed values
“FAO-56 fixed” but does not establish their exact table/page or crop-stage
provenance. The NDVI relation's calibration and reference basis likewise remain
undocumented. No explicit water-stress correction is made. `ndvi20_all` is not
independent of Sentinel-2-based RF25, and ST01/ST05 NDVI proxies are sensitivity
results in every scenario. Metadata also retain nonconforming installation
flags and ST01's position outside the basin.

Results and resumable field acquisition caches are isolated under
`outputs/evaluation/field_validation/`. Outputs include daily QC, period pairs,
overall metrics, per-station metrics, attrition, and JSON provenance with input
and code hashes. All four scenarios are always written together;
`--scenario historical_all_stations`, `--scenario fao_sensitivity`, or `--scenario ndvi20_all`
selects console reporting without changing saved rows. Metrics report n, R2,
RMSE, MAE, BIAS (prediction minus proxy), and KGE. R2/KGE are undefined when
their required sample size/variance is absent.

`available_sample` retains every valid pair of each scenario/product and is the
principal result. `common_sample` uses identical station-period keys across
scenarios for each product. The `main_vs_sensitivities` family intersects all
four scenarios on ST02, ST03, ST04; `all_station_sensitivities` intersects the three
ST01–ST05 sensitivities. MODIS common samples do not require RF25 availability.
Pair-key SHA-256 values document the intersections. RF25 pairs also require
native parent MODIS availability in the field comparison.

Sequential attrition is reported for every scenario and station, plus `ALL`:
candidate periods -> >=5 valid ETgage days -> proxy -> MODIS -> complete RF25
stack -> AOA -> eligible represented MODIS parents -> successful reconciliation
-> final published pair. Unknown diagnostics are counted separately from known
failures; an absent product does not establish a stack or AOA failure.

RF25 ET is read from the containing pixel of a verified final 20 m UTM raster,
without interpolation. All publication masks, float32 values, and native grid
phase are inherited from canonical production. Station `inside_basin` metadata
are retained as context; the published pixel mask determines eligibility.
The rejected local reconciliation proposal and its proof/cost audit are recorded
in [FIELD_RF25_AUDIT.md](FIELD_RF25_AUDIT.md).

`rf25_field_product_inventory.csv` records missing, incomplete and unattributed
dates. Existing products are reused only with an execution-time record binding
the final raster/metadata hashes to the current model, AOA and production code.
The current 2022-03-30 raster lacks that evidence and is excluded. Hashing today's
model next to an old raster does not retrospectively certify the run.

The default command samples verified products and leaves other dates pending.
`--raster-root <directory>` selects existing date directories read-only. Optional
`--produce-missing --project <project>` invokes `scripts/produce_rf25_rasters.py`
unchanged, with byte-identical model/AOA copies in
`outputs/evaluation/field_validation/production/<identity-prefix>/`.
`--production-date YYYY-MM-DD` limits production to specified pending field
dates (repeatable). Successful runs receive a separate field execution record;
canonical provenance code/metadata are not modified. No training is called.
The wrapper does not change tiles, supports, AOA, solver, or publication science.

For offline field-input reuse, supply `--daily-reference <csv>` and
`--satellite-table <csv>`; the latter contains `station_id`, `period_start`,
`number_days`, `ET_MODIS_mm_period`, and `NDVI_local_20m`. Alternatively reconstruct
reference ET with `--era5-hourly <csv> --station-support <csv>`. Without these
options, caches are checked before field acquisition with `--project`.
Acquisition caches are keyed to field inputs and acquisition code; explicitly
supplied existing CSVs are hashed and their station/date coverage validated.

## Data and output policy

The tracked portable inputs required by RF-25 are:

- `data/boundaries/fundacion_basin.geojson`
- `data/stations/fundacion_stations.geojson`

`data/field/field_etgage.csv` is tracked for the separate field-proxy comparison
but is not required to select, train, audit or produce RF-25.

All generated data are written below `outputs/` inside the repository. Generated outputs are ignored by Git and must not be uploaded to GitHub. Google Drive is not used by the workflow.
