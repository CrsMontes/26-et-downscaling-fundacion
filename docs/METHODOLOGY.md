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

No hyperparameter tuning or feature search occurs in the final training command. Candidate predictors are downloaded to a separate archive but are not silently introduced into RF-25.

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

## Data and output policy

The only tracked portable scientific inputs are:

- `data/boundaries/fundacion_basin.geojson`
- `data/stations/fundacion_stations.geojson`
- `data/field/field_etgage.csv`

All generated data are written below `outputs/` inside the repository. Generated outputs are ignored by Git and must not be uploaded to GitHub. Google Drive is not used by the workflow.
