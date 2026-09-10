# Scientific status ? Virtual Station V5

This is the authoritative scientific status after the September 2026 audit.

Historical methodological evolution is preserved under `docs/archive/`.
Detailed methodological provenance is preserved under `docs/decisions/`.

## Final design

- Target: `Kc_target = ET_MODIS / ETo`.
- Training: 10 virtual native-MODIS supports in 10 spatial blocks.
- Training population: 1454 support-period observations.
- Real field-station footprints are excluded from Virtual10 training.
- Model: StandardScaler + Ridge regression, alpha = 1.
- Predictors: 25.
- AOA: equal-weight standardized Euclidean dissimilarity.
- Frozen AOA threshold: 0.8007328515330622.
- Prediction grid: regular 20 m UTM cells.
- Final dates: 2020-03-13, 2021-11-25, 2022-03-30.

The 20 m grid is a model prediction support, not an independently observed
20 m ET support.

## Spatial transfer

Pooled spatial OOF:

- n = 1454
- R2 = 0.368752
- RMSE = 0.297795 Kc
- MAE = 0.226653 Kc
- bias = -0.002081 Kc
- KGE = 0.674625

Spatial transfer is heterogeneous:

- 6/10 held-out blocks have R2 < 0;
- median fold R2 = -0.801572;
- range = -6.072780 to 0.598579.

The pooled result shows aggregate predictive signal but does not demonstrate
uniform spatial generalization across the basin.

## Temporal transfer

Pooled LOYO:

- n = 1454
- R2 = 0.614995
- RMSE = 0.232568 Kc
- MAE = 0.163847 Kc
- bias = 0.002339 Kc
- KGE = 0.712861

Year-specific R2 ranges from 0.543150 to 0.690391.

LOYO evaluates a new year at spatial supports represented in other training
years. It is not simultaneous spatial-temporal extrapolation.

## Persistence

Previous-available persistence:

- n = 1444
- R2 = 0.658700
- RMSE = 0.219054 Kc
- MAE = 0.144429 Kc
- KGE = 0.829524

For 97.23% of these predictions, the previous observed Kc comes from the same
calendar year that LOYO excludes completely.

Persistence is therefore a sequential temporal baseline and not an equivalent
unseen-year test against LOYO.

## Field-proxy comparison

The comparison contains 21 station-period observations from five 2022 stations.

It is a field-derived ET proxy comparison, not independent ET validation at
20 m.

- ST01 uses ETo and fixed Kc.
- ST02-ST05 use ETr, converted to an ETo-equivalent basis before applying Kc.
- ST01-ST03 use fixed Kc.
- ST04-ST05 use NDVI-derived Kc as a sensitivity formulation.
- Only ST01 conforms to the instrument manual installation.
- ST04 is outside the basin but retained for external comparison.

The ETr-to-ETo-equivalent calculation and final ET proxy were independently
reconstructed to numerical precision.

Virtual10 own-domain, WITH AOA, >=5/8 valid days:

All stations:
- n = 20
- R2 = -0.579489
- RMSE = 13.095505 mm/period
- MAE = 11.209096 mm/period
- bias = 4.682093 mm/period
- KGE = 0.094110

Fixed-Kc ST01-ST03:
- n = 11
- R2 = -0.765152
- RMSE = 12.164081 mm/period
- MAE = 9.869177 mm/period
- bias = -1.998100 mm/period
- KGE = -0.036376

The external field-proxy evidence does not demonstrate a consistent accuracy
improvement over parent MODIS ET.

## Raster production

Fine-scale Kc provides the relative spatial pattern. MODIS ET supplies the
coarse-period magnitude.

Production uses proportional scaling followed by exact-overlap constrained
reconciliation.

MODIS conservation applies to the complete reconciled support before the final
publication mask. Exact conservation is not claimed for the final masked
raster.

The three final raster products passed the production audit.

A boundary implementation issue can classify truncated external MODIS parents
as represented. Counterfactual removal showed no detectable impact inside the
published basin:

- 2020: max change ~7.1e-15 mm
- 2021: max change ~7.1e-15 mm
- 2022: max change 0.0 mm

This is classified as future code cleanup, not an error in the frozen rasters.

## Field-reporting correction

The original `virtual_native_*_with_AOA` metrics inadvertently required a
Stable5 prediction.

This reporting error was corrected without modifying training data, model,
AOA, field pairs or rasters.

Correct own-domain sample sizes:

- primary all: 20
- primary fixed-Kc: 11
- strict all: 14
- strict fixed-Kc: 7

Matched Stable5-versus-Virtual10 comparisons remain restricted to common cases.

## Scientific closure

Closure commit:

`62eb897e5be8cf30bda59f77aa452284a5cd043c`

Scientific tag:

`virtual-station-v5-stable-v1`

Closure manifest SHA256:

`70d7e55305149699aea309f9b56b372d58faf04e5bed2d4323af47bef892648b`

## Manuscript guardrails

Supported claims:

- reproducible MODIS-constrained ET downscaling;
- moderate pooled spatial skill with strong between-support heterogeneity;
- more stable temporal transfer at represented spatial supports;
- strong sequential persistence;
- weak external field-proxy performance.

Not supported:

- independent validation at 20 m;
- uniform spatial generalization;
- causal interpretation of predictor effects;
- consistent superiority over MODIS for all land covers.
