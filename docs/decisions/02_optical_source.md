# Optical source

## Problem
Select an optical source for ET downscaling while preserving an alternative operational source.

## Alternatives
- Sentinel-2 SR Harmonized.
- HLS S30.
- HLS L30.
- Combined HLS S30 + L30.

## Evidence
Controlled diagnostics were performed over 138 MODIS periods and five station footprints.

At 90% optical coverage:
- Sentinel-2: 477 optical observations and 349 observations jointly supported by Sentinel-1.
- HLS combined: 361 optical observations and 256 observations jointly supported by Sentinel-1.

A paired leave-one-station-out comparison used exactly the same 243 station-period observations.

Sentinel-2:
- R2 = 0.2231
- RMSE = 0.3022
- MAE = 0.2229
- KGE = 0.2794

HLS combined:
- R2 = 0.1836
- RMSE = 0.3098
- MAE = 0.2377
- KGE = 0.3437

Neither source consistently dominated all predictive metrics.

## Decision
Keep both optical sources operational.

Default:
- S2
- working grid: 20 m

Alternative:
- HLS_COMBINED (S30 + L30)
- working grid: 30 m

The optical source must be selected explicitly by configuration or command-line argument.

Do not resample both sources to an artificial common production resolution.

Comparative experiments must use their common predictor set and matched observations.

## Important implementation requirement
HLS spatial filtering must use verified local MGRS tiles. filterBounds() alone was found insufficient during the diagnostic experiments and must not define the final HLS selection by itself.

## Status
Accepted.
