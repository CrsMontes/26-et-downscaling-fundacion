# Decision 08 — Local meteorological processing

## Problem

The diagnostic reproduction showed that repeatedly aggregating ERA5-Land hourly data and CHIRPS inside the Earth Engine graph for every MODIS footprint-period is computationally expensive and can trigger Earth Engine memory/quota failures. Those operations do not require server-side raster processing once the station-specific atmospheric support has been sampled.

## Alternatives

1. Keep all temporal meteorological aggregation and reference ET calculations in Earth Engine.
2. Export raw station-support ERA5-Land hourly values, CHIRPS daily precipitation, and static support information once, then calculate temporal aggregates, ETo/ETr, and Kc locally.

## Evidence and test

During the diagnostic reproduction, the local daily reference-ET checkpoint contained all 5,475 expected station-days for 2021–2023, all days had 24 ERA5-Land hours, all 690 MODIS footprint-periods produced complete ETo/ETr values, and Kc reconstruction returned MODIS ET with numerical error near machine precision.

## Decision

Use Earth Engine only for operations that require the source rasters: MODIS/optical/SAR extraction, resolving the nearest valid ERA5-Land land cell, sampling ERA5-Land/CHIRPS, and summarizing NASADEM elevation over the MODIS footprint. Perform all temporal aggregation, meteorological derivation, ASCE-EWRI ETo/ETr calculation, Kc construction, QA flags, and table joins locally.

This is a computational refactor, not a change to the diagnosed scientific formulation. ERA5-Land remains coarse atmospheric context and NASADEM is used only for representative footprint elevation; neither becomes a fine-resolution meteorological predictor.

## Temporal support detail

The historical meteorological predictors (`Tair_mean_C`, `Tair_max_C`, `VPD_mean_kPa`, `VPD_max_kPa`, `SolarRad_MJ_m2_day`, and `Wind_mean_ms`) are reproduced on the original MODIS UTC period window. Daily ETo/ETr are calculated on Colombia local calendar days (UTC-5), matching the reference-ET implementation. The raw ERA5-Land export therefore includes five extra UTC hours after `END_DATE` only to complete the final local day; those extra hours do not enter the MODIS-period meteorological predictors.

## ERA5-Land hourly solar-radiation QA

A complete audit of the raw ERA5-Land export identified 28 negative values in
`surface_solar_radiation_downwards_hourly` among 131,425 station-hour records.
All occurred at 00:00 UTC and ranged from -2.25 to -0.0625 J m-2.

The 28 observations were re-queried directly from
`ECMWF/ERA5_LAND/HOURLY` using the exact image timestamp, ERA5 sampling
location, native projection, and production sampling procedure. All 28
negative values were reproduced exactly. Therefore, they were not introduced
by CSV export, numeric casting, timestamp handling, or local processing.

The raw ERA5-Land files are preserved unchanged. During local processing,
negative downward hourly solar-radiation values are constrained to zero before
conversion to MJ m-2 and temporal aggregation.

A sensitivity test showed a maximum change of:

- 2.25e-6 MJ m-2 day-1 in daily solar radiation;
- 3.78e-7 mm day-1 in ETo;
- 3.75e-7 mm day-1 in ETr.

The correction therefore enforces physical consistency while having negligible
numerical influence on the reference-ET estimates.

## Implementation cleanup

The superseded server-side meteorology and reference-ET implementations were
removed from the active source tree after the local workflow was verified. The
authoritative modules are `meteorology_export.py` for source sampling and
`reference_et_local.py` / `local_training.py` for local derivation. Historical
implementations remain recoverable through Git.

## Status

Accepted and implemented.
