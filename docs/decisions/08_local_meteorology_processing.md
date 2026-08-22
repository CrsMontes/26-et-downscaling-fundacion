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
