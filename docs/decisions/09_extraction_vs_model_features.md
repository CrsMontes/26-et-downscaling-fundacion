# Decision 09 — Separate extraction features from model features

## Problem

The controlled Sentinel-2 versus HLS comparison intentionally used only
features with equivalent definitions in both optical sources. Reusing that
restricted comparison set as the production Earth Engine extraction schema
would discard source-specific variables before they can be evaluated locally.

The historical diagnostic workflow also separated a rich extracted table from
the later scale-transferable model subset.

## Alternatives

1. Export only the common S2/HLS variables.
2. Export every historical diagnostic variable, including thermal and
   within-footprint heterogeneity statistics.
3. Export a richer source-specific optical/satellite candidate set, but keep
   model selection local and explicit.

## Evidence

The diagnostic methodology retained only scale-transferable means for the final
downscaling model and excluded Sentinel-1 incidence angle from the predictive
set. It also found that the thermal block imposed a substantial sample cost
without useful predictive gain.

The source-selection diagnostic showed that S2 and combined HLS should remain
available operationally, while the matched comparison must use only equivalent
features.

Sentinel-2 provides red-edge bands that do not have a direct equivalent in
combined HLS S30 + L30. FVC and albedo can be calculated for both sources, but
their formulations/calibrations are source-specific.

## Decision

Earth Engine extraction and local model feature selection are separate stages.

### Common matched S2/HLS model features

- Blue
- Green
- Red
- NIR
- SWIR1
- SWIR2
- NDVI
- EVI
- SAVI
- NDWI
- NDMI
- Sentinel-1 VV
- Sentinel-1 VH
- Sentinel-1 VV minus VH

This set is used for controlled S2/HLS comparisons.

### Additional Sentinel-2 extraction candidates

- RedEdge1
- RedEdge2
- RedEdge3
- NIR_Broad
- NDRE
- Albedo
- FVC

### Additional combined-HLS extraction candidates

- Albedo
- FVC

These source-specific variables are candidates only. Their presence in the raw
table does not imply inclusion in the final model.

### QA-only variable

- Sentinel-1 incidence angle remains exported as `Angle_deg_mean` but is not a
  predictive feature.

### Excluded from the primary production extraction

- Landsat LST
- TVDI
- within-footprint standard deviation
- within-footprint percentiles

They remain diagnostic/experimental components rather than primary model
features.

## HLS FVC condition

The current HLS FVC calibration predates the corrected HLS MGRS spatial
selection. Combined-HLS production extraction must not be treated as final
until the HLS FVC endmembers are recalibrated with the corrected source
selection.

## Consequence

A raw satellite export preserves the scientifically relevant candidate
information that is expensive to recover from Earth Engine, while final feature
selection, leakage control, missingness analysis, validation, and model
comparison remain reproducible local operations.
