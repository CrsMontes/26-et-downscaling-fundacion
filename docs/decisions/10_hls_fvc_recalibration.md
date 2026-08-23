# Decision 10 — Recalibrate HLS FVC after the MGRS spatial-filter correction

## Problem

The original HLS FVC endmembers were calibrated before the diagnostic
reproduction identified a spatial-selection problem in HLS.

HLS assets selected only with `filterBounds()` could include products whose
MGRS tile identifier was not local to the station footprint. Therefore, the
previous HLS FVC calibration could have been influenced by observations outside
the intended local HLS tile support.

Because FVC is derived from source-specific NDVI endmembers, the previous HLS
calibration could not be treated as final after correcting the HLS spatial
selection.

## Alternatives

1. Retain the previous HLS FVC endmembers.
2. Remove HLS FVC from the production extraction.
3. Recalibrate HLS FVC using the corrected HLS spatial-selection procedure
   while preserving the diagnosed FVC formulation.

## Decision before recalibration

Recalibrate HLS FVC using the corrected MGRS spatial-selection procedure.

The recalibration must preserve the existing FVC formulation so that the test
isolates the effect of correcting HLS spatial support rather than changing
multiple methodological components simultaneously.

## Calibration strategy

The recalibration preserves the two-stage global percentile strategy used by
the existing FVC workflow.

For each HLS station-period observation:

1. use the MODIS 8-day periods from 2021–2023;
2. process HLS independently from Sentinel-2;
3. restrict HLS observations to the verified local MGRS tiles associated with
   each station footprint;
4. construct the optical temporal medoid using the production HLS processing
   workflow;
5. calculate valid optical coverage over the MODIS footprint;
6. retain observations with optical coverage >= 80%;
7. calculate NDVI and NDWI from the HLS medoid;
8. exclude water pixels using the existing NDWI-based rule;
9. calculate footprint NDVI P05 and P95 values as low- and high-endmember
   candidates;
10. combine all eligible station-period candidates and calculate global
    endmembers from the candidate distributions.

The calibration is performed using the HLS 30 m working grid.

## Spatial-selection correction

The recalibration uses the corrected HLS preprocessing in which source products
are restricted to verified local MGRS tiles before the temporal medoid is
constructed.

This correction prevents non-local HLS assets admitted by `filterBounds()` from
contributing to the station-period calibration sample.

The Sentinel-2 FVC calibration is not modified because the diagnosed
spatial-selection problem was specific to HLS.

## Safety rule

The diagnostic recalibration must not overwrite production FVC configuration
automatically.

Candidate values must first be inspected and accepted.

The production configuration is updated only after the recalibration result has
been inspected and accepted.

## Empirical result

The corrected HLS recalibration was completed using the verified local MGRS
selection and the preserved calibration method.

The diagnostic evaluated all 690 possible station-period combinations:

```text
5 stations × 138 MODIS periods = 690 station-period combinations