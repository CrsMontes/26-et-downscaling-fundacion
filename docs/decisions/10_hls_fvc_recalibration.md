# Decision 10 — Recalibrate HLS FVC after the MGRS spatial-filter correction

## Problem

The original HLS FVC endmembers were calibrated before the diagnostic
reproduction identified that `filterBounds()` alone could admit HLS products
whose MGRS tile was not local to the station footprint.

Because FVC depends on source-specific NDVI endmembers, the previous HLS
calibration could not remain final after correcting spatial selection.

## Alternatives

1. Retain the previous HLS FVC endmembers.
2. Remove HLS FVC from the operational alternative.
3. Recalibrate HLS FVC using verified local MGRS tiles while preserving the
   diagnosed FVC formulation.

## Method preserved

The recalibration retained the two-stage global percentile method:

1. evaluate all 138 MODIS periods from 2021–2023 at five station footprints;
2. restrict HLS to verified local MGRS tiles before constructing each medoid;
3. calculate coverage using the common valid support of Green, Red, and NIR;
4. retain station-period observations with coverage >= 80%;
5. calculate NDVI and NDWI;
6. exclude water using `NDWI > 0`;
7. use within-footprint NDVI P05 and P95 as low/high candidates;
8. use the 0.05 and 0.95 quantiles across eligible candidates as the global HLS
   endmembers.

The HLS working grid remains 30 m. Sentinel-2 calibration is unaffected by this
HLS-specific correction.

## Empirical result

The corrected diagnostic evaluated the complete design:

```text
5 stations x 138 MODIS periods = 690 station-period combinations
```

Eligible HLS observations after the 80% coverage criterion: **381**, covering
all five stations.

Accepted HLS endmembers:

- NDVI low endmember: `0.411908487478892`
- NDVI high endmember: `0.9082510914569858`

The accepted values are stored in `config/fvc_endmembers.json`.

## Decision

Use the recalibrated HLS endmembers whenever the combined-HLS operational
alternative is run. HLS remains an alternative source; Sentinel-2 remains the
primary production source.

## Status

Accepted and implemented.
