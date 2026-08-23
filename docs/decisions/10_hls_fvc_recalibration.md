# Decision 10 — Recalibrate HLS FVC after the MGRS spatial-filter correction

## Problem

The HLS FVC endmembers currently stored in `config/fvc_endmembers.json` were calibrated before the diagnostic reproduction identified a spatial-selection problem in HLS. In that earlier calibration, HLS assets were selected with `filterBounds()` over the calibration geometry, and the resulting HLS sample could include assets whose MGRS tile identifier was not local to the station footprint.

Because FVC is derived from source-specific NDVI endmembers, an HLS calibration produced from a potentially contaminated optical sample must not be treated as final production calibration.

## Historical calibration strategy to preserve

This correction does **not** redefine the FVC method. It preserves the diagnosed two-stage global percentile strategy:

1. use the 138 MODIS 8-day periods from 2021–2023;
2. calibrate HLS independently from Sentinel-2;
3. compute coverage from the common valid support of `Green`, `Red`, and `NIR`;
4. retain station-period observations with coverage >= 80%;
5. calculate NDVI and NDWI from the optical medoid;
6. exclude water pixels using the existing rule `NDWI > 0`;
7. within each valid footprint-period, use NDVI P05 and P95 as low/high candidates;
8. across all valid footprint-period candidates, use the 0.05 and 0.95 quantiles as the global HLS NDVI endmembers.

The Sentinel-2 FVC calibration is not changed by this correction because the diagnosed HLS MGRS-selection problem does not apply to the Sentinel-2 calibration.

## Correction

The HLS recalibration must use the **current production HLS preprocessing** and must restrict the HLS collection independently for every station footprint using the verified local MGRS tile set before building the medoid.

The recalibration script therefore calls:

- `get_hls_collection()` for the source collection;
- `filter_hls_collection_to_geometry()` for each station footprint and period;
- `get_local_hls_mgrs_tiles()` to retain spatial-selection provenance;
- `build_hls_medoid()` after local filtering.

The calibration is performed at the native HLS working scale of 30 m.

## Safety rule

The diagnostic script **must not overwrite** `config/fvc_endmembers.json` automatically. It writes:

- the complete 690-row HLS station-period diagnostic table;
- a summary comparing the current and candidate HLS calibration;
- a candidate FVC configuration that preserves the existing Sentinel-2 calibration.

The production configuration is updated only after the recalibration result has been inspected and accepted.

## Decision status

**Pending empirical recalibration result.**

The old HLS endmembers remain non-final until this test is completed. Sentinel-2 FVC remains available with its existing source-specific calibration.
