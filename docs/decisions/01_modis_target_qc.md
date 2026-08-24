# MODIS target and quality control

## Problem

Define the coarse evapotranspiration target without allowing a quality-control
filter to predetermine model performance or silently remove physically valid ET
observations.

## Alternatives

- Require the historical strict `ET_QC` filter.
- Retain every physically valid MOD16A2GF ET value and preserve QC fields for
  sensitivity analyses.

## Evidence

The diagnostic reproduction confirmed the MOD16A2GF ET scale factor and valid
ET range. The current MODIS preparation preserves the original QC information
and decodes its bit fields independently from ET-value validity.

When the source `ET_QC` band is masked, the exported table uses `255` only as an
explicit missing-QC sentinel. The separate field `modis_qc_present` identifies
that condition. Therefore, `ET_QC == 255` must not be interpreted as a measured
QC category or used by itself to invalidate an otherwise physically valid ET
observation.

## Decision

Use physically valid MODIS ET as the primary target criterion.

- `MODIS_REQUIRE_STRICT_QC = False` remains the primary configuration.
- Preserve `ET_QC`, `modis_qc_present`, and decoded QC fields.
- Do not discard observations only because source QC is missing.
- Strict MODIS QC remains available as a sensitivity analysis.

## Status

Accepted.
