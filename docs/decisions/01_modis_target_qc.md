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

A strict-QC sensitivity analysis was subsequently run using the final
25-predictor RF and the same spatial validation design. Strict QC retained
331 of 349 observations (94.84%) and all four spatial groups. On the same
331 strict-QC evaluation rows, retraining after QC filtering did not improve
performance relative to the primary model (R2 0.207 vs 0.225; RMSE 0.283 vs
0.279; MAE 0.213 vs 0.210; KGE 0.241 vs 0.247). The 18 excluded observations
also had a higher mean Kc (1.305) than the retained observations (1.036),
showing that strict filtering changes the target distribution rather than
acting as a neutral sample reduction.

## Decision

Use physically valid MODIS ET as the primary target criterion.

- `MODIS_REQUIRE_STRICT_QC = False` remains the primary configuration.
- Preserve `ET_QC`, `modis_qc_present`, and decoded QC fields.
- Do not discard observations only because source QC is missing.
- Strict MODIS QC remains available as a sensitivity analysis.

## Status

Accepted.
