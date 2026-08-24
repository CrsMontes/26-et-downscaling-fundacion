# Thermal predictors

## Problem

Determine whether Landsat land-surface temperature and TVDI should remain
mandatory predictors in the ET downscaling pipeline.

## Evidence

The historical diagnostic found that the thermal block had high marginal sample
cost and negligible or negative predictive contribution.

Reported diagnostic effects included:

- LST removed about 10% of otherwise usable training observations;
- thermal missingness substantially reduced fine-grid applicability;
- LST permutation contribution was approximately negligible;
- TVDI importance was negative.

The current selected production model does not use LST or TVDI.

## Decision

LST and TVDI are excluded from the production extraction and model pipeline.
The obsolete experimental Landsat-LST implementation is removed from the active
source tree; its history remains available through Git and the diagnostic
materials.

Reintroduction would require new empirical evidence of incremental predictive
value under the current S2 20 m workflow.

## Status

Accepted and implemented.
