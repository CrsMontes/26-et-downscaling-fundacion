# Thermal predictors

## Problem
Determine whether Landsat land-surface temperature and TVDI should remain mandatory model predictors.

## Evidence
The historical diagnostic found that the thermal block had high marginal sample cost and negligible or negative predictive contribution.

Reported diagnostic effects:
- LST removed 10.1% of training observations.
- Thermal missingness affected 16.6% of fine cells.
- LST permutation contribution was approximately negligible.
- TVDI importance was negative.

Removing the thermal block increased usable training observations and spatial prediction coverage.

## Decision
LST and TVDI are excluded from the primary predictor set.

The Landsat LST implementation is retained as an experimental diagnostic module and may be reconsidered only if new evidence demonstrates useful incremental information.

## Status
Accepted.
