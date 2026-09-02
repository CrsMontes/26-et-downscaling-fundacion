# Decision 15 - Ridge-25 production support and reconciliation

## Problem

The closed Random Forest production path used Sentinel-1 as both a predictor
and a production eligibility requirement. The accepted Ridge-25 model does not
use Sentinel-1 or CHIRPS. Reusing the old reconciliation function would
therefore reintroduce a discarded data requirement and unnecessarily reduce the
mapping domain.

## Decision

Ridge-25 production uses only the 25 predictors that define the fitted model:
16 Sentinel-2 optical variables, five ERA5-Land variables and four temporal
harmonics.

The production eligibility gate requires:

- Sentinel-2 valid fraction >= 90% at native MODIS support;
- complete Ridge-25 predictor-stack support under the same threshold;
- valid native MODIS ET.

Sentinel-1 and CHIRPS are not queried and are not eligibility gates.

## Reconciliation

Reconciliation is performed only after the fine Kc pattern has been predicted.
It is not part of Ridge fitting or spatial/temporal OOF validation.

The accepted three proportional correction passes are retained to compensate
for the non-nested native MODIS sinusoidal grid and 20 m UTM prediction grid.
The final product must be checked for MODIS-support conservation before it is
treated as an accepted output.

## Model lifecycle

The Earth Engine Ridge equation is built directly from the sklearn model fitted
during the current pipeline run. A serialized pre-trained model is not a
production input.
