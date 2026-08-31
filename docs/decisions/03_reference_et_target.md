# Reference ET and modelling target

## Problem
Normalize MODIS ET in a physically interpretable way and maintain consistency with the reference-ET observations available in the project.

## Alternatives
- Direct MODIS ET prediction.
- ET / ETo.
- ET / ETr.

## Evidence
The historical diagnosis found better generalization for the normalized target than for direct ET.

The diagnostic reproduction calculated complete daily ETo and ETr for all five stations and all days from 2021 through 2023.

All 690 MODIS station-period observations obtained complete reference ET.

Reconstruction check:
Kc * ETo reproduced MODIS ET to numerical precision at the training support.

## Decision
Primary modelling target:

Kc = ET_MODIS / ETo

At training support, the identity used to verify the target is:

ET_MODIS = Kc_target * ETo

The production map does not treat coarse ETo as an independently resolved
20 m ET magnitude. The predicted Kc field supplies relative subpixel weights,
and the final ET field is reconciled to the parent MODIS ET:

ET_fine,i = Kc_predicted,i * ET_MODIS / mean_parent(Kc_predicted)

Three proportional correction passes account for the non-nested MODIS and
20 m grids. Thus, ETo defines and normalizes the training target, while MODIS
sets the final coarse-scale ET magnitude.

ETo is the short-reference evapotranspiration used by the model.

ETr is retained for quality control and interpretation of tall-reference field instruments. It is not interchangeable with ETo in the Kc target.

Reference ET retains its coarse atmospheric support and must not be described as a 20 m or 30 m meteorological product.

The reconciled product must be described as conservative redistribution of
MODIS ET guided by predicted Kc, not as independent physical validation of ET
at 20 m.

## Status
Accepted.
