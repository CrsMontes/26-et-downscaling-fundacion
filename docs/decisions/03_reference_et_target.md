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

Fine-scale ET reconstruction:

ET_fine = Kc_predicted * ETo

ETo is the short-reference evapotranspiration used by the model.

ETr is retained for quality control and interpretation of tall-reference field instruments. It is not interchangeable with ETo in the Kc target.

Reference ET retains its coarse atmospheric support and must not be described as a 20 m or 30 m meteorological product.

## Status
Accepted.
