# Spatial support

## Problem
Avoid pseudoreplication and false claims of fine-resolution validation when the ET target is MODIS.

## Evidence
MODIS ET is the coarse target. Fine optical and radar pixels inside one MODIS footprint do not constitute independent ET observations.

The historical diagnostic strategy trained at the MODIS footprint support and predicted subsequently on a fine grid.

## Decision
Training unit:
one MODIS footprint x MODIS period observation.

Predictor statistics used for training must describe the same coarse support as the target.

Do not create local fine-scale rows that repeat the same parent MODIS target.

Fine grids are prediction supports only:
- S2 pipeline: 20 m.
- HLS pipeline: 30 m.

Meteorological variables retain their native/coarse support even when associated with fine prediction cells.

Independent validation must only be claimed at the spatial support provided by the independent observations.

## Status
Accepted.
