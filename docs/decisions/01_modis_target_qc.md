# MODIS target and quality control

## Problem
Define the coarse evapotranspiration target without allowing quality-control filters to predetermine model performance.

## Alternatives
- Require the historical strict ET_QC filter.
- Retain every physically valid MOD16A2GF ET value and preserve QC fields for sensitivity analyses.

## Evidence
The diagnostic reproduction confirmed the MOD16A2GF scale factor and valid ET range. Strict QC can be evaluated independently because all QC components are retained.

## Decision
Use physically valid MODIS ET as the extraction criterion. Preserve the original ET_QC value and decoded QC fields. Do not silently discard observations using strict QC.

Strict MODIS QC remains available as a sensitivity analysis.

## Status
Accepted.
