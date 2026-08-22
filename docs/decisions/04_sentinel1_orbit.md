# Sentinel-1 orbit

## Problem
Use a single Sentinel-1 acquisition geometry while maximizing temporal support across the five stations.

## Alternatives
- Relative orbit 142, descending.
- Relative orbit 77, ascending.
- Multiple-orbit combination.

## Evidence
The historical methodology reported relative orbit 142 descending.

The reproduction did not confirm the historical statement that orbit 142 was the only orbit covering all stations.

For 2021-2023:
- R077 ascending covered all five stations and supported 101 of 138 MODIS periods.
- R142 descending covered all five stations and supported 86 of 138 MODIS periods.

## Decision
Use Sentinel-1 relative orbit 77, ascending, as the primary radar geometry.

Do not combine ascending and descending geometries in the primary pipeline because this would introduce systematic acquisition-geometry differences.

## Status
Accepted correction to the historical diagnosis.
