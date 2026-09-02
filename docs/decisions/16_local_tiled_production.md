# Decision 16 - Local tiled raster production

## Problem

The accepted three-pass reconciliation works on a small local AOI and exactly
conserved native MODIS ET in the Stage 5 smoke test. However, evaluating the
same deep Earth Engine graph for the whole Fundación basin exceeded Earth
Engine memory limits both synchronously and as a batch table task.

This is a computational scaling problem, not evidence that the reconciliation
equations are invalid.

## Alternatives

1. Increase Earth Engine resource parameters.
2. Store intermediate rasters in Google Drive or persistent Earth Engine assets.
3. Change the reconciliation algorithm.
4. Evaluate the unchanged reconciliation on small spatial cores and mosaic the
   outputs locally.

## Decision

Use option 4.

The basin is divided into non-overlapping tiles aligned to the common 20 m UTM
prediction grid. Each tile is calculated independently with the established
production processing buffer, downloaded directly using
`ee.Image.getDownloadURL`, and written to the external local workspace.

If an Earth Engine request fails specifically because of memory/request-size
limits, only that tile is recursively subdivided. Other scientific errors are
raised and stop the run.

No Google Drive output and no persistent Earth Engine asset are used.

The final raster is assembled locally from non-overlapping, grid-aligned tile
cores. This changes computational execution only; Ridge-25, GE90 eligibility,
native MODIS support, and the three reconciliation passes are unchanged.

## Required verification

Before accepting basin-scale maps, the tiled implementation must pass:

- exact grid-alignment tests;
- an overlap/context test showing the same fine prediction when a location is
  evaluated from different neighboring tile contexts;
- MODIS conservation diagnostics on the resulting product;
- visual seam inspection.

The Stage 5 small-AOI reference test already passed with zero MODIS
conservation error for ST01 on 2022-04-07.
