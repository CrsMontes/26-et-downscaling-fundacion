# Final Ridge-25 production workflow

## Status

The 20 m production implementation was frozen after successful full-basin
production and quality control for three contrasting MODIS periods:

- 2020-03-13
- 2021-11-25
- 2022-04-07

All three dates completed 166/166 fixed 4 km tiles.

## Final rule

Fine-scale publication support is defined as:

    complete 25-predictor stack
    AND inside Ridge-25 AOA
    AND Kc_raw >= 0
    AND native MODIS usable support fraction >= 0.90

For eligible MODIS parents, non-publishable support up to 10% is filled
internally using the mean Kc of usable cells from the same parent solely for
mass conservation.

The fill is never published.

Reconciliation proceeds locally until the maximum absolute MODIS-parent
conservation error is <= 0.01 mm per period, with a maximum of 30 iterations.

Only original usable fine-resolution cells are published.

## Production architecture

- Primary model: Ridge regression
- Predictors: 25
- Training population: 799 observations
- Fine analysis grid: 20 m
- Tile core: fixed 4000 m
- AOA: hard publication constraint
- Production computation: local
- Earth Engine role: predictor and MODIS-data provision
- Google Drive: not used
- Persistent Earth Engine assets: not used
- Recursive tile subdivision: not used

## Final three-date QC

| Date | Tiles | Converged | Max iterations | Max error (mm) | Negative published ET | Published pixels |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2020-03-13 | 166/166 | yes | 11 | 0.00993357 | 0 | 717163 |
| 2021-11-25 | 166/166 | yes | 14 | 0.00993776 | 0 | 2386652 |
| 2022-04-07 | 166/166 | yes | 15 | 0.00998764 | 0 | 2642610 |

For all three dates:

- published pixels outside the complete predictor stack: 0
- published pixels outside AOA: 0
- published pixels with Kc_raw < 0: 0
- published pixels not marked usable: 0
- published pixels in ineligible MODIS parents: 0
- publication-mask mismatches: 0

The published-pixel counts exactly reproduce the preceding diagnostic
implementation for all three dates.

Small differences in the bookkeeping count of owned MODIS parents relative to
the earlier diagnostic remain unresolved at the ownership/accounting level.
They do not change the publication masks or published-pixel counts and
therefore do not represent a difference in the published ET product.

## Operational entry point

    python scripts/run_pipeline.py

Experimental scripts are retained for reproducibility but are not part of this
operational path.

