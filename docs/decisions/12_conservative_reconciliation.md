# Decision 12 — Conservative MODIS reconciliation

## Problem

Define the mathematical role of the 20 m Random Forest prediction and prevent
the production product from being misrepresented as independently observed or
validated ET at 20 m.

## Implemented formulation

The Random Forest predicts `Kc_raw` on the Sentinel-2 grid. For every eligible
MODIS parent, masked fine cells are filled with the mean valid `Kc_raw` of that
parent. The initial ET field is

```text
ET_fine,i = Kc_filled,i * ET_MODIS / mean_parent(Kc_filled)
```

Three proportional correction passes reconcile the non-nested MODIS
sinusoidal and 20 m UTM grids. Conservation is evaluated before clipping to
the basin.

ETo is indispensable for defining `Kc_target = ET_MODIS / ETo` during
training, but it is not multiplied as a spatial 20 m magnitude during current
production.

## Diagnostic evidence

The targeted 2022-04-07 ST02–ST03 diagnostic found:

- non-trivial within-parent `Kc_raw` variability driven mainly by optical
  predictors;
- exact rank preservation between `Kc_raw` and reconciled ET within every
  sampled parent with at least two observations and no fill;
- maximum sampled conservation error of 0.0000534 mm per MODIS period;
- parent-specific effective scaling capable of generating discontinuities at
  MODIS boundaries. Among 33 sampled pairs from adjacent MODIS parents and
  separated by at most 150 m, the reconciliation-scale component dominated
  more than half of the ET difference in 45.5% of pairs.

The bounded production QA for 2022-05-25 then closed the two previously open
implementation checks:

- ST02 was eligible with `fine_fill_fraction=0.06927`; 37 of 535 sampled fine
  cells were filled. ST03 was eligible with `fine_fill_fraction=0.02846`; 14
  of 532 cells were filled.
- No valid raw Kc value changed during fill, and every newly filled value
  equalled the valid parent Kc mean to the numerical precision of the sampled
  Earth Engine values.
- Within each tested parent, the Spearman relation between `Kc_filled` and
  reconciled ET was 1.0, including the filled cells.
- All 21 eligible MODIS parents in the bounded native grid met the 0.01 mm
  tolerance. Maximum absolute conservation error was 0.0000534 mm per period.

This boundary effect is a consequence of parent-wise conservation, not proof
that the RF spatial pattern is physically wrong. Removing or smoothing it
would alter the mathematical product and requires a separately approved
methodological decision.

## Decision

For the 2021-2023 architecture, parent-wise conservative reconciliation was
retained as the then-accepted reference implementation. The RF field was
interpreted as relative subpixel heterogeneity, and no independent validation
of absolute ET at 20 m was claimed.

This decision is retained as historical methodological evidence only. The
iterative raster reconciliation was subsequently rejected after the
non-nested-grid audit and was superseded for the current 2020-2024 workflow by
Decision 19.

The current accepted reconciliation is one global exact-overlap constrained
projection between the regular 20 m UTM grid and the native MODIS sinusoidal
grid. It does not use coarse-to-fine nearest-neighbour correction or arbitrary
reconciliation iterations.

## Status

Superseded by `19_exact_overlap_global_reconciliation.md`.

The legacy reconciliation implementation and its dedicated QA script have been
removed from the active codebase. This decision file is preserved to document
the methodological evolution rather than to describe the current production
algorithm.
