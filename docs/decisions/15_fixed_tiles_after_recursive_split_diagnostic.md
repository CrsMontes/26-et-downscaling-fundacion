# Decision 15 - Fixed production tiles after recursive-split diagnostic

## Problem

The local Ridge25 production workflow initially allowed a 4 km core tile to be
recursively subdivided if Earth Engine could not serve the requested data.

Because reconciliation operates on native MODIS parents, subdivision could
potentially make the final product depend on an operational download decision.

## Alternatives

1. Keep adaptive recursive subdivision.
2. Use fixed 4 km production tiles and fail explicitly after download retries.

## Evidence and test

For `r005_c008` on `2022-04-07`, the accepted 4 km parent product was compared
with a mosaic generated independently from its four 2 km children.

The four children covered the complete parent domain exactly once:

- missing pixels: 0
- overlapping pixels: 0

Predictor/model quantities were invariant:

- Kc_raw: exact
- dissimilarity index: exact
- stack validity: exact
- AOA: exact
- usable mask: exact
- owned eligible MODIS parents: 8 in the parent and 8 summed across children

However, the final reconciled product was not strictly invariant:

- parent published pixels: 2629
- child-mosaic published pixels: 2628
- publication-mask mismatch: 1 pixel
- maximum ET difference on common published pixels: 0.0172576904 mm/period
- mean ET difference: 0.0026339539 mm/period

All children converged and no negative ET was published, so the difference was
not caused by failed reconciliation. It resulted from changing the spatial
processing partition.

## Decision

Final Ridge25 production uses fixed 4 km core tiles.

Recursive subdivision is not allowed in the operational production path.
Transient Earth Engine/network failures may be retried, but if a required
4 km tile cannot be obtained after the configured retries, the run fails and
the product is not silently completed using a different spatial partition.

This is an operational reproducibility decision. It does not change the
scientific support rule, AOA definition, Ridge25 model, 90% usable-support
threshold, internal neutral fill, 0.01 mm conservation tolerance, or
30-iteration reconciliation cap.
