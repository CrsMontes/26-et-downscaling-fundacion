# Decision 19 — Exact-overlap global MODIS reconciliation

## Status

Accepted as the final reconciliation design after the 2022-04-07 diagnostic.
The exact-overlap production path is now wired as a separate module while the
previous iterative rasterio reconciliation remains in the repository only as
a diagnostic/legacy path. End-to-end acceptance still requires the 2022-04-07
production QA and the repeated spatial-OOF field comparison.

## Problem

The frozen local Ridge-25 production reconciled the 20 m field to MODIS by
alternating two raster reprojection operators:

- fine → MODIS: `Resampling.average`;
- MODIS → fine: `Resampling.nearest`.

The 20 m UTM grid and the native MODIS sinusoidal grid are not nested. Those
operators are therefore not inverses. On 2022-04-07 the iterative correction
changed station-scale ET strongly relative to the initial Ridge-25 pattern,
including approximately −48% at ST02.

A second audit found a separate local-geometry error: the WKT representation
returned for the MODIS projection was used in local UTM↔MODIS transformations.
The native MODIS grid is spherical sinusoidal (`R = 6371007.181 m`). Using the
WKT representation locally shifted the parent assignment in the Fundación
area by about 21 MODIS columns and 14 rows. Earth Engine point/footprint values
were correct; the error was confined to the final local grid correspondence.

## Alternatives considered

1. Keep the iterative rasterio average/nearest correction.
2. Normalize independently inside each MODIS parent.
3. Change model, AOA, support threshold or return to Random Forest.
4. Use a geostatistical area-to-point method.
5. Preserve Ridge-25 and solve one exact-overlap constrained reconciliation.

Alternatives 1–3 were rejected because they do not solve the non-nested
change-of-support problem. Alternative 4 is scientifically defensible but adds
complexity that is not required because Ridge-25 already supplies the fine
spatial pattern.

## Accepted method

Keep unchanged:

- 2020–2024 training data;
- Ridge-25 model and predictors;
- spatial and LOYO validation;
- AOA hard mask;
- `Kc_raw >= 0` applicability rule;
- 90% usable-support threshold.

For each date, retain sufficient external fine-grid support through mosaicking,
then reconcile once globally. Let `a_ji` be the real overlap area between fine
cell `i` and MODIS cell `j`. The coarse-support operator uses normalized area
weights. The initial ET field follows the Ridge-25 relative pattern. The final
field is the minimum Euclidean adjustment satisfying all eligible MODIS
constraints simultaneously:

`min ||ET - ET_initial||²`, subject to `C ET = ET_MODIS`.

A 20 m cell crossing a MODIS boundary always retains one single value; its
contribution to each coarse parent is controlled only by its overlap area.
No nearest-neighbour correction and no arbitrary reconciliation iterations are
used.

Local geometry must use the explicit spherical MODIS sinusoidal CRS:

`+proj=sinu +R=6371007.181 +nadgrids=@null +wktext +no_defs`

while Earth Engine downloads should use the exact native CRS identifier from
`projection.getInfo()["crs"]` and the native transform phase.

Small negative ET values produced by the unconstrained least-squares projection
are floored once to zero. Conservation is then recomputed with the exact
overlap operator. The date passes only if the existing 0.01 mm conservation
tolerance is still satisfied; otherwise production fails instead of adding a
new correction method.

## Diagnostic evidence — 2022-04-07

With corrected MODIS local geometry and exact overlaps:

- eligible MODIS parents: 5,064;
- active 20 m cells: 2,741,921;
- publishable 20 m cells: 2,610,357;
- initial maximum MODIS error: 1.231707 mm;
- final maximum MODIS error: 1.99e-13 mm;
- mean absolute adjustment from the initial field: 0.110588 mm/8 d;
- RMSE adjustment: 0.164690 mm/8 d;
- Pearson correlation final vs initial: 0.999911;
- negative publishable cells before flooring: 7;
- minimum ET before flooring: −0.299592 mm/8 d.

Station values (mm/8 d):

| Station | Old iterative | Overlap initial | Overlap final | Final / initial |
| --- | ---: | ---: | ---: | ---: |
| ST02 Oil palm | 18.056 | 34.959 | 34.767 | 0.9945 |
| ST03 Banana | 26.177 | 31.791 | 31.619 | 0.9946 |
| ST05 Dry forest | 58.105 | 52.151 | 52.493 | 1.0066 |

The corrected initial values reproduce the earlier simple proportional audit
(ST02 ~34.9, ST03 ~31.8, ST05 ~52.2), while the global exact-overlap correction
changes them by less than about 0.7% at these stations.

`GDAL average` is not the acceptance operator for final QA because it does not
implement the same exact area-intersection operator. Final conservation QA must
re-use the explicit overlap matrix after the complete mosaic has been written.

## Decision

Adopt exact-overlap global reconciliation as the only final conservation path.
Do not reopen Ridge vs RF, predictors, AOA or the 90% support threshold unless a
separate scientific result requires it. Production is wired through
`ridge25_overlap_production.py`; next steps are limited to one 2022-04-07
end-to-end QA, repeated field OOF evaluation, and the manuscript.
