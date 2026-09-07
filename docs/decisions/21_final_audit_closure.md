# Decision 21 — Final scientific audit closure

## Status

Accepted for the final Ridge-25 workflow after the September 2026 forensic
code/results audit. This decision closes the remaining methodological issues
that could affect final communication or product masking without reopening the
accepted Ridge-25 model.

## Problem

The final audit identified four issues requiring explicit closure before final
map production and manuscript writing:

1. the active applicability-domain implementation used equal predictor weights,
   whereas the model-informed AOA framework of Meyer and Pebesma (2021) weights
   predictor-space distances by model importance;
2. persistence had to be retained as a transparent coarse temporal baseline;
3. field-period aggregation allowed at least 5/8 valid daily ETgage readings,
   so complete 8/8 periods required sensitivity analysis;
4. exact MODIS conservation was demonstrated on the complete reconciled fine
   support, while the final publication mask removes part of that support.

The audit also found incomplete run-level provenance: `run_metadata.json` did
not record the repository commit or hashes of the exact local inputs and
outputs used by a run.

## Alternatives and evidence

### Applicability domain

The active domain standardizes all 25 Ridge predictors and computes Euclidean
DI with equal weights, using the same spatial-block grouping as validation.
Meyer and Pebesma (2021, *Methods in Ecology and Evolution*, 12, 1620–1633,
https://doi.org/10.1111/2041-210X.13650) define the DI/AOA using standardized
predictor-space distances with model-importance weighting and a threshold
derived from cross-validation training DI values.

A final project-specific sensitivity multiplied standardized predictors by the
absolute standardized Ridge coefficient, `|beta|`. This sensitivity was not
interpreted as an exact CAST implementation because `|beta|` is a specific
choice of importance and the Ridge predictors are strongly correlated.

Offline results on the 833-row audited population were:

| Diagnostic | Equal weight | `|beta|` sensitivity |
| --- | ---: | ---: |
| Training DI threshold | 0.608989 | 0.365495 |
| Training rows outside domain | 36 | 56 |
| Spearman DI vs absolute spatial-OOF error | -0.0589 | 0.0526 |
| Spatial-OOF RMSE inside domain | 0.249730 | 0.249865 |
| Spatial-OOF RMSE outside domain | 0.248767 | 0.247233 |

The coefficient-weighted distance contribution was 95.83% optical, 2.55%
meteorological and 1.62% harmonic. On the dry-period basin stack
(2020-03-13), 4,663,623 basin pixels had a complete predictor stack. The
accepted equal-weight domain retained 49.14% of these pixels and the `|beta|`
sensitivity retained 45.57%. A total of 824,552 pixels (17.68% of the valid
stack) changed classification. The change was therefore not a minor numerical
adjustment to the existing mask.

### Persistence baseline

Persistence was recalculated on rows matched exactly to Ridge spatial OOF:

| Definition | n | Persistence R2 | Ridge OOF R2 | Persistence RMSE | Ridge OOF RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous available observation | 828 | 0.501898 | 0.393887 | 0.226616 | 0.249982 |
| Strict previous 8-day composite | 622 | 0.539485 | 0.406367 | 0.211840 | 0.240517 |
| Previous observation within 16 days | 762 | 0.534721 | 0.403704 | 0.216419 | 0.245002 |

Persistence therefore remains an important coarse temporal baseline and must be
reported. It is not a fine-resolution downscaling alternative because it does
not generate subpixel spatial structure.

### Field valid-day sensitivity

The accepted field-period aggregation requires at least five valid ETgage days
per MODIS period. Requiring complete 8/8 periods reduced the all-cover AOA
sample from n=17 to n=12 and the pre-specified fixed-Kc ST01-ST03 subset from
n=9 to n=6. Metrics changed strongly at those small sample sizes. The 8/8 result
is retained as a sensitivity, while >=5/8 remains the primary aggregation rule
and must be reported explicitly.

### Conservation scope

A synthetic non-nested overlap test demonstrated that the complete reconciled
support can exactly reproduce the MODIS target while the publishable subset of
the same parent does not. In the real 2022-04-07 diagnostic:

- complete-support maximum conservation error: `1.99e-13 mm`;
- independently aggregated masked-raster median absolute difference:
  `0.482 mm`;
- masked-raster P95 absolute difference: `2.406 mm`;
- masked-raster maximum absolute difference: `6.596 mm`.

The UTM↔native MODIS sinusoidal round-trip error was below `1e-9 m`, and all
existing exact-overlap tests passed. The reconciliation solver itself therefore
did not require a numerical change.

## Decision

1. **Keep the equal-weight applicability domain.** It must be described as an
   equal-weight standardized multivariate DI/applicability domain adapted from
   the Meyer-Pebesma AOA framework, not as an exact model-importance-weighted
   implementation. The `|beta|` alternative is retained only as sensitivity
   evidence under `reproducibility/`.
2. **Report persistence as a coarse temporal baseline**, preferably the strict
   previous 8-day composite definition when making the direct comparison.
3. **Keep >=5/8 valid field days as the primary field aggregation rule** and
   report 8/8 as a small-sample sensitivity.
4. **Keep the exact-overlap reconciliation solver unchanged.** Exact MODIS
   conservation applies to the complete reconciled support before the final
   publication mask. The masked published raster is not claimed to conserve
   MODIS exactly by itself.
5. **Harden run provenance.** New run metadata must record Git state and
   SHA-256 hashes for canonical inputs, raw training sources, the rebuilt
   master, environment lock and generated run artifacts.

These decisions close the final scientific audit. Further model, predictor,
AOA or reconciliation alternatives should not be opened unless new evidence
reveals a concrete error or materially changes the scientific question.
