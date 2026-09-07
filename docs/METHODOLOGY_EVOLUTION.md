# Methodological evolution toward the parsimonious workflow

This document records why the active workflow differs from earlier branches.
Git history preserves the historical implementations; the active branch keeps
only the method required for reproducible execution and the code required to
reproduce the final methodological decisions.

## Evolution

| Stage | Main contribution | Limitation identified | Consequence for current workflow |
|---|---|---|---|
| `main` | Original MODIS, Sentinel, meteorology and ET extraction workflow | Fine pixels could be confused with independent ET observations; extraction and model selection were not fully separated | Training support is now explicitly MODIS footprint x period; fine pixels are prediction support only |
| diagnostic methodology | Expanded predictor inventory and explicit diagnostics | Excess predictor dimensionality, scale-incompatible statistics, restrictive extraction, alternate 30 m/HLS pathway | Extraction is permissive; model filtering is local; only scale-transferable predictors enter production |
| `diagnostic-reproduction` | Reproducible Kc target, blocked validation, AOA concept, conservative three-pass reconciliation | RF-25 depended on S1/CHIRPS availability and used only 349 GE90 observations | Preserve spatial support, validation and reconciliation; reopen algorithm/predictor selection |
| `experiment-5year` | 2020-2024 predictor-family tests on larger populations | Several physically plausible families did not improve spatial transfer; strong multicollinearity | Final primary configuration is Ridge with 25 predictors; row counts are rebuilt dynamically from accepted QC |

## Accepted current model

The accepted primary model is:

- target: `Kc_target = ET_MODIS / ETo`;
- training support: station x MODIS footprint x MODIS period;
- population: Sentinel-2 coverage >= 90%;
- algorithm: `StandardScaler` + Ridge, alpha = 1;
- predictors: 16 Sentinel-2 optical variables, five ERA5-Land/context
  variables and four temporal harmonics;
- primary validation: leave-one-spatial-block-out;
- complementary temporal validation: leave-one-year-out;
- prediction grid: regular 20 m UTM cells;
- conservative MODIS reconciliation: one global exact-overlap constrained
  projection after the raw 20 m support mosaic, using the native spherical
  MODIS sinusoidal grid;
- publication support: complete stack, inside the equal-weight standardized
  multivariate applicability domain adapted from the Meyer-Pebesma DI/AOA
  framework, `Kc_raw >= 0`, and >=90% usable support within an eligible MODIS
  parent.

The accepted population is rebuilt from the current source query and QC rather
than frozen to a historical row count. In the designated audited run
`20260906T192839Z_2020_2024`, 833 rows passed the final gate. Spatial-block OOF
performance was R2 = 0.393749, RMSE = 0.249689, MAE = 0.185496 and
KGE = 0.491569 Kc units. LOYO R2 was 0.527433 with RMSE = 0.220447.

## Parsimony principle

The repository does not store a fitted model as a scientific input. A run
rebuilds the eligible population, validates the fixed method and fits Ridge in
memory from the master dataset. Serialized models may be written only as
run-specific provenance artifacts.

The complete master dataset remains richer than the production model so that
the evidence behind predictor exclusions can be reproduced without repeating
all remote extraction.

## Final reconciliation and field comparison (September 2026)

The earlier iterative `average -> nearest` raster reconciliation was rejected
after a 2022-04-07 audit showed strong local distortion and a local MODIS grid
correspondence error. Decision 19 replaced it with a single global
area-overlap reconciliation. The accepted production version is
`ridge25_cs050_ge90_exact_overlap_support90_tol001_v3`.

Exact MODIS conservation is enforced and tested on the complete reconciled
fine support before the publication mask. The masked published raster is a
subset of that support and is not required to reaggregate exactly to MODIS.
For the 2022-04-07 audit, full-support maximum error was
1.99e-13 mm, while independent aggregation of the masked published raster had
a median absolute difference of 0.482 mm and a maximum of 6.596 mm.

The final field scenarios contain 17 field-proxy observations inside the
accepted AOA and 21 when pure AOA extrapolations are retained as sensitivity.
The pre-specified fixed-Kc ST01-ST03 subset contains 9 observations with AOA
and 11 without it. Ridge-25 and MODIS show mixed differences across metrics;
the field data do not demonstrate a consistent accuracy improvement over
MODIS. No independent 20 m validation is claimed.
