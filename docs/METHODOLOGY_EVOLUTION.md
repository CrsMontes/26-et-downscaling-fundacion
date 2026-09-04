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
| `experiment-5year` | 2020-2024 predictor-family tests on larger populations | Several physically plausible families did not improve spatial transfer; strong multicollinearity | Final primary configuration is Ridge with 25 predictors and 799 GE90 observations |

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
- publication support: complete stack, inside AOA, `Kc_raw >= 0`, and >=90%
  usable support within an eligible MODIS parent.

For the accepted 2020-2024 reference population, the expected gate is 799 rows.
The reference spatial OOF result is approximately R2 = 0.380 and
RMSE = 0.2535 Kc units.

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
`ridge25_exact_overlap_support90_tol001_v2`.

The 2022-04-07 end-to-end product passed the 0.01 mm conservation tolerance
after flooring 14 tiny negative active cells once to zero; the maximum final
conservation error was 0.000701 mm.

The spatial-OOF field comparison yielded 13 publishable fine observations. In
the fixed-Kc main subset (ST01-ST03, n=10), MODIS RMSE was 9.034 mm per period
and exact-overlap Ridge-25 RMSE was 8.746 mm. Other metrics were mixed, so the
field data do not demonstrate a consistent accuracy improvement over MODIS.
No independent 20 m validation is claimed.
