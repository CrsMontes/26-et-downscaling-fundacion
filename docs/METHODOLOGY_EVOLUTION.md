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
- prediction grid: 20 m;
- conservative MODIS reconciliation: three passes, applied only after
  fine-resolution prediction.

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
