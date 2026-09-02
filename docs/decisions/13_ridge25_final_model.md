# Decision 13 - Final primary Kc model: Ridge-25

## Problem

The previous closed workflow used a 25-predictor Random Forest requiring
Sentinel-1 and CHIRPS. The five-year experiment showed that algorithm choice and
predictor availability materially affected spatial transfer and usable sample
size.

## Alternatives evaluated

The experiments considered Ridge and Random Forest with common Sentinel-2,
meteorology and harmonic predictors, plus paired tests of red-edge/NDRE,
Sentinel-1, CHIRPS, Landsat LST, S2 FVC/albedo and VPD maximum.

## Evidence

On the fixed 799-row GE90 population, Ridge with the final 25 predictors
produced approximately:

- spatial OOF: R2 0.3800, RMSE 0.25346, MAE 0.18925, bias -0.01653,
  KGE 0.48763;
- LOYO: R2 0.52494, RMSE 0.22187, MAE 0.16397, bias -0.00565,
  KGE 0.63197.

The final five-variable red-edge/NDRE block improved Ridge modestly without
reducing the 799-row population. Sentinel-1 strongly degraded Ridge spatial
transfer; CHIRPS, LST and S2 FVC/albedo did not provide material spatial
improvement for the primary Ridge model.

Random Forest remains scientifically useful as a sensitivity model, especially
because its response to Sentinel-1 and the mangrove station differs from Ridge,
but it is not the primary production algorithm.

## Decision

Use `StandardScaler` + `Ridge(alpha=1.0)` with 25 predictors:

- 11 common S2 variables;
- RedEdge1, RedEdge2, RedEdge3, NIR_Broad and NDRE;
- Tair mean, Tair max, VPD mean, solar radiation and wind;
- four day-of-year harmonics.

The fitted model is rebuilt from the master dataset during each scientific run.
A pre-trained `.joblib` is never a required input.

## Important limitation

The spatial performance remains moderate and ST04/mangrove remains a difficult
transfer case. No independent observations validate ET at 20 m.
