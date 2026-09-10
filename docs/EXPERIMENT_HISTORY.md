# Experiment history and decision narrative

This is the single narrative record of methodological experiments retained on `main`. Detailed historical implementations remain recoverable through Git history and the `diagnostic`, `field` and `virtual` branches; they are not maintained as competing current-method documents.

## 1. Initial diagnostic downscaling

The project began with MOD16A2GF v6.1 ET aggregated over 8-day periods and fine-scale optical, SAR and meteorological predictors. Early work emphasized five real station footprints and Random Forest. The diagnostic phase exposed three recurring problems: limited effective spatial replication, sensitivity to predictor availability, and the risk of claiming 20–30 m validation when field observations did not independently observe ET at that raster support.

A 30 m workflow using HLS/other candidate information and RF was retained as historical evidence. Its field comparison on 26 complete observations produced approximately R² = -0.844, RMSE = 11.443 mm per period, MAE = 9.000, bias = +5.036 and KGE = 0.296. MODIS on the same 26 observations gave R² = -0.984 and RMSE = 11.870. The improvement was small and did not establish fine-resolution validation.

## 2. Five-year candidate-predictor screening

The study was extended to 2020–2024 and candidate families were materialized separately rather than forcing a single global complete-case table. Experiments examined Sentinel-2 versus HLS optical sources, Sentinel-2 red-edge variables, Sentinel-1 orbit-specific predictors, meteorological variables, CHIRPS precipitation, Landsat LST, FVC and albedo.

FVC/albedo and HLS did not provide a sufficiently consistent improvement to justify inclusion in the accepted model. Sentinel-1 and Landsat LST were useful as diagnostics/candidates but were not retained in the final 25-feature specification. The final feature set was reduced to 16 Sentinel-2 optical variables, five ERA5-Land variables and four seasonal harmonics. The candidate archive remains reproducible and does not determine the final population by missingness.

## 3. Ridge-25 phase

A standardized Ridge model using the final 25 predictors was adopted because it improved spatial transfer relative to the earlier RF configuration in the then-current design, remained interpretable and allowed a parsimonious production path. The production algorithm evolved from approximate/tiled conservation to a global exact-overlap reconciliation against the native MODIS sinusoidal grid.

The exact-overlap audit established the final conceptual production rule: fine predicted Kc determines the relative spatial pattern, MODIS ET determines the coarse magnitude, and one global constrained reconciliation enforces coarse conservation on the full represented support. A 90% usable-support gate and a 0.01 mm conservation tolerance were retained.

The first operational AOA implementation used an equal-weight standardized Euclidean DI over the 25 predictors. It respected the spatial validation blocks and produced a frozen threshold near 0.80073 in the previous Virtual Station run. This was explicitly an adaptation rather than the fully model-importance-weighted Meyer–Pebesma implementation.

## 4. Virtual Station design

To reduce dependence between training and field evaluation, 10 virtual MODIS footprints were selected across the basin using a fixed random seed, GE90 Sentinel-2 availability and distinct fixed 10 km UTM spatial blocks. The five real stations were excluded from model fitting. The target remained `Kc_target = MODIS_ET / ETo`, so internal CV continued to evaluate a MODIS-derived target rather than independent 20 m ET.

The audited Virtual10 Ridge population contained 1454 observations from 10 supports and 10 blocks. Ridge spatial OOF produced R² = 0.368752, RMSE = 0.297795, MAE = 0.226653 and KGE = 0.674625. LOYO produced R² = 0.614995, RMSE = 0.232568, MAE = 0.163847 and KGE = 0.712861. Spatial fold behavior was heterogeneous: 6 of 10 fold-specific R² values were negative.

A field-reporting audit found that a previously quoted fixed-Kc result with n = 9 (R² = 0.039130 for Ridge) was a matched subset conditioned on availability of the older Stable5 product. The correct Virtual10 own-domain fixed-Kc comparison used n = 11 and yielded Ridge R² = -0.765152 and RMSE = 12.164081 mm per period. The n = 9 result remains a valid matched sensitivity result, not the main own-domain field metric.

## 5. Reopening model selection: Random Forest versus Ridge

Model selection was reopened after running Random Forest under exactly the same Virtual10 population, 25 predictors, spatial folds and years. RF used 300 trees, `max_features=0.33`, `min_samples_leaf=3`, bootstrap sampling and random seed 42; no new tuning was performed.

Spatial OOF improved from Ridge R² = 0.368752 to RF R² = 0.526856. RMSE decreased from 0.297795 to 0.257819 and MAE from 0.226653 to 0.191418. RF had negative R² in 3 of 10 spatial folds versus 6 of 10 for Ridge and lower spatial-fold RMSE in 8 of 10 supports. One support (block 62_116) remained a clear RF weakness and is retained as a limitation.

LOYO also favored RF: R² increased from 0.614995 to 0.728444, RMSE decreased from 0.232568 to 0.195320, MAE from 0.163847 to 0.127113 and KGE increased from 0.712861 to 0.775315. RF had lower RMSE and MAE in all five held-out years. RF OOF predictions were non-negative but showed the expected compression of the highest Kc values, which remains relevant for interpretation of extremes.

The current fixed-Kc field subset did not uniformly favor RF. On the 11 own-domain observations, Ridge RMSE was 12.164 versus RF 12.505, while RF had slightly lower MAE and much smaller bias. On the older matched n = 9 subset, Ridge had R² = 0.039 and RMSE = 9.038 versus RF R² = -0.101 and RMSE = 9.675.

A broader retrospective test then applied current Virtual10 Ridge and RF to the historical 30 m field protocol. On the 24 observations valid for both current models under AOA, RF Virtual10 had R² = -1.007, RMSE = 11.898, MAE = 9.248, bias = +4.522 and KGE = 0.264; Ridge Virtual10 had R² = -1.552, RMSE = 13.415, MAE = 10.470, bias = +5.008 and KGE = 0.133. RF had lower absolute error in 17 of 24 observations and lower station-level RMSE in four of five stations. The historical 30 m RF remained very similar globally to current RF Virtual10. These repeated station-period observations are not treated as 24 independent spatial replicates.

**Decision:** RF-25 becomes the main estimator. Ridge remains a sensitivity/comparator in history and legacy branches. The decision is based on the joint spatial OOF, LOYO, fold stability and retrospective field-proxy evidence, not R² alone.

## 6. Final AOA/DI decision for RF-25

Because the final estimator changed to RF, the applicability-domain implementation was aligned more closely with Meyer & Pebesma (2021) and current CAST behavior. The final DI uses standardized predictors weighted by RF permutation importance, weighted Euclidean distance, normalization by the mean of all pairwise weighted training distances and a threshold derived from training DI under the same frozen spatial-CV blocks. The threshold rule is `min(max(training_DI), Q3 + 1.5*IQR)`.

AOA is a hard publication-support mask. Local Point Density is generated as a complementary diagnostic but is not used as a second hard mask. This avoids silently adding a new exclusion rule while preserving information about how densely a prediction environment is represented by training data.

The RF-weighted AOA threshold is deliberately recomputed from each fresh final training population; the old equal-weight threshold 0.80073 is historical and must not be reused as if both DI definitions shared a numeric scale.

## 7. Final clean-main policy

`main` contains the RF-25 method. The prior stages are preserved as branches: `virtual`, `field` and `diagnostic`. Generated data now live inside `outputs/` in the repository directory but are ignored by Git. A fresh run downloads and rebuilds data locally without Google Drive.

The final workflow does not reopen predictor selection or RF tuning. A fresh run first materializes the complete implemented predictor universe on the selected 10 Virtual10 supports, preserves that master, and then derives RF-25 from the frozen GE90 support-period whitelist using only the accepted 25 predictors. The five real stations remain external to training. Any future methodological change must be recorded explicitly rather than folded silently into production.
