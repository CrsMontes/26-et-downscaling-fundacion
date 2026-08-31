# Decision 11 — Final Sentinel-2 Kc model

## Problem

Select the final model used to transfer coarse MODIS Kc information to the
Sentinel-2 20 m prediction grid without adding unnecessary model complexity.

## Alternatives

Two pre-specified Random Forest models were compared on the same Sentinel-2
training population with optical coverage >= 90%:

1. `rf_common`: 25 scale-transferable predictors.
2. `rf_s2_full`: the same 25 predictors plus six Sentinel-2-specific variables
   (31 predictors total).

Two baselines were retained: training-fold global mean and previous-period MODIS
Kc persistence.

## Validation design

Primary validation used approximately 10 km spatial blocks and `GroupKFold`.
Five stations formed four effective spatial groups because Banana and Oil palm
occupy the same block.

The accepted MODIS target rule is physical ET validity rather than strict QC.
One candidate observation has missing source `ET_QC`; it is retained because
`ET_QC=255` is only the exported missing-QC sentinel and
`modis_qc_present=0` records that condition explicitly.

Final candidate population: **349 station-period observations**.

The final population is selected with
`training_candidate_source_ge_90`, not the common-only candidate flag. This
keeps the common and S2-full RF comparison on exactly the same complete
population, including availability of the six S2-specific candidate
variables. Changing that population would require a new controlled model
comparison and retraining.

## Result

| Model | R2 | RMSE | MAE | Bias | KGE |
|---|---:|---:|---:|---:|---:|
| `rf_common` | 0.234949 | 0.285317 | 0.213072 | -0.017163 | 0.250124 |
| `rf_s2_full` | 0.216928 | 0.288658 | 0.213690 | -0.028851 | 0.230684 |
| global mean | -0.138160 | 0.348004 | 0.263519 | -0.009524 | -0.685737 |
| MODIS persistence | 0.463209 | 0.238993 | 0.154116 | 0.010873 | 0.724387 |

The six additional Sentinel-2 variables did not provide a consistent gain. The
full model improved MAE only marginally while slightly worsening R2, RMSE, and
KGE.

The persistence baseline substantially outperformed both RF models at the
coarse target support. This is retained as a central limitation: temporal MODIS
memory is a stronger predictor of coarse Kc than the fitted RF. Persistence,
however, does not generate subpixel spatial structure and therefore is not a
replacement for the downscaling model.

## Decision

Use `rf_common` as the production model.

- target: `Kc_target = ET_MODIS / ETo`
- optical source: Sentinel-2
- prediction grid: 20 m
- predictors: 25 common scale-transferable variables
- trees: 300
- `max_features = 0.33`
- `min_samples_leaf = 3`
- `max_depth = None`
- random seed: 42

The predictor order and RF parameters are centralized in
`src/et_downscaling/model_spec.py` so training and spatial prediction consume
the same specification.

A 300-versus-500 tree sensitivity check under the same spatial validation showed practically equivalent performance; 300 trees were retained to reduce computational complexity without measurable predictive loss.

## Status

Accepted for production mapping.
