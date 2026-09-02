# Ridge-25 AOA basin diagnostic

Date evaluated: 2022-04-07

## AOA specification

- Model: Ridge-25
- Predictors: 25
- AOA: unweighted standardized predictor space
- Spatial-CV-derived DI threshold: 0.6026447300586807

## Basin-wide results

- Valid Kc pixels: 4,013,160
- Complete AOA predictors: 4,013,160
- Missing AOA predictors: 0
- Inside AOA: 3,352,197 (83.5301%)
- Outside AOA: 660,963 (16.4699%)

## Kc sign by AOA

| Domain | Kc >= 0 | Kc < 0 |
|---|---:|---:|
| Inside AOA | 3,350,835 | 1,362 |
| Outside AOA | 636,937 | 24,026 |

- Negative rate inside AOA: 0.04063%
- Negative rate outside AOA: 3.63500%
- Outside/inside negative-rate ratio: 89.47

Of all 25,388 negative Kc predictions, 94.64% occurred outside the AOA.

## DI distribution across basin

- min: 0.0289573
- median: 0.3899602
- p95: 1.0046518
- max: 9.1982502

## Interpretation

Negative Ridge-25 Kc predictions are strongly associated with extrapolation outside the spatially validated area of applicability.

This supports retaining Ridge-25 while restricting interpretation and eventual production to the AOA.

No final decision has yet been made regarding:
- the 1,362 negative Kc pixels occurring inside the AOA;
- post-prediction Kc validity thresholds;
- reconciliation changes;
- the final masking strategy.
