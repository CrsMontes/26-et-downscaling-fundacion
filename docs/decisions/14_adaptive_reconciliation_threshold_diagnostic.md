# Adaptive reconciliation threshold diagnostic

Evaluation date: 2022-04-07
Diagnostic tile: r005_c008

## Purpose

Evaluate whether MODIS reconciliation should use a fixed number of
correction passes or continue until the MODIS conservation tolerance
is satisfied, and compare usable-support thresholds of 80%, 90% and 95%.

Usable fine support was defined as:

AOA = TRUE AND Kc >= 0 AND complete Ridge-25 predictor stack.

Cells outside this condition were used only as internal neutral fill
when required to complete an eligible MODIS parent and were not
considered publishable pixels.

MODIS conservation tolerance: 0.01 mm per period.
Maximum iterations tested: 20.

## Results

| Minimum usable support | MODIS parents | Published pixels | Iterations to convergence | Mean fill fraction | Maximum fill fraction | Final maximum conservation error (mm) |
|---|---:|---:|---:|---:|---:|---:|
| 80% | 28 | 12,913 | 6 | 0.07955 | 0.19718 | 0.008746 |
| 90% | 18 | 8,760 | 5 | 0.03941 | 0.09069 | 0.002805 |
| 95% | 12 | 6,027 | 4 | 0.02024 | 0.04143 | 0.002208 |

All three thresholds converged within the 0.01 mm conservation
tolerance and produced zero negative published ET values.

## Convergence trajectories: maximum absolute conservation error

### 80%

2.25769 -> 0.87925 -> 0.35165 -> 0.14038 -> 0.05580 ->
0.02211 -> 0.00875 mm

### 90%

1.76110 -> 0.46557 -> 0.13286 -> 0.03687 -> 0.01017 ->
0.00281 mm

### 95%

0.85854 -> 0.20409 -> 0.04553 -> 0.01004 -> 0.00221 mm

## Published-ET sensitivity on common pixels

- 80% vs 90%:
  mean absolute difference = 0.5576 mm;
  P95 = 2.0702 mm;
  maximum = 2.8386 mm.

- 90% vs 95%:
  mean absolute difference = 0.3305 mm;
  P95 = 1.6399 mm;
  maximum = 2.1103 mm.

- 80% vs 95%:
  mean absolute difference = 0.9713 mm;
  P95 = 3.3303 mm;
  maximum = 4.8111 mm.

## Interpretation

The earlier poorer conservation at 80% and 90% was caused primarily by
stopping reconciliation after three fixed correction passes. All three
thresholds satisfy the conservation criterion when reconciliation is
allowed to continue until tolerance.

Therefore, the usable-support threshold must not be selected from
three-pass conservation performance.

The evidence supports replacing the fixed three-pass assumption with
a candidate convergence-to-tolerance rule, subject to basin-wide and
multi-date confirmation.

The final usable-support threshold (80%, 90% or 95%) remains unresolved.
It must be selected from the trade-off between spatial coverage,
internal fill fraction and stability of the published ET estimates.

No production methodology has yet been changed.
