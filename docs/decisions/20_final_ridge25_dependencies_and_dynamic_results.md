# Decision 20 — Final Ridge-25 dependencies and dynamic run results

## Problem

The final Ridge-25 model had already excluded Sentinel-1, CHIRPS, FVC, albedo,
HLS and other candidate predictors, but parts of the operational raw-data path
still queried or calculated some of them. In addition, early integration checks
treated a historical training-row count, OOF metrics and a numerical AOA
threshold as exact regression contracts even though public remote collections
can change between fresh queries.

## Alternatives

1. Keep the richer operational extraction and ignore rejected variables later.
2. Freeze historical raw files and require exact row counts/metrics.
3. Keep the scientific rules fixed while making the operational path depend
   only on the accepted model inputs; record row counts, metrics, AOA and hashes
   as run provenance rather than immutable algorithm constants.

## Evidence

The 2026-09-06 S2-only recheck recalibrated FVC with current Sentinel-2 data and
compared Ridge25, Ridge25 + albedo, Ridge25 + FVC, and Ridge25 + albedo + FVC.
The diagnostic was aligned to the operational 20 m analysis grid
(`EPSG:32618`) before the final comparison. The recalibrated FVC endmembers were
NDVI low=0.302990 and NDVI high=0.923539 (n=862 calibration observations across
five stations). The spatial OOF baseline exactly matched the accepted pipeline:
R2=0.393749, RMSE=0.249689, MAE=0.185496 and KGE=0.491569. Adding FVC reduced
spatial performance (R2=0.389312), while albedo produced only a negligible
spatial improvement (R2=0.394476; RMSE=0.249539, a change of about -0.000150).
The combined FVC+albedo model was also worse than the baseline (R2=0.389685).
These results do not justify increasing the final predictor set.

A separate reproducibility audit showed that two fresh Earth Engine queries
with the same code and CS=0.50 produced small optical differences in five
station-period rows. The eligible training keys and target values were
unchanged, but exact OOF and AOA values shifted slightly. Therefore query-
dependent results are run outputs, not mathematical invariants of Ridge-25.

## Test

The final path is required to:

- use Sentinel-2 SR Harmonized + Cloud Score+ `cs_cdf >= 0.50`;
- create deterministic same-day tile mosaics ordered by `system:index`;
- retain the 8-day temporal medoid;
- use only the 16 accepted S2 predictors, five ERA5-Land predictors and four
  seasonal harmonics in Ridge-25;
- calculate `Kc_target = ET_MODIS / ETo`;
- rebuild GE90 population, blocked spatial OOF, LOYO and AOA from the data
  returned by the current query;
- not query Sentinel-1 or CHIRPS in the accepted Ridge-25 operational path;
- not calculate rejected FVC/albedo in fine-resolution Ridge-25 production;
- retain rejected-variable experiments under `reproducibility/`.

## Result

The S2-only recheck supported keeping the 25-predictor Ridge model. FVC and
albedo remain excluded. HLS was not reopened. Numerical `n`, R2, RMSE, MAE,
bias, KGE and the AOA threshold are generated and recorded for each run.

## Decision

The operational model remains Ridge-25. Scientific configuration and QC rules
are fixed; query-dependent counts and performance statistics are not. The
normal pipeline may refresh its accepted external sources and regenerate its
training population. Provenance records what each run actually produced.
Rejected sources remain available only as reproducibility evidence and do not
form operational dependencies.


## Provenance hardening after final audit

The final audit confirmed that query-dependent numerical results must remain
run outputs. New runs therefore record the Git commit/dirty state and SHA-256
hashes for the three canonical repository inputs, the raw training source
caches, the rebuilt training master, `environment-lock.yml`, and the generated
run tables/AOA/figures. This does not freeze remote collections; it makes the
exact local realization of each scientific run auditable.
