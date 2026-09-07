# Decision 22 — Final execution and delivery contract

## Problem

The scientific method was frozen after the final audit, but the definitive run
still needed an explicit execution and delivery contract. Re-running training
once per map date would create unnecessary duplicate model fits and complicate
provenance. Field temporal-completeness sensitivity also needed to be generated
as a formal run output rather than as an ad-hoc diagnostic.

## Frozen scientific state

The definitive run does **not** reopen model selection or methodological tuning.
It uses:

- the accepted Ridge-25 training population and predictor set;
- spatial-block OOF and leave-one-year-out validation;
- the frozen equal-weight standardized DI applicability domain;
- the strict previous-8-day persistence baseline as the principal persistence
  comparison;
- field comparison with at least 5 valid ETgage days per MODIS period as the
  primary field protocol, with 8/8 complete periods as a temporal-completeness
  sensitivity;
- the four field scenarios already defined by the accepted workflow (all sites
  with/without AOA and ST01-ST03 with/without AOA);
- global exact-overlap reconciliation, with exact conservation referring to the
  complete reconciled MODIS support before the publication mask.

## Final map periods

One fitted Ridge-25/AOA state is used to produce all three final map periods:

1. `2020-03-13` — dry / high-atmospheric-demand contrast period;
2. `2021-11-25` — humid / low-atmospheric-demand contrast period;
3. `2022-03-30` — period with maximum simultaneous representation of the field
   campaign in the accepted comparison set.

`2022-04-07` remains a reproducible QA/diagnostic period and is not a fourth
primary map product.

## Execution decision

`run_pipeline.py` accepts repeated `--raster-date` arguments. Training,
validation and AOA fitting occur once; the same fitted in-memory model and AOA
are then used for every requested map period. Earth Engine is initialized once
for the multi-period production phase.

Run provenance is finalized only after field evaluation and all requested raster
products exist. SHA-256 provenance includes the final scientific rasters, their
production metadata and tile manifests, together with the model/run tables, AOA
artifacts, core figures and final field-comparison outputs. Raw tile caches and
raw-support mosaics are reproducibility caches rather than delivery products and
are not part of the final run-output hash manifest.

## Field output contract

The final field output contains the four accepted scenarios and a separate
temporal-completeness sensitivity table comparing:

- primary protocol: at least 5 of 8 valid field days;
- strict sensitivity: 8 of 8 valid field days.

The 5/8 protocol scales the mean of valid daily field-reference ET to the full
8-day MODIS period and must therefore be described as containing an implicit
representativeness assumption for missing days. The 8/8 sensitivity is retained
so this assumption is visible rather than hidden.

## Delivery contract

Generated products remain outside Git and are reproducible from the repository
plus the three canonical portable inputs and the required remote source data.
The scientific multiband rasters, run tables, diagnostics and metadata remain in
the external `current/` workspace where the production pipeline creates them.
They are not duplicated into a second delivery tree.

After the definitive run passes QA, a lightweight local finalization step creates
only `ET_fundacion_workspace/final/` with:

- `ET_2020-03-13_20m.tif`;
- `ET_2021-11-25_20m.tif`;
- `ET_2022-03-30_20m.tif`;
- `modis/MODIS_ET_2020-03-13_native.tif`;
- `modis/MODIS_ET_2021-11-25_native.tif`;
- `modis/MODIS_ET_2022-03-30_native.tif`;
- `raster_summary.csv`;
- `final_results_visualization.ipynb`.

The three ET-only GeoTIFFs are exact band-1 copies of the frozen scientific
multiband products; no ET is recalculated. The MODIS comparison GeoTIFFs retain
the native sinusoidal grid and original period ET values for native cells that
intersect the basin; they are not resampled to 20 m. `raster_summary.csv` records basic
published-support and common-three-date-support statistics. The notebook is a
read-only local visualizer and does not train, query Earth Engine, reconcile,
modify products, or export manuscript-specific figures or tables. Additional
publication graphics and tables are intentionally produced manually from the
frozen local outputs when needed.

No delivery README, separate manifest, checksum table, manuscript directory,
HTML export, duplicated scientific rasters, raw caches or tile intermediates are
created by this finalization step. Scientific provenance and SHA-256 hashes
remain in the production run's `run_metadata.json`.

## Decision

Do not run separate model fits for the three final map dates. Produce all final
maps from one definitive frozen run, finalize provenance after all scientific
outputs exist, QA that run, and only then assemble the human-facing delivery
package and final visualization notebook.
