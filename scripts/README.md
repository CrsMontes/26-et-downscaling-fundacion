# Operational scripts

The supported entry point on `main` is:

```powershell
python scripts/run_pipeline.py <command>
```

Commands:

- `preflight`: validate the RF-25 inputs, report the optional field input and print the final scientific configuration.
- `fresh`: delete generated outputs, select Virtual10, materialize only the RF-25 sources, derive the frozen 25-feature GE90 population, train RF-25, build weighted AOA/DI + LPD, produce illustrative rasters and write provenance.
- `run`: resume/re-run the same workflow without deleting outputs first.
- `select`: reproduce the frozen Virtual10 support-selection rule.
- `extract`: materialize the RF-25 Sentinel-2/MODIS/ERA5-Land sources and derive the frozen 25-feature GE90 training population. `--force` rebuilds valid cached exports.
- `train`: spatial OOF + LOYO validation, final RF fit, RF-weighted DI/AOA and LPD.
- `produce`: exact-overlap RF-25 raster production for one or more dates.
- `provenance`: hash the inputs and existing RF-25 products and write the run manifest without downloading data.
- `download-candidates`: materialize all already-implemented candidate predictor families on the selected Virtual10 supports; it does not fit a model.

Generated material lives below `outputs/` and is ignored by Git. No Google
Drive export is used. See `docs/METHODOLOGY.md` and
`docs/EXPERIMENT_HISTORY.md` for the scientific contract and history.
