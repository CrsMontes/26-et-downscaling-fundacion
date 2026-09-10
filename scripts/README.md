# Operational scripts

The supported entry point on `main` is:

```powershell
python scripts/run_pipeline.py <command>
```

Commands:

- `preflight`: validate the three portable inputs and print the final scientific configuration.
- `fresh`: delete generated outputs, select Virtual10, materialize the full implemented predictor archive on those 10 supports, derive the frozen 25-feature GE90 RF population, train RF-25, build weighted AOA/DI + LPD, and produce final rasters.
- `run`: resume/re-run the same workflow without deleting outputs first.
- `select`: reproduce the frozen Virtual10 support-selection rule.
- `extract`: materialize the complete Virtual10 candidate archive and derive the final frozen 25-feature GE90 training population.
- `train`: spatial OOF + LOYO validation, final RF fit, RF-weighted DI/AOA and LPD.
- `produce`: exact-overlap RF-25 raster production for one or more dates.
- `download-candidates`: materialize all already-implemented candidate predictor families on the selected Virtual10 supports; it does not fit a model.

Generated material lives below `outputs/` and is ignored by Git. No Google
Drive export is used. See `docs/METHODOLOGY.md` and
`docs/EXPERIMENT_HISTORY.md` for the scientific contract and history.
