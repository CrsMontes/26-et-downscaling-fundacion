# Operational scripts

The supported entry point on `main` is:

```powershell
python scripts/run_pipeline.py <command>
```

Commands:

- `preflight`: validate the three portable inputs and print the final scientific configuration.
- `fresh`: delete generated outputs and rebuild selection, training data, RF-25, weighted AOA/DI, final rasters and the full implemented candidate archive.
- `run`: resume/re-run the same workflow without deleting outputs first.
- `select`: reproduce the frozen Virtual10 support-selection rule.
- `extract`: reconstruct the final 25-feature GE90 training population only.
- `train`: spatial OOF + LOYO validation, final RF fit, RF-weighted DI/AOA and LPD.
- `produce`: exact-overlap RF-25 raster production for one or more dates.
- `download-candidates`: materialize all already-implemented candidate predictor families separately from final RF training.

Generated material lives below `outputs/` and is ignored by Git. No Google
Drive export is used. See `docs/METHODOLOGY.md` and
`docs/EXPERIMENT_HISTORY.md` for the scientific contract and history.
