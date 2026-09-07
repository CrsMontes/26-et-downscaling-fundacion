# Reproducibility scripts

These scripts are intentionally outside the operational `scripts/` directory.
They reproduce diagnostics, model-selection experiments, predictor-screening
experiments, legacy field analyses and superseded production paths.

They are retained as scientific provenance and must not be imported or
executed by `scripts/run_pipeline.py`.

Use `../script_manifest.md` to identify their role.


The current S2-only FVC/albedo decision audit is
`recheck_s2_fvc_albedo.py`. It is diagnostic only, does not modify the final
Ridge-25 specification, and intentionally excludes HLS.

`build_candidate_master.py` is retained here only to reconstruct the historical
five-year candidate predictor store. It is not part of the accepted operational
Ridge-25 workflow.


Final audit evidence is retained in `run_closure_diagnostics.py` and
`run_aoa_map_sensitivity.py`. The latter is intentionally a diagnostic and can
make many direct Earth Engine tile downloads; it must not be used as a routine
production command.
