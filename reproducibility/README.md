# Reproducibility and candidate archive

This directory contains historical experiments, audits, source-specific
exporters and candidate-predictor builders retained because they provide
reproducible evidence for the Fundación ET study.

The operational final workflow is `scripts/run_pipeline.py`; RF-25 production
code lives under `scripts/` and `src/et_downscaling/`.

A subset of the scripts here is intentionally reused by
`scripts/download_all_candidate_predictors.py` to materialize every predictor
family that was actually implemented during the study. Those files are an
audit/future-analysis archive only: they do not change the frozen 25-predictor
RF model.

Historical Ridge, HLS/FVC/albedo, S1, thermal, AOA-sensitivity and closure
experiments remain executable evidence. Their interpretation and final status
are consolidated in `docs/EXPERIMENT_HISTORY.md`; old experimental conclusions
must not be read as the current production configuration.
