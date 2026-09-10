# Candidate predictor archive

These scripts materialize predictor families that were already implemented and
audited during development. They do **not** select features, tune models, or
rerun historical model-comparison experiments.

The final model remains RF-25 with the frozen 25 predictors in
`src/et_downscaling/rf25.py`. Candidate data are written under `outputs/` for
future analysis and provenance only.
