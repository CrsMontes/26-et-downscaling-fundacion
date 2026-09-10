# Candidate predictor archive

These scripts materialize every predictor family already implemented and audited during development on the **10 selected Virtual10 MODIS supports** for 2020–2024. They do **not** select features, tune models, or rerun historical model-comparison experiments.

The support source is injected by `scripts/download_all_candidate_predictors.py` from `outputs/training/selection/virtual_points.geojson`. The five real field stations are not used for this archive and remain external to model training.

The final model remains RF-25 with the frozen 25 predictors in `src/et_downscaling/rf25.py`. The complete candidate master is kept below `outputs/current/master/` and copied to `outputs/training/master/` before the RF-25 GE90 population is derived. Missingness in unused candidates is not an eligibility gate for RF-25.
