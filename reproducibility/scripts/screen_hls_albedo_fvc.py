"""Screen HLS Albedo and fold-aware recalibrated FVC on paired GE90."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

import screen_feature_families as families
import screen_optical_algorithms as algorithms
import export_recalibrated_fvc_predictors as fvc_export


KEYS = ["station_id", "period_start"]
BASE = [f"hls_{name}_mean" for name in algorithms.COMMON_NAMES] + list(algorithms.ETO_DRIVERS) + list(families.SEASONALITY)
CONFIGURATIONS = {
    "HLS_BASE": [], "HLS_BASE_ALBEDO": ["hls_Albedo_mean"],
    "HLS_BASE_FVC": ["__fold_fvc__"],
    "HLS_BASE_ALBEDO_FVC": ["hls_Albedo_mean", "__fold_fvc__"],
}
ALGORITHMS = ("random_forest", "extra_trees")


def project_root(): return Path(__file__).resolve().parents[1]


def load_data():
    root = project_root(); diagnostic = root / "outputs/diagnostics/2020_2024"
    store = pd.read_csv(diagnostic / "experimental_feature_store/feature_store.csv", dtype={"station_id": str})
    population = pd.read_csv(diagnostic / "optical_source_experiment/population/paired_population_ge90.csv", dtype={"station_id": str})
    folds = pd.read_csv(diagnostic / "optical_source_experiment/folds/fold_assignments.csv", dtype={"station_id": str})
    recalibrated = pd.read_csv(diagnostic / "recalibrated_fvc_predictors/analysis/hls_recalibrated_fvc_station_period.csv", dtype={"station_id": str})
    for table in (store, population, folds, recalibrated):
        table.period_start = pd.to_datetime(table.period_start).dt.strftime("%Y-%m-%d")
    data = store.merge(population[KEYS], on=KEYS, validate="one_to_one").merge(
        recalibrated.drop(columns=["source", "year"], errors="ignore"), on=KEYS, validate="one_to_one")
    selected_folds = folds.loc[folds.threshold_pct.eq(90)].copy()
    algorithms.validate_folds(data, selected_folds)
    required = BASE + ["hls_Albedo_mean", "fvc_global_2020_2024_mean", "Kc_target"]
    required += [f"fvc_spatial_train_excl_fold{k}_mean" for k in range(1, 5)]
    required += [f"fvc_temporal_train_excl_{year}_mean" for year in range(2020, 2025)]
    data[required] = data[required].apply(pd.to_numeric, errors="coerce")
    population_audit = {
        "paired_ge90": len(data),
        "hls_albedo_valid": int(data.hls_Albedo_mean.notna().sum()),
        "hls_fvc_global_valid": int(data.fvc_global_2020_2024_mean.notna().sum()),
        "both_valid": int((data.hls_Albedo_mean.notna() & data.fvc_global_2020_2024_mean.notna()).sum()),
    }
    if population_audit != {"paired_ge90": 550, "hls_albedo_valid": 550,
                            "hls_fvc_global_valid": 550, "both_valid": 550}:
        raise RuntimeError(f"Paired screening population changed: {population_audit}")
    if data[required].isna().any().any(): raise RuntimeError("Screening matrix contains missing values")
    return data, selected_folds, diagnostic, population_audit


def clipping_extremes():
    root = fvc_export.output_root() / "analysis"; rows = []
    for source in fvc_export.SOURCES:
        table = pd.read_csv(root / f"{source.lower()}_recalibrated_fvc_station_period.csv", dtype={"station_id": str})
        table["source"] = source
        table["delta_FVC"] = table.fvc_global_2020_2024_mean - table.fvc_historical_mean
        table["abs_delta_FVC"] = table.delta_FVC.abs()
        table["delta_clipped_low"] = table.fvc_global_2020_2024_clipped_low_fraction - table.fvc_historical_clipped_low_fraction
        table["delta_clipped_high"] = table.fvc_global_2020_2024_clipped_high_fraction - table.fvc_historical_clipped_high_fraction
        definitions = (("clipped_low", "delta_clipped_low"), ("clipped_high", "delta_clipped_high"), ("FVC", "delta_FVC"))
        for ranking, column in definitions:
            selected = table.loc[table[column].notna()].assign(_absolute=table[column].abs()).nlargest(10, "_absolute")
            selected["ranking"] = f"top10_abs_delta_{ranking}"
            rows.append(selected)
    columns = ["ranking", "station_id", "period_start", "source", "valid_pixel_count",
        "optical_coverage_pct", "fvc_historical_mean", "fvc_global_2020_2024_mean", "delta_FVC",
        "fvc_historical_clipped_low_fraction", "fvc_global_2020_2024_clipped_low_fraction",
        "fvc_historical_clipped_high_fraction", "fvc_global_2020_2024_clipped_high_fraction",
        "delta_clipped_low", "delta_clipped_high"]
    return pd.concat(rows, ignore_index=True)[columns]


def fold_fvc_column(split_type, fold, global_sensitivity=False):
    if global_sensitivity: return "fvc_global_2020_2024_mean"
    return (f"fvc_spatial_train_excl_fold{fold}_mean" if split_type == "spatial"
            else f"fvc_temporal_train_excl_{2019 + fold}_mean")


def evaluate(data, folds):
    templates = algorithms.build_algorithms(); outputs, plausibility = [], []
    calibration = fvc_export.freeze_calibrations()
    for split_type in ("spatial", "temporal"):
        assignments = folds.loc[folds.split_type.eq(split_type), KEYS + ["fold"]]
        working = data.merge(assignments, on=KEYS, validate="one_to_one")
        for algorithm in ALGORITHMS:
            for configuration, extras in CONFIGURATIONS.items():
                modes = ("training_only", "global_sensitivity") if "__fold_fvc__" in extras else ("not_applicable",)
                for mode in modes:
                    for fold in sorted(working.fold.unique()):
                        fvc_column = fold_fvc_column(split_type, int(fold), mode == "global_sensitivity")
                        features = BASE + [fvc_column if item == "__fold_fvc__" else item for item in extras]
                        test_mask = working.fold.eq(fold); train, test = working.loc[~test_mask], working.loc[test_mask]
                        if mode == "training_only" and "__fold_fvc__" in extras:
                            record = (calibration["sources"]["HLS"]["spatial_training_only"][str(fold)]
                                      if split_type == "spatial" else calibration["sources"]["HLS"]["temporal_training_only"][str(2019 + fold)])
                            if split_type == "spatial" and train.station_id.isin(record["excluded_station_ids"]).any():
                                raise RuntimeError("Spatial calibration exclusion disagrees with fold")
                            if split_type == "temporal" and train.year.eq(record["excluded_year"]).any():
                                raise RuntimeError("Temporal calibration exclusion disagrees with fold")
                        model = clone(templates[algorithm]); model.fit(train[features], train.Kc_target)
                        prediction = model.predict(test[features])
                        output = test[KEYS + ["year", "spatial_block", "fold", "Kc_target"]].copy()
                        output["algorithm"] = algorithm; output["configuration"] = configuration
                        output["fvc_mode"] = mode; output["fvc_column"] = fvc_column if "__fold_fvc__" in extras else ""
                        output["split_type"] = split_type; output["prediction"] = prediction; outputs.append(output)
                        train_min, train_max = train.Kc_target.min(), train.Kc_target.max()
                        plausibility.append({"algorithm": algorithm, "configuration": configuration,
                            "fvc_mode": mode, "split_type": split_type, "fold": int(fold), "n": len(test),
                            "prediction_min": prediction.min(), "prediction_max": prediction.max(),
                            "negative_predictions": int((prediction < 0).sum()),
                            "outside_training_target_range": int(((prediction < train_min) | (prediction > train_max)).sum())})
    oof = pd.concat(outputs, ignore_index=True)
    if oof.duplicated(["algorithm", "configuration", "fvc_mode", "split_type", *KEYS]).any():
        raise RuntimeError("OOF predictions are not unique")
    return oof, pd.DataFrame(plausibility)


def metrics(oof):
    dimensions = ["algorithm", "configuration", "fvc_mode", "split_type"]
    overall = [dict(zip(dimensions, values)) | algorithms.calculate_metrics(group.Kc_target, group.prediction)
               for values, group in oof.groupby(dimensions, sort=True)]
    by_fold = [dict(zip(dimensions + ["fold"], values)) | algorithms.calculate_metrics(group.Kc_target, group.prediction)
               for values, group in oof.groupby(dimensions + ["fold"], sort=True)]
    return pd.DataFrame(overall), pd.DataFrame(by_fold)


def effect_deltas(metrics_table, fold_table):
    comparisons = {
        "albedo_effect": ("HLS_BASE_ALBEDO", "HLS_BASE"),
        "fvc_effect": ("HLS_BASE_FVC", "HLS_BASE"),
        "joint_effect": ("HLS_BASE_ALBEDO_FVC", "HLS_BASE"),
        "incremental_fvc_given_albedo": ("HLS_BASE_ALBEDO_FVC", "HLS_BASE_ALBEDO"),
        "incremental_albedo_given_fvc": ("HLS_BASE_ALBEDO_FVC", "HLS_BASE_FVC"),
    }
    rows = []
    main = metrics_table.loc[~metrics_table.fvc_mode.eq("global_sensitivity")]
    folds = fold_table.loc[~fold_table.fvc_mode.eq("global_sensitivity")]
    for algorithm in ALGORITHMS:
        for split_type in ("spatial", "temporal"):
            group = main.loc[(main.algorithm.eq(algorithm)) & main.split_type.eq(split_type)].set_index("configuration")
            fold_group = folds.loc[(folds.algorithm.eq(algorithm)) & folds.split_type.eq(split_type)]
            for effect, (variant, reference) in comparisons.items():
                v, r = group.loc[variant], group.loc[reference]
                paired = fold_group.loc[fold_group.configuration.isin([variant, reference])].pivot(index="fold", columns="configuration")
                simultaneous = ((paired.RMSE[variant] < paired.RMSE[reference]) & (paired.MAE[variant] < paired.MAE[reference])).sum()
                row = {"algorithm": algorithm, "split_type": split_type, "effect": effect,
                       "variant": variant, "reference": reference, "n": int(v.n),
                       "folds_improving_RMSE_and_MAE": int(simultaneous), "folds_total": len(paired)}
                for metric in ("RMSE", "MAE", "R2", "BIAS", "KGE"):
                    row[f"delta_{metric}_variant_minus_reference"] = v[metric] - r[metric]
                rows.append(row)
    return pd.DataFrame(rows)


def global_sensitivity(metrics_table):
    rows = []
    for algorithm in ALGORITHMS:
        for configuration in ("HLS_BASE_FVC", "HLS_BASE_ALBEDO_FVC"):
            for split_type in ("spatial", "temporal"):
                subset = metrics_table.loc[(metrics_table.algorithm.eq(algorithm)) & metrics_table.configuration.eq(configuration) & metrics_table.split_type.eq(split_type)].set_index("fvc_mode")
                row = {"algorithm": algorithm, "configuration": configuration, "split_type": split_type}
                for metric in ("RMSE", "MAE", "R2", "BIAS", "KGE"):
                    row[f"delta_{metric}_global_minus_training_only"] = subset.loc["global_sensitivity", metric] - subset.loc["training_only", metric]
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    data, folds, diagnostic, population = load_data(); extremes = clipping_extremes()
    oof, plausibility = evaluate(data, folds); overall, by_fold = metrics(oof)
    deltas = effect_deltas(overall, by_fold); sensitivity = global_sensitivity(overall)
    output = diagnostic / "hls_albedo_fvc_screening"; output.mkdir(parents=True, exist_ok=True)
    extremes.to_csv(output / "clipping_extreme_rows.csv", index=False)
    oof.to_csv(output / "oof_predictions.csv", index=False); overall.to_csv(output / "metrics_overall.csv", index=False)
    by_fold.to_csv(output / "metrics_by_fold.csv", index=False); deltas.to_csv(output / "effect_deltas.csv", index=False)
    sensitivity.to_csv(output / "global_vs_training_only_fvc_sensitivity.csv", index=False)
    plausibility.to_csv(output / "prediction_plausibility.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({"population": population, "features_base": BASE,
        "configurations": CONFIGURATIONS, "algorithms": {name: algorithms.serializable_parameters(algorithms.build_algorithms()[name]) for name in ALGORITHMS},
        "earth_engine_access": False, "tuning": False, "screening_exploratory": True,
        "fvc_validation_rule": "same training-only transform column for train and test within each fold"}, indent=2), encoding="utf-8")
    print(overall.to_string(index=False)); print(deltas.to_string(index=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
