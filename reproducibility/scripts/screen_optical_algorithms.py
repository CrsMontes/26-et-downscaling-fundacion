"""Screen preregistered regressors on the paired 2020-2024 GE90 population.

This is an exploratory diagnostic. It reuses local inputs and fixed validation
folds, performs no hyperparameter tuning, and does not save a production model.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


KEYS = ["station_id", "period_start"]
COMMON_NAMES = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
    "NDVI",
    "EVI",
    "SAVI",
    "NDWI",
    "NDMI",
]
ETO_DRIVERS = [
    "Tair_mean_C",
    "Tair_max_C",
    "VPD_mean_kPa",
    "SolarRad_MJ_m2_day",
    "Wind_mean_ms",
]
SOURCE_FEATURES = {
    source: [f"{source.lower()}_{name}_mean" for name in COMMON_NAMES]
    + ETO_DRIVERS
    for source in ("S2", "HLS")
}
EXPECTED_SPATIAL_FOLDS = {
    1: {"group": "-811_116", "stations": ["ST05"], "n": 120},
    2: {"group": "-814_118", "stations": ["ST02", "ST03"], "n": 180},
    3: {"group": "-814_119", "stations": ["ST04"], "n": 144},
    4: {"group": "-815_118", "stations": ["ST01"], "n": 106},
}
EXPECTED_TEMPORAL_FOLDS = {
    1: {"group": "2020", "n": 101},
    2: {"group": "2021", "n": 116},
    3: {"group": "2022", "n": 88},
    4: {"group": "2023", "n": 130},
    5: {"group": "2024", "n": 115},
}
RF_PARAMETERS = {
    "n_estimators": 300,
    "max_features": 0.33,
    "min_samples_leaf": 3,
    "max_depth": None,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}
EXTRA_TREES_PARAMETERS = {
    "n_estimators": 300,
    "max_features": 0.33,
    "min_samples_leaf": 3,
    "max_depth": None,
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
}
HIST_GRADIENT_BOOSTING_PARAMETERS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "early_stopping": False,
    "random_state": 42,
}
ALGORITHM_ORDER = [
    "dummy_mean",
    "ridge",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
]


def build_algorithms():
    """Return fresh preregistered estimator templates."""
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": RandomForestRegressor(**RF_PARAMETERS),
        "extra_trees": ExtraTreesRegressor(**EXTRA_TREES_PARAMETERS),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            **HIST_GRADIENT_BOOSTING_PARAMETERS
        ),
    }


def calculate_metrics(observed, predicted):
    """Calculate the fixed diagnostic metric set."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - observed
    observed_sd = observed.std(ddof=0)
    predicted_sd = predicted.std(ddof=0)
    if len(observed) < 2 or observed_sd == 0 or predicted_sd == 0:
        kge = np.nan
    else:
        correlation = np.corrcoef(observed, predicted)[0, 1]
        alpha = predicted_sd / observed_sd
        beta = predicted.mean() / observed.mean()
        kge = 1.0 - np.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    return {
        "n": len(observed),
        "R2": float(r2_score(observed, predicted)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "BIAS": float(np.mean(error)),
        "KGE": float(kge),
    }


def require_unique(table, columns, label):
    if table.duplicated(columns).any():
        raise RuntimeError(f"{label} contains duplicate keys: {columns}")


def load_inputs(project_root):
    root = project_root / "outputs" / "diagnostics" / "2020_2024"
    experiment = root / "optical_source_experiment"
    population = pd.read_csv(
        experiment / "population" / "paired_population_ge90.csv",
        dtype={"station_id": str},
    )
    meteorology = pd.read_csv(
        root / "meteorology_experiment" / "processed" / "period_meteorology.csv",
        dtype={"station_id": str},
    )
    folds = pd.read_csv(
        experiment / "folds" / "fold_assignments.csv",
        dtype={"station_id": str},
    )
    for table in (population, meteorology, folds):
        table["period_start"] = pd.to_datetime(
            table["period_start"], errors="raise"
        ).dt.strftime("%Y-%m-%d")
    require_unique(population, KEYS, "GE90 population")
    require_unique(meteorology, KEYS, "Period meteorology")
    if len(population) != 550:
        raise RuntimeError(f"Expected 550 GE90 rows, found {len(population)}")
    selected_folds = folds.loc[folds["threshold_pct"].eq(90)].copy()
    require_unique(
        selected_folds,
        ["split_type", *KEYS],
        "GE90 fold assignments",
    )
    data = population.merge(
        meteorology[KEYS + ETO_DRIVERS],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    used_columns = sorted(
        {column for columns in SOURCE_FEATURES.values() for column in columns}
    )
    data[used_columns + ["Kc_target"]] = data[
        used_columns + ["Kc_target"]
    ].apply(pd.to_numeric, errors="raise")
    if data[used_columns + ["Kc_target"]].isna().any().any():
        raise RuntimeError("The fixed GE90 matrices contain missing values")
    return data, selected_folds, experiment


def validate_folds(data, folds):
    """Require exact approved spatial and temporal test membership."""
    for split_type, expected in (
        ("spatial", EXPECTED_SPATIAL_FOLDS),
        ("temporal", EXPECTED_TEMPORAL_FOLDS),
    ):
        assignments = folds.loc[folds["split_type"].eq(split_type)]
        if len(assignments) != len(data):
            raise RuntimeError(f"Expected 550 {split_type} assignments")
        for fold, specification in expected.items():
            subset = assignments.loc[assignments["fold"].eq(fold)]
            groups = sorted(subset["group"].astype(str).unique())
            if len(subset) != specification["n"] or groups != [specification["group"]]:
                raise RuntimeError(f"Unexpected {split_type} fold {fold}")
            if split_type == "spatial":
                stations = sorted(subset["station_id"].unique())
                if stations != specification["stations"]:
                    raise RuntimeError(f"Unexpected stations in spatial fold {fold}")


def evaluate(data, folds):
    algorithms = build_algorithms()
    predictions = []
    plausibility = []
    for split_type in ("spatial", "temporal"):
        assignments = folds.loc[folds["split_type"].eq(split_type), KEYS + ["fold"]]
        working = data.merge(assignments, on=KEYS, validate="one_to_one")
        for source, features in SOURCE_FEATURES.items():
            if len(features) != 16:
                raise RuntimeError(f"Expected 16 predictors for {source}")
            for algorithm in ALGORITHM_ORDER:
                for fold in sorted(working["fold"].unique()):
                    test_mask = working["fold"].eq(fold)
                    train = working.loc[~test_mask]
                    test = working.loc[test_mask]
                    if set(train.index).intersection(test.index):
                        raise RuntimeError("Train/test row overlap")
                    model = clone(algorithms[algorithm])
                    model.fit(train[features], train["Kc_target"])
                    predicted = model.predict(test[features])
                    output = test[
                        KEYS
                        + ["year", "spatial_block", "fold", "Kc_target"]
                    ].copy()
                    output.insert(4, "source", source)
                    output.insert(5, "algorithm", algorithm)
                    output.insert(6, "split_type", split_type)
                    output["prediction"] = predicted
                    predictions.append(output)
                    train_min = float(train["Kc_target"].min())
                    train_max = float(train["Kc_target"].max())
                    plausibility.append(
                        {
                            "source": source,
                            "algorithm": algorithm,
                            "split_type": split_type,
                            "fold": int(fold),
                            "n": len(test),
                            "observed_Kc_min": float(test["Kc_target"].min()),
                            "observed_Kc_max": float(test["Kc_target"].max()),
                            "training_Kc_min": train_min,
                            "training_Kc_max": train_max,
                            "prediction_min": float(np.min(predicted)),
                            "prediction_max": float(np.max(predicted)),
                            "predictions_below_zero": int((predicted < 0).sum()),
                            "predictions_outside_training_range": int(
                                ((predicted < train_min) | (predicted > train_max)).sum()
                            ),
                        }
                    )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(plausibility)


def metric_tables(oof):
    dimensions = ["source", "algorithm", "split_type"]
    overall_rows = []
    for values, group in oof.groupby(dimensions, sort=True):
        overall_rows.append(
            dict(zip(dimensions, values))
            | calculate_metrics(group["Kc_target"], group["prediction"])
        )
    fold_dimensions = dimensions + ["fold"]
    fold_rows = []
    for values, group in oof.groupby(fold_dimensions, sort=True):
        fold_rows.append(
            dict(zip(fold_dimensions, values))
            | calculate_metrics(group["Kc_target"], group["prediction"])
        )
    return pd.DataFrame(overall_rows), pd.DataFrame(fold_rows)


def differences_from_rf(metrics):
    records = []
    metric_names = ["R2", "RMSE", "MAE", "BIAS", "KGE"]
    for (source, split_type), group in metrics.groupby(["source", "split_type"]):
        indexed = group.set_index("algorithm")
        rf = indexed.loc["random_forest"]
        for algorithm in ALGORITHM_ORDER:
            if algorithm == "random_forest":
                continue
            row = indexed.loc[algorithm]
            record = {
                "source": source,
                "algorithm": algorithm,
                "split_type": split_type,
                "n": int(row["n"]),
            }
            for metric in metric_names:
                record[f"{algorithm}_{metric}"] = row[metric]
                record[f"random_forest_{metric}"] = rf[metric]
                record[f"delta_{metric}_algorithm_minus_rf"] = row[metric] - rf[metric]
            records.append(record)
    return pd.DataFrame(records)


def paired_error_differences(oof):
    index_columns = KEYS + ["source", "split_type", "fold", "Kc_target"]
    wide = oof.pivot(index=index_columns, columns="algorithm", values="prediction")
    if wide.isna().any().any():
        raise RuntimeError("Algorithm predictions are not exactly paired")
    records = []
    for algorithm_a, algorithm_b in combinations(ALGORITHM_ORDER, 2):
        pair = wide[[algorithm_a, algorithm_b]].reset_index()
        error_a = pair[algorithm_a] - pair["Kc_target"]
        error_b = pair[algorithm_b] - pair["Kc_target"]
        result = pair[index_columns].copy()
        result["algorithm_a"] = algorithm_a
        result["algorithm_b"] = algorithm_b
        result["prediction_a"] = pair[algorithm_a]
        result["prediction_b"] = pair[algorithm_b]
        result["absolute_error_a"] = error_a.abs()
        result["absolute_error_b"] = error_b.abs()
        result["delta_absolute_error_a_minus_b"] = error_a.abs() - error_b.abs()
        result["squared_error_a"] = error_a**2
        result["squared_error_b"] = error_b**2
        result["delta_squared_error_a_minus_b"] = error_a**2 - error_b**2
        records.append(result)
    return pd.concat(records, ignore_index=True)


def serializable_parameters(estimator):
    return {
        key: value
        for key, value in estimator.get_params(deep=True).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def build_manifest(data, folds):
    algorithms = build_algorithms()
    fold_manifest = {}
    for split_type in ("spatial", "temporal"):
        rows = []
        subset = folds.loc[folds["split_type"].eq(split_type)]
        for fold, group in subset.groupby("fold", sort=True):
            rows.append(
                {
                    "fold": int(fold),
                    "test_group": sorted(group["group"].astype(str).unique()),
                    "station_ids": sorted(group["station_id"].unique()),
                    "n": len(group),
                }
            )
        fold_manifest[split_type] = rows
    return {
        "experiment": "exploratory_algorithm_screening",
        "period": "2020_2024",
        "population": "paired_population_ge90",
        "population_rows": len(data),
        "unique_station_period_keys": int(data[KEYS].drop_duplicates().shape[0]),
        "target": "Kc_target",
        "sources": list(SOURCE_FEATURES),
        "features": SOURCE_FEATURES,
        "feature_count_per_source": {source: len(features) for source, features in SOURCE_FEATURES.items()},
        "folds": fold_manifest,
        "algorithms": {
            name: serializable_parameters(estimator)
            for name, estimator in algorithms.items()
        },
        "hyperparameter_tuning_performed": False,
        "aoa_di_performed": False,
        "production_model_trained": False,
        "earth_engine_access": False,
        "persistence_in_primary_ranking": False,
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    data, folds, experiment = load_inputs(project_root)
    validate_folds(data, folds)
    oof, plausibility = evaluate(data, folds)
    expected_oof_rows = 550 * 2 * len(ALGORITHM_ORDER) * 2
    if len(oof) != expected_oof_rows:
        raise RuntimeError(f"Expected {expected_oof_rows} OOF rows, found {len(oof)}")
    require_unique(
        oof,
        ["source", "algorithm", "split_type", *KEYS],
        "OOF predictions",
    )
    overall, by_fold = metric_tables(oof)
    versus_rf = differences_from_rf(overall)
    paired = paired_error_differences(oof)
    output = experiment / "algorithm_screening"
    output.mkdir(parents=True, exist_ok=True)
    overall.to_csv(output / "metrics_overall.csv", index=False)
    by_fold.to_csv(output / "metrics_by_fold.csv", index=False)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    versus_rf.to_csv(output / "differences_vs_random_forest.csv", index=False)
    paired.to_csv(output / "paired_error_differences.csv", index=False)
    plausibility.to_csv(output / "prediction_plausibility_by_fold.csv", index=False)
    (output / "screening_manifest.json").write_text(
        json.dumps(build_manifest(data, folds), indent=2), encoding="utf-8"
    )
    print(overall.sort_values(["source", "split_type", "RMSE"]).to_string(index=False))
    print(f"\nOOF rows: {len(oof)}")
    print(f"Paired error rows: {len(paired)}")
    print(f"Output: {output}")
    print("hyperparameter_tuning_performed = false")
    print("production_model_trained = false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
