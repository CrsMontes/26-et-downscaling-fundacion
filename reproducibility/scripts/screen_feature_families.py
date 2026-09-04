"""Screen preregistered feature families on the fixed paired GE90 population.

This exploratory diagnostic compares incremental feature families within each
source and algorithm. It performs no tuning, Earth Engine access, AOA/DI, or
production-model training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

import screen_optical_algorithms as algorithm_screening


KEYS = ["station_id", "period_start"]
S2_COMMON = [f"s2_{name}_mean" for name in algorithm_screening.COMMON_NAMES]
HLS_COMMON = [f"hls_{name}_mean" for name in algorithm_screening.COMMON_NAMES]
ETO_DRIVERS = list(algorithm_screening.ETO_DRIVERS)
PRECIPITATION = ["Precip_period_mm", "Precip_prev30d_mm"]
SEASONALITY = ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
S2_RICH = [
    "s2_RedEdge1_mean",
    "s2_RedEdge2_mean",
    "s2_RedEdge3_mean",
    "s2_NIR_Broad_mean",
    "s2_NDRE_mean",
    "s2_Albedo_mean",
    "s2_FVC_mean",
]
SOURCE_CONFIGURATIONS = {
    "S2": {
        "s2_base": S2_COMMON + ETO_DRIVERS,
        "s2_base_plus_precip": S2_COMMON + ETO_DRIVERS + PRECIPITATION,
        "s2_base_plus_seasonality": S2_COMMON + ETO_DRIVERS + SEASONALITY,
        "s2_base_plus_full_context": S2_COMMON + ETO_DRIVERS + PRECIPITATION + SEASONALITY,
        "s2_base_plus_rich": S2_COMMON + ETO_DRIVERS + S2_RICH,
        "s2_base_plus_rich_full_context": (
            S2_COMMON + ETO_DRIVERS + S2_RICH + PRECIPITATION + SEASONALITY
        ),
    },
    "HLS": {
        "hls_base": HLS_COMMON + ETO_DRIVERS,
        "hls_base_plus_precip": HLS_COMMON + ETO_DRIVERS + PRECIPITATION,
        "hls_base_plus_seasonality": HLS_COMMON + ETO_DRIVERS + SEASONALITY,
        "hls_base_plus_full_context": HLS_COMMON + ETO_DRIVERS + PRECIPITATION + SEASONALITY,
    },
}
BASE_CONFIGURATION = {"S2": "s2_base", "HLS": "hls_base"}
SOURCE_ALGORITHMS = {
    "S2": ["random_forest", "extra_trees", "ridge"],
    "HLS": ["random_forest", "extra_trees"],
}
EXPECTED_FEATURE_COUNTS = {
    "s2_base": 16,
    "s2_base_plus_precip": 18,
    "s2_base_plus_seasonality": 20,
    "s2_base_plus_full_context": 22,
    "s2_base_plus_rich": 23,
    "s2_base_plus_rich_full_context": 29,
    "hls_base": 16,
    "hls_base_plus_precip": 18,
    "hls_base_plus_seasonality": 20,
    "hls_base_plus_full_context": 22,
}


def project_root():
    return Path(__file__).resolve().parents[1]


def key_digest(table):
    values = "\n".join(
        table.sort_values(KEYS)[KEYS].astype(str).agg("|".join, axis=1)
    )
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def load_fixed_population(root):
    diagnostic = root / "outputs" / "diagnostics" / "2020_2024"
    store_path = diagnostic / "experimental_feature_store" / "feature_store.csv"
    population_path = (
        diagnostic / "optical_source_experiment" / "population"
        / "paired_population_ge90.csv"
    )
    folds_path = (
        diagnostic / "optical_source_experiment" / "folds" / "fold_assignments.csv"
    )
    store = pd.read_csv(store_path, dtype={"station_id": str})
    population = pd.read_csv(population_path, dtype={"station_id": str})
    folds = pd.read_csv(folds_path, dtype={"station_id": str})
    for table in (store, population, folds):
        table["period_start"] = pd.to_datetime(
            table["period_start"], errors="raise"
        ).dt.strftime("%Y-%m-%d")
    if store.duplicated(KEYS).any() or len(store) != 1150:
        raise RuntimeError("Feature store must contain 1,150 unique base keys")
    if population.duplicated(KEYS).any() or len(population) != 550:
        raise RuntimeError("GE90 must contain exactly 550 unique keys")
    selected = store.merge(population[KEYS], on=KEYS, how="inner", validate="one_to_one")
    if len(selected) != 550 or key_digest(selected) != key_digest(population):
        raise RuntimeError("Feature store does not reproduce the exact GE90 keys")
    target_check = selected[KEYS + ["Kc_target"]].merge(
        population[KEYS + ["Kc_target"]],
        on=KEYS,
        suffixes=("_store", "_population"),
        validate="one_to_one",
    )
    if not np.allclose(
        target_check["Kc_target_store"],
        target_check["Kc_target_population"],
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError("Feature-store and GE90 targets differ")
    selected_folds = folds.loc[folds["threshold_pct"].eq(90)].copy()
    algorithm_screening.validate_folds(selected, selected_folds)
    return selected, selected_folds, diagnostic, store_path


def preflight_configurations(data):
    """Validate every matrix before fitting any estimator."""
    population_digest = key_digest(data)
    records = []
    for source, configurations in SOURCE_CONFIGURATIONS.items():
        for configuration, features in configurations.items():
            expected = EXPECTED_FEATURE_COUNTS[configuration]
            if len(features) != expected or len(set(features)) != expected:
                raise RuntimeError(
                    f"{configuration} must contain {expected} unique features"
                )
            missing_columns = sorted(set(features) - set(data.columns))
            if missing_columns:
                raise RuntimeError(
                    f"{configuration} is missing columns: {missing_columns}"
                )
            missing_counts = data[features].isna().sum()
            missing_counts = missing_counts.loc[missing_counts.gt(0)]
            if not missing_counts.empty:
                raise RuntimeError(
                    f"{configuration} introduces missing values: "
                    f"{missing_counts.to_dict()}"
                )
            if len(data) != 550 or key_digest(data) != population_digest:
                raise RuntimeError(f"{configuration} changed the GE90 population")
            records.append(
                {
                    "source": source,
                    "configuration": configuration,
                    "n_features": len(features),
                    "rows": len(data),
                    "unique_keys": len(data[KEYS].drop_duplicates()),
                    "key_sha256": population_digest,
                    "missing_values": 0,
                }
            )
    return pd.DataFrame(records)


def selected_algorithms():
    available = algorithm_screening.build_algorithms()
    return {
        name: available[name]
        for name in ("random_forest", "extra_trees", "ridge")
    }


def evaluate(data, folds):
    templates = selected_algorithms()
    outputs = []
    plausibility = []
    for split_type in ("spatial", "temporal"):
        assignments = folds.loc[
            folds["split_type"].eq(split_type), KEYS + ["fold"]
        ]
        working = data.merge(assignments, on=KEYS, validate="one_to_one")
        for source, configurations in SOURCE_CONFIGURATIONS.items():
            for algorithm in SOURCE_ALGORITHMS[source]:
                for configuration, features in configurations.items():
                    for fold in sorted(working["fold"].unique()):
                        test_mask = working["fold"].eq(fold)
                        train = working.loc[~test_mask]
                        test = working.loc[test_mask]
                        model = clone(templates[algorithm])
                        model.fit(train[features], train["Kc_target"])
                        prediction = model.predict(test[features])
                        output = test[
                            KEYS + ["year", "spatial_block", "fold", "Kc_target"]
                        ].copy()
                        output.insert(4, "source", source)
                        output.insert(5, "algorithm", algorithm)
                        output.insert(6, "configuration", configuration)
                        output.insert(7, "split_type", split_type)
                        output["prediction"] = prediction
                        outputs.append(output)
                        train_min = float(train["Kc_target"].min())
                        train_max = float(train["Kc_target"].max())
                        plausibility.append(
                            {
                                "source": source,
                                "algorithm": algorithm,
                                "configuration": configuration,
                                "split_type": split_type,
                                "fold": int(fold),
                                "n": len(test),
                                "training_Kc_min": train_min,
                                "training_Kc_max": train_max,
                                "prediction_min": float(np.min(prediction)),
                                "prediction_max": float(np.max(prediction)),
                                "predictions_below_zero": int((prediction < 0).sum()),
                                "predictions_outside_training_range": int(
                                    ((prediction < train_min) | (prediction > train_max)).sum()
                                ),
                            }
                        )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(plausibility)


def metric_tables(oof):
    dimensions = ["source", "algorithm", "configuration", "split_type"]
    overall = []
    for values, group in oof.groupby(dimensions, sort=True):
        overall.append(
            dict(zip(dimensions, values))
            | algorithm_screening.calculate_metrics(
                group["Kc_target"], group["prediction"]
            )
        )
    by_fold = []
    fold_dimensions = dimensions + ["fold"]
    for values, group in oof.groupby(fold_dimensions, sort=True):
        by_fold.append(
            dict(zip(fold_dimensions, values))
            | algorithm_screening.calculate_metrics(
                group["Kc_target"], group["prediction"]
            )
        )
    return pd.DataFrame(overall), pd.DataFrame(by_fold)


def metric_deltas_from_base(metrics):
    metric_names = ["R2", "RMSE", "MAE", "BIAS", "KGE"]
    records = []
    for (source, algorithm, split_type), group in metrics.groupby(
        ["source", "algorithm", "split_type"], sort=True
    ):
        indexed = group.set_index("configuration")
        base_name = BASE_CONFIGURATION[source]
        base = indexed.loc[base_name]
        for configuration, row in indexed.iterrows():
            if configuration == base_name:
                continue
            record = {
                "source": source,
                "algorithm": algorithm,
                "configuration": configuration,
                "base_configuration": base_name,
                "split_type": split_type,
                "n": int(row["n"]),
            }
            for metric in metric_names:
                record[f"configuration_{metric}"] = row[metric]
                record[f"base_{metric}"] = base[metric]
                record[f"delta_{metric}_configuration_minus_base"] = (
                    row[metric] - base[metric]
                )
            records.append(record)
    return pd.DataFrame(records)


def paired_deltas_from_base(oof):
    index = KEYS + ["year", "spatial_block", "source", "algorithm", "split_type", "fold", "Kc_target"]
    records = []
    for (source, algorithm, split_type), group in oof.groupby(
        ["source", "algorithm", "split_type"], sort=True
    ):
        wide = group.pivot(index=index, columns="configuration", values="prediction")
        base_name = BASE_CONFIGURATION[source]
        base = wide[base_name]
        for configuration in SOURCE_CONFIGURATIONS[source]:
            if configuration == base_name:
                continue
            result = wide[[configuration]].reset_index()
            variant = wide[configuration].to_numpy()
            base_values = base.to_numpy()
            observed = wide.index.get_level_values("Kc_target").to_numpy(dtype=float)
            variant_error = variant - observed
            base_error = base_values - observed
            result["configuration"] = configuration
            result["base_configuration"] = base_name
            result["prediction_configuration"] = variant
            result["prediction_base"] = base_values
            result["delta_absolute_error_configuration_minus_base"] = (
                np.abs(variant_error) - np.abs(base_error)
            )
            result["delta_squared_error_configuration_minus_base"] = (
                variant_error**2 - base_error**2
            )
            records.append(result)
    return pd.concat(records, ignore_index=True)


def manifest(data, folds, preflight, store_path):
    algorithms = selected_algorithms()
    return {
        "experiment": "exploratory_incremental_feature_family_screening",
        "period": "2020_2024",
        "population": "paired_GE90",
        "population_rows": len(data),
        "unique_keys": len(data[KEYS].drop_duplicates()),
        "key_sha256": key_digest(data),
        "feature_store": str(store_path.relative_to(project_root())).replace("\\", "/"),
        "configurations": SOURCE_CONFIGURATIONS,
        "feature_counts": EXPECTED_FEATURE_COUNTS,
        "source_algorithms": SOURCE_ALGORITHMS,
        "algorithms": {
            name: algorithm_screening.serializable_parameters(estimator)
            for name, estimator in algorithms.items()
        },
        "folds": {
            split: [
                {
                    "fold": int(fold),
                    "test_group": sorted(group["group"].astype(str).unique()),
                    "station_ids": sorted(group["station_id"].unique()),
                    "n": len(group),
                }
                for fold, group in folds.loc[folds["split_type"].eq(split)].groupby("fold", sort=True)
            ]
            for split in ("spatial", "temporal")
        },
        "preflight": preflight.to_dict("records"),
        "excluded_families": [
            "Sentinel-1", "Landsat LST", "elevation", "slope", "aspect",
            "direct ETo/ETr", "Kc_previous_modis",
        ],
        "hyperparameter_tuning_performed": False,
        "aoa_di_performed": False,
        "earth_engine_access": False,
        "production_model_trained": False,
        "winner_declared": False,
        "screening_is_exploratory": True,
    }


def main():
    root = project_root()
    data, folds, diagnostic, store_path = load_fixed_population(root)
    preflight = preflight_configurations(data)
    print(preflight.to_string(index=False))
    oof, plausibility = evaluate(data, folds)
    expected_groups = sum(
        len(SOURCE_CONFIGURATIONS[source]) * len(SOURCE_ALGORITHMS[source])
        for source in SOURCE_CONFIGURATIONS
    ) * 2
    expected_rows = expected_groups * 550
    if len(oof) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} OOF rows, found {len(oof)}")
    if oof.duplicated(["source", "algorithm", "configuration", "split_type", *KEYS]).any():
        raise RuntimeError("OOF predictions are not uniquely paired")
    overall, by_fold = metric_tables(oof)
    metric_deltas = metric_deltas_from_base(overall)
    paired_deltas = paired_deltas_from_base(oof)
    output = diagnostic / "feature_family_screening"
    output.mkdir(parents=True, exist_ok=True)
    preflight.to_csv(output / "population_preflight.csv", index=False)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    overall.to_csv(output / "metrics_overall.csv", index=False)
    by_fold.to_csv(output / "metrics_by_fold.csv", index=False)
    metric_deltas.to_csv(output / "metric_deltas_vs_base.csv", index=False)
    paired_deltas.to_csv(output / "paired_error_deltas_vs_base.csv", index=False)
    plausibility.to_csv(output / "prediction_plausibility_by_fold.csv", index=False)
    (output / "feature_family_manifest.json").write_text(
        json.dumps(manifest(data, folds, preflight, store_path), indent=2),
        encoding="utf-8",
    )
    print("\n", overall.sort_values(["source", "algorithm", "split_type", "RMSE"]).to_string(index=False))
    print(f"\nOOF rows: {len(oof)}")
    print(f"Paired delta rows: {len(paired_deltas)}")
    print(f"Output: {output}")
    print("hyperparameter_tuning_performed = false")
    print("earth_engine_access = false")
    print("winner_declared = false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
