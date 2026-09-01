"""Evaluate incremental Sentinel-1 value over the S2-common optical baseline."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


OPTICAL_FEATURES = [
    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
    "NDVI", "EVI", "SAVI", "NDWI", "NDMI",
]

SAR_FEATURES = [
    "VV_dB_mean",
    "VH_dB_mean",
    "VV_minus_VH_dB_mean",
]

RF_PARAMETERS = {
    "n_estimators": 300,
    "max_features": 0.33,
    "min_samples_leaf": 3,
    "max_depth": None,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

FOLD_KEYS = [
    "station_id",
    "period_start",
]


def calculate_kge(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    r = np.corrcoef(observed, predicted)[0, 1]
    alpha = np.std(predicted, ddof=0) / np.std(observed, ddof=0)
    beta = np.mean(predicted) / np.mean(observed)

    return float(
        1.0
        - np.sqrt(
            (r - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    )


def calculate_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - observed

    return {
        "n": len(observed),
        "R2": r2_score(observed, predicted),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "MAE": np.mean(np.abs(error)),
        "BIAS": np.mean(error),
        "KGE": calculate_kge(observed, predicted),
    }


def evaluate_population(
    population_name,
    data,
    fold_assignments,
    configurations,
):
    metric_records = []
    oof_records = []

    for split_type in ["spatial", "temporal"]:

        assignments = fold_assignments[
            (fold_assignments["threshold_pct"] == 90)
            & (fold_assignments["split_type"] == split_type)
        ][
            FOLD_KEYS + ["fold", "group"]
        ].copy()

        working = data.merge(
            assignments,
            on=FOLD_KEYS,
            how="inner",
            validate="one_to_one",
        )

        if len(working) != len(data):
            raise RuntimeError(
                f"Fold assignment mismatch for "
                f"{population_name}/{split_type}"
            )

        for configuration, features in configurations.items():

            predictions = []

            for fold in sorted(working["fold"].unique()):

                test_mask = working["fold"] == fold
                train = working.loc[~test_mask]
                test = working.loc[test_mask]

                model = RandomForestRegressor(
                    **RF_PARAMETERS
                )

                model.fit(
                    train[features].to_numpy(dtype=float),
                    train["Kc_target"].to_numpy(dtype=float),
                )

                predicted = model.predict(
                    test[features].to_numpy(dtype=float)
                )

                output = test[
                    [
                        "station_id",
                        "period_start",
                        "year",
                        "spatial_block",
                        "Kc_target",
                        "fold",
                    ]
                ].copy()

                output["population"] = population_name
                output["split_type"] = split_type
                output["configuration"] = configuration
                output["prediction"] = predicted

                predictions.append(output)

            oof = pd.concat(
                predictions,
                ignore_index=True,
            )

            metric_records.append(
                {
                    "population": population_name,
                    "split_type": split_type,
                    "configuration": configuration,
                    "n_features": len(features),
                    **calculate_metrics(
                        oof["Kc_target"],
                        oof["prediction"],
                    ),
                }
            )

            oof_records.append(oof)

    return metric_records, oof_records


def main():
    root = Path(
        "outputs/diagnostics/2020_2024"
    )

    population = pd.read_csv(
        root
        / "optical_source_experiment"
        / "population"
        / "paired_population_ge90.csv"
    )

    s1 = pd.read_csv(
        root
        / "s1_geometry_experiment"
        / "raw"
        / "s1_geometry_predictors.csv"
    )

    folds = pd.read_csv(
        root
        / "optical_source_experiment"
        / "folds"
        / "fold_assignments.csv"
    )

    merge_keys = [
        "station_id",
        "modis_pixel_id",
        "period_start",
    ]

    s1_columns = []

    for prefix in ["r077", "r142"]:
        s1_columns.extend(
            [f"{prefix}_{feature}" for feature in SAR_FEATURES]
        )
        s1_columns.append(
            f"{prefix}_products"
        )

    data = population.merge(
        s1[merge_keys + s1_columns],
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )

    optical_columns = [
        f"s2_{feature}_mean"
        for feature in OPTICAL_FEATURES
    ]

    def geometry_complete(prefix):
        columns = [
            f"{prefix}_{feature}"
            for feature in SAR_FEATURES
        ]

        return (
            data[columns].notna().all(axis=1)
            & data[columns].ne(-9999).all(axis=1)
            & data[f"{prefix}_products"].gt(0)
        )

    r077_complete = geometry_complete("r077")
    r142_complete = geometry_complete("r142")

    populations = {
        "r077": data.loc[r077_complete].copy(),
        "r142": data.loc[r142_complete].copy(),
        "both": data.loc[
            r077_complete & r142_complete
        ].copy(),
    }

    r077_columns = [
        f"r077_{feature}"
        for feature in SAR_FEATURES
    ]

    r142_columns = [
        f"r142_{feature}"
        for feature in SAR_FEATURES
    ]

    configurations = {
        "r077": {
            "s2_common": optical_columns,
            "s2_plus_r077": optical_columns + r077_columns,
        },
        "r142": {
            "s2_common": optical_columns,
            "s2_plus_r142": optical_columns + r142_columns,
        },
        "both": {
            "s2_common": optical_columns,
            "s2_plus_r077": optical_columns + r077_columns,
            "s2_plus_r142": optical_columns + r142_columns,
            "s2_plus_both": (
                optical_columns
                + r077_columns
                + r142_columns
            ),
        },
    }

    all_metrics = []
    all_oof = []

    for population_name in [
        "r077",
        "r142",
        "both",
    ]:
        metrics, oof = evaluate_population(
            population_name,
            populations[population_name],
            folds,
            configurations[population_name],
        )

        all_metrics.extend(metrics)
        all_oof.extend(oof)

    metrics = pd.DataFrame(all_metrics)
    oof = pd.concat(
        all_oof,
        ignore_index=True,
    )

    output_dir = (
        root
        / "s1_geometry_experiment"
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        output_dir / "metrics_overall.csv",
        index=False,
    )

    oof.to_csv(
        output_dir / "oof_predictions.csv",
        index=False,
    )

    print("\nS1 INCREMENTAL VALUE - GE90")
    print("===========================")

    print(
        metrics.sort_values(
            [
                "population",
                "split_type",
                "RMSE",
            ]
        ).to_string(index=False)
    )

    print()
    print("hyperparameter_tuning_performed = false")
    print("aoa_di_performed = false")
    print("production_model_saved = false")


if __name__ == "__main__":
    main()
