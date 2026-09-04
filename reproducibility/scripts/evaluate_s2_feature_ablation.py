"""Evaluate the added predictive value of Sentinel-2-specific optical features.

Primary population:
- paired optical population with >=90% coverage;
- identical target and folds used in the S2/HLS common-feature experiment;
- fixed Random Forest configuration;
- no hyperparameter tuning;
- no AOA/DI filtering;
- no production-model selection.

FVC is treated as exploratory because its calibration must be reassessed
for the expanded 2020-2024 analysis period.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


COMMON = [
    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
    "NDVI", "EVI", "SAVI", "NDWI", "NDMI",
]

CONFIGURATIONS = {
    "s2_common": [],
    "s2_plus_rededge_bands": [
        "RedEdge1",
        "RedEdge2",
        "RedEdge3",
        "NIR_Broad",
    ],
    "s2_plus_ndre": [
        "NDRE",
    ],
    "s2_plus_albedo": [
        "Albedo",
    ],
    "s2_plus_fvc_exploratory": [
        "FVC",
    ],
    "s2_full_rich_exploratory": [
        "RedEdge1",
        "RedEdge2",
        "RedEdge3",
        "NIR_Broad",
        "NDRE",
        "Albedo",
        "FVC",
    ],
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

KEYS = [
    "station_id",
    "period_start",
]


def feature_columns(extra_features):
    return [
        f"s2_{name}_mean"
        for name in COMMON + extra_features
    ]


def calculate_kge(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    r = np.corrcoef(observed, predicted)[0, 1]
    alpha = np.std(predicted, ddof=0) / np.std(observed, ddof=0)
    beta = np.mean(predicted) / np.mean(observed)

    return float(
        1.0 - np.sqrt(
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


def main():
    root = Path(
        "outputs/diagnostics/2020_2024/"
        "optical_source_experiment"
    )

    population = pd.read_csv(
        root / "population/paired_population_ge90.csv"
    )

    rich = pd.read_csv(
        root / "raw/s2_rich_optical.csv"
    )

    folds = pd.read_csv(
        root / "folds/fold_assignments.csv"
    )

    rich_columns = sorted({
        column
        for features in CONFIGURATIONS.values()
        for column in feature_columns(features)
        if column not in population.columns
    })

    merge_keys = [
        "station_id",
        "modis_pixel_id",
        "period_start",
    ]

    if rich_columns:
        population = population.merge(
            rich[merge_keys + rich_columns],
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )

    output_records = []
    oof_records = []

    for split_type in ["spatial", "temporal"]:

        assignment = folds[
            (folds["threshold_pct"] == 90)
            & (folds["split_type"] == split_type)
        ][KEYS + ["fold", "group"]].copy()

        data = population.merge(
            assignment,
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )

        if len(data) != len(population):
            raise RuntimeError(
                f"Fold assignment mismatch for {split_type}."
            )

        for configuration, extras in CONFIGURATIONS.items():

            columns = feature_columns(extras)

            if data[columns].isna().any().any():
                raise RuntimeError(
                    f"Missing predictors in {configuration}."
                )

            configuration_oof = []

            for fold in sorted(data["fold"].unique()):

                test_mask = data["fold"] == fold

                train = data.loc[~test_mask]
                test = data.loc[test_mask]

                model = RandomForestRegressor(
                    **RF_PARAMETERS
                )

                model.fit(
                    train[columns].to_numpy(dtype=float),
                    train["Kc_target"].to_numpy(dtype=float),
                )

                prediction = model.predict(
                    test[columns].to_numpy(dtype=float)
                )

                fold_output = test[
                    [
                        "station_id",
                        "modis_pixel_id",
                        "period_start",
                        "year",
                        "spatial_block",
                        "Kc_target",
                        "fold",
                    ]
                ].copy()

                fold_output["split_type"] = split_type
                fold_output["configuration"] = configuration
                fold_output["prediction"] = prediction

                configuration_oof.append(
                    fold_output
                )

            configuration_oof = pd.concat(
                configuration_oof,
                ignore_index=True,
            )

            metrics = calculate_metrics(
                configuration_oof["Kc_target"],
                configuration_oof["prediction"],
            )

            output_records.append({
                "split_type": split_type,
                "configuration": configuration,
                "n_features": len(columns),
                **metrics,
            })

            oof_records.append(
                configuration_oof
            )

    metrics = pd.DataFrame(output_records)

    oof = pd.concat(
        oof_records,
        ignore_index=True,
    )

    # Differences relative to S2-common within each validation design.
    baseline = (
        metrics[
            metrics["configuration"] == "s2_common"
        ]
        .set_index("split_type")
    )

    for metric in [
        "R2",
        "RMSE",
        "MAE",
        "BIAS",
        "KGE",
    ]:
        metrics[f"delta_{metric}_vs_common"] = metrics.apply(
            lambda row:
                row[metric]
                - baseline.loc[row["split_type"], metric],
            axis=1,
        )

    output_dir = root / "s2_ablation"
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

    with (
        output_dir / "ablation_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "threshold_pct": 90,
                "configurations": CONFIGURATIONS,
                "rf_parameters": RF_PARAMETERS,
                "same_population": True,
                "same_folds": True,
                "hyperparameter_tuning_performed": False,
                "aoa_di_performed": False,
                "production_model_saved": False,
                "fvc_status": "exploratory_pending_period_recalibration_review",
            },
            file,
            indent=2,
        )

    print("\nS2 FEATURE ABLATION - GE90")
    print("==========================")
    print(
        metrics.sort_values(
            ["split_type", "RMSE"]
        ).to_string(index=False)
    )

    print()
    print("hyperparameter_tuning_performed = false")
    print("production_model_saved = false")


if __name__ == "__main__":
    main()
