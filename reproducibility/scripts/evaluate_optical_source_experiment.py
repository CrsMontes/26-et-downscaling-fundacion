"""Evaluate Sentinel-2 versus HLS on an exactly paired population.

The experiment uses:
- the same observations for both optical sources;
- the same target (Kc_target);
- the same 11 common optical predictors;
- the same spatial and temporal folds;
- fixed Random Forest hyperparameters;
- no hyperparameter tuning;
- no AOA/DI filtering;
- no production-model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


COMMON_PREDICTORS = [
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

RF_PARAMETERS = {
    "n_estimators": 300,
    "max_features": 0.33,
    "min_samples_leaf": 3,
    "max_depth": None,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

KEY_COLUMNS = [
    "station_id",
    "period_start",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired S2-common versus HLS-common "
            "with spatial and temporal out-of-fold validation."
        )
    )
    parser.add_argument(
        "--period-label",
        default="2020_2024",
    )
    return parser.parse_args()


def project_root():
    return Path(__file__).resolve().parents[2]


def calculate_kge(observed, predicted):
    observed = np.asarray(
        observed,
        dtype=float,
    )
    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    observed_mean = observed.mean()
    predicted_mean = predicted.mean()

    observed_std = observed.std(ddof=0)
    predicted_std = predicted.std(ddof=0)

    observed_centered = observed - observed_mean
    predicted_centered = predicted - predicted_mean

    denominator = np.sqrt(
        np.sum(observed_centered ** 2)
        * np.sum(predicted_centered ** 2)
    )

    if (
        denominator == 0
        or observed_mean == 0
        or observed_std == 0
    ):
        return np.nan

    correlation = (
        np.sum(
            observed_centered
            * predicted_centered
        )
        / denominator
    )

    alpha = (
        predicted_std
        / observed_std
    )

    beta = (
        predicted_mean
        / observed_mean
    )

    return float(
        1.0
        - np.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    )


def calculate_metrics(
    observed,
    predicted,
):
    observed = np.asarray(
        observed,
        dtype=float,
    )
    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    residual = (
        predicted
        - observed
    )

    return {
        "n": int(len(observed)),
        "R2": float(
            r2_score(
                observed,
                predicted,
            )
        ),
        "RMSE": float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        ),
        "MAE": float(
            np.mean(
                np.abs(
                    residual
                )
            )
        ),
        "BIAS": float(
            np.mean(
                residual
            )
        ),
        "KGE": calculate_kge(
            observed,
            predicted,
        ),
    }


def make_model():
    return RandomForestRegressor(
        **RF_PARAMETERS
    )


def feature_columns(source):
    prefix = (
        "s2"
        if source == "S2"
        else "hls"
    )

    return [
        f"{prefix}_{name}_mean"
        for name in COMMON_PREDICTORS
    ]


def validate_population(
    population,
    threshold,
):
    if population.duplicated(
        KEY_COLUMNS
    ).any():
        raise RuntimeError(
            f"Duplicate population keys at threshold {threshold}."
        )

    required = (
        ["Kc_target"]
        + feature_columns("S2")
        + feature_columns("HLS")
    )

    missing_columns = [
        column
        for column in required
        if column not in population.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    numeric = (
        population[required]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    if numeric.isna().any().any():
        bad = (
            numeric.columns[
                numeric.isna().any()
            ]
            .tolist()
        )
        raise RuntimeError(
            "Non-finite required values in: "
            + ", ".join(
                bad
            )
        )

    if (
        numeric
        .eq(-9999)
        .any()
        .any()
    ):
        bad = (
            numeric.columns[
                numeric.eq(-9999).any()
            ]
            .tolist()
        )
        raise RuntimeError(
            "Sentinel missing-value code found in: "
            + ", ".join(
                bad
            )
        )


def validate_fold_counts(
    definitions,
    threshold,
    split_type,
    fold,
    train_rows,
    test_rows,
):
    expected = definitions[
        (definitions["threshold_pct"] == threshold)
        & (definitions["split_type"] == split_type)
        & (definitions["fold"] == fold)
    ]

    if len(expected) != 1:
        raise RuntimeError(
            "Fold definition is missing or duplicated: "
            f"threshold={threshold}, "
            f"split={split_type}, "
            f"fold={fold}"
        )

    row = expected.iloc[0]

    if (
        int(row["train_rows"]) != train_rows
        or int(row["test_rows"]) != test_rows
    ):
        raise RuntimeError(
            "Fold population count mismatch: "
            f"threshold={threshold}, "
            f"split={split_type}, "
            f"fold={fold}"
        )


def metrics_by_group(
    oof,
    group_column,
):
    records = []

    grouping_columns = [
        "threshold_pct",
        "split_type",
        group_column,
    ]

    for keys, group in oof.groupby(
        grouping_columns,
        dropna=False,
    ):
        threshold, split_type, group_value = keys

        for source, prediction_column in [
            ("S2", "s2_prediction"),
            ("HLS", "hls_prediction"),
        ]:
            record = {
                "threshold_pct": int(
                    threshold
                ),
                "split_type": (
                    split_type
                ),
                group_column: (
                    group_value
                ),
                "source": source,
            }

            record.update(
                calculate_metrics(
                    group["Kc_target"],
                    group[
                        prediction_column
                    ],
                )
            )

            records.append(
                record
            )

    return pd.DataFrame(
        records
    )


def paired_error_summary(oof):
    records = []

    for (
        threshold,
        split_type,
    ), group in oof.groupby(
        [
            "threshold_pct",
            "split_type",
        ]
    ):
        observed = (
            group["Kc_target"]
            .to_numpy(
                dtype=float
            )
        )

        s2_prediction = (
            group["s2_prediction"]
            .to_numpy(
                dtype=float
            )
        )

        hls_prediction = (
            group["hls_prediction"]
            .to_numpy(
                dtype=float
            )
        )

        s2_error = (
            s2_prediction
            - observed
        )

        hls_error = (
            hls_prediction
            - observed
        )

        s2_absolute = np.abs(
            s2_error
        )

        hls_absolute = np.abs(
            hls_error
        )

        difference = (
            s2_absolute
            - hls_absolute
        )

        squared_difference = (
            s2_error ** 2
            - hls_error ** 2
        )

        records.append(
            {
                "threshold_pct": int(
                    threshold
                ),
                "split_type": (
                    split_type
                ),
                "n": int(
                    len(group)
                ),
                "mean_abs_error_difference_s2_minus_hls": float(
                    difference.mean()
                ),
                "median_abs_error_difference_s2_minus_hls": float(
                    np.median(
                        difference
                    )
                ),
                "mean_squared_error_difference_s2_minus_hls": float(
                    squared_difference.mean()
                ),
                "s2_lower_absolute_error_pct": float(
                    100.0
                    * np.mean(
                        s2_absolute
                        < hls_absolute
                    )
                ),
                "hls_lower_absolute_error_pct": float(
                    100.0
                    * np.mean(
                        hls_absolute
                        < s2_absolute
                    )
                ),
                "equal_absolute_error_pct": float(
                    100.0
                    * np.mean(
                        np.isclose(
                            s2_absolute,
                            hls_absolute,
                            rtol=0.0,
                            atol=1e-12,
                        )
                    )
                ),
            }
        )

    return pd.DataFrame(
        records
    )


def main():
    args = parse_arguments()

    root = (
        project_root()
        / "outputs"
        / "diagnostics"
        / args.period_label
        / "optical_source_experiment"
    )

    population_dir = (
        root
        / "population"
    )

    folds_dir = (
        root
        / "folds"
    )

    output_dir = (
        root
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_assignments = pd.read_csv(
        folds_dir
        / "fold_assignments.csv"
    )

    fold_definitions = pd.read_csv(
        folds_dir
        / "fold_definitions.csv"
    )

    all_oof = []

    for threshold in [
        80,
        90,
        99,
    ]:
        population = pd.read_csv(
            population_dir
            / (
                f"paired_population_ge"
                f"{threshold}.csv"
            )
        )

        validate_population(
            population,
            threshold,
        )

        for split_type in [
            "spatial",
            "temporal",
        ]:
            assignments = (
                fold_assignments[
                    (
                        fold_assignments[
                            "threshold_pct"
                        ]
                        == threshold
                    )
                    & (
                        fold_assignments[
                            "split_type"
                        ]
                        == split_type
                    )
                ][
                    KEY_COLUMNS
                    + [
                        "group",
                        "fold",
                    ]
                ]
                .copy()
            )

            if assignments.duplicated(
                KEY_COLUMNS
            ).any():
                raise RuntimeError(
                    "Duplicate fold assignments: "
                    f"threshold={threshold}, "
                    f"split={split_type}"
                )

            data = population.merge(
                assignments,
                on=KEY_COLUMNS,
                how="inner",
                validate="one_to_one",
            )

            if len(data) != len(
                population
            ):
                raise RuntimeError(
                    "Population/fold assignment "
                    "key mismatch: "
                    f"threshold={threshold}, "
                    f"split={split_type}"
                )

            for fold in sorted(
                data["fold"].unique()
            ):
                test_mask = (
                    data["fold"]
                    == fold
                )

                train = data.loc[
                    ~test_mask
                ].copy()

                test = data.loc[
                    test_mask
                ].copy()

                validate_fold_counts(
                    fold_definitions,
                    threshold,
                    split_type,
                    int(fold),
                    len(train),
                    len(test),
                )

                observed_test = (
                    pd.to_numeric(
                        test["Kc_target"]
                    )
                    .to_numpy(
                        dtype=float
                    )
                )

                predictions = {}

                for source in [
                    "S2",
                    "HLS",
                ]:
                    columns = (
                        feature_columns(
                            source
                        )
                    )

                    x_train = (
                        train[columns]
                        .to_numpy(
                            dtype=float
                        )
                    )

                    y_train = (
                        pd.to_numeric(
                            train[
                                "Kc_target"
                            ]
                        )
                        .to_numpy(
                            dtype=float
                        )
                    )

                    x_test = (
                        test[columns]
                        .to_numpy(
                            dtype=float
                        )
                    )

                    model = make_model()

                    model.fit(
                        x_train,
                        y_train,
                    )

                    predictions[
                        source
                    ] = model.predict(
                        x_test
                    )

                keep_columns = [
                    column
                    for column in [
                        "station",
                        "station_id",
                        "modis_pixel_id",
                        "period_start",
                        "period_end",
                        "year",
                        "spatial_block",
                        "Kc_target",
                        "s2_coverage_pct",
                        "hls_coverage_pct",
                    ]
                    if column in test.columns
                ]

                fold_output = (
                    test[
                        keep_columns
                    ]
                    .copy()
                    .reset_index(
                        drop=True
                    )
                )

                fold_output[
                    "threshold_pct"
                ] = threshold

                fold_output[
                    "split_type"
                ] = split_type

                fold_output[
                    "fold"
                ] = int(fold)

                fold_output[
                    "test_group"
                ] = (
                    test["group"]
                    .astype(str)
                    .reset_index(
                        drop=True
                    )
                )

                fold_output[
                    "s2_prediction"
                ] = predictions["S2"]

                fold_output[
                    "hls_prediction"
                ] = predictions["HLS"]

                fold_output[
                    "s2_error"
                ] = (
                    fold_output[
                        "s2_prediction"
                    ]
                    - fold_output[
                        "Kc_target"
                    ]
                )

                fold_output[
                    "hls_error"
                ] = (
                    fold_output[
                        "hls_prediction"
                    ]
                    - fold_output[
                        "Kc_target"
                    ]
                )

                all_oof.append(
                    fold_output
                )

    oof = pd.concat(
        all_oof,
        ignore_index=True,
    )

    overall_records = []

    for (
        threshold,
        split_type,
    ), group in oof.groupby(
        [
            "threshold_pct",
            "split_type",
        ]
    ):
        for source, prediction_column in [
            ("S2", "s2_prediction"),
            ("HLS", "hls_prediction"),
        ]:
            record = {
                "threshold_pct": int(
                    threshold
                ),
                "split_type": (
                    split_type
                ),
                "source": source,
            }

            record.update(
                calculate_metrics(
                    group[
                        "Kc_target"
                    ],
                    group[
                        prediction_column
                    ],
                )
            )

            overall_records.append(
                record
            )

    metrics_overall = pd.DataFrame(
        overall_records
    )

    metrics_by_fold = (
        metrics_by_group(
            oof,
            "fold",
        )
    )

    metrics_by_station = (
        metrics_by_group(
            oof,
            "station_id",
        )
    )

    metrics_by_year = (
        metrics_by_group(
            oof,
            "year",
        )
    )

    paired_summary = (
        paired_error_summary(
            oof
        )
    )

    oof.to_csv(
        output_dir
        / "oof_predictions.csv",
        index=False,
    )

    metrics_overall.to_csv(
        output_dir
        / "metrics_overall.csv",
        index=False,
    )

    metrics_by_fold.to_csv(
        output_dir
        / "metrics_by_fold.csv",
        index=False,
    )

    metrics_by_station.to_csv(
        output_dir
        / "metrics_by_station.csv",
        index=False,
    )

    metrics_by_year.to_csv(
        output_dir
        / "metrics_by_year.csv",
        index=False,
    )

    paired_summary.to_csv(
        output_dir
        / "paired_error_summary.csv",
        index=False,
    )

    manifest = {
        "period_label": (
            args.period_label
        ),
        "target": "Kc_target",
        "common_predictors": (
            COMMON_PREDICTORS
        ),
        "sources": [
            "S2",
            "HLS",
        ],
        "thresholds_pct": [
            80,
            90,
            99,
        ],
        "split_types": [
            "spatial",
            "temporal",
        ],
        "rf_parameters": (
            RF_PARAMETERS
        ),
        "paired_population": True,
        "same_folds_between_sources": True,
        "hyperparameter_tuning_performed": False,
        "model_selection_performed": False,
        "aoa_di_performed": False,
        "training_performed": True,
        "production_model_saved": False,
    }

    with (
        output_dir
        / "evaluation_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print()
    print("OVERALL METRICS")
    print("================")
    print(
        metrics_overall
        .sort_values(
            [
                "threshold_pct",
                "split_type",
                "source",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print("PAIRED ERROR SUMMARY")
    print("====================")
    print(
        paired_summary
        .sort_values(
            [
                "threshold_pct",
                "split_type",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "hyperparameter_tuning_performed = false"
    )
    print(
        "model_selection_performed = false"
    )
    print(
        "aoa_di_performed = false"
    )


if __name__ == "__main__":
    main()
