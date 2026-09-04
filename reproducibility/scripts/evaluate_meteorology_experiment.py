"""Evaluate meteorological predictor blocks over the S2-common baseline.

All comparisons use the exact same GE90 population and fold assignments.
No hyperparameter tuning, AOA/DI filtering, or production model is created.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from et_downscaling.model_spec import add_doy_harmonics


OPTICAL_FEATURES = [
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

HARMONIC_FEATURES = [
    "doy_sin1",
    "doy_cos1",
    "doy_sin2",
    "doy_cos2",
]

PRECIPITATION_FEATURES = [
    "Precip_period_mm",
    "Precip_prev30d_mm",
]

ETO_DRIVER_FEATURES = [
    "Tair_mean_C",
    "Tair_max_C",
    "VPD_mean_kPa",
    "SolarRad_MJ_m2_day",
    "Wind_mean_ms",
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

    correlation = np.corrcoef(
        observed,
        predicted,
    )[0, 1]

    alpha = (
        np.std(predicted, ddof=0)
        / np.std(observed, ddof=0)
    )

    beta = (
        np.mean(predicted)
        / np.mean(observed)
    )

    return float(
        1.0
        - np.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    )


def calculate_metrics(data):
    observed = data["Kc_target"].to_numpy(
        dtype=float
    )

    predicted = data["prediction"].to_numpy(
        dtype=float
    )

    error = predicted - observed

    return {
        "n": len(data),
        "R2": r2_score(
            observed,
            predicted,
        ),
        "RMSE": float(
            np.sqrt(
                np.mean(error ** 2)
            )
        ),
        "MAE": float(
            np.mean(
                np.abs(error)
            )
        ),
        "BIAS": float(
            np.mean(error)
        ),
        "KGE": calculate_kge(
            observed,
            predicted,
        ),
    }


def evaluate_configuration(
    data,
    assignments,
    split_type,
    configuration,
    features,
):
    working = data.merge(
        assignments[
            FOLD_KEYS + ["fold", "group"]
        ],
        on=FOLD_KEYS,
        how="inner",
        validate="one_to_one",
    )

    if len(working) != len(data):
        raise RuntimeError(
            f"Fold mismatch for "
            f"{split_type}/{configuration}"
        )

    oof_parts = []

    for fold in sorted(
        working["fold"].unique()
    ):
        test_mask = (
            working["fold"] == fold
        )

        train = working.loc[
            ~test_mask
        ]

        test = working.loc[
            test_mask
        ]

        model = RandomForestRegressor(
            **RF_PARAMETERS
        )

        model.fit(
            train[features].to_numpy(
                dtype=float
            ),
            train["Kc_target"].to_numpy(
                dtype=float
            ),
        )

        prediction = model.predict(
            test[features].to_numpy(
                dtype=float
            )
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

        output["split_type"] = (
            split_type
        )

        output["configuration"] = (
            configuration
        )

        output["prediction"] = (
            prediction
        )

        oof_parts.append(
            output
        )

    return pd.concat(
        oof_parts,
        ignore_index=True,
    )


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

    meteorology = pd.read_csv(
        root
        / "meteorology_experiment"
        / "processed"
        / "period_meteorology.csv"
    )

    folds = pd.read_csv(
        root
        / "optical_source_experiment"
        / "folds"
        / "fold_assignments.csv"
    )

    merge_keys = [
        "station_id",
        "period_start",
    ]

    meteorology_columns = (
        merge_keys
        + ETO_DRIVER_FEATURES
        + PRECIPITATION_FEATURES
        + [
            "meteo_complete",
        ]
    )

    data = population.merge(
        meteorology[
            meteorology_columns
        ],
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )

    if len(data) != 550:
        raise RuntimeError(
            f"Expected 550 GE90 rows, "
            f"found {len(data)}"
        )

    if not data[
        "meteo_complete"
    ].eq(1).all():
        raise RuntimeError(
            "Incomplete meteorology "
            "inside GE90 population."
        )

    data = add_doy_harmonics(
        data
    )

    optical = [
        f"s2_{feature}_mean"
        for feature in OPTICAL_FEATURES
    ]

    configurations = {
        "s2_common": (
            optical
        ),
        "s2_plus_seasonality": (
            optical
            + HARMONIC_FEATURES
        ),
        "s2_plus_precipitation": (
            optical
            + PRECIPITATION_FEATURES
        ),
        "s2_plus_seasonality_precip": (
            optical
            + HARMONIC_FEATURES
            + PRECIPITATION_FEATURES
        ),
        "s2_plus_eto_drivers": (
            optical
            + ETO_DRIVER_FEATURES
        ),
        "s2_plus_full_meteorology": (
            optical
            + HARMONIC_FEATURES
            + PRECIPITATION_FEATURES
            + ETO_DRIVER_FEATURES
        ),
    }

    all_oof = []

    for split_type in [
        "spatial",
        "temporal",
    ]:
        assignments = folds[
            (folds["threshold_pct"] == 90)
            & (
                folds["split_type"]
                == split_type
            )
        ].copy()

        for (
            configuration,
            features,
        ) in configurations.items():

            oof = evaluate_configuration(
                data=data,
                assignments=assignments,
                split_type=split_type,
                configuration=configuration,
                features=features,
            )

            oof["n_features"] = (
                len(features)
            )

            all_oof.append(
                oof
            )

    oof = pd.concat(
        all_oof,
        ignore_index=True,
    )

    overall_records = []

    for (
        split_type,
        configuration,
    ), group in oof.groupby(
        [
            "split_type",
            "configuration",
        ],
        sort=True,
    ):
        overall_records.append(
            {
                "split_type": split_type,
                "configuration": (
                    configuration
                ),
                "n_features": int(
                    group[
                        "n_features"
                    ].iloc[0]
                ),
                **calculate_metrics(
                    group
                ),
            }
        )

    metrics_overall = pd.DataFrame(
        overall_records
    )

    fold_records = []

    for (
        split_type,
        configuration,
        fold,
    ), group in oof.groupby(
        [
            "split_type",
            "configuration",
            "fold",
        ],
        sort=True,
    ):
        fold_records.append(
            {
                "split_type": split_type,
                "configuration": (
                    configuration
                ),
                "fold": fold,
                **calculate_metrics(
                    group
                ),
            }
        )

    metrics_by_fold = pd.DataFrame(
        fold_records
    )

    baseline = (
        metrics_overall[
            metrics_overall[
                "configuration"
            ]
            == "s2_common"
        ][
            [
                "split_type",
                "R2",
                "RMSE",
                "MAE",
                "BIAS",
                "KGE",
            ]
        ]
        .rename(
            columns={
                "R2": "baseline_R2",
                "RMSE": "baseline_RMSE",
                "MAE": "baseline_MAE",
                "BIAS": "baseline_BIAS",
                "KGE": "baseline_KGE",
            }
        )
    )

    delta = metrics_overall.merge(
        baseline,
        on="split_type",
        how="left",
        validate="many_to_one",
    )

    delta["delta_R2"] = (
        delta["R2"]
        - delta["baseline_R2"]
    )

    delta["delta_RMSE"] = (
        delta["RMSE"]
        - delta["baseline_RMSE"]
    )

    delta["delta_MAE"] = (
        delta["MAE"]
        - delta["baseline_MAE"]
    )

    delta["delta_BIAS"] = (
        delta["BIAS"]
        - delta["baseline_BIAS"]
    )

    delta["delta_KGE"] = (
        delta["KGE"]
        - delta["baseline_KGE"]
    )

    output_dir = (
        root
        / "meteorology_experiment"
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    delta.to_csv(
        output_dir
        / "metrics_delta_vs_s2.csv",
        index=False,
    )

    manifest = {
        "threshold_pct": 90,
        "population_rows": len(data),
        "configurations": {
            key: value
            for key, value
            in configurations.items()
        },
        "rf_parameters": (
            RF_PARAMETERS
        ),
        "hyperparameter_tuning_performed": (
            False
        ),
        "aoa_di_performed": False,
        "production_model_saved": False,
        "earth_engine_access": False,
        "interpretation_note": (
            "ETo-driver predictors are "
            "evaluated as sensitivity because "
            "Kc_target contains ETo in its "
            "denominator."
        ),
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

    display_columns = [
        "split_type",
        "configuration",
        "n_features",
        "n",
        "R2",
        "RMSE",
        "MAE",
        "BIAS",
        "KGE",
    ]

    print()
    print(
        "METEOROLOGY INCREMENTAL VALUE - GE90"
    )
    print(
        "====================================="
    )

    print(
        metrics_overall[
            display_columns
        ]
        .sort_values(
            [
                "split_type",
                "RMSE",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "DELTA VS S2 COMMON"
    )
    print(
        "=================="
    )

    print(
        delta[
            [
                "split_type",
                "configuration",
                "delta_R2",
                "delta_RMSE",
                "delta_MAE",
                "delta_BIAS",
                "delta_KGE",
            ]
        ]
        .sort_values(
            [
                "split_type",
                "delta_RMSE",
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
        "aoa_di_performed = false"
    )
    print(
        "production_model_saved = false"
    )
    print(
        "earth_engine_access = false"
    )


if __name__ == "__main__":
    main()
