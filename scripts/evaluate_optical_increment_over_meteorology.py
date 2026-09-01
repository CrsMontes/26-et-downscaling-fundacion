"""Test the incremental value of S2 over meteorological predictors.

All models use the same GE90 population, fold assignments, and fixed RF.
The purpose is to distinguish meteorological predictability from
fine-resolution optical information relevant to spatial downscaling.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from et_downscaling.model_spec import add_doy_harmonics


OPTICAL = [
    "s2_Blue_mean",
    "s2_Green_mean",
    "s2_Red_mean",
    "s2_NIR_mean",
    "s2_SWIR1_mean",
    "s2_SWIR2_mean",
    "s2_NDVI_mean",
    "s2_EVI_mean",
    "s2_SAVI_mean",
    "s2_NDWI_mean",
    "s2_NDMI_mean",
]

SEASONALITY = [
    "doy_sin1",
    "doy_cos1",
    "doy_sin2",
    "doy_cos2",
]

PRECIPITATION = [
    "Precip_period_mm",
    "Precip_prev30d_mm",
]

ETO_DRIVERS = [
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


def calculate_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - observed

    correlation = np.corrcoef(observed, predicted)[0, 1]
    alpha = np.std(predicted, ddof=0) / np.std(observed, ddof=0)
    beta = np.mean(predicted) / np.mean(observed)

    kge = 1.0 - np.sqrt(
        (correlation - 1.0) ** 2
        + (alpha - 1.0) ** 2
        + (beta - 1.0) ** 2
    )

    return {
        "n": len(observed),
        "R2": r2_score(observed, predicted),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "MAE": np.mean(np.abs(error)),
        "BIAS": np.mean(error),
        "KGE": kge,
    }


def evaluate(data, folds, split_type, configuration, features):
    assignments = folds[
        (folds["threshold_pct"] == 90)
        & (folds["split_type"] == split_type)
    ][["station_id", "period_start", "fold"]]

    working = data.merge(
        assignments,
        on=["station_id", "period_start"],
        how="inner",
        validate="one_to_one",
    )

    if len(working) != len(data):
        raise RuntimeError(
            f"Fold mismatch: {split_type}/{configuration}"
        )

    predictions = []

    for fold in sorted(working["fold"].unique()):
        test_mask = working["fold"] == fold

        train = working.loc[~test_mask]
        test = working.loc[test_mask]

        model = RandomForestRegressor(**RF_PARAMETERS)

        model.fit(
            train[features].to_numpy(float),
            train["Kc_target"].to_numpy(float),
        )

        predicted = model.predict(
            test[features].to_numpy(float)
        )

        output = test[
            [
                "station_id",
                "period_start",
                "Kc_target",
                "fold",
            ]
        ].copy()

        output["split_type"] = split_type
        output["configuration"] = configuration
        output["prediction"] = predicted
        predictions.append(output)

    return pd.concat(predictions, ignore_index=True)


def main():
    root = Path("outputs/diagnostics/2020_2024")

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

    meteorological_columns = (
        ["station_id", "period_start"]
        + PRECIPITATION
        + ETO_DRIVERS
    )

    data = population.merge(
        meteorology[meteorological_columns],
        on=["station_id", "period_start"],
        how="left",
        validate="one_to_one",
    )

    if len(data) != 550:
        raise RuntimeError(
            f"Expected 550 rows, found {len(data)}"
        )

    data = add_doy_harmonics(data)

    season_precip = SEASONALITY + PRECIPITATION
    full_meteorology = (
        SEASONALITY
        + PRECIPITATION
        + ETO_DRIVERS
    )

    configurations = {
        "s2_common": OPTICAL,

        "seasonality_precip_only":
            season_precip,

        "s2_plus_seasonality_precip":
            OPTICAL + season_precip,

        "eto_drivers_only":
            ETO_DRIVERS,

        "s2_plus_eto_drivers":
            OPTICAL + ETO_DRIVERS,

        "full_meteorology_only":
            full_meteorology,

        "s2_plus_full_meteorology":
            OPTICAL + full_meteorology,
    }

    all_oof = []

    for split_type in ["spatial", "temporal"]:
        for configuration, features in configurations.items():
            oof = evaluate(
                data,
                folds,
                split_type,
                configuration,
                features,
            )

            oof["n_features"] = len(features)
            all_oof.append(oof)

    oof = pd.concat(
        all_oof,
        ignore_index=True,
    )

    records = []

    for (
        split_type,
        configuration,
    ), group in oof.groupby(
        ["split_type", "configuration"],
        sort=True,
    ):
        records.append(
            {
                "split_type": split_type,
                "configuration": configuration,
                "n_features": int(
                    group["n_features"].iloc[0]
                ),
                **calculate_metrics(
                    group["Kc_target"],
                    group["prediction"],
                ),
            }
        )

    metrics = pd.DataFrame(records)

    pairs = [
        (
            "seasonality_precip_only",
            "s2_plus_seasonality_precip",
        ),
        (
            "eto_drivers_only",
            "s2_plus_eto_drivers",
        ),
        (
            "full_meteorology_only",
            "s2_plus_full_meteorology",
        ),
    ]

    incremental_records = []

    for split_type in ["spatial", "temporal"]:
        subset = metrics[
            metrics["split_type"] == split_type
        ].set_index("configuration")

        for meteorology_only, combined in pairs:
            base = subset.loc[meteorology_only]
            model = subset.loc[combined]

            incremental_records.append(
                {
                    "split_type": split_type,
                    "meteorology_configuration":
                        meteorology_only,
                    "combined_configuration":
                        combined,
                    "delta_R2_from_S2":
                        model["R2"] - base["R2"],
                    "delta_RMSE_from_S2":
                        model["RMSE"] - base["RMSE"],
                    "delta_MAE_from_S2":
                        model["MAE"] - base["MAE"],
                    "delta_BIAS_from_S2":
                        model["BIAS"] - base["BIAS"],
                    "delta_KGE_from_S2":
                        model["KGE"] - base["KGE"],
                }
            )

    incremental = pd.DataFrame(
        incremental_records
    )

    output = (
        root
        / "meteorology_experiment"
        / "optical_increment"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        output / "metrics_overall.csv",
        index=False,
    )

    incremental.to_csv(
        output / "s2_increment_over_meteorology.csv",
        index=False,
    )

    oof.to_csv(
        output / "oof_predictions.csv",
        index=False,
    )

    print()
    print("S2 INCREMENT OVER METEOROLOGY - GE90")
    print("====================================")

    print(
        metrics.sort_values(
            ["split_type", "RMSE"]
        ).to_string(index=False)
    )

    print()
    print("INCREMENTAL EFFECT OF ADDING S2")
    print("==============================")

    print(
        incremental.to_string(index=False)
    )

    print()
    print("hyperparameter_tuning_performed = false")
    print("aoa_di_performed = false")
    print("production_model_saved = false")


if __name__ == "__main__":
    main()
