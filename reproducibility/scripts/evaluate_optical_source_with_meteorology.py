"""Compare S2 and HLS under identical meteorological predictor contexts.

All comparisons use the exact paired GE90 population, identical folds,
and fixed RF parameters. No tuning, AOA/DI filtering, or productive
model selection is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from et_downscaling.model_spec import add_doy_harmonics


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

KEYS = [
    "station_id",
    "period_start",
]


def calculate_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = predicted - observed

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

    kge = 1.0 - np.sqrt(
        (correlation - 1.0) ** 2
        + (alpha - 1.0) ** 2
        + (beta - 1.0) ** 2
    )

    return {
        "n": len(observed),
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
        "KGE": float(kge),
    }


def evaluate(
    data,
    folds,
    split_type,
    source,
    context,
    features,
):
    assignments = folds[
        (folds["threshold_pct"] == 90)
        & (folds["split_type"] == split_type)
    ][
        KEYS + ["fold", "group"]
    ]

    working = data.merge(
        assignments,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    if len(working) != len(data):
        raise RuntimeError(
            f"Fold mismatch: "
            f"{source}/{context}/{split_type}"
        )

    outputs = []

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

        output["split_type"] = split_type
        output["source"] = source
        output["context"] = context
        output["n_features"] = len(features)
        output["prediction"] = prediction

        outputs.append(output)

    return pd.concat(
        outputs,
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

    data = population.merge(
        meteorology[
            KEYS
            + PRECIPITATION
            + ETO_DRIVERS
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    if len(data) != 550:
        raise RuntimeError(
            f"Expected 550 rows, "
            f"found {len(data)}"
        )

    data = add_doy_harmonics(
        data
    )

    source_features = {
        "S2": [
            f"s2_{name}_mean"
            for name in COMMON_NAMES
        ],
        "HLS": [
            f"hls_{name}_mean"
            for name in COMMON_NAMES
        ],
    }

    contexts = {
        "optical_only": [],
        "seasonality_precip": (
            SEASONALITY
            + PRECIPITATION
        ),
        "eto_drivers": (
            ETO_DRIVERS
        ),
        "full_meteorology": (
            SEASONALITY
            + PRECIPITATION
            + ETO_DRIVERS
        ),
    }

    all_oof = []

    for split_type in [
        "spatial",
        "temporal",
    ]:
        for source, optical in (
            source_features.items()
        ):
            for context, additional in (
                contexts.items()
            ):
                features = (
                    optical
                    + additional
                )

                all_oof.append(
                    evaluate(
                        data=data,
                        folds=folds,
                        split_type=split_type,
                        source=source,
                        context=context,
                        features=features,
                    )
                )

    oof = pd.concat(
        all_oof,
        ignore_index=True,
    )

    overall_records = []

    for (
        split_type,
        source,
        context,
    ), group in oof.groupby(
        [
            "split_type",
            "source",
            "context",
        ],
        sort=True,
    ):
        overall_records.append(
            {
                "split_type": split_type,
                "source": source,
                "context": context,
                "n_features": int(
                    group[
                        "n_features"
                    ].iloc[0]
                ),
                **calculate_metrics(
                    group["Kc_target"],
                    group["prediction"],
                ),
            }
        )

    overall = pd.DataFrame(
        overall_records
    )

    fold_records = []

    for (
        split_type,
        source,
        context,
        fold,
    ), group in oof.groupby(
        [
            "split_type",
            "source",
            "context",
            "fold",
        ],
        sort=True,
    ):
        fold_records.append(
            {
                "split_type": split_type,
                "source": source,
                "context": context,
                "fold": fold,
                **calculate_metrics(
                    group["Kc_target"],
                    group["prediction"],
                ),
            }
        )

    by_fold = pd.DataFrame(
        fold_records
    )

    comparison_records = []

    for split_type in [
        "spatial",
        "temporal",
    ]:
        for context in contexts:

            subset = overall[
                (
                    overall["split_type"]
                    == split_type
                )
                & (
                    overall["context"]
                    == context
                )
            ].set_index("source")

            s2 = subset.loc["S2"]
            hls = subset.loc["HLS"]

            comparison_records.append(
                {
                    "split_type": split_type,
                    "context": context,
                    "S2_R2": s2["R2"],
                    "HLS_R2": hls["R2"],
                    "delta_R2_S2_minus_HLS": (
                        s2["R2"]
                        - hls["R2"]
                    ),
                    "S2_RMSE": s2["RMSE"],
                    "HLS_RMSE": hls["RMSE"],
                    "delta_RMSE_S2_minus_HLS": (
                        s2["RMSE"]
                        - hls["RMSE"]
                    ),
                    "S2_MAE": s2["MAE"],
                    "HLS_MAE": hls["MAE"],
                    "delta_MAE_S2_minus_HLS": (
                        s2["MAE"]
                        - hls["MAE"]
                    ),
                    "S2_BIAS": s2["BIAS"],
                    "HLS_BIAS": hls["BIAS"],
                    "S2_KGE": s2["KGE"],
                    "HLS_KGE": hls["KGE"],
                    "delta_KGE_S2_minus_HLS": (
                        s2["KGE"]
                        - hls["KGE"]
                    ),
                }
            )

    comparison = pd.DataFrame(
        comparison_records
    )

    output = (
        root
        / "optical_source_experiment"
        / "meteorology_comparison"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall.to_csv(
        output / "metrics_overall.csv",
        index=False,
    )

    by_fold.to_csv(
        output / "metrics_by_fold.csv",
        index=False,
    )

    comparison.to_csv(
        output / "s2_vs_hls.csv",
        index=False,
    )

    oof.to_csv(
        output / "oof_predictions.csv",
        index=False,
    )

    print()
    print(
        "S2 VS HLS WITH METEOROLOGY - GE90"
    )
    print(
        "================================="
    )

    print(
        overall.sort_values(
            [
                "split_type",
                "context",
                "RMSE",
            ]
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "PAIRED SOURCE DIFFERENCE"
    )
    print(
        "========================"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        "Negative delta_RMSE_S2_minus_HLS "
        "= S2 lower RMSE"
    )
    print(
        "Positive delta_R2_S2_minus_HLS "
        "= S2 higher R2"
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


if __name__ == "__main__":
    main()
