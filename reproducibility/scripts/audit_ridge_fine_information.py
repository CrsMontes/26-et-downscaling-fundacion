"""Audit fine-information dependence of the fixed S2 Ridge candidate.

The analysis is local and diagnostic. It reuses paired GE80/GE90/GE99
populations and their existing folds without tuning or row filtering.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

import evaluate_coverage_threshold_sensitivity as threshold_sensitivity
import screen_feature_families as family_screening
import screen_optical_algorithms as algorithm_screening


KEYS = ["station_id", "period_start"]
THRESHOLDS = (80, 90, 99)
OPTICAL = list(family_screening.S2_COMMON)
ETO_DRIVERS = list(family_screening.ETO_DRIVERS)
SEASONALITY = list(family_screening.SEASONALITY)
FEATURES = {
    "ridge_optical_only": OPTICAL,
    "ridge_coarse_only": ETO_DRIVERS + SEASONALITY,
    "ridge_eto_drivers_only": ETO_DRIVERS,
    "ridge_optical_plus_eto": OPTICAL + ETO_DRIVERS,
    "ridge_full_candidate": OPTICAL + ETO_DRIVERS + SEASONALITY,
}
FEATURE_FAMILY = (
    {feature: "optical_common11" for feature in OPTICAL}
    | {feature: "eto_drivers" for feature in ETO_DRIVERS}
    | {feature: "seasonality" for feature in SEASONALITY}
)
COMPARISONS = (
    {
        "comparison": "full_candidate_vs_coarse_only",
        "variant": "ridge_full_candidate",
        "reference": "ridge_coarse_only",
    },
    {
        "comparison": "optical_plus_eto_vs_eto_drivers_only",
        "variant": "ridge_optical_plus_eto",
        "reference": "ridge_eto_drivers_only",
    },
)


def project_root():
    return Path(__file__).resolve().parents[2]


def load_and_preflight(root):
    populations, folds, _, diagnostic, store_path = (
        threshold_sensitivity.load_and_preflight(root)
    )
    records = []
    for threshold in THRESHOLDS:
        data = populations[threshold]
        for configuration, features in FEATURES.items():
            missing_columns = sorted(set(features) - set(data.columns))
            if missing_columns:
                raise RuntimeError(
                    f"GE{threshold} {configuration} missing columns: {missing_columns}"
                )
            missing_count = int(data[features].isna().sum().sum())
            if missing_count:
                raise RuntimeError(
                    f"GE{threshold} {configuration} has {missing_count} missing values"
                )
            records.append(
                {
                    "threshold": threshold,
                    "configuration": configuration,
                    "rows": len(data),
                    "unique_keys": len(data[KEYS].drop_duplicates()),
                    "n_features": len(features),
                    "missing_values": missing_count,
                    "key_sha256": threshold_sensitivity.key_digest(data),
                }
            )
    return populations, folds, pd.DataFrame(records), diagnostic, store_path


def evaluate(populations, fold_tables):
    template = algorithm_screening.build_algorithms()["ridge"]
    predictions = []
    plausibility = []
    coefficients = []
    for threshold in THRESHOLDS:
        data = populations[threshold]
        for split_type in ("spatial", "temporal"):
            assignments = fold_tables[threshold].loc[
                lambda frame: frame["split_type"].eq(split_type), KEYS + ["fold"]
            ]
            working = data.merge(assignments, on=KEYS, validate="one_to_one")
            for configuration, features in FEATURES.items():
                for fold in sorted(working["fold"].unique()):
                    test_mask = working["fold"].eq(fold)
                    train = working.loc[~test_mask]
                    test = working.loc[test_mask]
                    model = clone(template)
                    model.fit(train[features], train["Kc_target"])
                    prediction = model.predict(test[features])
                    output = test[
                        KEYS + ["year", "spatial_block", "fold", "Kc_target"]
                    ].copy()
                    output.insert(4, "threshold", threshold)
                    output.insert(5, "configuration", configuration)
                    output.insert(6, "split_type", split_type)
                    output["prediction"] = prediction
                    predictions.append(output)

                    train_min = float(train["Kc_target"].min())
                    train_max = float(train["Kc_target"].max())
                    plausibility.append(
                        {
                            "threshold": threshold,
                            "configuration": configuration,
                            "split_type": split_type,
                            "fold": int(fold),
                            "n": len(test),
                            "prediction_min": float(np.min(prediction)),
                            "prediction_max": float(np.max(prediction)),
                            "predictions_below_zero": int((prediction < 0).sum()),
                            "predictions_outside_training_range": int(
                                ((prediction < train_min) | (prediction > train_max)).sum()
                            ),
                        }
                    )
                    ridge = model.named_steps["regressor"]
                    for feature, coefficient in zip(features, ridge.coef_):
                        coefficients.append(
                            {
                                "threshold": threshold,
                                "configuration": configuration,
                                "split_type": split_type,
                                "fold": int(fold),
                                "feature_name": feature,
                                "feature_family": FEATURE_FAMILY[feature],
                                "standardized_coefficient": float(coefficient),
                                "sign": (
                                    "positive" if coefficient > 0
                                    else "negative" if coefficient < 0 else "zero"
                                ),
                                "absolute_magnitude": float(abs(coefficient)),
                            }
                        )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(plausibility),
        pd.DataFrame(coefficients),
    )


def metric_tables(oof):
    dimensions = ["threshold", "configuration", "split_type"]
    overall = [
        dict(zip(dimensions, values))
        | algorithm_screening.calculate_metrics(group["Kc_target"], group["prediction"])
        for values, group in oof.groupby(dimensions, sort=True)
    ]
    fold_dimensions = dimensions + ["fold"]
    by_fold = [
        dict(zip(fold_dimensions, values))
        | algorithm_screening.calculate_metrics(group["Kc_target"], group["prediction"])
        for values, group in oof.groupby(fold_dimensions, sort=True)
    ]
    return pd.DataFrame(overall), pd.DataFrame(by_fold)


def comparison_table(metrics, include_fold):
    keys = ["threshold", "split_type"] + (["fold"] if include_fold else [])
    metric_names = ["R2", "RMSE", "MAE", "BIAS", "KGE"]
    records = []
    for definition in COMPARISONS:
        variant = metrics.loc[
            metrics["configuration"].eq(definition["variant"])
        ].set_index(keys)
        reference = metrics.loc[
            metrics["configuration"].eq(definition["reference"])
        ].set_index(keys)
        if not variant.index.equals(reference.index):
            raise RuntimeError(f"Unpaired metrics for {definition['comparison']}")
        for index, row in variant.iterrows():
            index_values = index if isinstance(index, tuple) else (index,)
            record = dict(zip(keys, index_values)) | definition | {"n": int(row["n"])}
            reference_row = reference.loc[index]
            for metric in metric_names:
                record[f"variant_{metric}"] = row[metric]
                record[f"reference_{metric}"] = reference_row[metric]
                record[f"delta_{metric}_variant_minus_reference"] = (
                    row[metric] - reference_row[metric]
                )
            records.append(record)
    return pd.DataFrame(records)


def coefficient_summaries(coefficients):
    keys = [
        "threshold", "configuration", "split_type", "feature_name", "feature_family"
    ]
    stability = coefficients.groupby(keys, as_index=False).agg(
        fold_count=("fold", "nunique"),
        mean_coefficient=("standardized_coefficient", "mean"),
        coefficient_sd=("standardized_coefficient", "std"),
        mean_absolute_magnitude=("absolute_magnitude", "mean"),
        positive_folds=("standardized_coefficient", lambda values: int((values > 0).sum())),
        negative_folds=("standardized_coefficient", lambda values: int((values < 0).sum())),
        zero_folds=("standardized_coefficient", lambda values: int((values == 0).sum())),
    )
    stability["positive_proportion"] = stability["positive_folds"] / stability["fold_count"]
    stability["negative_proportion"] = stability["negative_folds"] / stability["fold_count"]

    per_fold_family = coefficients.groupby(
        ["threshold", "configuration", "split_type", "fold", "feature_family"],
        as_index=False,
    ).agg(
        predictor_count=("feature_name", "size"),
        mean_absolute_coefficient=("absolute_magnitude", "mean"),
        sum_absolute_coefficients=("absolute_magnitude", "sum"),
        coefficient_l2_norm=(
            "standardized_coefficient",
            lambda values: float(np.sqrt(np.sum(np.asarray(values) ** 2))),
        ),
    )
    family_stability = per_fold_family.groupby(
        ["threshold", "configuration", "split_type", "feature_family"],
        as_index=False,
    ).agg(
        fold_count=("fold", "nunique"),
        mean_fold_absolute_coefficient=("mean_absolute_coefficient", "mean"),
        sd_fold_absolute_coefficient=("mean_absolute_coefficient", "std"),
        mean_fold_sum_absolute_coefficients=("sum_absolute_coefficients", "mean"),
        mean_fold_l2_norm=("coefficient_l2_norm", "mean"),
        sd_fold_l2_norm=("coefficient_l2_norm", "std"),
    )
    return stability, per_fold_family, family_stability


def manifest(populations, preflight, store_path):
    return {
        "experiment": "ridge_fine_information_audit",
        "thresholds": list(THRESHOLDS),
        "population_rows": {
            f"GE{threshold}": len(populations[threshold]) for threshold in THRESHOLDS
        },
        "population_key_hashes": {
            f"GE{threshold}": threshold_sensitivity.key_digest(populations[threshold])
            for threshold in THRESHOLDS
        },
        "feature_store": str(store_path.relative_to(project_root())).replace("\\", "/"),
        "features": FEATURES,
        "feature_families": FEATURE_FAMILY,
        "comparisons": list(COMPARISONS),
        "ridge": algorithm_screening.serializable_parameters(
            algorithm_screening.build_algorithms()["ridge"]
        ),
        "preflight": preflight.to_dict("records"),
        "standardized_coefficients_are_not_causal_importances": True,
        "hyperparameter_tuning_performed": False,
        "earth_engine_access": False,
        "aoa_di_performed": False,
        "production_model_trained": False,
        "winner_declared": False,
    }


def main():
    root = project_root()
    populations, folds, preflight, diagnostic, store_path = load_and_preflight(root)
    print(
        preflight.groupby("threshold", as_index=False).agg(
            rows=("rows", "first"), unique_keys=("unique_keys", "first"),
            configurations=("configuration", "size"), missing_values=("missing_values", "sum"),
        ).to_string(index=False)
    )
    oof, plausibility, coefficients = evaluate(populations, folds)
    expected = sum(len(populations[t]) * len(FEATURES) * 2 for t in THRESHOLDS)
    if len(oof) != expected:
        raise RuntimeError(f"Expected {expected} OOF rows, found {len(oof)}")
    if oof.duplicated(["threshold", "configuration", "split_type", *KEYS]).any():
        raise RuntimeError("OOF predictions contain duplicate comparison keys")
    overall, by_fold = metric_tables(oof)
    deltas = comparison_table(overall, include_fold=False)
    fold_deltas = comparison_table(by_fold, include_fold=True)
    coefficient_stability, family_by_fold, family_stability = coefficient_summaries(
        coefficients
    )
    output = diagnostic / "ridge_fine_information_audit"
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "population_preflight.csv": preflight,
        "oof_predictions.csv": oof,
        "metrics_overall.csv": overall,
        "metrics_by_fold.csv": by_fold,
        "comparison_deltas_overall.csv": deltas,
        "comparison_deltas_by_fold.csv": fold_deltas,
        "prediction_plausibility_by_fold.csv": plausibility,
        "standardized_coefficients_by_fold.csv": coefficients,
        "coefficient_stability.csv": coefficient_stability,
        "coefficient_family_by_fold.csv": family_by_fold,
        "coefficient_family_stability.csv": family_stability,
    }
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)
    (output / "ridge_fine_information_manifest.json").write_text(
        json.dumps(manifest(populations, preflight, store_path), indent=2),
        encoding="utf-8",
    )
    print("\n", overall.sort_values(["threshold", "split_type", "RMSE"]).to_string(index=False))
    print(f"\nOOF rows: {len(oof)}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
