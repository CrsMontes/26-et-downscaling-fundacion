"""Evaluate local GE80/GE90/GE99 sensitivity for fixed model candidates.

All comparisons are internal to one paired threshold population. The script
performs no tuning, Earth Engine access, AOA/DI, or production-model training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

import screen_feature_families as family_screening
import screen_optical_algorithms as algorithm_screening


KEYS = ["station_id", "period_start"]
THRESHOLDS = (80, 90, 99)
FEATURES = {
    "S2": {
        "base": family_screening.S2_COMMON + family_screening.ETO_DRIVERS,
        "seasonality": (
            family_screening.S2_COMMON
            + family_screening.ETO_DRIVERS
            + family_screening.SEASONALITY
        ),
        "rich7": (
            family_screening.S2_COMMON
            + family_screening.ETO_DRIVERS
            + family_screening.S2_RICH
        ),
    },
    "HLS": {
        "base": family_screening.HLS_COMMON + family_screening.ETO_DRIVERS,
        "seasonality": (
            family_screening.HLS_COMMON
            + family_screening.ETO_DRIVERS
            + family_screening.SEASONALITY
        ),
    },
}
CANDIDATES = [
    {"source": "S2", "algorithm": "random_forest", "configuration": "base"},
    {"source": "S2", "algorithm": "random_forest", "configuration": "seasonality"},
    {"source": "S2", "algorithm": "random_forest", "configuration": "rich7"},
    {"source": "S2", "algorithm": "extra_trees", "configuration": "base"},
    {"source": "S2", "algorithm": "extra_trees", "configuration": "seasonality"},
    {"source": "S2", "algorithm": "ridge", "configuration": "base"},
    {"source": "S2", "algorithm": "ridge", "configuration": "seasonality"},
    {"source": "HLS", "algorithm": "random_forest", "configuration": "base"},
    {"source": "HLS", "algorithm": "random_forest", "configuration": "seasonality"},
    {"source": "HLS", "algorithm": "extra_trees", "configuration": "base"},
    {"source": "HLS", "algorithm": "extra_trees", "configuration": "seasonality"},
]


def project_root():
    return Path(__file__).resolve().parents[2]


def candidate_name(source, algorithm, configuration):
    return f"{source.lower()}__{algorithm}__{configuration}"


def key_digest(table):
    values = "\n".join(
        table.sort_values(KEYS)[KEYS].astype(str).agg("|".join, axis=1)
    )
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def read_keyed(path, label, require_unique_keys=True):
    table = pd.read_csv(path, dtype={"station_id": str})
    table["period_start"] = pd.to_datetime(
        table["period_start"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    if require_unique_keys and table.duplicated(KEYS).any():
        raise RuntimeError(f"{label} contains duplicate station-period keys")
    return table


def validate_threshold_folds(data, folds, threshold):
    selected = folds.loc[folds["threshold_pct"].eq(threshold)].copy()
    if selected.duplicated(["split_type", *KEYS]).any():
        raise RuntimeError(f"GE{threshold} fold assignments contain duplicates")
    for split_type, group_column, expected_groups in (
        ("spatial", "spatial_block", 4),
        ("temporal", "year", 5),
    ):
        assignments = selected.loc[selected["split_type"].eq(split_type)]
        if len(assignments) != len(data):
            raise RuntimeError(
                f"GE{threshold} {split_type} assignments do not cover the population"
            )
        joined = data[KEYS + [group_column]].merge(
            assignments[KEYS + ["group", "fold"]],
            on=KEYS,
            validate="one_to_one",
        )
        if len(joined) != len(data) or joined["fold"].nunique() != expected_groups:
            raise RuntimeError(f"GE{threshold} has invalid {split_type} folds")
        if not joined[group_column].astype(str).eq(joined["group"].astype(str)).all():
            raise RuntimeError(
                f"GE{threshold} {split_type} fold groups disagree with the population"
            )
    return selected


def load_and_preflight(root):
    diagnostic = root / "outputs" / "diagnostics" / "2020_2024"
    feature_store_path = diagnostic / "experimental_feature_store" / "feature_store.csv"
    population_root = diagnostic / "optical_source_experiment" / "population"
    folds_path = diagnostic / "optical_source_experiment" / "folds" / "fold_assignments.csv"
    store = read_keyed(feature_store_path, "feature store")
    if len(store) != 1150:
        raise RuntimeError("Feature store must preserve all 1,150 base rows")
    folds = read_keyed(folds_path, "fold assignments", require_unique_keys=False)
    populations = {}
    selected_folds = {}
    preflight = []
    for threshold in THRESHOLDS:
        population_path = population_root / f"paired_population_ge{threshold}.csv"
        population = read_keyed(population_path, f"GE{threshold} population")
        if population.empty:
            raise RuntimeError(f"GE{threshold} population is empty")
        data = store.merge(population[KEYS], on=KEYS, how="inner", validate="one_to_one")
        if len(data) != len(population) or key_digest(data) != key_digest(population):
            raise RuntimeError(f"Feature store does not reproduce exact GE{threshold} keys")
        target = data[KEYS + ["Kc_target"]].merge(
            population[KEYS + ["Kc_target"]],
            on=KEYS,
            suffixes=("_store", "_population"),
            validate="one_to_one",
        )
        if not np.allclose(
            target["Kc_target_store"], target["Kc_target_population"],
            atol=1e-12, rtol=0,
        ):
            raise RuntimeError(f"GE{threshold} target differs from feature store")
        threshold_folds = validate_threshold_folds(data, folds, threshold)
        for candidate in CANDIDATES:
            features = FEATURES[candidate["source"]][candidate["configuration"]]
            missing_columns = sorted(set(features) - set(data.columns))
            if missing_columns:
                raise RuntimeError(
                    f"GE{threshold} {candidate} missing columns: {missing_columns}"
                )
            missing = data[features].isna().sum()
            missing = missing.loc[missing.gt(0)]
            if not missing.empty:
                raise RuntimeError(
                    f"GE{threshold} {candidate} contains missing values: {missing.to_dict()}"
                )
            preflight.append(
                {
                    "threshold": threshold,
                    **candidate,
                    "n_features": len(features),
                    "rows": len(data),
                    "unique_keys": len(data[KEYS].drop_duplicates()),
                    "key_sha256": key_digest(data),
                    "missing_values": 0,
                }
            )
        populations[threshold] = data
        selected_folds[threshold] = threshold_folds
    return populations, selected_folds, pd.DataFrame(preflight), diagnostic, feature_store_path


def algorithm_templates():
    available = algorithm_screening.build_algorithms()
    return {
        name: available[name]
        for name in ("random_forest", "extra_trees", "ridge")
    }


def evaluate(populations, fold_tables):
    templates = algorithm_templates()
    predictions = []
    plausibility = []
    for threshold in THRESHOLDS:
        data = populations[threshold]
        folds = fold_tables[threshold]
        for split_type in ("spatial", "temporal"):
            assignments = folds.loc[
                folds["split_type"].eq(split_type), KEYS + ["fold"]
            ]
            working = data.merge(assignments, on=KEYS, validate="one_to_one")
            for candidate in CANDIDATES:
                source = candidate["source"]
                algorithm = candidate["algorithm"]
                configuration = candidate["configuration"]
                features = FEATURES[source][configuration]
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
                    output.insert(4, "threshold", threshold)
                    output.insert(5, "source", source)
                    output.insert(6, "algorithm", algorithm)
                    output.insert(7, "configuration", configuration)
                    output.insert(8, "split_type", split_type)
                    output["prediction"] = prediction
                    predictions.append(output)
                    train_min = float(train["Kc_target"].min())
                    train_max = float(train["Kc_target"].max())
                    plausibility.append(
                        {
                            "threshold": threshold,
                            "source": source,
                            "algorithm": algorithm,
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
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(plausibility)


def metric_tables(oof):
    dimensions = ["threshold", "source", "algorithm", "configuration", "split_type"]
    overall = []
    for values, group in oof.groupby(dimensions, sort=True):
        overall.append(
            dict(zip(dimensions, values))
            | algorithm_screening.calculate_metrics(
                group["Kc_target"], group["prediction"]
            )
        )
    fold_dimensions = dimensions + ["fold"]
    by_fold = []
    for values, group in oof.groupby(fold_dimensions, sort=True):
        by_fold.append(
            dict(zip(fold_dimensions, values))
            | algorithm_screening.calculate_metrics(
                group["Kc_target"], group["prediction"]
            )
        )
    return pd.DataFrame(overall), pd.DataFrame(by_fold)


def comparison_definitions():
    definitions = []
    for source, algorithms in (
        ("S2", ("random_forest", "extra_trees", "ridge")),
        ("HLS", ("random_forest", "extra_trees")),
    ):
        for algorithm in algorithms:
            definitions.append(
                {
                    "comparison": "seasonality_vs_base",
                    "source": source,
                    "algorithm": algorithm,
                    "configuration": "seasonality",
                    "base_configuration": "base",
                }
            )
    definitions.append(
        {
            "comparison": "rich7_vs_rf_base",
            "source": "S2",
            "algorithm": "random_forest",
            "configuration": "rich7",
            "base_configuration": "base",
        }
    )
    return definitions


def comparison_outputs(overall, by_fold):
    metric_names = ["R2", "RMSE", "MAE", "BIAS", "KGE"]
    delta_rows = []
    win_rows = []
    for threshold in THRESHOLDS:
        for definition in comparison_definitions():
            for split_type in ("spatial", "temporal"):
                selector = (
                    overall["threshold"].eq(threshold)
                    & overall["source"].eq(definition["source"])
                    & overall["algorithm"].eq(definition["algorithm"])
                    & overall["split_type"].eq(split_type)
                )
                subset = overall.loc[selector].set_index("configuration")
                variant = subset.loc[definition["configuration"]]
                base = subset.loc[definition["base_configuration"]]
                record = {
                    "threshold": threshold,
                    **definition,
                    "split_type": split_type,
                    "n": int(variant["n"]),
                }
                for metric in metric_names:
                    record[f"variant_{metric}"] = variant[metric]
                    record[f"base_{metric}"] = base[metric]
                    record[f"delta_{metric}_variant_minus_base"] = (
                        variant[metric] - base[metric]
                    )
                delta_rows.append(record)

                fold_selector = (
                    by_fold["threshold"].eq(threshold)
                    & by_fold["source"].eq(definition["source"])
                    & by_fold["algorithm"].eq(definition["algorithm"])
                    & by_fold["split_type"].eq(split_type)
                )
                folds = by_fold.loc[fold_selector]
                variant_folds = folds.loc[
                    folds["configuration"].eq(definition["configuration"])
                ].set_index("fold")
                base_folds = folds.loc[
                    folds["configuration"].eq(definition["base_configuration"])
                ].set_index("fold")
                delta_rmse = variant_folds["RMSE"] - base_folds["RMSE"]
                delta_mae = variant_folds["MAE"] - base_folds["MAE"]
                win_rows.append(
                    {
                        "threshold": threshold,
                        **definition,
                        "split_type": split_type,
                        "folds": len(delta_rmse),
                        "folds_better_RMSE": int(delta_rmse.lt(0).sum()),
                        "folds_better_MAE": int(delta_mae.lt(0).sum()),
                        "folds_better_both": int(
                            (delta_rmse.lt(0) & delta_mae.lt(0)).sum()
                        ),
                    }
                )
    return pd.DataFrame(delta_rows), pd.DataFrame(win_rows)


def spatial_rankings(overall):
    spatial = overall.loc[overall["split_type"].eq("spatial")].copy()
    spatial["candidate"] = spatial.apply(
        lambda row: candidate_name(row.source, row.algorithm, row.configuration),
        axis=1,
    )
    rankings = []
    for threshold, group in spatial.groupby("threshold", sort=True):
        ranked = group.sort_values(["RMSE", "MAE", "candidate"]).copy()
        ranked["spatial_rank"] = np.arange(1, len(ranked) + 1)
        rankings.append(ranked)
    rankings = pd.concat(rankings, ignore_index=True)
    stability = (
        rankings.pivot(index="candidate", columns="threshold", values="spatial_rank")
        .reset_index()
    )
    for threshold in THRESHOLDS:
        if threshold not in stability.columns:
            stability[threshold] = np.nan
    stability = stability.rename(
        columns={threshold: f"rank_GE{threshold}" for threshold in THRESHOLDS}
    )
    rank_columns = [f"rank_GE{threshold}" for threshold in THRESHOLDS]
    stability["rank_min"] = stability[rank_columns].min(axis=1)
    stability["rank_max"] = stability[rank_columns].max(axis=1)
    stability["rank_range"] = stability["rank_max"] - stability["rank_min"]
    stability["rank_mean"] = stability[rank_columns].mean(axis=1)
    return rankings, stability.sort_values(["rank_mean", "candidate"])


def manifest(populations, preflight, feature_store_path):
    return {
        "experiment": "coverage_threshold_sensitivity",
        "thresholds": list(THRESHOLDS),
        "population_rows": {
            f"GE{threshold}": len(populations[threshold]) for threshold in THRESHOLDS
        },
        "population_key_hashes": {
            f"GE{threshold}": key_digest(populations[threshold]) for threshold in THRESHOLDS
        },
        "feature_store": str(feature_store_path.relative_to(project_root())).replace("\\", "/"),
        "candidates": CANDIDATES,
        "features": FEATURES,
        "algorithms": {
            name: algorithm_screening.serializable_parameters(estimator)
            for name, estimator in algorithm_templates().items()
        },
        "comparisons_are_within_threshold_only": True,
        "cross_threshold_metrics_are_not_treated_as_same_population": True,
        "preflight": preflight.to_dict("records"),
        "hyperparameter_tuning_performed": False,
        "earth_engine_access": False,
        "aoa_di_performed": False,
        "production_model_trained": False,
        "winner_declared": False,
        "screening_is_exploratory": True,
    }


def main():
    root = project_root()
    populations, fold_tables, preflight, diagnostic, store_path = load_and_preflight(root)
    print(
        preflight.groupby("threshold", as_index=False)
        .agg(rows=("rows", "first"), unique_keys=("unique_keys", "first"),
             configurations=("configuration", "size"), missing_values=("missing_values", "sum"))
        .to_string(index=False)
    )
    oof, plausibility = evaluate(populations, fold_tables)
    expected_rows = sum(len(populations[threshold]) * len(CANDIDATES) * 2 for threshold in THRESHOLDS)
    if len(oof) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} OOF rows, found {len(oof)}")
    if oof.duplicated(
        ["threshold", "source", "algorithm", "configuration", "split_type", *KEYS]
    ).any():
        raise RuntimeError("OOF predictions contain duplicate comparison keys")
    overall, by_fold = metric_tables(oof)
    deltas, fold_wins = comparison_outputs(overall, by_fold)
    rankings, ranking_stability = spatial_rankings(overall)
    output = diagnostic / "coverage_threshold_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    preflight.to_csv(output / "population_preflight.csv", index=False)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    overall.to_csv(output / "metrics_overall.csv", index=False)
    by_fold.to_csv(output / "metrics_by_fold.csv", index=False)
    deltas.to_csv(output / "incremental_deltas_within_threshold.csv", index=False)
    fold_wins.to_csv(output / "incremental_fold_wins.csv", index=False)
    rankings.to_csv(output / "spatial_rankings_within_threshold.csv", index=False)
    ranking_stability.to_csv(output / "spatial_ranking_stability.csv", index=False)
    plausibility.to_csv(output / "prediction_plausibility_by_fold.csv", index=False)
    (output / "threshold_sensitivity_manifest.json").write_text(
        json.dumps(manifest(populations, preflight, store_path), indent=2),
        encoding="utf-8",
    )
    print("\n", overall.sort_values(["threshold", "split_type", "RMSE"]).to_string(index=False))
    print("\nSPATIAL RANK STABILITY")
    print(ranking_stability.to_string(index=False))
    print(f"\nOOF rows: {len(oof)}")
    print(f"Output: {output}")
    print("hyperparameter_tuning_performed = false")
    print("earth_engine_access = false")
    print("winner_declared = false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
