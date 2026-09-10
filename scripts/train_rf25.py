"""Train and validate the final RF-25 model from the Virtual10 population.

This script performs no feature selection and no hyperparameter tuning. The
25 predictors and Random-Forest hyperparameters are frozen in
``et_downscaling.rf25``. Spatial OOF uses the 10 frozen spatial blocks and
LOYO uses calendar year. The final fitted model is then used to derive the
model-weighted DI/AOA specification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from et_downscaling.aoa_rf25 import build_rf_weighted_aoa, importance_table
from et_downscaling.metrics import TARGET_COLUMN, calculate_metrics
from et_downscaling.rf25 import (
    RF25_AOA_FILENAME,
    RF25_METADATA_FILENAME,
    RF25_MODEL_FEATURES,
    RF25_MODEL_FILENAME,
    RF25_PARAMETERS,
    build_rf25_model,
    rf25_model_signature,
)
from et_downscaling.workspace import get_workspace_paths


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        default=None,
        help="Optional training population CSV. Defaults to outputs/evaluation/results/virtual10_training_population.csv.",
    )
    parser.add_argument("--importance-repeats", type=int, default=20)
    return parser.parse_args()


def oof_by_group(
    data: pd.DataFrame,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    predictions = np.full(len(data), np.nan, dtype=float)
    fold_rows: list[dict[str, object]] = []
    groups = data[group_column].astype(str)

    for fold_number, group in enumerate(sorted(groups.unique()), start=1):
        test_mask = groups.eq(group).to_numpy()
        train_mask = ~test_mask
        model = build_rf25_model()
        model.fit(
            data.loc[train_mask, RF25_MODEL_FEATURES],
            data.loc[train_mask, TARGET_COLUMN],
        )
        prediction = model.predict(data.loc[test_mask, RF25_MODEL_FEATURES])
        predictions[test_mask] = prediction
        fold_rows.append(
            {
                "fold": fold_number,
                "group": str(group),
                **calculate_metrics(data.loc[test_mask, TARGET_COLUMN], prediction),
            }
        )

    if np.isnan(predictions).any():
        raise RuntimeError(f"OOF predictions contain missing values for {group_column}.")

    output = data[
        ["station_id", "period_start", "spatial_block", "year", TARGET_COLUMN]
    ].copy()
    output["prediction"] = predictions
    output["error"] = predictions - data[TARGET_COLUMN].to_numpy(dtype=float)
    return output, pd.DataFrame(fold_rows), calculate_metrics(
        output[TARGET_COLUMN], output["prediction"]
    )


def persistence_table(population: pd.DataFrame) -> pd.DataFrame:
    ordered = population.copy()
    ordered["period_start"] = pd.to_datetime(ordered["period_start"], errors="raise")
    ordered = ordered.sort_values(["station_id", "period_start"]).reset_index(drop=True)
    ordered["previous_target"] = ordered.groupby("station_id")[TARGET_COLUMN].shift(1)
    ordered["previous_date"] = ordered.groupby("station_id")["period_start"].shift(1)
    ordered["lag_days"] = (ordered["period_start"] - ordered["previous_date"]).dt.days
    masks = {
        "previous_available": ordered["previous_target"].notna(),
        "previous_exact_8_days": ordered["previous_target"].notna() & ordered["lag_days"].eq(8),
        "previous_within_16_days": ordered["previous_target"].notna() & ordered["lag_days"].le(16),
    }
    rows = []
    for name, mask in masks.items():
        subset = ordered.loc[mask]
        if subset.empty:
            continue
        rows.append(
            {
                "baseline": name,
                **calculate_metrics(subset[TARGET_COLUMN], subset["previous_target"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = project_root()
    workspace = get_workspace_paths(root).ensure()
    results_root = root / "outputs" / "evaluation" / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    model_root = workspace.models
    model_root.mkdir(parents=True, exist_ok=True)

    population_path = (
        Path(args.population).expanduser().resolve()
        if args.population
        else results_root / "virtual10_training_population.csv"
    )
    if not population_path.is_file():
        raise FileNotFoundError(
            f"Training population not found: {population_path}\n"
            "Run the fresh extraction stage first."
        )

    population = pd.read_csv(population_path, dtype={"station_id": str})
    population["period_start"] = pd.to_datetime(population["period_start"], errors="raise")
    if "year" not in population.columns:
        population["year"] = population["period_start"].dt.year.astype(int)

    required = set(RF25_MODEL_FEATURES + [TARGET_COLUMN, "station_id", "spatial_block", "year"])
    missing = sorted(required - set(population.columns))
    if missing:
        raise ValueError("Training population missing columns: " + ", ".join(missing))
    matrix = population[RF25_MODEL_FEATURES + [TARGET_COLUMN]].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("RF-25 training matrix contains non-finite values.")
    if population["station_id"].nunique() != 10 or population["spatial_block"].nunique() != 10:
        raise ValueError("Final Virtual10 training requires exactly 10 supports and 10 spatial blocks.")

    print("=" * 96)
    print("FINAL RF-25 TRAINING")
    print("=" * 96)
    print("Population:", population_path)
    print("Rows:", len(population))
    print("Supports:", population["station_id"].nunique())
    print("Spatial blocks:", population["spatial_block"].nunique())
    print("Features:", len(RF25_MODEL_FEATURES))
    print("Hyperparameters:", RF25_PARAMETERS)

    spatial_oof, spatial_folds, spatial_metrics = oof_by_group(population, "spatial_block")
    temporal_oof, temporal_folds, temporal_metrics = oof_by_group(population, "year")

    model = build_rf25_model()
    model.fit(population[RF25_MODEL_FEATURES], population[TARGET_COLUMN])
    aoa = build_rf_weighted_aoa(
        population,
        model,
        target_column=TARGET_COLUMN,
        group_column="spatial_block",
        importance_repeats=args.importance_repeats,
        random_state=42,
    )

    model_path = model_root / RF25_MODEL_FILENAME
    aoa_path = model_root / RF25_AOA_FILENAME
    joblib.dump(model, model_path)
    joblib.dump(aoa, aoa_path)

    metrics = pd.DataFrame(
        [
            {"model": "RF25", "evaluation": "spatial_block_OOF", **spatial_metrics},
            {"model": "RF25", "evaluation": "leave_one_year_out", **temporal_metrics},
        ]
    )
    metrics.to_csv(results_root / "rf25_metrics.csv", index=False)
    spatial_oof.to_csv(results_root / "rf25_spatial_oof.csv", index=False)
    spatial_folds.to_csv(results_root / "rf25_spatial_fold_metrics.csv", index=False)
    temporal_oof.to_csv(results_root / "rf25_temporal_oof.csv", index=False)
    temporal_folds.to_csv(results_root / "rf25_temporal_fold_metrics.csv", index=False)
    persistence_table(population).to_csv(results_root / "persistence_baselines.csv", index=False)
    importance_table(aoa).to_csv(results_root / "rf25_aoa_weights.csv", index=False)
    pd.DataFrame(
        {
            "station_id": population["station_id"].astype(str),
            "period_start": population["period_start"].dt.strftime("%Y-%m-%d"),
            "spatial_block": population["spatial_block"].astype(str),
            "training_DI": aoa.training_di,
            "training_LPD": aoa.training_lpd,
        }
    ).to_csv(results_root / "rf25_training_aoa.csv", index=False)

    metadata = {
        "model": "RandomForestRegressor",
        "model_name": "rf25_virtual10_ge90",
        "training_rows": int(len(population)),
        "supports": int(population["station_id"].nunique()),
        "spatial_blocks": int(population["spatial_block"].nunique()),
        "target": "Kc_target = MODIS_ET / ETo",
        "features": RF25_MODEL_FEATURES,
        "hyperparameters": RF25_PARAMETERS,
        "spatial_metrics": spatial_metrics,
        "temporal_metrics": temporal_metrics,
        "model_signature": rf25_model_signature(model),
        "aoa": {
            "method": aoa.method,
            "importance": "sklearn permutation importance, neg_mean_squared_error, final fitted RF",
            "importance_repeats": int(args.importance_repeats),
            "scaling": "training mean and sample SD (ddof=1)",
            "distance": "weighted Euclidean L2",
            "normalization": "mean of all pairwise weighted training distances",
            "threshold_training_reference": "nearest point outside the same frozen spatial CV block",
            "threshold_rule": "min(max(training_DI), Q3 + 1.5*IQR)",
            "threshold": float(aoa.threshold),
            "LPD": "count of training points within threshold * mean_training_distance; diagnostic layer",
            "hard_mask": True,
        },
        "guardrails": [
            "No feature selection is performed here; the 25-feature set is frozen.",
            "No RF hyperparameter tuning is performed here.",
            "Permutation importance can distribute importance across correlated predictors; raw weights are saved.",
            "Spatial OOF evaluates transfer among 10 Virtual10 supports, not independent 20 m ET validation.",
        ],
    }
    metadata_path = model_root / RF25_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print(metrics.to_string(index=False))
    print("AOA threshold:", aoa.threshold)
    print("Mean training LPD:", float(np.mean(aoa.training_lpd)))
    print("Model:", model_path)
    print("AOA:", aoa_path)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
