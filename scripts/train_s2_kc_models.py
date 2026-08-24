"""Compare Sentinel-2 Kc models and train the selected production model.

The model-selection experiment is intentionally small:

1. Random Forest using the common scale-transferable predictor set.
2. Random Forest adding Sentinel-2-specific predictors.
3. Training-fold global-mean baseline.
4. Previous-MODIS-period Kc persistence baseline.

The primary validation is spatial GroupKFold using approximately 10 km blocks.
The final production model is the parsimonious common-feature Random Forest,
selected from the observed spatial-validation results.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from et_downscaling.model_spec import (
    CANDIDATE_COLUMN,
    COMMON_MODEL_FEATURES,
    FULL_S2_MODEL_FEATURES,
    PRODUCTION_MODEL_FILENAME,
    RF_PARAMETERS,
    SELECTED_MODEL_NAME,
    SPATIAL_BLOCK_SIZE_KM,
    TARGET_COLUMN,
    add_doy_harmonics,
    build_random_forest,
)


def add_previous_modis_kc(master: pd.DataFrame) -> pd.DataFrame:
    """Add previous-period Kc within station from the complete MODIS series.

    The lag is calculated before optical candidate filtering so optical
    missingness cannot change which MODIS period is considered previous.
    MODIS QC availability is not used as a hard filter because the accepted
    target definition is based on physical validity of the ET value itself.
    """
    result = master.copy()
    result["period_start"] = pd.to_datetime(result["period_start"], errors="raise")
    result = result.sort_values(["station_id", "period_start"])
    result["Kc_previous_modis"] = (
        result.groupby("station_id")[TARGET_COLUMN].shift(1)
    )
    return result.sort_index()


def resolve_coordinate_columns(data: pd.DataFrame) -> tuple[str, str]:
    """Resolve the best available longitude/latitude pair."""
    candidates = [
        ("station_longitude", "station_latitude"),
        ("longitude", "latitude"),
        ("footprint_centroid_longitude", "footprint_centroid_latitude"),
    ]
    for longitude_column, latitude_column in candidates:
        if longitude_column in data.columns and latitude_column in data.columns:
            return longitude_column, latitude_column
    raise ValueError("No usable longitude/latitude columns were found.")


def add_spatial_blocks(data: pd.DataFrame) -> pd.DataFrame:
    """Assign observations to approximately 10 km spatial blocks."""
    result = data.copy()
    longitude_column, latitude_column = resolve_coordinate_columns(result)
    longitude = pd.to_numeric(result[longitude_column], errors="raise")
    latitude = pd.to_numeric(result[latitude_column], errors="raise")

    km_per_degree_latitude = 111.32
    km_per_degree_longitude = 111.32 * np.cos(np.radians(latitude.mean()))

    block_x = np.floor(
        longitude * km_per_degree_longitude / SPATIAL_BLOCK_SIZE_KM
    ).astype(int)
    block_y = np.floor(
        latitude * km_per_degree_latitude / SPATIAL_BLOCK_SIZE_KM
    ).astype(int)

    result["spatial_block"] = block_x.astype(str) + "_" + block_y.astype(str)
    return result


def calculate_kge(observed, predicted) -> float:
    """Calculate Kling-Gupta Efficiency."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(observed) < 2:
        return np.nan

    observed_sd = observed.std(ddof=0)
    predicted_sd = predicted.std(ddof=0)
    if observed_sd == 0 or predicted_sd == 0:
        return np.nan

    correlation = float(np.corrcoef(observed, predicted)[0, 1])
    alpha = predicted_sd / observed_sd
    observed_mean = observed.mean()
    if observed_mean == 0:
        return np.nan
    beta = predicted.mean() / observed_mean

    return float(
        1.0
        - np.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    )


def calculate_metrics(observed, predicted) -> dict[str, float]:
    """Calculate model-performance metrics."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - observed
    return {
        "n": int(len(observed)),
        "R2": float(r2_score(observed, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(observed, predicted))),
        "MAE": float(mean_absolute_error(observed, predicted)),
        "BIAS": float(error.mean()),
        "KGE": calculate_kge(observed, predicted),
    }


def run_spatial_validation(candidate: pd.DataFrame):
    """Run identical spatial folds for the two RFs and both baselines."""
    number_blocks = int(candidate["spatial_block"].nunique())
    if number_blocks < 3:
        raise RuntimeError("Spatial validation requires at least three spatial blocks.")

    splitter = GroupKFold(n_splits=number_blocks)
    groups = candidate["spatial_block"].to_numpy()
    y = candidate[TARGET_COLUMN].to_numpy(dtype=float)
    number_rows = len(candidate)

    predictions = {
        "rf_common": np.full(number_rows, np.nan, dtype=float),
        "rf_s2_full": np.full(number_rows, np.nan, dtype=float),
        "global_mean": np.full(number_rows, np.nan, dtype=float),
        "modis_persistence": np.full(number_rows, np.nan, dtype=float),
    }
    fold_values = np.full(number_rows, -1, dtype=int)
    fold_rows = []
    split_input = np.zeros((number_rows, 1))

    for fold, (train_index, test_index) in enumerate(
        splitter.split(split_input, y, groups=groups),
        start=1,
    ):
        train = candidate.iloc[train_index]
        test = candidate.iloc[test_index]

        common_model = build_random_forest()
        common_model.fit(train[COMMON_MODEL_FEATURES], train[TARGET_COLUMN])
        predictions["rf_common"][test_index] = common_model.predict(
            test[COMMON_MODEL_FEATURES]
        )

        full_model = build_random_forest()
        full_model.fit(train[FULL_S2_MODEL_FEATURES], train[TARGET_COLUMN])
        predictions["rf_s2_full"][test_index] = full_model.predict(
            test[FULL_S2_MODEL_FEATURES]
        )

        training_mean = float(train[TARGET_COLUMN].mean())
        predictions["global_mean"][test_index] = training_mean
        predictions["modis_persistence"][test_index] = (
            pd.to_numeric(test["Kc_previous_modis"], errors="coerce")
            .fillna(training_mean)
            .to_numpy(dtype=float)
        )
        fold_values[test_index] = fold

        common_metrics = calculate_metrics(
            y[test_index], predictions["rf_common"][test_index]
        )
        full_metrics = calculate_metrics(
            y[test_index], predictions["rf_s2_full"][test_index]
        )
        persistence_metrics = calculate_metrics(
            y[test_index], predictions["modis_persistence"][test_index]
        )

        fold_rows.append(
            {
                "fold": fold,
                "test_rows": int(len(test_index)),
                "stations": ";".join(sorted(test["station"].drop_duplicates())),
                "blocks": ";".join(sorted(test["spatial_block"].drop_duplicates())),
                "common_R2": common_metrics["R2"],
                "common_RMSE": common_metrics["RMSE"],
                "full_R2": full_metrics["R2"],
                "full_RMSE": full_metrics["RMSE"],
                "persistence_R2": persistence_metrics["R2"],
                "persistence_RMSE": persistence_metrics["RMSE"],
            }
        )

    metrics = pd.DataFrame(
        [
            {"model": name, **calculate_metrics(y, values)}
            for name, values in predictions.items()
        ]
    )
    folds = pd.DataFrame(fold_rows)

    oof_columns = [
        "station",
        "station_id",
        "period_start",
        "spatial_block",
        TARGET_COLUMN,
        "ET_mm_period",
        "ETo_mm_period",
        "ET_QC",
        "modis_qc_present",
        "optical_union_coverage_pct",
        "s1_union_coverage_pct",
        "Kc_previous_modis",
    ]
    oof = candidate[oof_columns].copy()
    oof["fold"] = fold_values
    for model_name, predicted_values in predictions.items():
        oof[f"Kc_predicted_{model_name}"] = predicted_values
        oof[f"Kc_residual_{model_name}"] = predicted_values - y

    return metrics, folds, oof


def main():
    project_root = Path(__file__).resolve().parents[1]
    input_path = (
        project_root
        / "outputs"
        / "processed"
        / "training"
        / "S2"
        / "ET_S2_S1_METEO_KC_FOOTPRINT_2021_2023.csv"
    )
    output_directory = project_root / "outputs" / "processed" / "models" / "S2"
    output_directory.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {input_path}")

    print("Loading:", input_path)
    master = pd.read_csv(input_path, dtype={"station_id": "string"}).copy()
    print("Master rows:", len(master))

    if len(master) != 690:
        raise RuntimeError("Expected 690 rows in the Sentinel-2 training master.")
    if master.duplicated(["station_id", "period_start"]).any():
        raise RuntimeError("Duplicate station-period rows found in training master.")

    master = add_previous_modis_kc(master)
    master = add_doy_harmonics(master)

    if CANDIDATE_COLUMN not in master.columns:
        raise ValueError(f"Missing candidate column: {CANDIDATE_COLUMN}")
    candidate = master.loc[master[CANDIDATE_COLUMN] == 1].copy()

    print("S2 >=90% candidate rows:", len(candidate))
    missing_qc = int((pd.to_numeric(candidate["modis_qc_present"], errors="coerce") == 0).sum())
    print("Candidates with missing MODIS ET_QC:", missing_qc)
    print(
        "Candidates with S1 coverage <90%:",
        int((candidate["s1_union_coverage_pct"] < 90.0).sum()),
    )
    print("Common model features:", len(COMMON_MODEL_FEATURES))
    print("Full S2 model features:", len(FULL_S2_MODEL_FEATURES))

    if len(COMMON_MODEL_FEATURES) != 25:
        raise RuntimeError("Expected 25 common model features.")
    if len(FULL_S2_MODEL_FEATURES) != 31:
        raise RuntimeError("Expected 31 full Sentinel-2 model features.")

    required_columns = list(
        dict.fromkeys(
            FULL_S2_MODEL_FEATURES
            + [
                TARGET_COLUMN,
                "station",
                "station_id",
                "period_start",
                "ET_QC",
                "modis_qc_present",
                "ET_mm_period",
                "ETo_mm_period",
                "optical_union_coverage_pct",
                "s1_union_coverage_pct",
                "Kc_previous_modis",
            ]
        )
    )
    missing_columns = [column for column in required_columns if column not in candidate.columns]
    if missing_columns:
        raise ValueError(f"Missing required model columns: {sorted(missing_columns)}")

    incomplete_model_rows = candidate[
        FULL_S2_MODEL_FEATURES + [TARGET_COLUMN]
    ].isna().any(axis=1)
    if incomplete_model_rows.any():
        raise RuntimeError(
            "Final candidate population contains "
            f"{int(incomplete_model_rows.sum())} rows with incomplete model values."
        )

    candidate = add_spatial_blocks(candidate)
    station_blocks = (
        candidate[["station", "spatial_block"]]
        .drop_duplicates()
        .sort_values(["spatial_block", "station"])
    )
    print("\n=== SPATIAL BLOCKS ===")
    print(station_blocks.to_string(index=False))
    number_spatial_groups = int(candidate["spatial_block"].nunique())
    print("Spatial groups:", number_spatial_groups)

    metrics, fold_metrics, oof = run_spatial_validation(candidate)
    print("\n=== SPATIAL VALIDATION ===")
    print(metrics.to_string(index=False))
    print("\n=== PERFORMANCE BY SPATIAL FOLD ===")
    print(fold_metrics.to_string(index=False))

    common_final_model = build_random_forest()
    common_final_model.fit(candidate[COMMON_MODEL_FEATURES], candidate[TARGET_COLUMN])

    full_final_model = build_random_forest()
    full_final_model.fit(candidate[FULL_S2_MODEL_FEATURES], candidate[TARGET_COLUMN])

    common_model_path = output_directory / "rf_kc_s2_common_ge90.joblib"
    full_model_path = output_directory / "rf_kc_s2_full_ge90.joblib"
    production_model_path = output_directory / PRODUCTION_MODEL_FILENAME
    metrics_path = output_directory / "kc_model_comparison_ge90.csv"
    folds_path = output_directory / "kc_model_spatial_folds_ge90.csv"
    oof_path = output_directory / "kc_model_oof_predictions_ge90.csv"
    training_population_path = output_directory / "kc_model_training_population_ge90.csv"
    metadata_path = output_directory / "kc_model_comparison_ge90.json"

    joblib.dump(common_final_model, common_model_path)
    joblib.dump(full_final_model, full_model_path)
    joblib.dump(common_final_model, production_model_path)

    metrics.to_csv(metrics_path, index=False)
    fold_metrics.to_csv(folds_path, index=False)
    oof.to_csv(oof_path, index=False)

    training_population_columns = list(
        dict.fromkeys(
            [
                "station",
                "station_id",
                "period_start",
                "spatial_block",
                TARGET_COLUMN,
                "ET_QC",
                "modis_qc_present",
                "optical_union_coverage_pct",
                "s1_union_coverage_pct",
            ]
            + FULL_S2_MODEL_FEATURES
        )
    )
    candidate[training_population_columns].to_csv(training_population_path, index=False)

    metric_records = metrics.set_index("model").to_dict(orient="index")
    metadata = {
        "optical_source": "S2",
        "target": TARGET_COLUMN,
        "training_support": "MODIS footprint x MODIS period",
        "prediction_grid_m": 20,
        "candidate_column": CANDIDATE_COLUMN,
        "optical_coverage_threshold_pct": 90,
        "training_rows": int(len(candidate)),
        "missing_modis_qc_rows_retained": missing_qc,
        "spatial_block_size_km": SPATIAL_BLOCK_SIZE_KM,
        "spatial_groups": number_spatial_groups,
        "random_forest": RF_PARAMETERS,
        "common_features": COMMON_MODEL_FEATURES,
        "full_s2_features": FULL_S2_MODEL_FEATURES,
        "n_common_features": len(COMMON_MODEL_FEATURES),
        "n_full_s2_features": len(FULL_S2_MODEL_FEATURES),
        "selected_model": SELECTED_MODEL_NAME,
        "production_model_filename": PRODUCTION_MODEL_FILENAME,
        "selection_reason": (
            "The 25-feature common RF had slightly better aggregate spatial "
            "R2, RMSE, and KGE than the 31-feature S2 RF; the added S2-specific "
            "features did not provide a consistent improvement."
        ),
        "spatial_validation_metrics": metric_records,
        "baseline_definitions": {
            "global_mean": "Mean Kc of the training folds.",
            "modis_persistence": (
                "Previous MODIS-period Kc at the same station, derived from the "
                "complete 690-row MODIS master before optical candidate filtering."
            ),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nSelected production model:", SELECTED_MODEL_NAME)
    print("Saved production RF:", production_model_path)
    print("Saved common RF:", common_model_path)
    print("Saved full S2 RF:", full_model_path)
    print("Saved comparison:", metrics_path)
    print("Saved folds:", folds_path)
    print("Saved OOF predictions:", oof_path)
    print("Saved training population:", training_population_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
