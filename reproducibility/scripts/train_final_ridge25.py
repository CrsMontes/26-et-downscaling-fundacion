"""Train and audit the final Ridge-25 Kc model on the fixed S2 GE90 population.

This script is intentionally local-only. It does not access Earth Engine and it
never modifies earlier diagnostic gates. It reads the master predictor store if
available, otherwise the experimental feature store, rebuilds the fixed 799-row
GE90 population, evaluates the final Ridge-25 model with spatial leave-one-block-
out and LOYO validation, then fits and saves the production sklearn pipeline.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KEYS = ["station_id", "period_start"]
TARGET = "Kc_target"
EXPECTED_ROWS = 799
EXPECTED_SPATIAL_COUNTS = {
    "-811_116": 168,
    "-814_118": 297,
    "-814_119": 169,
    "-815_118": 165,
}
EXPECTED_YEAR_COUNTS = {2020: 171, 2021: 151, 2022: 142, 2023: 185, 2024: 150}

MASTER_FEATURES = [
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
    "s2_RedEdge1_mean",
    "s2_RedEdge2_mean",
    "s2_RedEdge3_mean",
    "s2_NIR_Broad_mean",
    "s2_NDRE_mean",
    "Tair_mean_C",
    "Tair_max_C",
    "VPD_mean_kPa",
    "SolarRad_MJ_m2_day",
    "Wind_mean_ms",
    "doy_sin1",
    "doy_cos1",
    "doy_sin2",
    "doy_cos2",
]

PRODUCTION_FEATURES = [
    "Blue_mean",
    "Green_mean",
    "Red_mean",
    "NIR_mean",
    "SWIR1_mean",
    "SWIR2_mean",
    "NDVI_mean",
    "EVI_mean",
    "SAVI_mean",
    "NDWI_mean",
    "NDMI_mean",
    "RedEdge1_mean",
    "RedEdge2_mean",
    "RedEdge3_mean",
    "NIR_Broad_mean",
    "NDRE_mean",
    "Tair_mean_C",
    "Tair_max_C",
    "VPD_mean_kPa",
    "SolarRad_MJ_m2_day",
    "Wind_mean_ms",
    "doy_sin1",
    "doy_cos1",
    "doy_sin2",
    "doy_cos2",
]

RENAME_TO_PRODUCTION = dict(zip(MASTER_FEATURES, PRODUCTION_FEATURES, strict=True))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - observed
    observed_sd = observed.std(ddof=0)
    predicted_sd = predicted.std(ddof=0)
    if len(observed) < 2 or observed_sd == 0 or predicted_sd == 0:
        kge = np.nan
    else:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
        alpha = float(predicted_sd / observed_sd)
        beta = float(predicted.mean() / observed.mean())
        kge = 1.0 - np.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    return {
        "n": int(len(observed)),
        "R2": float(r2_score(observed, predicted)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "BIAS": float(np.mean(error)),
        "KGE": float(kge),
    }


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, fit_intercept=True)),
        ]
    )


def locate_input(root: Path) -> Path:
    candidates = [
        root
        / "outputs"
        / "diagnostics"
        / "2020_2024"
        / "predictor_availability_ladder"
        / "master_predictor_store.parquet",
        root
        / "outputs"
        / "diagnostics"
        / "2020_2024"
        / "experimental_feature_store"
        / "feature_store.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("No supported master predictor store was found.")


def load_store(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        data = pd.read_parquet(path)
    else:
        data = pd.read_csv(path, dtype={"station_id": str})
    data["station_id"] = data["station_id"].astype(str)
    data["period_start"] = pd.to_datetime(data["period_start"], errors="raise").dt.strftime("%Y-%m-%d")
    if data.duplicated(KEYS).any():
        raise RuntimeError("Master predictor store contains duplicate station-period keys.")
    return data


def build_population(store: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + [TARGET, "modis_good", "target_complete", "s2_coverage_pct", "spatial_block", "year"] + MASTER_FEATURES)
    missing = sorted(required - set(store.columns))
    if missing:
        raise RuntimeError(f"Master predictor store is missing required columns: {missing}")

    working = store.copy()
    numeric = MASTER_FEATURES + [TARGET, "s2_coverage_pct"]
    working[numeric] = working[numeric].apply(pd.to_numeric, errors="coerce")
    eligible = (
        working["modis_good"].eq(1)
        & working["target_complete"].eq(1)
        & working[TARGET].notna()
        & working["s2_coverage_pct"].ge(90.0)
        & working[MASTER_FEATURES].notna().all(axis=1)
    )
    selected = working.loc[eligible].copy().sort_values(KEYS).reset_index(drop=True)
    if len(selected) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} S2 GE90 rows, found {len(selected)}.")
    if selected[MASTER_FEATURES + [TARGET]].isna().any().any():
        raise RuntimeError("Final Ridge-25 matrix contains missing values.")
    if not np.isfinite(selected[MASTER_FEATURES + [TARGET]].to_numpy(dtype=float)).all():
        raise RuntimeError("Final Ridge-25 matrix contains non-finite values.")

    spatial_counts = selected.groupby("spatial_block").size().to_dict()
    if spatial_counts != EXPECTED_SPATIAL_COUNTS:
        raise RuntimeError(f"Unexpected spatial fold population: {spatial_counts}")
    year_counts = {int(key): int(value) for key, value in selected.groupby("year").size().to_dict().items()}
    if year_counts != EXPECTED_YEAR_COUNTS:
        raise RuntimeError(f"Unexpected LOYO population: {year_counts}")

    selected = selected.rename(columns=RENAME_TO_PRODUCTION)
    return selected


def oof_predictions(data: pd.DataFrame, split_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = np.full(len(data), np.nan, dtype=float)
    fold_rows = []
    groups = data[split_column].astype(str)
    for fold_id, group in enumerate(sorted(groups.unique()), start=1):
        test_mask = groups.eq(group).to_numpy()
        train_mask = ~test_mask
        model = build_model()
        model.fit(data.loc[train_mask, PRODUCTION_FEATURES], data.loc[train_mask, TARGET])
        predicted = model.predict(data.loc[test_mask, PRODUCTION_FEATURES])
        predictions[test_mask] = predicted
        fold_metrics = calculate_metrics(data.loc[test_mask, TARGET].to_numpy(), predicted)
        fold_rows.append({"fold": fold_id, "group": group, **fold_metrics})
    if np.isnan(predictions).any():
        raise RuntimeError("OOF prediction vector contains missing values.")
    output = data[KEYS + ["spatial_block", "year", TARGET]].copy()
    output["prediction"] = predictions
    output["error"] = predictions - data[TARGET].to_numpy(dtype=float)
    return output, pd.DataFrame(fold_rows)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    input_path = locate_input(root)
    store = load_store(input_path)
    data = build_population(store)

    spatial_oof, spatial_folds = oof_predictions(data, "spatial_block")
    temporal_oof, temporal_folds = oof_predictions(data, "year")
    spatial_metrics = calculate_metrics(spatial_oof[TARGET], spatial_oof["prediction"])
    temporal_metrics = calculate_metrics(temporal_oof[TARGET], temporal_oof["prediction"])

    # Guardrails from the completed red-edge decomposition experiment.
    if abs(spatial_metrics["R2"] - 0.3800) > 0.01 or abs(spatial_metrics["RMSE"] - 0.2535) > 0.005:
        raise RuntimeError(f"Ridge-25 spatial metrics do not reproduce the accepted gate: {spatial_metrics}")

    final_model = build_model()
    final_model.fit(data[PRODUCTION_FEATURES], data[TARGET])

    model_dir = root / "outputs" / "processed" / "models" / "S2" / "2020_2024"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ridge_kc_s2_rededge25_ge90.joblib"
    spec_path = model_dir / "ridge25_model_spec.json"
    population_path = model_dir / "ridge25_training_population.csv"
    spatial_oof_path = model_dir / "ridge25_spatial_oof.csv"
    temporal_oof_path = model_dir / "ridge25_temporal_oof.csv"
    fold_path = model_dir / "ridge25_fold_metrics.csv"

    joblib.dump(final_model, model_path)
    data[KEYS + ["spatial_block", "year", TARGET] + PRODUCTION_FEATURES].to_csv(population_path, index=False)
    spatial_oof.to_csv(spatial_oof_path, index=False)
    temporal_oof.to_csv(temporal_oof_path, index=False)
    pd.concat(
        [
            spatial_folds.assign(split_type="spatial"),
            temporal_folds.assign(split_type="temporal"),
        ],
        ignore_index=True,
    ).to_csv(fold_path, index=False)

    scaler = final_model.named_steps["scaler"]
    regressor = final_model.named_steps["regressor"]
    coefficient_weights = np.abs(regressor.coef_.astype(float))
    coefficient_weights = coefficient_weights / coefficient_weights.sum()

    spec = {
        "model_name": "ridge25_s2_rededge_ge90",
        "algorithm": "Ridge",
        "alpha": 1.0,
        "target": TARGET,
        "training_rows": len(data),
        "ge90_threshold": 0.90,
        "features": PRODUCTION_FEATURES,
        "input_feature_names": MASTER_FEATURES,
        "standard_scaler_mean": scaler.mean_.astype(float).tolist(),
        "standard_scaler_scale": scaler.scale_.astype(float).tolist(),
        "ridge_coefficients_standardized": regressor.coef_.astype(float).tolist(),
        "ridge_intercept": float(regressor.intercept_),
        "aoa_candidate_weights_abs_standardized_coefficients_normalized": coefficient_weights.tolist(),
        "validation": {
            "spatial": spatial_metrics,
            "temporal_loyo": temporal_metrics,
        },
        "input": {
            "path": str(input_path.relative_to(root)),
            "sha256": sha256_file(input_path),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "winner_frozen": True,
        "product_frozen": False,
        "notes": [
            "Predictor selection is frozen at Ridge-25 after the availability ladder and red-edge decomposition gate.",
            "AOA, Earth Engine transfer, cartographic gate, reconciliation, and field comparison remain downstream checks.",
        ],
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    print("FINAL RIDGE-25 TRAINING COMPLETE")
    print(f"Input: {input_path}")
    print(f"Rows: {len(data)}")
    print(f"Predictors: {len(PRODUCTION_FEATURES)}")
    print("Spatial:", json.dumps(spatial_metrics, indent=2))
    print("Temporal LOYO:", json.dumps(temporal_metrics, indent=2))
    print(f"Model: {model_path}")
    print(f"Spec: {spec_path}")


if __name__ == "__main__":
    main()
