"""Smoke-test exact Ridge-25 transfer from scikit-learn to Earth Engine.

This script does not read satellite imagery and does not modify the production
pipeline. It sends deterministic training rows to Earth Engine as feature
properties, evaluates the exact StandardScaler + Ridge equation server-side,
and compares those predictions with the saved scikit-learn pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd


MODEL_FILENAME = "ridge_kc_s2_rededge25_ge90.joblib"
SPEC_FILENAME = "ridge25_model_spec.json"
POPULATION_FILENAME = "ridge25_training_population.csv"
OUTPUT_FILENAME = "ridge25_ee_transfer_smoke_test.json"
DEFAULT_SAMPLE_SIZE = 10
ABSOLUTE_TOLERANCE = 1e-9


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved Ridge-25 predictions with Earth Engine."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Google Earth Engine project used for initialization.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of deterministic training rows to test.",
    )
    return parser.parse_args()


def get_model_directory(project_root: Path) -> Path:
    return (
        project_root
        / "outputs"
        / "processed"
        / "models"
        / "S2"
        / "2020_2024"
    )


def load_artifacts(model_directory: Path):
    model_path = model_directory / MODEL_FILENAME
    spec_path = model_directory / SPEC_FILENAME
    population_path = model_directory / POPULATION_FILENAME

    missing = [
        str(path)
        for path in (model_path, spec_path, population_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Ridge-25 artifact(s):\n" + "\n".join(missing)
        )

    model = joblib.load(model_path)
    specification = json.loads(spec_path.read_text(encoding="utf-8"))
    population = pd.read_csv(population_path)

    return model, specification, population


def validate_pipeline(model, features: list[str]) -> tuple[np.ndarray, ...]:
    if not hasattr(model, "named_steps"):
        raise TypeError("Expected a scikit-learn Pipeline.")

    if "scaler" not in model.named_steps or "regressor" not in model.named_steps:
        raise ValueError("Pipeline must contain scaler and regressor steps.")

    scaler = model.named_steps["scaler"]
    regressor = model.named_steps["regressor"]

    means = np.asarray(scaler.mean_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)
    coefficients = np.asarray(regressor.coef_, dtype=float)
    intercept = float(regressor.intercept_)

    expected_count = len(features)
    for name, values in (
        ("means", means),
        ("scales", scales),
        ("coefficients", coefficients),
    ):
        if len(values) != expected_count:
            raise ValueError(
                f"Feature-count mismatch for {name}: "
                f"{len(values)} != {expected_count}."
            )

    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("Invalid StandardScaler scales.")

    return means, scales, coefficients, intercept


def choose_sample(population: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if sample_size < 1:
        raise ValueError("--sample-size must be at least 1.")
    if sample_size > len(population):
        raise ValueError(
            f"--sample-size {sample_size} exceeds population {len(population)}."
        )

    indices = np.linspace(
        0,
        len(population) - 1,
        num=sample_size,
        dtype=int,
    )
    return population.iloc[indices].copy().reset_index(drop=True)


def build_ee_feature_collection(
    sample: pd.DataFrame,
    features: list[str],
) -> ee.FeatureCollection:
    ee_features = []
    for row_id, row in sample.iterrows():
        properties = {"row_id": int(row_id)}
        properties.update(
            {
                feature: float(row[feature])
                for feature in features
            }
        )
        ee_features.append(ee.Feature(None, properties))
    return ee.FeatureCollection(ee_features)


def predict_ee(
    feature_collection: ee.FeatureCollection,
    features: list[str],
    means: np.ndarray,
    scales: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
) -> list[float]:
    def predict_feature(feature):
        feature = ee.Feature(feature)
        prediction = ee.Number(intercept)

        for name, mean, scale, coefficient in zip(
            features,
            means,
            scales,
            coefficients,
            strict=True,
        ):
            standardized = (
                ee.Number(feature.get(name))
                .subtract(float(mean))
                .divide(float(scale))
            )
            prediction = prediction.add(
                standardized.multiply(float(coefficient))
            )

        return feature.set("ee_prediction", prediction)

    evaluated = feature_collection.map(predict_feature)
    return [
        float(value)
        for value in evaluated.aggregate_array("ee_prediction").getInfo()
    ]


def main() -> None:
    args = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    model_directory = get_model_directory(project_root)

    model, specification, population = load_artifacts(model_directory)
    features = list(specification["features"])

    missing_features = [
        feature
        for feature in features
        if feature not in population.columns
    ]
    if missing_features:
        raise ValueError(
            f"Training population is missing features: {missing_features}"
        )

    means, scales, coefficients, intercept = validate_pipeline(
        model,
        features,
    )

    sample = choose_sample(population, args.sample_size)
    local_predictions = np.asarray(
        model.predict(sample[features]),
        dtype=float,
    )

    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()

    feature_collection = build_ee_feature_collection(sample, features)
    ee_predictions = np.asarray(
        predict_ee(
            feature_collection=feature_collection,
            features=features,
            means=means,
            scales=scales,
            coefficients=coefficients,
            intercept=intercept,
        ),
        dtype=float,
    )

    differences = ee_predictions - local_predictions
    max_absolute_difference = float(np.max(np.abs(differences)))
    mean_absolute_difference = float(np.mean(np.abs(differences)))

    result = {
        "status": (
            "PASS"
            if max_absolute_difference <= ABSOLUTE_TOLERANCE
            else "FAIL"
        ),
        "sample_size": int(len(sample)),
        "feature_count": int(len(features)),
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "max_absolute_difference": max_absolute_difference,
        "mean_absolute_difference": mean_absolute_difference,
        "local_predictions": local_predictions.tolist(),
        "earth_engine_predictions": ee_predictions.tolist(),
        "differences": differences.tolist(),
    }

    output_path = model_directory / OUTPUT_FILENAME
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("RIDGE-25 EARTH ENGINE TRANSFER SMOKE TEST")
    print("Rows:", result["sample_size"])
    print("Predictors:", result["feature_count"])
    print("Maximum absolute difference:", max_absolute_difference)
    print("Mean absolute difference:", mean_absolute_difference)
    print("Tolerance:", ABSOLUTE_TOLERANCE)
    print("Status:", result["status"])
    print("Output:", output_path)

    if result["status"] != "PASS":
        raise RuntimeError(
            "Earth Engine Ridge transfer exceeds numerical tolerance."
        )


if __name__ == "__main__":
    main()
