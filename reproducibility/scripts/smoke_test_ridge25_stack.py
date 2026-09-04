"""Integration smoke test for the real Ridge-25 Earth Engine stack.

The test builds the actual Sentinel-2 + ERA5-Land + harmonic stack for one
MODIS period, evaluates Ridge-25 in Earth Engine, samples real predictor pixels,
and recomputes predictions locally from the exact sampled predictor values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd

from et_downscaling.production import load_basin_geometry
from et_downscaling.ridge25 import (
    RIDGE25_MODEL_FEATURES,
    RIDGE25_MODEL_FILENAME,
    build_ee_ridge25_prediction,
)
from et_downscaling.ridge25_production import build_ridge25_production_stack


DEFAULT_SAMPLE_SIZE = 100
ABSOLUTE_TOLERANCE = 1e-9


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the real Ridge-25 production stack."
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--period-start", required=True)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
    )
    return parser.parse_args()


def model_path(project_root: Path) -> Path:
    return (
        project_root
        / "outputs"
        / "processed"
        / "models"
        / "S2"
        / "2020_2024"
        / RIDGE25_MODEL_FILENAME
    )


def feature_collection_to_dataframe(
    collection: ee.FeatureCollection,
) -> pd.DataFrame:
    payload = collection.getInfo()
    rows = [feature["properties"] for feature in payload["features"]]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_arguments()
    if args.sample_size < 1:
        raise ValueError("--sample-size must be at least 1.")

    project_root = Path(__file__).resolve().parents[2]
    path = model_path(project_root)
    if not path.is_file():
        raise FileNotFoundError(f"Ridge-25 model not found: {path}")

    model = joblib.load(path)

    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()

    basin = load_basin_geometry(project_root)
    context = build_ridge25_production_stack(
        period_start_text=args.period_start,
        basin_geometry=basin,
    )

    band_names = context["stack"].bandNames().getInfo()
    expected = list(RIDGE25_MODEL_FEATURES)
    if band_names != expected:
        raise RuntimeError(
            "Ridge-25 production stack order mismatch.\n"
            f"Expected: {expected}\n"
            f"Found: {band_names}"
        )

    s2_count = int(context["optical_period"].size().getInfo())
    if s2_count == 0:
        raise RuntimeError("No Sentinel-2 observations are available.")

    kc_raw = build_ee_ridge25_prediction(
        model_stack=context["stack"],
        model=model,
    )

    sample_image = context["stack"].toDouble().addBands(kc_raw)
    sample = sample_image.sample(
        region=basin,
        projection=context["fine_projection"],
        scale=20,
        numPixels=args.sample_size,
        seed=42,
        dropNulls=True,
        tileScale=8,
        geometries=False,
    )
    frame = feature_collection_to_dataframe(sample)

    if frame.empty:
        raise RuntimeError("No valid Ridge-25 predictor pixels were sampled.")

    missing = [
        feature
        for feature in expected + ["Kc_raw"]
        if feature not in frame.columns
    ]
    if missing:
        raise RuntimeError(f"Sample is missing expected fields: {missing}")

    local_prediction = model.predict(frame[expected].astype(float))
    ee_prediction = frame["Kc_raw"].to_numpy(dtype=float)
    differences = ee_prediction - local_prediction

    max_absolute_difference = float(np.max(np.abs(differences)))
    mean_absolute_difference = float(np.mean(np.abs(differences)))
    status = (
        "PASS"
        if max_absolute_difference <= ABSOLUTE_TOLERANCE
        else "FAIL"
    )

    print("RIDGE-25 REAL STACK SMOKE TEST")
    print("Period:", args.period_start)
    print("Sentinel-2 products:", s2_count)
    print("Sampled valid pixels:", len(frame))
    print("Predictors:", len(expected))
    print("Kc minimum:", float(np.min(ee_prediction)))
    print("Kc mean:", float(np.mean(ee_prediction)))
    print("Kc maximum:", float(np.max(ee_prediction)))
    print("Maximum absolute local-EE difference:", max_absolute_difference)
    print("Mean absolute local-EE difference:", mean_absolute_difference)
    print("Tolerance:", ABSOLUTE_TOLERANCE)
    print("Status:", status)

    if status != "PASS":
        raise RuntimeError(
            "Real-stack Earth Engine Ridge prediction exceeds tolerance."
        )


if __name__ == "__main__":
    main()
