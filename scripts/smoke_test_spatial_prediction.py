"""Run one end-to-end 20 m ET production smoke test.

This is intentionally the only pre-production raster test. It verifies:

1. all fitted Random Forest trees are transferred to Earth Engine;
2. Earth Engine and local sklearn predictions agree numerically;
3. the 25-band spatial predictor stack uses the accepted definitions;
4. Sentinel-2 and Sentinel-1 support are sufficient for the test period;
5. all eligible parent MODIS cells in the local smoke window conserve ET after proportional reconciliation;
6. optionally, one 20 m GeoTIFF is downloaded locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import ee
import joblib
import numpy as np
import pandas as pd

from et_downscaling.config import ANALYSIS_CRS
from et_downscaling.model_spec import (
    COMMON_MODEL_FEATURES,
    PRODUCTION_MODEL_FILENAME,
    RF_PARAMETERS,
)
from et_downscaling.model_transfer import build_ee_regressor
from et_downscaling.stations import load_station_dataframe
from et_downscaling.production import (
    MODIS_CONSERVATION_TOLERANCE_MM,
    PREDICTION_SCALE_M,
    build_modis_constrained_et,
    build_production_stack,
    load_basin_geometry,
)


TRANSFER_MAX_ABS_TOLERANCE = 1e-4
TRANSFER_RMSE_TOLERANCE = 1e-5
CONSERVATION_MAX_ABS_TOLERANCE_MM = MODIS_CONSERVATION_TOLERANCE_MM
TRANSFER_SAMPLE_ROWS = 30
SMOKE_STATION_ID = "ST01"
SMOKE_BUFFER_M = 2000


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Validate and optionally download one 20 m ET period."
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Google Cloud Project ID. If omitted, ask interactively.",
    )
    parser.add_argument(
        "--period-start",
        default=None,
        help=(
            "MODIS period start (YYYY-MM-DD). If omitted, select the strongest "
            "period from the accepted 349-row training population."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the smoke-test ET image as a local GeoTIFF.",
    )
    return parser.parse_args()


def initialize_earth_engine(project_id: str | None) -> str:
    if project_id is None:
        project_id = input("Google Cloud Project ID: ").strip()
    if not project_id:
        raise ValueError("Google Cloud Project ID cannot be empty.")

    ee.Initialize(project=project_id)
    ee.Number(1).getInfo()
    print("Earth Engine initialized with project:", project_id)
    return project_id


def get_paths(project_root: Path) -> dict[str, Path]:
    model_directory = (
        project_root
        / "outputs"
        / "processed"
        / "models"
        / "S2"
    )
    return {
        "model": model_directory / PRODUCTION_MODEL_FILENAME,
        "training": model_directory / "kc_model_training_population_ge90.csv",
        "output": (
            project_root
            / "outputs"
            / "processed"
            / "predictions"
            / "S2"
        ),
    }


def select_smoke_period(
    training: pd.DataFrame,
) -> str:
    """Choose a well-supported period without introducing another experiment."""
    required = {
        "period_start",
        "station_id",
        "optical_union_coverage_pct",
    }
    missing = required - set(training.columns)
    if missing:
        raise ValueError(
            "Training-population file is missing columns needed for automatic "
            f"period selection: {sorted(missing)}"
        )

    summary = (
        training
        .groupby("period_start", as_index=False)
        .agg(
            station_count=("station_id", "nunique"),
            mean_optical_coverage=("optical_union_coverage_pct", "mean"),
        )
    )

    max_station_count = int(summary["station_count"].max())
    best = (
        summary.loc[summary["station_count"] == max_station_count]
        .sort_values(
            ["mean_optical_coverage", "period_start"],
            ascending=[False, True],
        )
        .iloc[0]
    )

    selected = str(best["period_start"])
    print(
        "Auto-selected smoke period:",
        selected,
        f"({int(best['station_count'])} stations, "
        f"mean station optical coverage {best['mean_optical_coverage']:.2f}%)",
    )
    return selected


def validate_local_model(
    model,
) -> None:
    if int(model.n_features_in_) != len(COMMON_MODEL_FEATURES):
        raise RuntimeError(
            "Production model feature count does not match model_spec.py."
        )
    expected_trees = int(RF_PARAMETERS["n_estimators"])
    if int(model.n_estimators) != expected_trees:
        raise RuntimeError(
            "Production Random Forest tree count does not match "
            f"model_spec.py: expected {expected_trees}, "
            f"found {int(model.n_estimators)}."
        )


def validate_ee_model_transfer(
    model,
    classifier: ee.Classifier,
    training: pd.DataFrame,
) -> None:
    """Compare local sklearn and EE predictions on identical feature rows."""
    sample = (
        training
        .sample(
            n=min(TRANSFER_SAMPLE_ROWS, len(training)),
            random_state=42,
        )
        .reset_index(drop=True)
    )

    x = sample[COMMON_MODEL_FEATURES].astype(float)
    local_prediction = model.predict(x)

    features = []
    for row_id, (_, row) in enumerate(x.iterrows()):
        properties = {
            feature: float(row[feature])
            for feature in COMMON_MODEL_FEATURES
        }
        properties["row_id"] = row_id
        features.append(ee.Feature(None, properties))

    predicted = (
        ee.FeatureCollection(features)
        .classify(classifier, "Kc_ee")
        .sort("row_id")
        .aggregate_array("Kc_ee")
        .getInfo()
    )

    ee_prediction = np.asarray(predicted, dtype=float)
    error = ee_prediction - np.asarray(local_prediction, dtype=float)
    max_abs = float(np.max(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    print()
    print("=== RF TRANSFER VALIDATION ===")
    print("Rows:", len(sample))
    print("Maximum absolute difference:", max_abs)
    print("RMSE difference:", rmse)

    if (
        max_abs > TRANSFER_MAX_ABS_TOLERANCE
        or rmse > TRANSFER_RMSE_TOLERANCE
    ):
        raise RuntimeError(
            "Earth Engine Random Forest transfer failed numerical equivalence. "
            "Do not produce ET maps."
        )

    print("RF transfer: PASS")


def get_max_conservation_error(
    conservation_error: ee.Image,
    geometry: ee.Geometry,
    modis_projection: ee.Projection,
) -> tuple[float, int]:
    """Evaluate all eligible MODIS parent pixels in the local smoke window."""
    error = ee.Image(conservation_error).abs()

    stats = (
        error
        .reduceRegion(
            reducer=(
                ee.Reducer.max()
                .combine(
                    reducer2=ee.Reducer.count(),
                    sharedInputs=True,
                )
            ),
            geometry=geometry,
            crs=modis_projection,
            scale=modis_projection.nominalScale(),
            maxPixels=10000,
            tileScale=4,
        )
        .getInfo()
    )

    maximum = stats.get("ET_conservation_error_mm_max")
    count = stats.get("ET_conservation_error_mm_count")

    if maximum is None or count is None or int(count) == 0:
        raise RuntimeError(
            "No MODIS pixels passed production eligibility "
            "inside the local smoke-test window."
        )

    return float(maximum), int(count)


def get_support_summary(
    outputs: dict[str, ee.Image],
    geometry: ee.Geometry,
    modis_projection: ee.Projection,
) -> dict[str, float]:
    """Summarize coarse support over the local smoke-test window."""
    qa = (
        outputs["optical_valid_fraction"]
        .addBands(outputs["s1_valid_fraction"])
        .addBands(outputs["model_stack_valid_fraction"])
        .addBands(outputs["fine_fill_fraction"])
    )

    values = (
        qa
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            crs=modis_projection,
            scale=modis_projection.nominalScale(),
            maxPixels=10000,
            tileScale=4,
        )
        .getInfo()
    )

    return {
        key: float(value)
        for key, value in values.items()
        if value is not None
    }


def download_et_image(
    et_image: ee.Image,
    basin: ee.Geometry,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tif.part")

    url = et_image.toFloat().getDownloadURL(
        {
            "name": output_path.stem,
            "scale": PREDICTION_SCALE_M,
            "crs": ANALYSIS_CRS,
            "region": basin,
            "format": "GEO_TIFF",
        }
    )

    print("Downloading smoke-test GeoTIFF...")
    urlretrieve(url, temporary_path)
    temporary_path.replace(output_path)
    print("Saved ET GeoTIFF:", output_path)



def build_smoke_geometry() -> ee.Geometry:
    """Build a small deterministic production window for integration QA."""
    stations = load_station_dataframe()

    selected = stations.loc[
        stations["station_id"] == SMOKE_STATION_ID
    ]

    if len(selected) != 1:
        raise RuntimeError(
            f"Expected exactly one {SMOKE_STATION_ID} station."
        )

    row = selected.iloc[0]

    return (
        ee.Geometry.Point(
            [
                float(row["longitude"]),
                float(row["latitude"]),
            ]
        )
        .buffer(SMOKE_BUFFER_M)
        .bounds()
    )

def main():
    args = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    paths = get_paths(project_root)

    if not paths["model"].is_file():
        raise FileNotFoundError(
            f"Production model not found: {paths['model']}\n"
            "Run first: python scripts/train_s2_kc_models.py"
        )
    if not paths["training"].is_file():
        raise FileNotFoundError(
            f"Training-population QA file not found: {paths['training']}"
        )

    training = pd.read_csv(
        paths["training"],
        dtype={"station_id": "string"},
    )
    model = joblib.load(paths["model"])
    validate_local_model(model)

    period_start = args.period_start or select_smoke_period(training)

    initialize_earth_engine(args.project)

    print()
    print("Transferring production RF to Earth Engine...")
    classifier, tree_strings = build_ee_regressor(
        model,
        COMMON_MODEL_FEATURES,
    )
    total_tree_bytes = sum(len(tree.encode("utf-8")) for tree in tree_strings)
    print("Trees transferred:", len(tree_strings))
    print("Serialized model size (MB):", round(total_tree_bytes / 1024**2, 3))

    validate_ee_model_transfer(
        model=model,
        classifier=classifier,
        training=training,
    )

    basin = load_basin_geometry(project_root)
    smoke_geometry = build_smoke_geometry()

    print()
    print("Building LOCAL 20 m production stack for:", period_start)
    print("Smoke station:", SMOKE_STATION_ID)
    print("Smoke buffer (m):", SMOKE_BUFFER_M)

    context = build_production_stack(
        period_start_text=period_start,
        basin_geometry=smoke_geometry,
    )

    modis_count = int(context["collection"].size().getInfo())
    s2_count = int(context["optical_period"].size().getInfo())
    s1_count = int(context["s1_period"].size().getInfo())

    print("MODIS images:", modis_count)
    print("Sentinel-2 products:", s2_count)
    print("Sentinel-1 products:", s1_count)

    if modis_count != 1:
        raise RuntimeError(
            f"Expected one MODIS image for {period_start}; found {modis_count}."
        )
    if s2_count == 0:
        raise RuntimeError("Selected period contains no Sentinel-2 observations.")
    if s1_count == 0:
        raise RuntimeError(
            "Selected period contains no R077 ascending Sentinel-1 acquisition. "
            "This confirms an operational temporal-coverage gap; choose another "
            "smoke period before designing any fallback."
        )

    stack_band_names = context["stack"].bandNames().getInfo()
    if stack_band_names != list(COMMON_MODEL_FEATURES):
        raise RuntimeError(
            "Spatial predictor order differs from the fitted model schema.\n"
            f"Expected: {COMMON_MODEL_FEATURES}\nFound: {stack_band_names}"
        )

    kc_raw = (
        context["stack"]
        .classify(classifier, "Kc_raw")
        .toFloat()
    )

    outputs = build_modis_constrained_et(
        kc_raw=kc_raw,
        optical_predictors=context["optical"],
        s1_predictors=context["s1"],
        model_stack=context["stack"],
        modis_et=context["modis_et"],
        modis_projection=context["modis_projection"],
        fine_projection=context["fine_projection"],
        basin_geometry=smoke_geometry,
    )

    max_conservation_error, conservation_pixels = (
        get_max_conservation_error(
            outputs["conservation_error"],
            smoke_geometry,
            context["modis_projection"],
        )
    )

    support = get_support_summary(
        outputs,
        smoke_geometry,
        context["modis_projection"],
    )

    print()
    print("=== LOCAL SPATIAL SUPPORT QA ===")
    for key, value in support.items():
        print(f"{key}: {value:.6f}")

    print()
    print("=== LOCAL MODIS CONSERVATION ===")
    print(
        "Eligible MODIS pixels:",
        conservation_pixels,
    )
    print(
        "Maximum absolute error (mm/period):",
        max_conservation_error,
    )

    if max_conservation_error > CONSERVATION_MAX_ABS_TOLERANCE_MM:
        raise RuntimeError(
            "MODIS mass-conservation tolerance was not met in the local smoke window. "
            "Do not export the ET product."
        )

    print("MODIS conservation: PASS")

    if args.download:
        output_path = (
            paths["output"]
            / f"ET_S2_20m_{period_start.replace('-', '')}.tif"
        )
        download_et_image(
            outputs["et_final"],
            smoke_geometry,
            output_path,
        )

    print()
    print("Smoke test: PASS")
    print("Period:", period_start)
    print("Production grid (m):", PREDICTION_SCALE_M)
    print("Production model trees:", len(tree_strings))


if __name__ == "__main__":
    main()

