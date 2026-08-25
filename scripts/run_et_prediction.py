"""Generate one MODIS-period ET product from the accepted production model.

The script does not retrain the model. It builds the real 25-predictor stack
for one MODIS period, predicts Kc, reconciles fine ET to native MODIS support,
calculates DI/AOA, reports map-support QA, and optionally starts a Google Drive
GeoTIFF export.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import ee
import joblib

from et_downscaling.aoa import build_aoa_images, load_aoa_spec
from et_downscaling.config import ANALYSIS_CRS, END_DATE, START_DATE
from et_downscaling.model_spec import (
    COMMON_MODEL_FEATURES,
    PRODUCTION_MODEL_FILENAME,
)
from et_downscaling.model_transfer import build_ee_regressor
from et_downscaling.modis import get_modis_collection
from et_downscaling.production import (
    MODIS_RECONCILIATION_PASSES,
    PREDICTION_SCALE_M,
    build_modis_constrained_et,
    build_production_stack,
    load_basin_geometry,
)


AOA_SAMPLE_PIXELS = 10000


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate one 20 m ET product for a MODIS period."
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--period-start", required=True)
    parser.add_argument(
        "--export",
        action="store_true",
        help="Start a Google Drive GeoTIFF export.",
    )
    parser.add_argument(
        "--drive-folder",
        default="ET_Fundacion",
    )
    return parser.parse_args()


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
        "aoa": model_directory / "aoa_spec.json",
    }


def require_files(paths: dict[str, Path]) -> None:
    missing = [
        str(path)
        for path in paths.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Production artifacts are missing:\n"
            + "\n".join(missing)
            + "\nRun scripts/run_pipeline.py first."
        )


def calculate_screening(
    basin: ee.Geometry,
    stack: ee.Image,
    aoa_outputs: dict[str, ee.Image],
    fine_projection: ee.Projection,
    seed: int,
) -> dict[str, float]:
    """Estimate predictor support and AOA from one reproducible spatial sample.

    This screening is intentionally lightweight. It avoids an exact
    full-basin 20 m reduceRegion, which is unnecessary for deciding whether
    to export a production map and can exceed interactive Earth Engine limits.
    """

    valid_mask = (
        stack
        .mask()
        .reduce(ee.Reducer.min())
        .gt(0)
        .rename("predictor_valid")
        .toByte()
    )

    diagnostic = (
        valid_mask
        .unmask(0)
        .addBands(
            aoa_outputs["di"]
            .rename("DI")
            .unmask(-9999)
            .toFloat()
        )
        .addBands(
            aoa_outputs["aoa"]
            .rename("AOA")
            .unmask(0)
            .toByte()
        )
    )

    sample = diagnostic.sample(
        region=basin,
        projection=fine_projection,
        scale=PREDICTION_SCALE_M,
        numPixels=AOA_SAMPLE_PIXELS,
        seed=seed,
        dropNulls=False,
        tileScale=16,
        geometries=False,
    )

    sample_n = int(
        sample.size().getInfo()
    )

    if sample_n == 0:
        raise RuntimeError(
            "Spatial screening sample contains no pixels."
        )

    valid_fraction = float(
        sample.aggregate_mean(
            "predictor_valid"
        ).getInfo()
    )

    valid_sample = sample.filter(
        ee.Filter.eq(
            "predictor_valid",
            1,
        )
    )

    valid_sample_n = int(
        valid_sample.size().getInfo()
    )

    if valid_sample_n == 0:
        raise RuntimeError(
            "Spatial screening found no pixels with all 25 predictors."
        )

    inside_fraction = float(
        valid_sample.aggregate_mean(
            "AOA"
        ).getInfo()
    )

    di_stats = (
        valid_sample
        .reduceColumns(
            reducer=ee.Reducer.percentile(
                [50, 95]
            ),
            selectors=["DI"],
        )
        .getInfo()
    )

    return {
        "sample_n": sample_n,
        "valid_sample_n": valid_sample_n,
        "valid_pct": (
            100.0 * valid_fraction
        ),
        "inside_aoa_pct": (
            100.0 * inside_fraction
        ),
        "outside_aoa_pct": (
            100.0 * (1.0 - inside_fraction)
        ),
        "di_p50": float(
            di_stats["p50"]
        ),
        "di_p95": float(
            di_stats["p95"]
        ),
    }


def build_product(
    project_root: Path,
    period_start_text: str,
):
    requested_date = date.fromisoformat(period_start_text)
    requested_start = ee.Date(period_start_text)

    modis_count = int(
        get_modis_collection()
        .filterDate(
            requested_start,
            requested_start.advance(1, "day"),
        )
        .size()
        .getInfo()
    )
    if modis_count != 1:
        raise ValueError(
            f"{period_start_text} is not an available MODIS period start."
        )

    basin = load_basin_geometry(project_root)
    context = build_production_stack(
        period_start_text=period_start_text,
        basin_geometry=basin,
    )

    s2_count = int(context["optical_period"].size().getInfo())
    s1_count = int(context["s1_period"].size().getInfo())
    number_days = int(round(float(context["number_days"].getInfo())))

    if s2_count == 0:
        raise RuntimeError("No Sentinel-2 observations are available.")
    if s1_count == 0:
        raise RuntimeError(
            "No R077 ascending Sentinel-1 acquisition is available. "
            "No fallback is applied."
        )

    paths = get_paths(project_root)
    require_files(paths)

    model = joblib.load(paths["model"])
    classifier, tree_strings = build_ee_regressor(
        model,
        COMMON_MODEL_FEATURES,
    )
    aoa_spec = load_aoa_spec(paths["aoa"])

    band_names = context["stack"].bandNames().getInfo()
    if band_names != list(COMMON_MODEL_FEATURES):
        raise RuntimeError(
            "Production predictor order differs from the fitted model schema."
        )

    kc_raw = (
        context["stack"]
        .classify(classifier, "Kc_raw")
        .toFloat()
    )

    et_outputs = build_modis_constrained_et(
        kc_raw=kc_raw,
        optical_predictors=context["optical"],
        s1_predictors=context["s1"],
        model_stack=context["stack"],
        modis_et=context["modis_et"],
        modis_projection=context["modis_projection"],
        fine_projection=context["fine_projection"],
        basin_geometry=basin,
    )

    aoa_outputs = build_aoa_images(
        model_stack=context["stack"],
        specification=aoa_spec,
    )

    screening = calculate_screening(
        basin=basin,
        stack=context["stack"],
        aoa_outputs=aoa_outputs,
        fine_projection=context["fine_projection"],
        seed=int(requested_date.strftime("%Y%m%d")),
    )

    eligible = (
        et_outputs["eligible"]
        .unmask(0)
        .reproject(context["fine_projection"])
        .clip(basin)
        .rename("eligible")
        .toFloat()
    )

    product = (
        et_outputs["et_final"]
        .rename("ET_mm_period")
        .toFloat()
        .addBands(
            aoa_outputs["di"]
            .clip(basin)
            .rename("DI")
            .toFloat()
        )
        .addBands(
            aoa_outputs["aoa"]
            .clip(basin)
            .rename("AOA")
            .toFloat()
        )
        .addBands(eligible)
    )

    metadata = {
        "period_start": period_start_text,
        "number_days": number_days,
        "modis_count": modis_count,
        "s2_count": s2_count,
        "s1_count": s1_count,
        "trees": len(tree_strings),
        "aoa_threshold": float(aoa_spec["threshold"]),
        **screening,
    }
    return product, basin, metadata


def print_summary(metadata: dict) -> None:
    print()
    print("=== PERIOD PRODUCT QA ===")
    print("Period start:", metadata["period_start"])
    print("Period duration (days):", metadata["number_days"])
    print("MODIS images:", metadata["modis_count"])
    print("Sentinel-2 products:", metadata["s2_count"])
    print("Sentinel-1 products:", metadata["s1_count"])
    print("RF trees:", metadata["trees"])

    print()
    print("=== PREDICTOR SUPPORT ===")
    print("Screening sample n:", metadata["sample_n"])
    print("Valid predictor pixels in sample:", metadata["valid_sample_n"])
    print(
        "Estimated valid 25-band coverage (% basin):",
        f"{metadata['valid_pct']:.2f}",
    )

    print()
    print("=== AOA SCREENING ===")
    print("Sample n:", metadata["sample_n"])
    print("DI threshold:", metadata["aoa_threshold"])
    print("Inside AOA (% valid sample):", f"{metadata['inside_aoa_pct']:.2f}")
    print("Outside AOA (% valid sample):", f"{metadata['outside_aoa_pct']:.2f}")
    print("DI median:", f"{metadata['di_p50']:.4f}")
    print("DI p95:", f"{metadata['di_p95']:.4f}")

    current = date.fromisoformat(metadata["period_start"])
    training_start = date.fromisoformat(START_DATE)
    training_end = date.fromisoformat(END_DATE)
    if not (training_start <= current < training_end):
        print()
        print(
            "WARNING: period is outside the model training interval "
            f"{START_DATE} to {END_DATE} (end exclusive)."
        )

    print()
    print(
        "AOA and predictor coverage are support diagnostics, "
        "not independent 20 m accuracy metrics."
    )
    print(
        "ET uses",
        MODIS_RECONCILIATION_PASSES,
        "MODIS reconciliation passes.",
    )


def start_drive_export(
    product: ee.Image,
    basin: ee.Geometry,
    period_start_text: str,
    drive_folder: str,
):
    compact_date = period_start_text.replace("-", "")
    prefix = f"ET_S2_20m_{compact_date}"

    task = ee.batch.Export.image.toDrive(
        image=product,
        description=prefix,
        folder=drive_folder,
        fileNamePrefix=prefix,
        region=basin,
        crs=ANALYSIS_CRS,
        scale=PREDICTION_SCALE_M,
        maxPixels=1_000_000_000,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    print()
    print("Export task started.")
    print("Google Drive folder:", drive_folder)
    print("Task ID:", task.id)


def main():
    args = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]

    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()
    print("Earth Engine initialized with project:", args.project)

    product, basin, metadata = build_product(
        project_root,
        args.period_start,
    )
    print_summary(metadata)

    if args.export:
        start_drive_export(
            product,
            basin,
            args.period_start,
            args.drive_folder,
        )
    else:
        print()
        print("No raster export requested.")


if __name__ == "__main__":
    main()
