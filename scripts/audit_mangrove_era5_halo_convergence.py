"""Audit ST01 halo convergence with nearest-valid ERA5-Land coastal fill."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import ee
import joblib
import pandas as pd
from rasterio.transform import Affine

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

import audit_mangrove_halo_convergence as audit
import produce_field_rf25_halo as base

import et_downscaling.rf25_overlap_production as overlap_production
import et_downscaling.rf25_production as rf25_production

from et_downscaling.meteorology_export import (
    get_era5_collection,
    get_nearest_valid_era5_point,
)
from et_downscaling.rf25 import (
    RF25_METEOROLOGICAL_FEATURES,
    RF25_MODEL_FEATURES,
)


PROJECT = "ee-sneiderquintero"
DATE = "2022-03-30"
STATION_ID = "ST01"

REFERENCE_RADIUS = 4  # 9x9
RADII = [3, 4]        # 7x7 and 9x9

EXTENSION_METHOD = (
    "st04_validation_extension_"
    "era5_nearest_valid_land_pixel_fill_v1"
)

MAX_CONVERGENCE_DIFFERENCE_MM = 0.001
CONSERVATION_TOLERANCE_MM = 0.01


def main():
    stations = base.load_stations(ROOT)
    station = stations[STATION_ID]

    if station["inside_basin"]:
        raise RuntimeError(
            "ST01 was expected outside the official basin domain."
        )

    workspace = base.get_workspace_paths(ROOT).ensure()

    model = joblib.load(
        workspace.models / base.RF25_MODEL_FILENAME
    )
    aoa = joblib.load(
        workspace.models / base.RF25_AOA_FILENAME
    )

    base.validate_rf25_model(model)

    ee.Initialize(project=PROJECT)
    ee.Number(1).getInfo()

    station_point = ee.Geometry.Point(
        [
            station["longitude"],
            station["latitude"],
        ]
    )

    # ------------------------------------------------------------------
    # Resolve exactly the nearest valid ERA5-Land support used for ST01.
    # ------------------------------------------------------------------
    era5_collection = get_era5_collection()

    era5_reference = (
        ee.Image(era5_collection.first())
        .select("temperature_2m")
    )

    era5_projection = era5_reference.projection()
    era5_scale = era5_projection.nominalScale()

    support = get_nearest_valid_era5_point(
        station_point,
        era5_reference,
        era5_projection,
        era5_scale,
    )

    support_point = support.geometry()
    support_coordinates = support_point.coordinates().getInfo()
    support_distance_m = float(
        support.get("era5_distance_m").getInfo()
    )

    support_lon = float(support_coordinates[0])
    support_lat = float(support_coordinates[1])

    print()
    print("=" * 82)
    print("ST01 ERA5-LAND VALIDATION EXTENSION")
    print("=" * 82)
    print("Station:", STATION_ID, station["station"])
    print("ERA5 support longitude:", support_lon)
    print("ERA5 support latitude :", support_lat)
    print("ERA5 distance (m)     :", support_distance_m)

    # ------------------------------------------------------------------
    # Keep the original production builder untouched on disk.
    # Monkeypatch only this audit process.
    # ------------------------------------------------------------------
    original_builder = (
        overlap_production.build_rf25_production_stack
    )

    def build_stack_with_era5_coastal_fill(
        period_start_text,
        basin_geometry,
    ):
        context = original_builder(
            period_start_text=period_start_text,
            basin_geometry=basin_geometry,
        )

        # Period-aggregated ERA5-Land at the same valid land support.
        fallback_source = (
            rf25_production.build_rf25_meteorological_predictors(
                period_start=context["period_start"],
                period_end=context["period_end"],
                number_days=context["number_days"],
                processing_geometry=support_point.buffer(2000),
            )
        )

        fallback_dictionary = fallback_source.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=support_point.buffer(100),
            crs=era5_projection,
            scale=era5_scale,
            maxPixels=1000,
        )

        fallback_values = fallback_dictionary.values(
            RF25_METEOROLOGICAL_FEATURES
        )

        fallback_image = (
            ee.Image.constant(fallback_values)
            .rename(RF25_METEOROLOGICAL_FEATURES)
            .toFloat()
        )

        # Preserve every existing ERA5-Land pixel.
        # Only masked meteorological values receive the ST01
        # nearest-valid-land support.
        meteorology_filled = (
            context["meteorology"]
            .unmask(fallback_image)
            .clip(context["processing_geometry"])
            .toFloat()
        )

        stack = (
            context["optical"]
            .addBands(meteorology_filled)
            .addBands(context["harmonics"])
            .select(RF25_MODEL_FEATURES)
            .reproject(context["fine_projection"])
            .toFloat()
        )

        updated = dict(context)
        updated["meteorology"] = meteorology_filled
        updated["stack"] = stack
        return updated

    overlap_production.build_rf25_production_stack = (
        build_stack_with_era5_coastal_fill
    )

    # Give the validation extension its own scientific identity.
    base_signature = (
        base.build_production_scientific_signature(
            model,
            aoa,
        )
    )

    signature_text = (
        f"{base_signature}|"
        f"{EXTENSION_METHOD}|"
        f"{support_lon:.12f}|"
        f"{support_lat:.12f}|"
        f"{support_distance_m:.6f}"
    )

    extension_signature = hashlib.sha256(
        signature_text.encode("utf-8")
    ).hexdigest()

    print("Extension method       :", EXTENSION_METHOD)
    print("Extension signature    :", extension_signature)

    # ------------------------------------------------------------------
    # Native MODIS parent and one largest (9x9) raw support download.
    # ------------------------------------------------------------------
    context = base.build_modis_period_context(
        DATE,
        station_point.buffer(5000),
    )

    projection_info = (
        context["modis_projection"].getInfo()
    )

    (
        parent_row,
        parent_col,
        native_transform,
    ) = base.native_parent(
        projection_info,
        station["longitude"],
        station["latitude"],
    )

    print(
        "Native MODIS parent    :",
        f"r{parent_row}_c{parent_col}",
    )

    base.HALO_RADIUS = REFERENCE_RADIUS
    base.HALO_SIZE = 2 * REFERENCE_RADIUS + 1

    transform9 = base.halo_transform(
        native_transform,
        parent_row,
        parent_col,
    )

    fine_tile = base.fine_tile_for_halo(
        "ST01_era5_land_halo9_convergence",
        transform9,
    )

    output_root = (
        ROOT
        / "outputs"
        / "evaluation"
        / "field_validation"
        / "st01_validation_extension"
        / "era5_nearest_land_halo_convergence"
        / DATE
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Fine support           :",
        f"{fine_tile.width_m:.0f} x "
        f"{fine_tile.height_m:.0f} m",
    )

    completed = base._download_raw_tile(
        period_start=DATE,
        model=model,
        aoa_parameters=aoa,
        tile=fine_tile,
        tile_directory=output_root / "raw",
        timeout_seconds=600,
        scientific_signature=extension_signature,
    )

    modis9 = base.download_modis_halo(
        context=context,
        projection_info=projection_info,
        target_transform=transform9,
        timeout_seconds=600,
    )

    raw, fine_transform, fine_crs = (
        base.read_raw_tile(completed.path)
    )

    # ------------------------------------------------------------------
    # Solve 7x7 and 9x9 using the same largest raw support.
    # ------------------------------------------------------------------
    results = {}
    summaries = []

    for radius in RADII:
        size = 2 * radius + 1
        offset = REFERENCE_RADIUS - radius

        modis = modis9[
            offset:offset + size,
            offset:offset + size,
        ]

        transform = (
            transform9
            * Affine.translation(
                offset,
                offset,
            )
        )

        print()
        print("=" * 82)
        print(
            f"ST01 {DATE}: solving {size}x{size} "
            "with ERA5 nearest-land fill"
        )
        print("=" * 82)

        result = audit.solve_halo(
            radius=radius,
            raw=raw,
            fine_transform=fine_transform,
            fine_crs=fine_crs,
            modis_et=modis,
            modis_transform=transform,
        )

        results[size] = result
        summaries.append(result["summary"])

        summary = result["summary"]

        print(
            "Central parent eligible :",
            summary["central_parent_eligible"],
        )
        print(
            "Usable fraction         :",
            summary["central_parent_usable_fraction"],
        )
        print(
            "Published pixels        :",
            summary["central_published_pixels"],
        )
        print(
            "Central conservation err:",
            summary["central_conservation_error_mm"],
        )

    comparison = audit.compare(
        results[9],
        results[7],
        "7x7_vs_9x9",
    )

    summary_df = pd.DataFrame(summaries)
    comparison_df = pd.DataFrame([comparison])

    # ------------------------------------------------------------------
    # Pre-declared operational acceptance criteria.
    # ------------------------------------------------------------------
    masks_match = (
        comparison["candidate_only_pixels"] == 0
        and comparison["reference_only_pixels"] == 0
    )

    has_common_pixels = (
        comparison["common_published_pixels"] > 0
    )

    convergence_ok = (
        math.isfinite(
            comparison["max_abs_difference_mm"]
        )
        and comparison["max_abs_difference_mm"]
        <= MAX_CONVERGENCE_DIFFERENCE_MM
    )

    eligible_ok = all(
        bool(row["central_parent_eligible"])
        for row in summaries
    )

    conservation_ok = all(
        math.isfinite(
            row["central_conservation_error_mm"]
        )
        and abs(
            row["central_conservation_error_mm"]
        )
        <= CONSERVATION_TOLERANCE_MM
        for row in summaries
    )

    accepted = all(
        [
            masks_match,
            has_common_pixels,
            convergence_ok,
            eligible_ok,
            conservation_ok,
        ]
    )

    summary_path = (
        output_root
        / "st01_era5_halo_summary.csv"
    )

    comparison_path = (
        output_root
        / "st01_era5_halo_comparison.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    metadata = {
        "date": DATE,
        "station_id": STATION_ID,
        "station": station["station"],
        "inside_official_basin": station["inside_basin"],
        "method": EXTENSION_METHOD,
        "era5_sampling_method": "nearest_valid_land_pixel",
        "era5_sampling_longitude": support_lon,
        "era5_sampling_latitude": support_lat,
        "era5_sampling_distance_m": support_distance_m,
        "base_rf25_scientific_signature": base_signature,
        "extension_scientific_signature": extension_signature,
        "tested_halos": ["7x7", "9x9"],
        "max_convergence_difference_mm": (
            MAX_CONVERGENCE_DIFFERENCE_MM
        ),
        "conservation_tolerance_mm": (
            CONSERVATION_TOLERANCE_MM
        ),
        "accepted": accepted,
    }

    (
        output_root
        / "audit_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 82)
    print("HALO SUMMARY")
    print("=" * 82)
    print(summary_df.to_string(index=False))

    print()
    print("=" * 82)
    print("7x7 VS 9x9")
    print("=" * 82)
    print(comparison_df.to_string(index=False))

    print()
    print("=" * 82)
    print("PRE-DECLARED ACCEPTANCE")
    print("=" * 82)
    print("Central eligible both :", eligible_ok)
    print("Mask agreement        :", masks_match)
    print("Common pixels > 0     :", has_common_pixels)
    print(
        "Max diff <= 0.001 mm :",
        convergence_ok,
    )
    print(
        "Conservation <= 0.01 :",
        conservation_ok,
    )
    print()
    print(
        "ST01 HALO-7 ACCEPTED :",
        accepted,
    )

    print()
    print("Saved:")
    print(summary_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
