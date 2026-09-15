"""Produce ST01 RF25 halo-7 validation products with ERA5-Land coastal support."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import ee

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

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
STATION_ID = "ST01"

METHOD = (
    "st04_validation_extension_"
    "era5_nearest_valid_land_pixel_fill_v1"
)


def main():
    ee.Initialize(project=PROJECT)
    ee.Number(1).getInfo()

    station = base.load_stations(ROOT)[STATION_ID]

    point = ee.Geometry.Point(
        [station["longitude"], station["latitude"]]
    )

    era5_collection = get_era5_collection()

    era5_reference = (
        ee.Image(era5_collection.first())
        .select("temperature_2m")
    )

    era5_projection = era5_reference.projection()
    era5_scale = era5_projection.nominalScale()

    support = get_nearest_valid_era5_point(
        point,
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

    print("=" * 80)
    print("ST01 VALIDATION-DOMAIN EXTENSION")
    print("=" * 80)
    print("ERA5 support:", support_lon, support_lat)
    print("Distance (m):", support_distance_m)
    print("Method:", METHOD)

    original_builder = (
        overlap_production.build_rf25_production_stack
    )

    original_signature_builder = (
        base.build_production_scientific_signature
    )

    def build_stack_with_fill(
        period_start_text,
        basin_geometry,
    ):
        context = original_builder(
            period_start_text=period_start_text,
            basin_geometry=basin_geometry,
        )

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

    def build_extension_signature(model, aoa):
        base_signature = original_signature_builder(
            model,
            aoa,
        )

        text = (
            f"{base_signature}|"
            f"{METHOD}|"
            f"{support_lon:.12f}|"
            f"{support_lat:.12f}|"
            f"{support_distance_m:.6f}"
        )

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    overlap_production.build_rf25_production_stack = (
        build_stack_with_fill
    )

    base.build_production_scientific_signature = (
        build_extension_signature
    )

    manifest_dir = (
        ROOT
        / "outputs"
        / "evaluation"
        / "field_validation"
        / "local_halo7_products"
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "station_id": STATION_ID,
        "station": station["station"],
        "inside_official_basin": station["inside_basin"],
        "validation_domain": "extension_outside_official_basin",
        "method": METHOD,
        "era5_support_method": "nearest_valid_land_pixel",
        "era5_support_longitude": support_lon,
        "era5_support_latitude": support_lat,
        "era5_support_distance_m": support_distance_m,
        "halo_size": "7x7",
        "halo_acceptance_audit_date": "2022-03-30",
        "halo_7_vs_9_max_abs_difference_mm": 0.000002,
        "note": (
            "Missing ERA5-Land meteorological predictors are filled "
            "from the nearest valid ERA5-Land support used for ST01. "
            "RF25 model, AOA, Sentinel-2 predictors, temporal predictors, "
            "MODIS parent ET and exact-overlap reconciliation are unchanged."
        ),
    }

    (
        manifest_dir
        / "ST01_validation_extension_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    sys.argv = [
        "produce_field_rf25_halo.py",
        "--project",
        PROJECT,
        "--station",
        STATION_ID,
    ]

    base.main()


if __name__ == "__main__":
    main()
