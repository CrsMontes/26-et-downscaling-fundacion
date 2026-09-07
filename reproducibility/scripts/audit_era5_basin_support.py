"""Audit native ERA5-Land support inside the Fundación production domain.

Purpose
-------
Determine whether the coastal ERA5-Land mask issue observed at external ST04
also affects the mapped basin or its production buffer.

This script does not alter production. It evaluates the native masks of the
five ERA5-Land source bands used by Ridge-25 on:
1. the basin itself;
2. the basin + the production processing buffer;
3. the five station points.

It also checks representative dates from 2020-2024 against the reference mask.

Decision rule
-------------
- Basin and buffer fully valid, stable mask:
    keep native ERA5-Land in basin production; nearest-valid fallback remains
    an external-ST04 diagnostic only.
- Basin valid but buffer not fully valid:
    review boundary-parent support before mapping.
- Basin itself contains masked ERA5-Land support:
    production requires a general, documented nearest-valid fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee
import pandas as pd

from et_downscaling.production import (
    ERA5_COLLECTION_ID,
    PROCESSING_BUFFER_M,
    load_basin_geometry,
)
from et_downscaling.workspace import get_workspace_paths, require_portable_inputs


ERA5_SOURCE_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_hourly",
]

CHECK_DATES = [
    "2020-01-15",
    "2020-07-15",
    "2021-01-15",
    "2021-07-15",
    "2022-01-15",
    "2022-07-15",
    "2023-01-15",
    "2023-07-15",
    "2024-01-15",
    "2024-07-15",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    return parser.parse_args()


def valid_mask(image: ee.Image) -> ee.Image:
    return (
        ee.Image(image)
        .select(ERA5_SOURCE_BANDS)
        .mask()
        .reduce(ee.Reducer.min())
        .rename("era5_native_valid")
        .unmask(0)
        .toByte()
    )


def image_for_date(collection: ee.ImageCollection, date_text: str) -> ee.Image:
    start = ee.Date(date_text)
    subset = collection.filterDate(start, start.advance(1, "day"))
    return ee.Image(subset.first())


def region_stats(
    mask: ee.Image,
    geometry: ee.Geometry,
    projection: ee.Projection,
    scale: ee.Number,
    region_name: str,
) -> dict:
    pixel_area = ee.Image.pixelArea()

    total = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        crs=projection,
        scale=scale,
        maxPixels=1e7,
        tileScale=4,
    ).get("area")

    valid = (
        pixel_area
        .multiply(mask)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            crs=projection,
            scale=scale,
            maxPixels=1e7,
            tileScale=4,
        )
        .get("area")
    )

    invalid_pixels = (
        mask.eq(0)
        .rename("invalid")
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            crs=projection,
            scale=scale,
            maxPixels=1e7,
            tileScale=4,
        )
        .get("invalid")
    )

    payload = ee.Dictionary(
        {
            "region": region_name,
            "total_area_m2": total,
            "valid_area_m2": valid,
            "invalid_native_pixels": invalid_pixels,
        }
    ).getInfo()

    total_area = float(payload["total_area_m2"] or 0.0)
    valid_area = float(payload["valid_area_m2"] or 0.0)
    payload["valid_fraction"] = (
        valid_area / total_area if total_area > 0 else float("nan")
    )
    payload["invalid_area_m2"] = max(total_area - valid_area, 0.0)
    payload["invalid_area_km2"] = payload["invalid_area_m2"] / 1e6
    return payload


def load_station_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    records = []
    for feature in data.get("features", []):
        props = dict(feature.get("properties", {}))
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates")
        if not coords or geometry.get("type") != "Point":
            continue
        records.append(
            {
                "station_id": str(props.get("station_id", "")),
                "station": props.get("station", ""),
                "longitude": float(coords[0]),
                "latitude": float(coords[1]),
                "inside_basin_metadata": props.get("inside_basin"),
            }
        )
    return records


def sample_stations(
    mask: ee.Image,
    stations: list[dict],
    projection: ee.Projection,
    scale: ee.Number,
) -> pd.DataFrame:
    features = []
    for record in stations:
        point = ee.Geometry.Point(
            [record["longitude"], record["latitude"]]
        )
        value = mask.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            crs=projection,
            scale=scale,
            maxPixels=100,
        ).get("era5_native_valid")
        features.append(
            ee.Feature(
                None,
                {
                    **record,
                    "era5_native_valid": value,
                },
            )
        )

    payload = ee.FeatureCollection(features).getInfo()
    return pd.DataFrame(
        feature["properties"] for feature in payload.get("features", [])
    )


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    inputs = require_portable_inputs(root)
    workspace = get_workspace_paths(root).ensure()

    print("Initializing Earth Engine...")
    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()

    basin = load_basin_geometry(root)
    processing_geometry = basin.buffer(PROCESSING_BUFFER_M)

    collection = (
        ee.ImageCollection(ERA5_COLLECTION_ID)
        .filterDate("2020-01-01", "2025-01-01")
        .select(ERA5_SOURCE_BANDS)
    )

    reference = image_for_date(collection, CHECK_DATES[0])
    reference_mask = valid_mask(reference)
    projection = reference.select(ERA5_SOURCE_BANDS[0]).projection()
    scale = projection.nominalScale()

    image_count = int(collection.size().getInfo())
    scale_m = float(scale.getInfo())

    print()
    print("=" * 88)
    print("ERA5-LAND NATIVE SUPPORT AUDIT")
    print("=" * 88)
    print("Collection:", ERA5_COLLECTION_ID)
    print("Period: 2020-01-01 to 2025-01-01")
    print("Hourly images:", image_count)
    print("Native nominal scale (m):", f"{scale_m:.2f}")
    print("Production buffer (m):", PROCESSING_BUFFER_M)

    region_rows = [
        region_stats(
            reference_mask,
            basin,
            projection,
            scale,
            "basin",
        ),
        region_stats(
            reference_mask,
            processing_geometry,
            projection,
            scale,
            "basin_plus_processing_buffer",
        ),
    ]
    region_table = pd.DataFrame(region_rows)

    print()
    print("REFERENCE MASK SUPPORT")
    print(
        region_table[
            [
                "region",
                "valid_fraction",
                "invalid_area_km2",
                "invalid_native_pixels",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )

    temporal_rows = []
    for date_text in CHECK_DATES:
        image = image_for_date(collection, date_text)
        mask = valid_mask(image)
        difference = mask.neq(reference_mask).rename("mask_changed")

        changed = difference.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=processing_geometry,
            crs=projection,
            scale=scale,
            maxPixels=1e7,
            tileScale=4,
        ).get("mask_changed")

        changed_value = int(ee.Number(changed).getInfo() or 0)
        temporal_rows.append(
            {
                "date": date_text,
                "mask_differs_from_reference_in_buffer": changed_value,
            }
        )

    temporal_table = pd.DataFrame(temporal_rows)

    print()
    print("REPRESENTATIVE-DATE MASK STABILITY")
    print(temporal_table.to_string(index=False))

    stations = load_station_records(inputs["stations"])
    station_table = sample_stations(
        reference_mask,
        stations,
        projection,
        scale,
    ).sort_values("station_id")

    print()
    print("STATION NATIVE ERA5 SUPPORT")
    print(
        station_table[
            [
                "station_id",
                "station",
                "inside_basin_metadata",
                "era5_native_valid",
            ]
        ].to_string(index=False)
    )

    basin_fraction = float(
        region_table.loc[
            region_table["region"].eq("basin"),
            "valid_fraction",
        ].iloc[0]
    )
    buffer_fraction = float(
        region_table.loc[
            region_table["region"].eq("basin_plus_processing_buffer"),
            "valid_fraction",
        ].iloc[0]
    )
    mask_stable = not temporal_table[
        "mask_differs_from_reference_in_buffer"
    ].astype(bool).any()

    tolerance = 1e-12
    basin_complete = basin_fraction >= 1.0 - tolerance
    buffer_complete = buffer_fraction >= 1.0 - tolerance

    if basin_complete and buffer_complete and mask_stable:
        decision = (
            "KEEP_NATIVE_ERA5_FOR_BASIN_PRODUCTION; "
            "nearest-valid fallback remains external-ST04 only."
        )
    elif basin_complete and not buffer_complete:
        decision = (
            "BASIN_NATIVE_ERA5_COMPLETE_BUT_BUFFER_MASKED; "
            "audit boundary-parent support before mapping."
        )
    else:
        decision = (
            "BASIN_CONTAINS_NATIVE_ERA5_MASKS; "
            "implement a general documented nearest-valid fallback before mapping."
        )

    print()
    print("=" * 88)
    print("DECISION")
    print("=" * 88)
    print(decision)

    out_dir = workspace.diagnostics / "era5_basin_support_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    region_path = out_dir / "era5_native_support_regions.csv"
    temporal_path = out_dir / "era5_native_mask_stability.csv"
    station_path = out_dir / "era5_native_support_stations.csv"
    decision_path = out_dir / "decision.txt"

    region_table.to_csv(region_path, index=False)
    temporal_table.to_csv(temporal_path, index=False)
    station_table.to_csv(station_path, index=False)
    decision_path.write_text(decision + "\n", encoding="utf-8")

    print()
    print("Saved:")
    print(" -", region_path)
    print(" -", temporal_path)
    print(" -", station_path)
    print(" -", decision_path)


if __name__ == "__main__":
    main()
