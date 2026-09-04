"""Audit MODIS native-grid extraction alignment for 2022-04-07.

Diagnostic only. It compares, at each selected field station:

1) MODIS ET sampled directly at the station point in Earth Engine, using the
   source image native projection. This matches the target-extraction logic
   used by the training workflow.
2) MODIS ET averaged over the previously constructed station footprint.
3) MODIS ET from the current local-download path, which requests the grid
   using the WKT representation returned by projection.wkt().
4) MODIS ET from an otherwise identical download that requests the grid using
   the exact Earth Engine CRS identifier from projection.getInfo()["crs"].

The purpose is to determine whether the local MODIS download is being
implicitly resampled because an equivalent WKT CRS is supplied instead of the
source image's exact CRS identifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform

from et_downscaling.modis import build_modis_inputs
from et_downscaling.production import build_modis_period_context
from et_downscaling.ridge25_local_production import (
    _build_modis_grid,
    _download_ee_bytes,
    _read_downloaded_array,
)


DEFAULT_DATE = "2022-04-07"
DEFAULT_STATIONS = ("ST01", "ST02", "ST03", "ST05")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit native MODIS extraction alignment at field stations."
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--raster", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--stations", nargs="+", default=list(DEFAULT_STATIONS))
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_raster(root: Path, date_text: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    raster_directory = (
        root.parent
        / "ET_fundacion_workspace"
        / "current"
        / "rasters"
        / date_text
    )
    candidates = sorted(
        raster_directory.glob(
            f"ET_ridge25_local_aoa_support90_tol001_v1_{date_text}_20m.tif"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one final raster for {date_text}; found {len(candidates)} "
            f"in {raster_directory}"
        )
    return candidates[0]


def load_stations(root: Path) -> dict[str, dict]:
    path = root / "data" / "stations" / "fundacion_stations.geojson"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    result: dict[str, dict] = {}
    for feature in payload["features"]:
        properties = dict(feature["properties"])
        longitude, latitude = feature["geometry"]["coordinates"]
        properties["longitude"] = float(longitude)
        properties["latitude"] = float(latitude)
        result[str(properties["station_id"])] = properties

    return result


def point_et(
    date_text: str,
    longitude: float,
    latitude: float,
) -> float:
    point = ee.Geometry.Point([longitude, latitude])
    context = build_modis_period_context(date_text, point)
    projection = ee.Projection(context["modis_projection"])

    value = (
        ee.Image(context["modis_et"])
        .reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            crs=projection,
            scale=projection.nominalScale(),
            maxPixels=100,
        )
        .get("ET_mm_period")
        .getInfo()
    )

    return float(value) if value is not None else np.nan


def footprint_et(
    date_text: str,
    footprints: ee.FeatureCollection,
    station_id: str,
) -> tuple[float, object]:
    footprint = ee.Feature(
        footprints.filter(ee.Filter.eq("station_id", station_id)).first()
    )
    context = build_modis_period_context(date_text, footprint.geometry())
    projection = ee.Projection(context["modis_projection"])

    value = (
        ee.Image(context["modis_et"])
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=footprint.geometry(),
            crs=projection,
            scale=projection.nominalScale(),
            maxPixels=100,
        )
        .get("ET_mm_period")
        .getInfo()
    )

    return (
        float(value) if value is not None else np.nan,
        footprint.get("modis_pixel_id").getInfo(),
    )


def download_modis(
    image: ee.Image,
    crs_parameter: str,
    transform,
    shape: tuple[int, int],
    timeout_seconds: int,
) -> tuple[np.ndarray, object, object]:
    height, width = shape

    parameters = {
        "bands": ["ET_mm_period"],
        "crs": crs_parameter,
        "crs_transform": [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ],
        "dimensions": [width, height],
        "format": "GEO_TIFF",
    }

    payload = _download_ee_bytes(
        image,
        parameters,
        timeout_seconds,
    )

    bands, downloaded_transform, downloaded_crs = _read_downloaded_array(payload)

    if bands.shape != (1, height, width):
        raise RuntimeError(
            f"Unexpected MODIS download shape {bands.shape}; "
            f"expected {(1, height, width)}."
        )

    if not downloaded_transform.almost_equals(transform):
        raise RuntimeError("Downloaded MODIS transform differs from requested grid.")

    if downloaded_crs is None:
        raise RuntimeError("Downloaded MODIS raster has no CRS.")

    return (
        bands[0].filled(np.nan).astype(np.float64),
        downloaded_transform,
        downloaded_crs,
    )


def sample_download(
    array: np.ndarray,
    transform,
    crs,
    longitude: float,
    latitude: float,
) -> tuple[float, int, int]:
    xs, ys = warp_transform(
        "EPSG:4326",
        crs,
        [longitude],
        [latitude],
    )
    row, col = rowcol(transform, xs[0], ys[0])

    inside = (
        0 <= row < array.shape[0]
        and 0 <= col < array.shape[1]
    )

    value = (
        float(array[row, col])
        if inside and np.isfinite(array[row, col])
        else np.nan
    )

    return value, int(row), int(col)


def main() -> None:
    args = parse_arguments()
    root = project_root()
    raster_path = find_raster(root, args.date, args.raster)
    stations = load_stations(root)

    print("Initializing Earth Engine...")
    ee.Initialize(project=args.project)

    modis_inputs = build_modis_inputs()
    footprints = ee.FeatureCollection(modis_inputs["station_footprints"])

    with rasterio.open(raster_path) as fine_dataset:
        bounds = fine_dataset.bounds
        fine_crs = fine_dataset.crs

        if fine_crs is None:
            raise RuntimeError("Fine raster has no CRS.")

        processing_geometry = ee.Geometry.Rectangle(
            [bounds.left, bounds.bottom, bounds.right, bounds.top],
            proj=str(fine_crs),
            geodesic=False,
        )

        context = build_modis_period_context(args.date, processing_geometry)
        projection = ee.Projection(context["modis_projection"])
        projection_info = projection.getInfo()
        exact_ee_crs = str(projection_info["crs"])

        current_wkt_crs, modis_transform, modis_shape = _build_modis_grid(
            {"modis_projection": projection},
            (bounds.left, bounds.bottom, bounds.right, bounds.top),
        )

        image = ee.Image(context["modis_et"])

        current_array, current_transform, current_crs = download_modis(
            image=image,
            crs_parameter=current_wkt_crs,
            transform=modis_transform,
            shape=modis_shape,
            timeout_seconds=args.timeout_seconds,
        )

        exact_crs_array, exact_crs_transform, exact_crs_raster_crs = download_modis(
            image=image,
            crs_parameter=exact_ee_crs,
            transform=modis_transform,
            shape=modis_shape,
            timeout_seconds=args.timeout_seconds,
        )

        print("Raster:", raster_path)
        print("Fine CRS:", fine_crs)
        print("Earth Engine native CRS identifier:", exact_ee_crs)
        print("Current download CRS representation:", current_wkt_crs)
        print("MODIS shape:", modis_shape)
        print("MODIS transform:", modis_transform)
        print()

        rows_output: list[dict[str, object]] = []

        for station_id in args.stations:
            meta = stations[station_id]
            longitude = meta["longitude"]
            latitude = meta["latitude"]

            ee_point = point_et(
                args.date,
                longitude,
                latitude,
            )

            ee_footprint, modis_pixel_id = footprint_et(
                args.date,
                footprints,
                station_id,
            )

            current_value, current_row, current_col = sample_download(
                current_array,
                current_transform,
                current_crs,
                longitude,
                latitude,
            )

            exact_crs_value, exact_row, exact_col = sample_download(
                exact_crs_array,
                exact_crs_transform,
                exact_crs_raster_crs,
                longitude,
                latitude,
            )

            rows_output.append(
                {
                    "station_id": station_id,
                    "station": meta["station"],
                    "modis_pixel_id": modis_pixel_id,
                    "EE_point_ET_mm_period": ee_point,
                    "EE_footprint_mean_ET_mm_period": ee_footprint,
                    "current_WKT_download_ET_mm_period": current_value,
                    "exact_CRS_download_ET_mm_period": exact_crs_value,
                    "current_minus_point_mm": (
                        current_value - ee_point
                        if np.isfinite(current_value) and np.isfinite(ee_point)
                        else np.nan
                    ),
                    "exact_CRS_minus_point_mm": (
                        exact_crs_value - ee_point
                        if np.isfinite(exact_crs_value) and np.isfinite(ee_point)
                        else np.nan
                    ),
                    "current_row": current_row,
                    "current_col": current_col,
                    "exact_CRS_row": exact_row,
                    "exact_CRS_col": exact_col,
                }
            )

        table = pd.DataFrame(rows_output)

        print("MODIS EXTRACTION ALIGNMENT")
        print(table.to_string(index=False))
        print()

        for label in (
            "current_minus_point_mm",
            "exact_CRS_minus_point_mm",
        ):
            values = table[label].to_numpy(dtype=float)
            finite = np.isfinite(values)

            if finite.any():
                print(
                    f"Max |{label}|: "
                    f"{np.max(np.abs(values[finite])):.12g} mm"
                )


if __name__ == "__main__":
    main()
