"""Audit the 2022-04-07 Ridge-25 reconciliation against exact MODIS footprints.

Purpose
-------
This is a diagnostic only. It does not modify the frozen production workflow.

For each selected field station it compares:
1. Kc_raw at the station from the final 20 m raster.
2. Area-weighted Kc_raw mean over the exact native MODIS footprint.
3. A direct one-step proportional reconciliation:
       ET_simple = Kc_station * ET_MODIS / mean(Kc_parent)
4. The ET value stored in the final iteratively reconciled raster.

If ET_simple is close to the historical three-year result but the final raster
is very different, the divergence is introduced by the current iterative
non-nested-grid reconciliation. If ET_simple is already very different, the
main divergence is in the current fine-scale Kc field / parent mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform as transform_coordinates
from rasterio.warp import transform_geom
from shapely.geometry import box, shape

from et_downscaling.modis import build_modis_inputs
from et_downscaling.production import build_modis_period_context


DEFAULT_DATE = "2022-04-07"
DEFAULT_STATIONS = ("ST01", "ST02", "ST03", "ST05")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit current Ridge-25 reconciliation at exact MODIS footprints."
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument(
        "--stations",
        nargs="+",
        default=list(DEFAULT_STATIONS),
    )
    parser.add_argument(
        "--raster",
        default=None,
        help="Optional explicit path to final 20 m raster.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    stations = {}
    for feature in payload["features"]:
        props = dict(feature["properties"])
        lon, lat = feature["geometry"]["coordinates"]
        props["longitude"] = float(lon)
        props["latitude"] = float(lat)
        stations[str(props["station_id"])] = props
    return stations


def clean(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= -9990:
        return np.nan
    return value


def band_index(dataset: rasterio.io.DatasetReader, name: str) -> int:
    descriptions = list(dataset.descriptions)
    if name not in descriptions:
        raise RuntimeError(
            f"Band {name!r} not found. Available bands: {descriptions}"
        )
    return descriptions.index(name) + 1


def station_sample(
    dataset: rasterio.io.DatasetReader,
    longitude: float,
    latitude: float,
) -> dict[str, float]:
    xs, ys = transform_coordinates(
        "EPSG:4326",
        dataset.crs,
        [longitude],
        [latitude],
    )
    values = next(dataset.sample([(xs[0], ys[0])]))
    return {
        name: clean(values[index])
        for index, name in enumerate(dataset.descriptions)
    }


def weighted_parent_statistics(
    dataset: rasterio.io.DatasetReader,
    footprint_wgs84: dict,
) -> dict[str, float]:
    footprint_raster_crs = transform_geom(
        "EPSG:4326",
        dataset.crs,
        footprint_wgs84,
        precision=-1,
    )
    polygon = shape(footprint_raster_crs)

    raw_window = from_bounds(
        *polygon.bounds,
        transform=dataset.transform,
    )
    window = raw_window.round_offsets().round_lengths()
    window = window.intersection(
        rasterio.windows.Window(
            0,
            0,
            dataset.width,
            dataset.height,
        )
    )

    row_start = int(window.row_off)
    row_stop = int(window.row_off + window.height)
    col_start = int(window.col_off)
    col_stop = int(window.col_off + window.width)

    names = [
        "Kc_raw",
        "stack_valid",
        "AOA_inside",
        "usable",
        "ET_mm_period",
        "coarse_eligible",
    ]
    arrays = {
        name: dataset.read(
            band_index(dataset, name),
            window=window,
        ).astype(float)
        for name in names
    }
    for name, array in arrays.items():
        array[array <= -9990] = np.nan

    weights = np.zeros(
        (row_stop - row_start, col_stop - col_start),
        dtype=float,
    )

    for local_row, row in enumerate(range(row_start, row_stop)):
        for local_col, col in enumerate(range(col_start, col_stop)):
            x0, y0 = dataset.transform * (col, row)
            x1, y1 = dataset.transform * (col + 1, row + 1)
            pixel = box(
                min(x0, x1),
                min(y0, y1),
                max(x0, x1),
                max(y0, y1),
            )
            area = polygon.intersection(pixel).area
            if area > 0:
                weights[local_row, local_col] = area

    def weighted_mean(values: np.ndarray, mask: np.ndarray) -> float:
        valid = (
            (weights > 0)
            & mask
            & np.isfinite(values)
        )
        if not valid.any():
            return np.nan
        return float(
            np.sum(values[valid] * weights[valid])
            / np.sum(weights[valid])
        )

    kc = arrays["Kc_raw"]
    stack = arrays["stack_valid"] >= 0.5
    aoa = arrays["AOA_inside"] >= 0.5
    usable = arrays["usable"] >= 0.5
    eligible = arrays["coarse_eligible"] >= 0.5
    et = arrays["ET_mm_period"]

    footprint_area = float(polygon.area)
    represented_area = float(weights.sum())

    return {
        "footprint_area_m2": footprint_area,
        "represented_area_m2": represented_area,
        "represented_fraction": (
            represented_area / footprint_area
            if footprint_area > 0
            else np.nan
        ),
        "Kc_mean_all_finite": weighted_mean(
            kc,
            np.isfinite(kc),
        ),
        "Kc_mean_stack_valid": weighted_mean(
            kc,
            stack,
        ),
        "Kc_mean_stack_and_AOA": weighted_mean(
            kc,
            stack & aoa & (kc >= 0),
        ),
        "Kc_mean_usable": weighted_mean(
            kc,
            usable,
        ),
        "usable_area_fraction": (
            float(
                np.sum(weights[usable & (weights > 0)])
                / represented_area
            )
            if represented_area > 0
            else np.nan
        ),
        "ET_final_mean_published": weighted_mean(
            et,
            np.isfinite(et),
        ),
        "eligible_area_fraction": (
            float(
                np.sum(weights[eligible & (weights > 0)])
                / represented_area
            )
            if represented_area > 0
            else np.nan
        ),
    }


def get_exact_footprint(
    footprints: ee.FeatureCollection,
    station_id: str,
) -> ee.Feature:
    return ee.Feature(
        footprints
        .filter(
            ee.Filter.eq(
                "station_id",
                station_id,
            )
        )
        .first()
    )


def get_modis_et(
    date_text: str,
    footprint: ee.Feature,
) -> float:
    context = build_modis_period_context(
        date_text,
        footprint.geometry(),
    )
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
    return float(value)


def main() -> None:
    args = parse_arguments()
    root = project_root()
    raster_path = find_raster(
        root,
        args.date,
        args.raster,
    )
    stations = load_stations(root)

    print("Initializing Earth Engine...")
    ee.Initialize(project=args.project)
    modis_inputs = build_modis_inputs()
    footprints = ee.FeatureCollection(
        modis_inputs["station_footprints"]
    )

    rows = []

    with rasterio.open(raster_path) as dataset:
        print("Raster:", raster_path)
        print("CRS:", dataset.crs)
        print()

        for station_id in args.stations:
            if station_id not in stations:
                raise KeyError(f"Unknown station: {station_id}")

            meta = stations[station_id]
            footprint = get_exact_footprint(
                footprints,
                station_id,
            )
            footprint_wgs84 = (
                footprint.geometry()
                .transform("EPSG:4326", 1)
                .getInfo()
            )
            modis_et = get_modis_et(
                args.date,
                footprint,
            )

            point = station_sample(
                dataset,
                meta["longitude"],
                meta["latitude"],
            )
            stats = weighted_parent_statistics(
                dataset,
                footprint_wgs84,
            )

            kc_station = point.get("Kc_raw", np.nan)
            kc_mean_stack = stats["Kc_mean_stack_valid"]
            kc_mean_usable = stats["Kc_mean_usable"]

            simple_stack = (
                kc_station * modis_et / kc_mean_stack
                if (
                    np.isfinite(kc_station)
                    and np.isfinite(kc_mean_stack)
                    and abs(kc_mean_stack) > 1e-12
                )
                else np.nan
            )
            simple_usable = (
                kc_station * modis_et / kc_mean_usable
                if (
                    np.isfinite(kc_station)
                    and np.isfinite(kc_mean_usable)
                    and abs(kc_mean_usable) > 1e-12
                )
                else np.nan
            )
            et_final = point.get("ET_mm_period", np.nan)

            rows.append(
                {
                    "station_id": station_id,
                    "station": meta["station"],
                    "MODIS_ET_mm_period": modis_et,
                    "Kc_station": kc_station,
                    "station_stack_valid": point.get(
                        "stack_valid",
                        np.nan,
                    ),
                    "station_AOA_inside": point.get(
                        "AOA_inside",
                        np.nan,
                    ),
                    "station_usable": point.get(
                        "usable",
                        np.nan,
                    ),
                    "station_coarse_eligible": point.get(
                        "coarse_eligible",
                        np.nan,
                    ),
                    "Kc_parent_mean_all_finite":
                        stats["Kc_mean_all_finite"],
                    "Kc_parent_mean_stack_valid":
                        kc_mean_stack,
                    "Kc_parent_mean_stack_AOA":
                        stats["Kc_mean_stack_and_AOA"],
                    "Kc_parent_mean_usable":
                        kc_mean_usable,
                    "usable_area_fraction_exact":
                        stats["usable_area_fraction"],
                    "ET_simple_stack_mm_period":
                        simple_stack,
                    "ET_simple_usable_mm_period":
                        simple_usable,
                    "ET_final_iterative_mm_period":
                        et_final,
                    "final_over_simple_usable":
                        (
                            et_final / simple_usable
                            if (
                                np.isfinite(et_final)
                                and np.isfinite(simple_usable)
                                and abs(simple_usable) > 1e-12
                            )
                            else np.nan
                        ),
                    "ET_final_parent_mean_published":
                        stats["ET_final_mean_published"],
                    "represented_fraction":
                        stats["represented_fraction"],
                }
            )

    result = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print("=" * 120)
    print("RECONCILIATION AUDIT")
    print("=" * 120)
    print(result.to_string(index=False))

    output_directory = (
        root.parent
        / "ET_fundacion_workspace"
        / "current"
        / "diagnostics"
        / "reconciliation_audit"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = (
        output_directory
        / f"reconciliation_audit_{args.date}.csv"
    )
    result.to_csv(
        output_path,
        index=False,
    )
    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
