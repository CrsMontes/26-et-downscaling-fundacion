"""Produce station-centered RF25 validation products with a 7x7 MODIS halo."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("PROJ_NETWORK", "OFF")

import ee
import joblib
import numpy as np
import rasterio
from rasterio.transform import Affine, array_bounds
from rasterio.windows import Window
from rasterio.warp import transform as warp_transform, transform_bounds

from et_downscaling.config import ANALYSIS_CRS
from et_downscaling.ee_download import (
    OUTPUT_NODATA,
    download_ee_bytes,
    read_downloaded_array,
)
from et_downscaling.overlap_reconciliation import (
    MODIS_SINUSOIDAL_LOCAL_CRS,
    build_overlap_edges,
    materialize_active_values,
    solve_overlap_reconciliation,
)
from et_downscaling.production import (
    PREDICTION_SCALE_M,
    build_modis_period_context,
)
from et_downscaling.rf25 import (
    RF25_AOA_FILENAME,
    RF25_MODEL_FILENAME,
    validate_rf25_model,
)
from et_downscaling.rf25_local_state import (
    RF25_RECONCILIATION_TOLERANCE_MM,
    RF25_USABLE_SUPPORT_FRACTION,
)
from et_downscaling.rf25_overlap_production import (
    RAW_TILE_BANDS,
    _download_raw_tile,
    build_production_scientific_signature,
)
from et_downscaling.tiles import Tile
from et_downscaling.workspace import get_workspace_paths


FIELD_DATES = [
    "2022-03-14",
    "2022-03-22",
    "2022-03-30",
    "2022-04-07",
    "2022-04-15",
    "2022-04-23",
    "2022-05-01",
    "2022-05-09",
    "2022-05-17",
    "2022-05-25",
    "2022-06-02",
    "2022-06-10",
    "2022-06-18",
    "2022-06-26",
]

DEFAULT_STATIONS = ["ST02", "ST03", "ST04", "ST05"]

HALO_RADIUS = 3
HALO_SIZE = 2 * HALO_RADIUS + 1

# Extra fine-grid support beyond the transformed 7x7 footprint.
# This is not an additional MODIS halo; it only prevents fine-grid
# truncation at transformed parent boundaries.
FINE_SUPPORT_PADDING_M = 200.0

OUTPUT_BANDS = [
    "ET_reconciled_published_mm_period",
    "ET_reconciled_support_mm_period",
    "ET_initial_pre_exact_overlap_mm_period",
    "Kc_raw",
    "dissimilarity_index",
    "local_point_density",
    "stack_valid",
    "AOA_inside",
    "usable",
    "publishable",
    "MODIS_ET_parent_mm_period",
    "parent_usable_fraction",
    "parent_coarse_eligible",
    "parent_conservation_error_mm",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--date", dest="dates", action="append")
    parser.add_argument("--station", dest="stations", action="append")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


def snap_floor(value: float, step: float) -> float:
    return math.floor(value / step) * step


def snap_ceil(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def load_stations(root: Path) -> dict[str, dict[str, object]]:
    from et_downscaling.field_station_identity import validate_station_geometry
    path = root / "data" / "stations" / "fundacion_stations.geojson"
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_station_geometry(payload)

    result = {}

    for feature in payload["features"]:
        properties = feature["properties"]
        station_id = str(properties["station_id"])
        longitude, latitude = feature["geometry"]["coordinates"]

        result[station_id] = {
            **properties,
            "station_id": station_id,
            "station": properties.get("station", station_id),
            "inside_basin": bool(properties.get("inside_basin", False)),
            "longitude": float(longitude),
            "latitude": float(latitude),
        }

    return result


def native_parent(
    projection_info: dict[str, object],
    longitude: float,
    latitude: float,
) -> tuple[int, int, Affine]:
    native_transform = Affine(
        *[float(value) for value in projection_info["transform"]]
    )

    xs, ys = warp_transform(
        "EPSG:4326",
        MODIS_SINUSOIDAL_LOCAL_CRS,
        [longitude],
        [latitude],
    )

    col_float, row_float = (
        ~native_transform
    ) * (float(xs[0]), float(ys[0]))

    parent_col = int(math.floor(col_float))
    parent_row = int(math.floor(row_float))

    return parent_row, parent_col, native_transform


def halo_transform(
    native_transform: Affine,
    parent_row: int,
    parent_col: int,
) -> Affine:
    return native_transform * Affine.translation(
        parent_col - HALO_RADIUS,
        parent_row - HALO_RADIUS,
    )


def fine_tile_for_halo(
    station_id: str,
    modis_transform: Affine,
) -> Tile:
    left, bottom, right, top = array_bounds(
        HALO_SIZE,
        HALO_SIZE,
        modis_transform,
    )

    xmin, ymin, xmax, ymax = transform_bounds(
        MODIS_SINUSOIDAL_LOCAL_CRS,
        ANALYSIS_CRS,
        left,
        bottom,
        right,
        top,
        densify_pts=41,
    )

    scale = float(PREDICTION_SCALE_M)

    xmin = snap_floor(
        xmin - FINE_SUPPORT_PADDING_M,
        scale,
    )
    ymin = snap_floor(
        ymin - FINE_SUPPORT_PADDING_M,
        scale,
    )
    xmax = snap_ceil(
        xmax + FINE_SUPPORT_PADDING_M,
        scale,
    )
    ymax = snap_ceil(
        ymax + FINE_SUPPORT_PADDING_M,
        scale,
    )

    return Tile(
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        tile_id=f"{station_id}_halo7",
        level=0,
    )


def download_modis_halo(
    context: dict[str, object],
    projection_info: dict[str, object],
    target_transform: Affine,
    timeout_seconds: int,
) -> np.ndarray:
    image = (
        context["modis_et"]
        .rename("ET_MODIS_mm_period")
        .toFloat()
    )

    parameters = {
        "bands": ["ET_MODIS_mm_period"],
        "crs": str(projection_info["crs"]),
        "crs_transform": [
            target_transform.a,
            target_transform.b,
            target_transform.c,
            target_transform.d,
            target_transform.e,
            target_transform.f,
        ],
        "dimensions": [HALO_SIZE, HALO_SIZE],
        "format": "GEO_TIFF",
    }

    payload = download_ee_bytes(
        image,
        parameters,
        timeout_seconds,
    )

    bands, downloaded_transform, _ = read_downloaded_array(
        payload
    )

    if bands.shape != (1, HALO_SIZE, HALO_SIZE):
        raise RuntimeError(
            f"Unexpected MODIS halo shape: {bands.shape}"
        )

    if not downloaded_transform.almost_equals(
        target_transform
    ):
        raise RuntimeError(
            "Downloaded MODIS halo transform differs from "
            "the requested native grid."
        )

    return (
        bands[0]
        .filled(np.nan)
        .astype(np.float64)
    )


def read_raw_tile(path: Path):
    with rasterio.open(path) as src:
        if tuple(src.descriptions) != tuple(RAW_TILE_BANDS):
            raise RuntimeError(
                "Local raw tile band contract mismatch."
            )

        raw = (
            src.read(masked=True)
            .filled(np.nan)
            .astype(np.float64)
        )

        return raw, src.transform, src.crs


def central_parent_mask(
    domain: np.ndarray,
    fine_transform: Affine,
    fine_crs,
    modis_transform: Affine,
) -> np.ndarray:
    center = HALO_RADIUS

    central_transform = (
        modis_transform
        * Affine.translation(center, center)
    )

    # Geometry only: finite dummy value ensures that overlap
    # edges are constructed even if MODIS ET itself is missing.
    central_edges = build_overlap_edges(
        domain=domain,
        fine_transform=fine_transform,
        fine_crs=fine_crs,
        modis_et=np.ones((1, 1), dtype=float),
        modis_transform=central_transform,
        modis_crs=MODIS_SINUSOIDAL_LOCAL_CRS,
        progress_every=0,
    )

    mask = np.zeros(domain.size, dtype=bool)
    mask[np.unique(central_edges.fine_index)] = True

    return mask.reshape(domain.shape)


def write_modis_halo(
    path: Path,
    modis_et: np.ndarray,
    transform: Affine,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "width": HALO_SIZE,
        "height": HALO_SIZE,
        "count": 1,
        "dtype": "float32",
        "crs": MODIS_SINUSOIDAL_LOCAL_CRS,
        "transform": transform,
        "nodata": OUTPUT_NODATA,
        "compress": "deflate",
    }

    prepared = np.where(
        np.isfinite(modis_et),
        modis_et,
        OUTPUT_NODATA,
    ).astype(np.float32)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(prepared, 1)
        dst.set_band_description(
            1,
            "ET_MODIS_mm_period",
        )


def write_central_product(
    output_path: Path,
    metadata_path: Path,
    period_start: str,
    station: dict[str, object],
    scientific_signature: str,
    raw_path: Path,
    modis_et: np.ndarray,
    modis_transform: Affine,
    parent_row: int,
    parent_col: int,
) -> None:
    raw, fine_transform, fine_crs = read_raw_tile(raw_path)

    kc_raw = raw[RAW_TILE_BANDS.index("Kc_raw")]
    dissimilarity = raw[
        RAW_TILE_BANDS.index("dissimilarity_index")
    ]
    lpd = raw[
        RAW_TILE_BANDS.index("local_point_density")
    ]
    stack_valid = (
        raw[RAW_TILE_BANDS.index("stack_valid")]
        > 0.5
    )
    aoa_inside = (
        raw[RAW_TILE_BANDS.index("AOA_inside")]
        > 0.5
    )
    usable = (
        raw[RAW_TILE_BANDS.index("usable")]
        > 0.5
    )
    domain = (
        raw[RAW_TILE_BANDS.index("support_domain")]
        > 0.5
    )

    fine_shape = kc_raw.shape

    central_mask = central_parent_mask(
        domain=domain,
        fine_transform=fine_transform,
        fine_crs=fine_crs,
        modis_transform=modis_transform,
    )

    rows, cols = np.where(central_mask)

    if rows.size == 0:
        raise RuntimeError(
            "No fine pixels overlap the central MODIS parent."
        )

    row0 = int(rows.min())
    row1 = int(rows.max()) + 1
    col0 = int(cols.min())
    col1 = int(cols.max()) + 1

    result = None
    reconciliation_status = "not_available"

    finite_modis = np.isfinite(modis_et)

    if finite_modis.any():
        edges = build_overlap_edges(
            domain=domain,
            fine_transform=fine_transform,
            fine_crs=fine_crs,
            modis_et=modis_et,
            modis_transform=modis_transform,
            modis_crs=MODIS_SINUSOIDAL_LOCAL_CRS,
            progress_every=0,
        )

        missing_representation = (
            finite_modis
            & ~edges.represented_coarse
        )

        if missing_representation.any():
            raise RuntimeError(
                "Fine support does not fully represent every "
                "finite MODIS parent in the 7x7 halo."
            )

        try:
            result = solve_overlap_reconciliation(
                kc_raw=kc_raw,
                usable=usable,
                modis_et=modis_et,
                edges=edges,
                usable_support_fraction=(
                    RF25_USABLE_SUPPORT_FRACTION
                ),
                tolerance_mm=(
                    RF25_RECONCILIATION_TOLERANCE_MM
                ),
            )
            reconciliation_status = "solved"
        except RuntimeError as exc:
            reconciliation_status = (
                "not_solved: " + str(exc)
            )

    empty = np.full(
        fine_shape,
        np.nan,
        dtype=float,
    )

    et_initial = empty.copy()
    et_reconciled_support = empty.copy()
    et_reconciled_published = empty.copy()
    publishable = np.zeros(
        fine_shape,
        dtype=float,
    )

    center = HALO_RADIUS
    center_flat = (
        center * HALO_SIZE + center
    )

    parent_modis_et = (
        float(modis_et[center, center])
        if np.isfinite(modis_et[center, center])
        else np.nan
    )

    parent_usable_fraction = np.nan
    parent_eligible = False
    parent_conservation_error = np.nan
    parent_kc_valid_mean = np.nan

    if result is not None:
        et_initial = materialize_active_values(
            fine_shape=fine_shape,
            active_fine=result.active_fine,
            values=result.et_initial,
        )

        et_reconciled_support = (
            materialize_active_values(
                fine_shape=fine_shape,
                active_fine=result.active_fine,
                values=result.et_final_nonnegative,
            )
        )

        et_reconciled_published = (
            materialize_active_values(
                fine_shape=fine_shape,
                active_fine=result.active_fine,
                values=result.et_final_nonnegative,
                selected_active=result.publishable_active,
            )
        )

        publishable_flat = publishable.ravel()
        publishable_flat[
            result.active_fine[
                result.publishable_active
            ]
        ] = 1.0
        publishable = publishable_flat.reshape(
            fine_shape
        )

        parent_usable_fraction = float(
            result.usable_fraction[
                center,
                center,
            ]
        )

        parent_eligible = bool(
            result.eligible_coarse_mask[
                center,
                center,
            ]
        )

        parent_kc_valid_mean = float(
            result.kc_valid_mean[
                center,
                center,
            ]
        )

        if parent_eligible:
            positions = np.flatnonzero(
                result.eligible_coarse
                == center_flat
            )

            if positions.size != 1:
                raise RuntimeError(
                    "Central eligible parent mapping is ambiguous."
                )

            parent_conservation_error = float(
                result.final_error_after_nonnegative[
                    positions[0]
                ]
            )

    def central(array):
        masked = np.where(
            central_mask,
            np.asarray(array, dtype=float),
            np.nan,
        )
        return masked[row0:row1, col0:col1]

    parent_et_grid = np.where(
        central_mask,
        parent_modis_et,
        np.nan,
    )

    usable_fraction_grid = np.where(
        central_mask,
        parent_usable_fraction,
        np.nan,
    )

    eligible_grid = np.where(
        central_mask,
        float(parent_eligible),
        np.nan,
    )

    conservation_grid = np.where(
        central_mask,
        parent_conservation_error,
        np.nan,
    )

    arrays = [
        central(et_reconciled_published),
        central(et_reconciled_support),
        central(et_initial),
        central(kc_raw),
        central(dissimilarity),
        central(lpd),
        central(stack_valid.astype(float)),
        central(aoa_inside.astype(float)),
        central(usable.astype(float)),
        central(publishable),
        central(parent_et_grid),
        central(usable_fraction_grid),
        central(eligible_grid),
        central(conservation_grid),
    ]

    crop_window = Window(
        col0,
        row0,
        col1 - col0,
        row1 - row0,
    )

    output_transform = rasterio.windows.transform(
        crop_window,
        fine_transform,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = {
        "driver": "GTiff",
        "width": arrays[0].shape[1],
        "height": arrays[0].shape[0],
        "count": len(OUTPUT_BANDS),
        "dtype": "float32",
        "crs": fine_crs,
        "transform": output_transform,
        "nodata": OUTPUT_NODATA,
        "compress": "deflate",
        "tiled": True,
    }

    prepared = np.stack(arrays, axis=0)
    prepared = np.where(
        np.isfinite(prepared),
        prepared,
        OUTPUT_NODATA,
    ).astype(np.float32)

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:
        dst.write(prepared)

        for index, name in enumerate(
            OUTPUT_BANDS,
            start=1,
        ):
            dst.set_band_description(
                index,
                name,
            )

        dst.update_tags(
            period_start=period_start,
            station_id=str(
                station["station_id"]
            ),
            validation_product="local_halo7",
            reconciliation_scope=(
                "7x7_native_MODIS_halo"
            ),
        )

    metadata = {
        "period_start": period_start,
        "station_id": station["station_id"],
        "station": station["station"],
        "inside_basin": station["inside_basin"],
        "longitude": station["longitude"],
        "latitude": station["latitude"],
        "halo_radius_modis_pixels": HALO_RADIUS,
        "halo_size_modis_pixels": HALO_SIZE,
        "fine_support_padding_m": (
            FINE_SUPPORT_PADDING_M
        ),
        "scientific_signature": (
            scientific_signature
        ),
        "parent_native_row": parent_row,
        "parent_native_col": parent_col,
        "central_modis_et_mm_period": (
            parent_modis_et
        ),
        "central_parent_usable_fraction": (
            parent_usable_fraction
        ),
        "central_parent_eligible": (
            parent_eligible
        ),
        "central_parent_kc_valid_mean": (
            parent_kc_valid_mean
        ),
        "central_parent_conservation_error_mm": (
            parent_conservation_error
        ),
        "reconciliation_status": (
            reconciliation_status
        ),
        "published_pixels": int(
            np.isfinite(
                central(
                    et_reconciled_published
                )
            ).sum()
        ),
        "output_bands": OUTPUT_BANDS,
        "raster": str(output_path),
        "raw_tile": str(raw_path),
        "method_note": (
            "Operational 7x7 station-validation halo. "
            "For ST02, ST03, ST04 and ST05 the halo "
            "was empirically checked against full-basin "
            "products before operational use."
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )


def product_is_current(
    raster_path: Path,
    metadata_path: Path,
    scientific_signature: str,
    period_start: str,
    station_id: str,
) -> bool:
    if (
        not raster_path.is_file()
        or not metadata_path.is_file()
    ):
        return False

    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            metadata.get("scientific_signature")
            != scientific_signature
        ):
            return False

        if metadata.get("period_start") != period_start:
            return False

        if metadata.get("station_id") != station_id:
            return False

        if (
            metadata.get("halo_size_modis_pixels")
            != HALO_SIZE
        ):
            return False

        with rasterio.open(raster_path) as src:
            return (
                tuple(src.descriptions)
                == tuple(OUTPUT_BANDS)
            )

    except Exception:
        return False


def main() -> None:
    args = parse_args()
    root = project_root()

    dates = list(
        dict.fromkeys(
            args.dates or FIELD_DATES
        )
    )
    station_ids = list(
        dict.fromkeys(
            args.stations or DEFAULT_STATIONS
        )
    )

    stations = load_stations(root)

    missing = [
        station_id
        for station_id in station_ids
        if station_id not in stations
    ]

    if missing:
        raise ValueError(
            "Unknown station(s): "
            + ", ".join(missing)
        )

    workspace = get_workspace_paths(
        root
    ).ensure()

    model_path = (
        workspace.models
        / RF25_MODEL_FILENAME
    )
    aoa_path = (
        workspace.models
        / RF25_AOA_FILENAME
    )

    if (
        not model_path.is_file()
        or not aoa_path.is_file()
    ):
        raise FileNotFoundError(
            "RF25 model/AOA missing.\n"
            f"Model: {model_path}\n"
            f"AOA: {aoa_path}"
        )

    model = joblib.load(model_path)
    aoa = joblib.load(aoa_path)
    validate_rf25_model(model)

    scientific_signature = (
        build_production_scientific_signature(
            model,
            aoa,
        )
    )

    output_root = (
        root
        / "outputs"
        / "evaluation"
        / "field_validation"
        / "local_halo7_products"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()

    failures = []

    total = len(dates) * len(station_ids)
    counter = 0

    for period_start in dates:
        # MODIS projection is native and constant, but deriving
        # it from each requested period keeps provenance explicit.
        context = build_modis_period_context(
            period_start,
            ee.Geometry.Point(
                [
                    stations[
                        station_ids[0]
                    ]["longitude"],
                    stations[
                        station_ids[0]
                    ]["latitude"],
                ]
            ).buffer(5000),
        )

        projection_info = (
            context["modis_projection"]
            .getInfo()
        )

        for station_id in station_ids:
            counter += 1
            station = stations[station_id]

            print()
            print("=" * 88)
            print(
                f"[{counter}/{total}] "
                f"{period_start} {station_id} "
                f"HALO 7x7"
            )
            print("=" * 88)

            station_dir = (
                output_root
                / period_start
                / station_id
            )
            station_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            raster_path = (
                station_dir
                / (
                    f"RF25_halo7_{station_id}_"
                    f"{period_start}_20m.tif"
                )
            )
            metadata_path = (
                station_dir
                / "metadata.json"
            )

            if product_is_current(
                raster_path,
                metadata_path,
                scientific_signature,
                period_start,
                station_id,
            ):
                print("SKIP: current product exists.")
                continue

            try:
                (
                    parent_row,
                    parent_col,
                    native_transform,
                ) = native_parent(
                    projection_info,
                    station["longitude"],
                    station["latitude"],
                )

                target_modis_transform = (
                    halo_transform(
                        native_transform,
                        parent_row,
                        parent_col,
                    )
                )

                fine_tile = fine_tile_for_halo(
                    station_id,
                    target_modis_transform,
                )

                print(
                    "Parent:",
                    f"native_r{parent_row}_c{parent_col}",
                )
                print(
                    "Fine support:",
                    f"{fine_tile.width_m:.0f} x "
                    f"{fine_tile.height_m:.0f} m",
                )

                raw_directory = (
                    station_dir / "raw"
                )

                completed = _download_raw_tile(
                    period_start=period_start,
                    model=model,
                    aoa_parameters=aoa,
                    tile=fine_tile,
                    tile_directory=raw_directory,
                    timeout_seconds=args.timeout_seconds,
                    scientific_signature=(
                        scientific_signature
                    ),
                )

                modis_et = download_modis_halo(
                    context=context,
                    projection_info=projection_info,
                    target_transform=(
                        target_modis_transform
                    ),
                    timeout_seconds=(
                        args.timeout_seconds
                    ),
                )

                modis_path = (
                    station_dir
                    / (
                        f"MODIS_halo7_{station_id}_"
                        f"{period_start}_native.tif"
                    )
                )

                write_modis_halo(
                    modis_path,
                    modis_et,
                    target_modis_transform,
                )

                write_central_product(
                    output_path=raster_path,
                    metadata_path=metadata_path,
                    period_start=period_start,
                    station=station,
                    scientific_signature=(
                        scientific_signature
                    ),
                    raw_path=completed.path,
                    modis_et=modis_et,
                    modis_transform=(
                        target_modis_transform
                    ),
                    parent_row=parent_row,
                    parent_col=parent_col,
                )

                metadata = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )

                print(
                    "DONE:",
                    raster_path,
                )
                print(
                    "MODIS ET:",
                    metadata[
                        "central_modis_et_mm_period"
                    ],
                )
                print(
                    "Eligible:",
                    metadata[
                        "central_parent_eligible"
                    ],
                )
                print(
                    "Published pixels:",
                    metadata[
                        "published_pixels"
                    ],
                )

            except Exception as exc:
                failures.append(
                    {
                        "period_start": period_start,
                        "station_id": station_id,
                        "error": repr(exc),
                    }
                )

                print(
                    "FAILED:",
                    period_start,
                    station_id,
                    repr(exc),
                )

    summary = {
        "dates": dates,
        "stations": station_ids,
        "halo_size": HALO_SIZE,
        "scientific_signature": (
            scientific_signature
        ),
        "failures": failures,
    }

    summary_path = (
        output_root
        / "production_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("HALO-7 FIELD PRODUCTION FINISHED")
    print("=" * 88)
    print("Failures:", len(failures))
    print("Summary:", summary_path)

    if failures:
        print()
        print(
            "Re-run the same command to retry only "
            "unfinished products; completed products "
            "will be reused."
        )


if __name__ == "__main__":
    main()
