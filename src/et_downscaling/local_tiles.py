"""Local tiled production for the accepted Ridge-25 ET workflow.

Earth Engine is used only to compute small spatial chunks. Each chunk is
downloaded directly to the local workspace with `ee.Image.getDownloadURL`.
Google Drive and persistent Earth Engine assets are not used.

Why tiles?
----------
The three-pass fine-to-MODIS reconciliation is scientifically local but creates
a deep Earth Engine graph when evaluated for the whole Fundación basin at once.
A small-AOI smoke test demonstrated that the same graph is valid and conserves
MODIS ET. This module therefore evaluates the identical method independently
for aligned core tiles and mosaics the non-overlapping cores locally.

The tile calculation always includes the production module's processing buffer
around the core. If Earth Engine reports a memory/size error, the core tile is
split recursively. Scientific/model errors are not silently converted into
smaller tiles.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
import urllib.request
from urllib.error import HTTPError, URLError
from dataclasses import dataclass
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.features import bounds as geometry_bounds
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.warp import transform_geom

from .config import ANALYSIS_CRS
from .production import (
    MODIS_CONSERVATION_TOLERANCE_MM,
    PREDICTION_SCALE_M,
    load_basin_geometry,
)
from .ridge25 import (
    RIDGE25_MODEL_FEATURES,
    build_ee_ridge25_prediction,
)
from .ridge25_production import build_ridge25_production_stack
from .ridge25_spatial import build_ridge25_constrained_et
from .workspace import get_workspace_paths


DEFAULT_TILE_SIZE_M = 4000
DEFAULT_MIN_TILE_SIZE_M = 500

DIRECT_DOWNLOAD_MAX_ATTEMPTS = 6
TRANSIENT_HTTP_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


def _is_transient_http_status(code: int) -> bool:
    return int(code) in TRANSIENT_HTTP_STATUS_CODES


def _download_retry_delay_seconds(attempt: int) -> float:
    base = min(
        30.0,
        2.0 ** attempt,
    )
    return base + random.uniform(
        0.0,
        min(1.0, base * 0.25),
    )
OUTPUT_BANDS = [
    "ET_mm_period",
    "Kc_raw",
    "coarse_eligible",
    "ET_conservation_error_mm",
]
OUTPUT_NODATA = -9999.0


@dataclass(frozen=True)
class Tile:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    tile_id: str
    level: int = 0

    @property
    def width_m(self) -> float:
        return self.xmax - self.xmin

    @property
    def height_m(self) -> float:
        return self.ymax - self.ymin

    @property
    def width_px(self) -> int:
        return int(round(self.width_m / PREDICTION_SCALE_M))

    @property
    def height_px(self) -> int:
        return int(round(self.height_m / PREDICTION_SCALE_M))


@dataclass(frozen=True)
class CompletedTile:
    tile: Tile
    path: Path


def _snap_floor(value: float, step: float) -> float:
    return math.floor(value / step) * step


def _snap_ceil(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def _read_basin_geometry(project_root: Path) -> dict:
    path = (
        Path(project_root)
        / "data"
        / "boundaries"
        / "fundacion_basin.geojson"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("type") == "FeatureCollection":
        features = payload.get("features", [])
        if len(features) != 1:
            raise ValueError(
                "Fundación basin GeoJSON must contain exactly one feature."
            )
        return features[0]["geometry"]

    if payload.get("type") == "Feature":
        return payload["geometry"]

    return payload


def _analysis_geometry(project_root: Path) -> dict:
    geometry = _read_basin_geometry(project_root)
    return transform_geom(
        "EPSG:4326",
        ANALYSIS_CRS,
        geometry,
        precision=3,
    )


def _aligned_grid_bounds(geometry: dict) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = geometry_bounds(geometry)
    scale = float(PREDICTION_SCALE_M)
    return (
        _snap_floor(xmin, scale),
        _snap_floor(ymin, scale),
        _snap_ceil(xmax, scale),
        _snap_ceil(ymax, scale),
    )


def _normalize_tile_size(tile_size_m: int) -> int:
    scale = int(PREDICTION_SCALE_M)
    if tile_size_m < scale:
        raise ValueError(
            f"Tile size must be >= {scale} m."
        )
    return int(math.ceil(tile_size_m / scale) * scale)


def build_initial_tiles(
    project_root: Path,
    tile_size_m: int = DEFAULT_TILE_SIZE_M,
) -> tuple[
    list[Tile],
    tuple[float, float, float, float],
]:
    """Create analysis-grid-aligned core tiles intersecting the basin."""
    geometry = _analysis_geometry(project_root)
    xmin, ymin, xmax, ymax = _aligned_grid_bounds(geometry)
    tile_size = _normalize_tile_size(tile_size_m)

    width = int(math.ceil((xmax - xmin) / tile_size))
    height = int(math.ceil((ymax - ymin) / tile_size))

    grid_xmax = xmin + width * tile_size
    grid_ymin = ymax - height * tile_size

    transform = from_origin(
        xmin,
        ymax,
        tile_size,
        tile_size,
    )
    touched = rasterize(
        [(geometry, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    tiles: list[Tile] = []
    for row in range(height):
        for col in range(width):
            if touched[row, col] == 0:
                continue

            x0 = xmin + col * tile_size
            x1 = x0 + tile_size
            y1 = ymax - row * tile_size
            y0 = y1 - tile_size

            tiles.append(
                Tile(
                    xmin=x0,
                    ymin=y0,
                    xmax=x1,
                    ymax=y1,
                    tile_id=f"r{row:03d}_c{col:03d}",
                    level=0,
                )
            )

    return tiles, (xmin, grid_ymin, grid_xmax, ymax)


def tile_intersects_geometry(
    tile: Tile,
    geometry: dict,
) -> bool:
    transform = from_origin(
        tile.xmin,
        tile.ymax,
        tile.width_m,
        tile.height_m,
    )
    touched = rasterize(
        [(geometry, 1)],
        out_shape=(1, 1),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return bool(touched[0, 0])


def split_tile(
    tile: Tile,
    project_root: Path,
) -> list[Tile]:
    """Split one aligned core into four aligned children."""
    geometry = _analysis_geometry(project_root)

    mid_x = (tile.xmin + tile.xmax) / 2.0
    mid_y = (tile.ymin + tile.ymax) / 2.0

    scale = float(PREDICTION_SCALE_M)
    mid_x = _snap_floor(mid_x, scale)
    mid_y = _snap_floor(mid_y, scale)

    candidates = [
        Tile(
            tile.xmin,
            mid_y,
            mid_x,
            tile.ymax,
            f"{tile.tile_id}_nw",
            tile.level + 1,
        ),
        Tile(
            mid_x,
            mid_y,
            tile.xmax,
            tile.ymax,
            f"{tile.tile_id}_ne",
            tile.level + 1,
        ),
        Tile(
            tile.xmin,
            tile.ymin,
            mid_x,
            mid_y,
            f"{tile.tile_id}_sw",
            tile.level + 1,
        ),
        Tile(
            mid_x,
            tile.ymin,
            tile.xmax,
            mid_y,
            f"{tile.tile_id}_se",
            tile.level + 1,
        ),
    ]

    return [
        child
        for child in candidates
        if child.width_px > 0
        and child.height_px > 0
        and tile_intersects_geometry(
            child,
            geometry,
        )
    ]


def _tile_geometry(tile: Tile) -> ee.Geometry:
    return ee.Geometry.Rectangle(
        [
            tile.xmin,
            tile.ymin,
            tile.xmax,
            tile.ymax,
        ],
        proj=ANALYSIS_CRS,
        geodesic=False,
    )


def build_tile_image(
    project_root: Path,
    period_start: str,
    model,
    tile: Tile,
) -> ee.Image:
    """Build the four-band final product for one non-overlapping core tile."""
    core = _tile_geometry(tile)
    basin = load_basin_geometry(project_root)
    output_geometry = core.intersection(
        basin,
        maxError=10,
    )

    # The stack builder adds the established production buffer around this
    # core. Therefore fine-to-MODIS operations have neighboring support while
    # only the non-overlapping core is written.
    context = build_ridge25_production_stack(
        period_start_text=period_start,
        basin_geometry=core,
    )

    kc_raw = build_ee_ridge25_prediction(
        model_stack=context["stack"],
        model=model,
        feature_names=RIDGE25_MODEL_FEATURES,
        output_name="Kc_raw",
    ).toFloat()

    outputs = build_ridge25_constrained_et(
        kc_raw=kc_raw,
        optical_predictors=context["optical"],
        model_stack=context["stack"],
        modis_et=context["modis_et"],
        modis_projection=context["modis_projection"],
        fine_projection=context["fine_projection"],
        basin_geometry=output_geometry,
    )

    return (
        ee.Image.cat(
            [
                outputs["et_final"].rename(
                    "ET_mm_period"
                ),
                outputs["kc_raw"].rename(
                    "Kc_raw"
                ),
                outputs["eligible"].rename(
                    "coarse_eligible"
                ),
                outputs["conservation_error"].rename(
                    "ET_conservation_error_mm"
                ),
            ]
        )
        .clip(output_geometry)
        .toFloat()
    )


def _tile_download_parameters(tile: Tile) -> dict:
    return {
        "bands": OUTPUT_BANDS,
        "crs": ANALYSIS_CRS,
        "crs_transform": [
            PREDICTION_SCALE_M,
            0,
            tile.xmin,
            0,
            -PREDICTION_SCALE_M,
            tile.ymax,
        ],
        "dimensions": [
            tile.width_px,
            tile.height_px,
        ],
        "format": "GEO_TIFF",
    }


def _is_splittable_error(error: Exception) -> bool:
    message = str(error).lower()
    tokens = (
        "memory limit",
        "request size",
        "too many pixels",
        "computation timed out",
        "payload",
        "response too large",
    )
    return any(token in message for token in tokens)


def _validate_tile_file(
    path: Path,
    tile: Tile,
) -> None:
    with rasterio.open(path) as dataset:
        if dataset.count != len(OUTPUT_BANDS):
            raise RuntimeError(
                f"{path} has {dataset.count} bands; "
                f"expected {len(OUTPUT_BANDS)}."
            )
        if dataset.width != tile.width_px:
            raise RuntimeError(
                f"{path} width mismatch."
            )
        if dataset.height != tile.height_px:
            raise RuntimeError(
                f"{path} height mismatch."
            )


def download_tile(
    project_root: Path,
    period_start: str,
    model,
    tile: Tile,
    tile_directory: Path,
    timeout_seconds: int = 600,
) -> CompletedTile:
    """Download one tile directly from Earth Engine to local disk."""
    tile_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = tile_directory / f"{tile.tile_id}.tif"

    if path.is_file():
        try:
            _validate_tile_file(path, tile)
            return CompletedTile(tile, path)
        except Exception:
            path.unlink(missing_ok=True)

    image = build_tile_image(
        project_root=project_root,
        period_start=period_start,
        model=model,
        tile=tile,
    )

    parameters = _tile_download_parameters(tile)

    temporary = path.with_suffix(".part")
    temporary.unlink(missing_ok=True)

    for attempt in range(
        1,
        DIRECT_DOWNLOAD_MAX_ATTEMPTS + 1,
    ):
        try:
            url = image.getDownloadURL(
                parameters
            )

            with urllib.request.urlopen(
                url,
                timeout=timeout_seconds,
            ) as response:
                with temporary.open("wb") as output:
                    while True:
                        chunk = response.read(
                            1024 * 1024
                        )
                        if not chunk:
                            break
                        output.write(chunk)

            break

        except HTTPError as error:
            try:
                body = error.read().decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                body = ""

            temporary.unlink(
                missing_ok=True
            )

            if (
                _is_transient_http_status(
                    error.code
                )
                and attempt
                < DIRECT_DOWNLOAD_MAX_ATTEMPTS
            ):
                delay = (
                    _download_retry_delay_seconds(
                        attempt
                    )
                )
                print(
                    "Transient Earth Engine "
                    f"HTTP {error.code}; "
                    f"retry {attempt}/"
                    f"{DIRECT_DOWNLOAD_MAX_ATTEMPTS} "
                    f"in {delay:.1f} s..."
                )
                time.sleep(delay)
                continue

            raise RuntimeError(
                "Earth Engine direct download failed "
                f"(HTTP {error.code}): "
                f"{body or error.reason}"
            ) from error

        except (URLError, TimeoutError) as error:
            temporary.unlink(
                missing_ok=True
            )

            if (
                attempt
                < DIRECT_DOWNLOAD_MAX_ATTEMPTS
            ):
                delay = (
                    _download_retry_delay_seconds(
                        attempt
                    )
                )
                print(
                    "Transient network error during "
                    "Earth Engine download; "
                    f"retry {attempt}/"
                    f"{DIRECT_DOWNLOAD_MAX_ATTEMPTS} "
                    f"in {delay:.1f} s: {error}"
                )
                time.sleep(delay)
                continue

            raise RuntimeError(
                "Earth Engine direct download failed "
                "after repeated network errors: "
                f"{error}"
            ) from error

    temporary.replace(path)
    _validate_tile_file(path, tile)

    return CompletedTile(tile, path)


def download_tile_adaptive(
    project_root: Path,
    period_start: str,
    model,
    tile: Tile,
    tile_directory: Path,
    min_tile_size_m: int = DEFAULT_MIN_TILE_SIZE_M,
) -> list[CompletedTile]:
    """Download a tile, recursively splitting only resource-limit failures."""
    try:
        return [
            download_tile(
                project_root=project_root,
                period_start=period_start,
                model=model,
                tile=tile,
                tile_directory=tile_directory,
            )
        ]
    except Exception as error:
        if (
            not _is_splittable_error(error)
            or tile.width_m
            <= _normalize_tile_size(
                min_tile_size_m
            )
            or tile.height_m
            <= _normalize_tile_size(
                min_tile_size_m
            )
        ):
            raise

        children = split_tile(
            tile,
            project_root,
        )
        if not children:
            raise

        completed: list[CompletedTile] = []
        for child in children:
            completed.extend(
                download_tile_adaptive(
                    project_root=project_root,
                    period_start=period_start,
                    model=model,
                    tile=child,
                    tile_directory=tile_directory,
                    min_tile_size_m=min_tile_size_m,
                )
            )
        return completed


def _write_manifest(
    completed: list[CompletedTile],
    path: Path,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile_id",
                "level",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "width_px",
                "height_px",
                "path",
            ],
        )
        writer.writeheader()
        for item in completed:
            tile = item.tile
            writer.writerow(
                {
                    "tile_id": tile.tile_id,
                    "level": tile.level,
                    "xmin": tile.xmin,
                    "ymin": tile.ymin,
                    "xmax": tile.xmax,
                    "ymax": tile.ymax,
                    "width_px": tile.width_px,
                    "height_px": tile.height_px,
                    "path": str(item.path),
                }
            )


def mosaic_tiles(
    completed: list[CompletedTile],
    output_path: Path,
    grid_bounds: tuple[
        float,
        float,
        float,
        float,
    ],
) -> Path:
    """Write the aligned non-overlapping tile cores to one local GeoTIFF."""
    if not completed:
        raise ValueError(
            "No tiles are available to mosaic."
        )

    xmin, ymin, xmax, ymax = grid_bounds
    scale = float(PREDICTION_SCALE_M)

    width = int(round((xmax - xmin) / scale))
    height = int(round((ymax - ymin) / scale))
    transform = from_origin(
        xmin,
        ymax,
        scale,
        scale,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": len(OUTPUT_BANDS),
        "dtype": "float32",
        "crs": ANALYSIS_CRS,
        "transform": transform,
        "nodata": OUTPUT_NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as destination:
        for index, band_name in enumerate(
            OUTPUT_BANDS,
            start=1,
        ):
            destination.set_band_description(
                index,
                band_name,
            )

        for item in completed:
            tile = item.tile
            col_off = int(
                round(
                    (tile.xmin - xmin)
                    / scale
                )
            )
            row_off = int(
                round(
                    (ymax - tile.ymax)
                    / scale
                )
            )

            with rasterio.open(
                item.path
            ) as source:
                data = source.read(
                    masked=True
                ).filled(OUTPUT_NODATA)
                destination.write(
                    data.astype(
                        np.float32,
                        copy=False,
                    ),
                    window=Window(
                        col_off,
                        row_off,
                        tile.width_px,
                        tile.height_px,
                    ),
                )

    return output_path


def download_ridge25_basin(
    project_root: Path,
    period_start: str,
    model,
    tile_size_m: int = DEFAULT_TILE_SIZE_M,
    min_tile_size_m: int = DEFAULT_MIN_TILE_SIZE_M,
) -> dict[str, object]:
    """Download and mosaic one final Ridge-25 period entirely to local disk."""
    project_root = Path(
        project_root
    ).resolve()
    workspace = get_workspace_paths(
        project_root
    ).ensure()

    period_directory = (
        workspace.rasters
        / period_start
    )
    tile_directory = (
        period_directory
        / "tiles"
    )
    period_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    initial_tiles, grid_bounds = (
        build_initial_tiles(
            project_root,
            tile_size_m=tile_size_m,
        )
    )

    completed: list[CompletedTile] = []
    for index, tile in enumerate(
        initial_tiles,
        start=1,
    ):
        print(
            f"[{index}/{len(initial_tiles)}] "
            f"{tile.tile_id} "
            f"({int(tile.width_m)} m core)"
        )
        completed.extend(
            download_tile_adaptive(
                project_root=project_root,
                period_start=period_start,
                model=model,
                tile=tile,
                tile_directory=tile_directory,
                min_tile_size_m=min_tile_size_m,
            )
        )

    completed = sorted(
        completed,
        key=lambda item: item.tile.tile_id,
    )

    manifest_path = (
        period_directory
        / "tile_manifest.csv"
    )
    _write_manifest(
        completed,
        manifest_path,
    )

    raster_path = (
        period_directory
        / f"ET_Ridge25_{period_start}_20m.tif"
    )
    mosaic_tiles(
        completed,
        raster_path,
        grid_bounds,
    )

    metadata = {
        "period_start": period_start,
        "analysis_crs": ANALYSIS_CRS,
        "prediction_scale_m": (
            PREDICTION_SCALE_M
        ),
        "initial_tile_size_m": (
            _normalize_tile_size(
                tile_size_m
            )
        ),
        "minimum_tile_size_m": (
            _normalize_tile_size(
                min_tile_size_m
            )
        ),
        "initial_tiles": len(
            initial_tiles
        ),
        "completed_tiles": len(
            completed
        ),
        "output_bands": OUTPUT_BANDS,
        "conservation_tolerance_mm": (
            MODIS_CONSERVATION_TOLERANCE_MM
        ),
        "google_drive_used": False,
        "earth_engine_asset_created": False,
        "model_source": (
            "fitted_in_current_run"
        ),
        "raster": str(raster_path),
        "tile_manifest": str(
            manifest_path
        ),
    }

    metadata_path = (
        period_directory
        / "production_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "raster": raster_path,
        "manifest": manifest_path,
        "metadata": metadata_path,
        "completed_tiles": completed,
        "initial_tiles": initial_tiles,
    }
