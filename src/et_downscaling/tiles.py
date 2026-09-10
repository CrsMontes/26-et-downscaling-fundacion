"""Generic 20 m analysis-grid tiling utilities for final RF production."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from rasterio.features import bounds as geometry_bounds
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_geom

from .config import ANALYSIS_CRS
from .production import PREDICTION_SCALE_M

DEFAULT_TILE_SIZE_M = 4000
DEFAULT_MIN_TILE_SIZE_M = 500


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
    path = Path(project_root) / "data" / "boundaries" / "fundacion_basin.geojson"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features", [])
        if len(features) != 1:
            raise ValueError("Fundación basin GeoJSON must contain exactly one feature.")
        return features[0]["geometry"]
    if payload.get("type") == "Feature":
        return payload["geometry"]
    return payload


def _analysis_geometry(project_root: Path) -> dict:
    return transform_geom("EPSG:4326", ANALYSIS_CRS, _read_basin_geometry(project_root), precision=3)


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
        raise ValueError(f"Tile size must be >= {scale} m.")
    return int(math.ceil(tile_size_m / scale) * scale)


def build_initial_tiles(
    project_root: Path,
    tile_size_m: int = DEFAULT_TILE_SIZE_M,
) -> tuple[list[Tile], tuple[float, float, float, float]]:
    """Create analysis-grid-aligned core tiles intersecting the basin."""
    geometry = _analysis_geometry(project_root)
    xmin, ymin, xmax, ymax = _aligned_grid_bounds(geometry)
    tile_size = _normalize_tile_size(tile_size_m)
    width = int(math.ceil((xmax - xmin) / tile_size))
    height = int(math.ceil((ymax - ymin) / tile_size))
    grid_xmax = xmin + width * tile_size
    grid_ymin = ymax - height * tile_size
    transform = from_origin(xmin, ymax, tile_size, tile_size)
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
            tiles.append(Tile(x0, y0, x1, y1, f"r{row:03d}_c{col:03d}", 0))
    return tiles, (xmin, grid_ymin, grid_xmax, ymax)
