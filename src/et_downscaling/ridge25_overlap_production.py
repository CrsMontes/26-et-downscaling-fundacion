"""Final Ridge-25 production with one global exact-overlap reconciliation.

The legacy tiled module remains unchanged as a diagnostic path. This module
uses tiles only to obtain the raw Ridge-25/AOA fields, retains an external
support halo, mosaics those raw fields, and reconciles once globally against
native-grid MODIS ET using real fine/coarse overlap areas.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from rasterio.windows import Window

from .config import ANALYSIS_CRS
from .local_reconciliation import (
    RIDGE25_RECONCILIATION_TOLERANCE_MM,
    RIDGE25_USABLE_SUPPORT_FRACTION,
    score_local_ridge25,
)
from .local_tiles import (
    CompletedTile,
    Tile,
    _analysis_geometry,
    _normalize_tile_size,
    build_initial_tiles,
)
from .overlap_reconciliation import (
    build_native_modis_grid,
    build_overlap_edges,
    materialize_active_values,
    solve_overlap_reconciliation,
)
from .production import (
    PREDICTION_SCALE_M,
    PROCESSING_BUFFER_M,
    build_modis_period_context,
)
from .ridge25 import RIDGE25_MODEL_FEATURES
from .ridge25_local_production import (
    DIRECT_DOWNLOAD_MAX_ATTEMPTS,
    OUTPUT_NODATA,
    _download_ee_bytes,
    _read_downloaded_array,
)
from .ridge25_production import build_ridge25_production_stack
from .workspace import get_workspace_paths


RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION = (
    "ridge25_exact_overlap_support90_tol001_v2"
)

RAW_TILE_BANDS = [
    "Kc_raw",
    "dissimilarity_index",
    "stack_valid",
    "AOA_inside",
    "usable",
    "support_domain",
]

OUTPUT_BANDS = [
    "ET_mm_period",
    "Kc_raw",
    "dissimilarity_index",
    "stack_valid",
    "AOA_inside",
    "usable",
    "usable_fraction",
    "coarse_eligible",
    "ET_conservation_error_mm",
]


def _processing_grid(tile: Tile) -> tuple[float, float, float, float, int, int, Affine]:
    scale = float(PREDICTION_SCALE_M)
    buffer_m = float(PROCESSING_BUFFER_M)
    xmin = tile.xmin - buffer_m
    xmax = tile.xmax + buffer_m
    ymin = tile.ymin - buffer_m
    ymax = tile.ymax + buffer_m
    width = int(round((xmax - xmin) / scale))
    height = int(round((ymax - ymin) / scale))
    transform = from_origin(xmin, ymax, scale, scale)
    return xmin, ymin, xmax, ymax, width, height, transform


def _expected_tile_transform(tile: Tile) -> Affine:
    return from_origin(
        tile.xmin,
        tile.ymax,
        PREDICTION_SCALE_M,
        PREDICTION_SCALE_M,
    )


def _support_tiles(
    project_root: Path,
    tile_size_m: int,
) -> tuple[list[Tile], tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Return basin tiles plus a deterministic halo at least 1 km wide."""
    basin_tiles, basin_grid_bounds = build_initial_tiles(
        project_root,
        tile_size_m=tile_size_m,
    )
    if not basin_tiles:
        raise RuntimeError("No basin tiles were generated.")

    tile_size = float(basin_tiles[0].width_m)
    halo_rings = max(1, int(math.ceil(float(PROCESSING_BUFFER_M) / tile_size)))

    by_origin: dict[tuple[float, float], Tile] = {}
    for tile in basin_tiles:
        for dy in range(-halo_rings, halo_rings + 1):
            for dx in range(-halo_rings, halo_rings + 1):
                xmin = tile.xmin + dx * tile_size
                ymin = tile.ymin + dy * tile_size
                key = (xmin, ymin)
                if key in by_origin:
                    continue
                xmax = xmin + tile_size
                ymax = ymin + tile_size
                by_origin[key] = Tile(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    tile_id=f"sx{int(round(xmin))}_sy{int(round(ymin))}",
                    level=0,
                )

    tiles = sorted(
        by_origin.values(),
        key=lambda item: (-item.ymax, item.xmin),
    )
    support_bounds = (
        min(item.xmin for item in tiles),
        min(item.ymin for item in tiles),
        max(item.xmax for item in tiles),
        max(item.ymax for item in tiles),
    )
    return tiles, support_bounds, basin_grid_bounds


def _build_raw_tile(
    period_start: str,
    model,
    aoa_parameters,
    tile: Tile,
    timeout_seconds: int,
) -> tuple[np.ndarray, dict[str, object]]:
    core = ee.Geometry.Rectangle(
        [tile.xmin, tile.ymin, tile.xmax, tile.ymax],
        proj=ANALYSIS_CRS,
        geodesic=False,
    )
    context = build_ridge25_production_stack(
        period_start_text=period_start,
        basin_geometry=core,
    )

    (
        source_xmin,
        source_ymin,
        source_xmax,
        source_ymax,
        source_width,
        source_height,
        requested_transform,
    ) = _processing_grid(tile)

    predictor_image = context["stack"].select(RIDGE25_MODEL_FEATURES).toFloat()
    parameters = {
        "bands": RIDGE25_MODEL_FEATURES,
        "crs": ANALYSIS_CRS,
        "crs_transform": [
            PREDICTION_SCALE_M,
            0,
            source_xmin,
            0,
            -PREDICTION_SCALE_M,
            source_ymax,
        ],
        "dimensions": [source_width, source_height],
        "format": "GEO_TIFF",
    }
    payload = _download_ee_bytes(
        predictor_image,
        parameters,
        timeout_seconds,
    )
    predictor_bands, transform, crs = _read_downloaded_array(payload)

    expected_shape = (
        len(RIDGE25_MODEL_FEATURES),
        source_height,
        source_width,
    )
    if predictor_bands.shape != expected_shape:
        raise RuntimeError(
            f"Downloaded predictor shape {predictor_bands.shape}; "
            f"expected {expected_shape}."
        )
    if crs is None:
        raise RuntimeError("Downloaded predictor stack has no CRS.")
    if not transform.almost_equals(requested_transform):
        raise RuntimeError("Downloaded predictor transform differs from requested 20 m grid.")

    cube = np.moveaxis(
        predictor_bands.filled(np.nan),
        0,
        -1,
    ).astype(np.float64)
    state = score_local_ridge25(
        predictor_cube=cube,
        model=model,
        aoa_parameters=aoa_parameters,
    )

    buffer_pixels_float = float(PROCESSING_BUFFER_M) / float(PREDICTION_SCALE_M)
    buffer_pixels = int(round(buffer_pixels_float))
    if not math.isclose(buffer_pixels_float, buffer_pixels, abs_tol=1e-9):
        raise RuntimeError("Production buffer is not aligned with the 20 m grid.")

    rs = slice(buffer_pixels, buffer_pixels + tile.height_px)
    cs = slice(buffer_pixels, buffer_pixels + tile.width_px)
    support = np.ones((tile.height_px, tile.width_px), dtype=np.float64)

    output = np.stack(
        [
            state.kc_raw[rs, cs],
            state.dissimilarity_index[rs, cs],
            state.stack_valid[rs, cs].astype(np.float64),
            state.aoa_inside[rs, cs].astype(np.float64),
            state.usable[rs, cs].astype(np.float64),
            support,
        ],
        axis=0,
    )

    return output, {
        "production_method_version": RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION,
        "period_start": period_start,
        "tile_id": tile.tile_id,
        "tile_role": "raw_support",
        "stack_valid_pixels": int(state.stack_valid[rs, cs].sum()),
        "usable_pixels": int(state.usable[rs, cs].sum()),
    }


def _validate_raw_tile(path: Path, tile: Tile) -> None:
    with rasterio.open(path) as dataset:
        if dataset.count != len(RAW_TILE_BANDS):
            raise RuntimeError(f"{path} band-count mismatch.")
        if dataset.width != tile.width_px or dataset.height != tile.height_px:
            raise RuntimeError(f"{path} tile dimensions mismatch.")
        if tuple(dataset.descriptions) != tuple(RAW_TILE_BANDS):
            raise RuntimeError(f"{path} raw band descriptions mismatch.")
        if dataset.crs is None or dataset.crs.to_string() != ANALYSIS_CRS:
            raise RuntimeError(f"{path} CRS mismatch.")
        if not dataset.transform.almost_equals(_expected_tile_transform(tile)):
            raise RuntimeError(f"{path} transform mismatch.")


def _write_raw_tile(path: Path, tile: Tile, data: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "width": tile.width_px,
        "height": tile.height_px,
        "count": len(RAW_TILE_BANDS),
        "dtype": "float32",
        "crs": ANALYSIS_CRS,
        "transform": _expected_tile_transform(tile),
        "nodata": OUTPUT_NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    prepared = np.where(np.isfinite(data), data, OUTPUT_NODATA).astype(np.float32)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(prepared)
        for index, name in enumerate(RAW_TILE_BANDS, start=1):
            destination.set_band_description(index, name)


def _download_raw_tile(
    period_start: str,
    model,
    aoa_parameters,
    tile: Tile,
    tile_directory: Path,
    timeout_seconds: int,
) -> CompletedTile:
    tile_directory.mkdir(parents=True, exist_ok=True)
    path = tile_directory / f"{tile.tile_id}.tif"
    metadata_path = tile_directory / f"{tile.tile_id}.json"

    if path.is_file() and metadata_path.is_file():
        try:
            _validate_raw_tile(path, tile)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("production_method_version") != RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION:
                raise RuntimeError("Cached tile version mismatch.")
            if metadata.get("tile_role") != "raw_support":
                raise RuntimeError("Cached tile role mismatch.")
            return CompletedTile(tile, path)
        except Exception:
            path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)

    data, metadata = _build_raw_tile(
        period_start=period_start,
        model=model,
        aoa_parameters=aoa_parameters,
        tile=tile,
        timeout_seconds=timeout_seconds,
    )
    temporary = path.with_suffix(".part.tif")
    temporary_meta = metadata_path.with_suffix(".part.json")
    temporary.unlink(missing_ok=True)
    temporary_meta.unlink(missing_ok=True)
    _write_raw_tile(temporary, tile, data)
    _validate_raw_tile(temporary, tile)
    temporary_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(path)
    temporary_meta.replace(metadata_path)
    return CompletedTile(tile, path)


def _write_manifest(completed: list[CompletedTile], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile_id",
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
                    "xmin": tile.xmin,
                    "ymin": tile.ymin,
                    "xmax": tile.xmax,
                    "ymax": tile.ymax,
                    "width_px": tile.width_px,
                    "height_px": tile.height_px,
                    "path": str(item.path),
                }
            )


def _mosaic_raw_tiles(
    completed: list[CompletedTile],
    output_path: Path,
    bounds: tuple[float, float, float, float],
) -> Path:
    xmin, ymin, xmax, ymax = bounds
    scale = float(PREDICTION_SCALE_M)
    width = int(round((xmax - xmin) / scale))
    height = int(round((ymax - ymin) / scale))
    transform = from_origin(xmin, ymax, scale, scale)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": len(RAW_TILE_BANDS),
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as destination:
        for index, name in enumerate(RAW_TILE_BANDS, start=1):
            destination.set_band_description(index, name)
        for item in completed:
            tile = item.tile
            col_off = int(round((tile.xmin - xmin) / scale))
            row_off = int(round((ymax - tile.ymax) / scale))
            with rasterio.open(item.path) as source:
                data = source.read(masked=True).filled(OUTPUT_NODATA)
            destination.write(
                data.astype(np.float32, copy=False),
                window=Window(col_off, row_off, tile.width_px, tile.height_px),
            )
    return output_path


def _download_native_modis(
    period_start: str,
    support_bounds: tuple[float, float, float, float],
    timeout_seconds: int,
):
    geometry = ee.Geometry.Rectangle(
        list(support_bounds),
        proj=ANALYSIS_CRS,
        geodesic=False,
    )
    context = build_modis_period_context(period_start, geometry)
    projection_info = context["modis_projection"].getInfo()
    grid = build_native_modis_grid(
        projection_info=projection_info,
        processing_bounds=support_bounds,
    )
    height, width = grid.shape
    image = context["modis_et"].rename("ET_MODIS_mm_period").toFloat()
    parameters = {
        "bands": ["ET_MODIS_mm_period"],
        "crs": grid.earth_engine_crs,
        "crs_transform": [
            grid.transform.a,
            grid.transform.b,
            grid.transform.c,
            grid.transform.d,
            grid.transform.e,
            grid.transform.f,
        ],
        "dimensions": [width, height],
        "format": "GEO_TIFF",
    }
    payload = _download_ee_bytes(image, parameters, timeout_seconds)
    bands, downloaded_transform, downloaded_crs = _read_downloaded_array(payload)
    if bands.shape != (1, height, width):
        raise RuntimeError("Downloaded MODIS ET grid has unexpected shape.")
    if downloaded_crs is None:
        raise RuntimeError("Downloaded MODIS ET has no CRS.")
    if not downloaded_transform.almost_equals(grid.transform):
        raise RuntimeError("Downloaded MODIS transform differs from requested native grid.")
    return bands[0].filled(np.nan).astype(np.float64), grid


def _fine_diagnostics(edges, result, fine_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coarse = np.asarray(edges.coarse_index, dtype=np.int64)
    fine = np.asarray(edges.fine_index, dtype=np.int64)
    area = np.asarray(edges.overlap_area_m2, dtype=float)

    usable_fraction_coarse = result.usable_fraction.ravel()
    eligible = result.eligible_coarse_mask.ravel()

    minimum_support = np.full(fine_size, np.inf, dtype=np.float64)
    finite_support_edge = np.isfinite(usable_fraction_coarse[coarse])
    np.minimum.at(
        minimum_support,
        fine[finite_support_edge],
        usable_fraction_coarse[coarse[finite_support_edge]],
    )
    minimum_support[~np.isfinite(minimum_support)] = np.nan

    touched = np.bincount(fine, minlength=fine_size) > 0
    bad = np.bincount(
        fine,
        weights=(~eligible[coarse]).astype(np.int32),
        minlength=fine_size,
    ) > 0
    all_eligible = touched & ~bad

    coarse_error = np.full(eligible.size, np.nan, dtype=np.float64)
    coarse_error[result.eligible_coarse] = result.final_error_after_nonnegative
    absolute_error = np.abs(coarse_error)
    maximum_error = np.full(fine_size, -np.inf, dtype=np.float64)
    finite_error_edge = np.isfinite(absolute_error[coarse])
    np.maximum.at(
        maximum_error,
        fine[finite_error_edge],
        absolute_error[coarse[finite_error_edge]],
    )
    maximum_error[~np.isfinite(maximum_error)] = np.nan

    return minimum_support, all_eligible.astype(np.float64), maximum_error


def _crop_window(
    support_transform: Affine,
    target_bounds: tuple[float, float, float, float],
) -> Window:
    xmin, ymin, xmax, ymax = target_bounds
    scale = float(PREDICTION_SCALE_M)
    col_off = int(round((xmin - support_transform.c) / scale))
    row_off = int(round((support_transform.f - ymax) / scale))
    width = int(round((xmax - xmin) / scale))
    height = int(round((ymax - ymin) / scale))
    return Window(col_off, row_off, width, height)


def _write_final_raster(
    path: Path,
    arrays: list[np.ndarray],
    transform: Affine,
) -> None:
    height, width = arrays[0].shape
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
    prepared = np.stack(arrays, axis=0)
    prepared = np.where(np.isfinite(prepared), prepared, OUTPUT_NODATA).astype(np.float32)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(prepared)
        for index, name in enumerate(OUTPUT_BANDS, start=1):
            destination.set_band_description(index, name)


def _reconcile_raw_mosaic(
    project_root: Path,
    period_start: str,
    raw_mosaic_path: Path,
    final_path: Path,
    basin_grid_bounds: tuple[float, float, float, float],
    timeout_seconds: int,
) -> dict[str, object]:
    with rasterio.open(raw_mosaic_path) as dataset:
        if tuple(dataset.descriptions) != tuple(RAW_TILE_BANDS):
            raise RuntimeError("Raw support mosaic band contract mismatch.")
        raw = dataset.read(masked=True).filled(np.nan).astype(np.float64)
        support_transform = dataset.transform
        support_crs = dataset.crs
        support_bounds = (
            dataset.bounds.left,
            dataset.bounds.bottom,
            dataset.bounds.right,
            dataset.bounds.top,
        )
        fine_shape = (dataset.height, dataset.width)

    if support_crs is None or support_crs.to_string() != ANALYSIS_CRS:
        raise RuntimeError("Raw support mosaic CRS mismatch.")

    kc_raw = raw[RAW_TILE_BANDS.index("Kc_raw")]
    dissimilarity = raw[RAW_TILE_BANDS.index("dissimilarity_index")]
    stack_valid = raw[RAW_TILE_BANDS.index("stack_valid")] > 0.5
    aoa_inside = raw[RAW_TILE_BANDS.index("AOA_inside")] > 0.5
    usable = raw[RAW_TILE_BANDS.index("usable")] > 0.5
    domain = raw[RAW_TILE_BANDS.index("support_domain")] > 0.5

    modis_et, modis_grid = _download_native_modis(
        period_start=period_start,
        support_bounds=support_bounds,
        timeout_seconds=timeout_seconds,
    )
    print("Building global exact-overlap operator...")
    edges = build_overlap_edges(
        domain=domain,
        fine_transform=support_transform,
        fine_crs=support_crs,
        modis_et=modis_et,
        modis_transform=modis_grid.transform,
        modis_crs=modis_grid.local_crs,
        progress_every=1000,
    )
    print("Solving one global exact-overlap reconciliation...")
    result = solve_overlap_reconciliation(
        kc_raw=kc_raw,
        usable=usable,
        modis_et=modis_et,
        edges=edges,
        usable_support_fraction=RIDGE25_USABLE_SUPPORT_FRACTION,
        tolerance_mm=RIDGE25_RECONCILIATION_TOLERANCE_MM,
    )

    et_support = materialize_active_values(
        fine_shape=fine_shape,
        active_fine=result.active_fine,
        values=result.et_final_nonnegative,
        selected_active=result.publishable_active,
    )
    minimum_support, all_eligible, maximum_error = _fine_diagnostics(
        edges=edges,
        result=result,
        fine_size=kc_raw.size,
    )
    minimum_support = minimum_support.reshape(fine_shape)
    all_eligible = all_eligible.reshape(fine_shape)
    maximum_error = maximum_error.reshape(fine_shape)

    crop = _crop_window(support_transform, basin_grid_bounds)
    rs = slice(int(crop.row_off), int(crop.row_off + crop.height))
    cs = slice(int(crop.col_off), int(crop.col_off + crop.width))
    final_transform = from_origin(
        basin_grid_bounds[0],
        basin_grid_bounds[3],
        PREDICTION_SCALE_M,
        PREDICTION_SCALE_M,
    )

    basin_geometry = _analysis_geometry(project_root)
    basin_mask = rasterize(
        [(basin_geometry, 1)],
        out_shape=(int(crop.height), int(crop.width)),
        transform=final_transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)

    source_arrays = [
        et_support,
        kc_raw,
        dissimilarity,
        stack_valid.astype(np.float64),
        aoa_inside.astype(np.float64),
        usable.astype(np.float64),
        minimum_support,
        all_eligible,
        maximum_error,
    ]
    final_arrays = []
    for array in source_arrays:
        cropped = np.asarray(array[rs, cs], dtype=np.float64)
        final_arrays.append(np.where(basin_mask, cropped, np.nan))

    published = final_arrays[0][np.isfinite(final_arrays[0])]
    if published.size and np.any(published < 0):
        raise RuntimeError("Negative ET reached final published raster.")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    _write_final_raster(final_path, final_arrays, final_transform)

    adjustment = result.et_final_nonnegative - result.et_initial
    finite_pair = np.isfinite(result.et_initial) & np.isfinite(result.et_final_nonnegative)
    if int(finite_pair.sum()) > 1:
        correlation = float(
            np.corrcoef(
                result.et_initial[finite_pair],
                result.et_final_nonnegative[finite_pair],
            )[0, 1]
        )
    else:
        correlation = float("nan")

    return {
        "eligible_modis_parents": int(result.eligible_coarse.size),
        "active_fine_cells": int(result.active_fine.size),
        "publishable_active_cells_support": int(result.publishable_active.sum()),
        "published_basin_pixels": int(published.size),
        "negative_active_before_floor": int(result.negative_active_cells),
        "negative_publishable_before_floor": int(result.negative_publishable_cells),
        "max_abs_conservation_error_before_floor_mm": float(result.max_abs_final_error_mm),
        "max_abs_conservation_error_after_floor_mm": float(result.max_abs_error_after_nonnegative_mm),
        "mae_adjustment_mm": float(np.mean(np.abs(adjustment))),
        "rmse_adjustment_mm": float(np.sqrt(np.mean(adjustment ** 2))),
        "pearson_final_vs_initial": correlation,
    }


def download_ridge25_basin(
    project_root: Path,
    period_start: str,
    model,
    aoa_parameters,
    tile_size_m: int = 4000,
    min_tile_size_m: int = 500,
) -> dict[str, object]:
    """Produce the final 20 m basin raster using global exact overlaps."""
    project_root = Path(project_root).resolve()
    workspace = get_workspace_paths(project_root).ensure()
    period_directory = workspace.rasters / period_start
    period_directory.mkdir(parents=True, exist_ok=True)

    support_tiles, support_bounds, basin_grid_bounds = _support_tiles(
        project_root=project_root,
        tile_size_m=tile_size_m,
    )
    tile_directory = period_directory / (
        "tiles_" + RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION
    )

    completed: list[CompletedTile] = []
    for index, tile in enumerate(support_tiles, start=1):
        print(
            f"[{index}/{len(support_tiles)}] {tile.tile_id} "
            f"({int(tile.width_m)} m raw support core)"
        )
        completed.append(
            _download_raw_tile(
                period_start=period_start,
                model=model,
                aoa_parameters=aoa_parameters,
                tile=tile,
                tile_directory=tile_directory,
                timeout_seconds=600,
            )
        )

    manifest_path = period_directory / (
        "tile_manifest_" + RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION + ".csv"
    )
    _write_manifest(completed, manifest_path)

    raw_mosaic_path = period_directory / (
        "raw_support_" + RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION + f"_{period_start}_20m.tif"
    )
    _mosaic_raw_tiles(completed, raw_mosaic_path, support_bounds)

    raster_path = period_directory / (
        "ET_" + RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION + f"_{period_start}_20m.tif"
    )
    reconciliation = _reconcile_raw_mosaic(
        project_root=project_root,
        period_start=period_start,
        raw_mosaic_path=raw_mosaic_path,
        final_path=raster_path,
        basin_grid_bounds=basin_grid_bounds,
        timeout_seconds=600,
    )

    metadata = {
        "period_start": period_start,
        "production_method_version": RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION,
        "analysis_crs": ANALYSIS_CRS,
        "prediction_scale_m": PREDICTION_SCALE_M,
        "tile_size_m": _normalize_tile_size(tile_size_m),
        "minimum_tile_size_m": _normalize_tile_size(min_tile_size_m),
        "support_halo_rule": "one_or_more_full_tile_rings_covering_at_least_1000_m",
        "support_tile_count": len(support_tiles),
        "output_bands": OUTPUT_BANDS,
        "usable_support_fraction": RIDGE25_USABLE_SUPPORT_FRACTION,
        "conservation_tolerance_mm": RIDGE25_RECONCILIATION_TOLERANCE_MM,
        "applicability_rule": "complete_stack AND AOA_inside AND Kc_raw >= 0",
        "reconciliation": "single_global_exact_overlap_after_raw_mosaic",
        "negative_et_rule": "floor_once_to_zero_then_fail_if_conservation_exceeds_tolerance",
        "google_drive_used": False,
        "earth_engine_asset_created": False,
        "model_source": "fitted_in_current_run",
        "raw_support_mosaic": str(raw_mosaic_path),
        "raster": str(raster_path),
        "tile_manifest": str(manifest_path),
        **reconciliation,
    }
    metadata_path = period_directory / (
        "production_metadata_" + RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION + ".json"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "raster": raster_path,
        "manifest": manifest_path,
        "metadata": metadata_path,
        "completed_tiles": completed,
        "support_tiles": support_tiles,
        "raw_support_mosaic": raw_mosaic_path,
    }
