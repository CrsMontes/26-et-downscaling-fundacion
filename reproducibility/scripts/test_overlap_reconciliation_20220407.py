"""Test a single global overlap-based MODIS reconciliation on 2022-04-07.

This script is diagnostic only. It leaves the frozen production path unchanged.
It uses the existing 20 m production raster only for Kc_raw and QC masks,
downloads native-grid MODIS ET for the same period, builds actual area overlaps
between the 20 m UTM cells and MODIS cells, and solves one constrained least-
squares projection:

    minimize ||ET - ET0||^2
    subject to C @ ET = MODIS_ET

where C contains area fractions from the real grid overlap. No nearest-neighbour
correction and no iterative reconciliation are used.

Only MODIS cells fully represented by the existing basin-masked raster are used
in this diagnostic. This is intentional: the frozen raster does not retain the
20 m predictor field outside the basin, so boundary MODIS cells cannot yet be
reconciled on their complete footprint. A final production implementation must
retain a one-MODIS-cell buffer until after the global reconciliation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import warnings

import ee
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, rowcol
from rasterio.warp import transform as warp_transform, transform_bounds
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import shapely

from et_downscaling.config import ANALYSIS_CRS
from et_downscaling.local_reconciliation import aggregate_average_to_grid
from et_downscaling.production import build_modis_period_context
from et_downscaling.ridge25_local_production import (
    _download_ee_bytes,
    _read_downloaded_array,
)


DEFAULT_DATE = "2022-04-07"
DEFAULT_STATIONS = ("ST01", "ST02", "ST03", "ST05")
USABLE_SUPPORT_FRACTION = 0.90
OVERLAP_AREA_EPSILON_M2 = 1e-6

MODIS_LOCAL_CRS = CRS.from_string(
    "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext +no_defs"
)


def build_native_modis_grid(
    context: dict[str, object],
    processing_bounds: tuple[float, float, float, float],
) -> tuple[str, CRS, Affine, tuple[int, int]]:
    """Build a native MODIS subset with the correct spherical CRS locally."""
    projection = context["modis_projection"]
    projection_info = projection.getInfo()

    ee_crs = str(projection_info["crs"])
    native_transform = Affine(
        *[float(value) for value in projection_info["transform"]]
    )

    xmin, ymin, xmax, ymax = processing_bounds
    left, bottom, right, top = transform_bounds(
        ANALYSIS_CRS,
        MODIS_LOCAL_CRS,
        xmin,
        ymin,
        xmax,
        ymax,
        densify_pts=21,
    )

    inverse = ~native_transform
    corners = [
        inverse * (x, y)
        for x in (left, right)
        for y in (bottom, top)
    ]
    columns = [item[0] for item in corners]
    rows = [item[1] for item in corners]

    col0 = math.floor(min(columns)) - 1
    col1 = math.ceil(max(columns)) + 1
    row0 = math.floor(min(rows)) - 1
    row1 = math.ceil(max(rows)) + 1

    destination_transform = (
        native_transform * Affine.translation(col0, row0)
    )

    return (
        ee_crs,
        MODIS_LOCAL_CRS,
        destination_transform,
        (row1 - row0, col1 - col0),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test one global overlap-based Ridge25/MODIS reconciliation."
        )
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--raster", default=None)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=list(DEFAULT_STATIONS),
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


def band_index(dataset: rasterio.io.DatasetReader, name: str) -> int:
    descriptions = list(dataset.descriptions)
    if name not in descriptions:
        raise RuntimeError(
            f"Band {name!r} not found. Available: {descriptions}"
        )
    return descriptions.index(name) + 1


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


def native_pixel_polygon(
    transform,
    row: int,
    col: int,
    source_crs,
    destination_crs,
):
    """Return a densified native MODIS pixel transformed to the 20 m CRS."""

    # Eight segments per edge are ample for a ~463 m MODIS cell and avoid
    # treating the cross-CRS boundary as a four-corner straight quadrilateral.
    segments = 8
    fractions = np.linspace(0.0, 1.0, segments + 1)

    x00, y00 = transform * (col, row)
    x10, y10 = transform * (col + 1, row)
    x11, y11 = transform * (col + 1, row + 1)
    x01, y01 = transform * (col, row + 1)

    points: list[tuple[float, float]] = []
    edges = [
        ((x00, y00), (x10, y10)),
        ((x10, y10), (x11, y11)),
        ((x11, y11), (x01, y01)),
        ((x01, y01), (x00, y00)),
    ]
    for edge_index, (start, end) in enumerate(edges):
        use_fractions = fractions if edge_index == 0 else fractions[1:]
        for fraction in use_fractions:
            points.append(
                (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
            )

    xs, ys = warp_transform(
        source_crs,
        destination_crs,
        [item[0] for item in points],
        [item[1] for item in points],
    )
    return shapely.Polygon(np.column_stack([xs, ys]))


def candidate_center_mask(
    modis_shape: tuple[int, int],
    modis_transform,
    modis_crs,
    fine_crs,
    fine_transform,
    fine_height: int,
    fine_width: int,
    domain: np.ndarray,
) -> np.ndarray:
    rows, columns = np.indices(modis_shape)
    column_centers = columns.astype(np.float64) + 0.5
    row_centers = rows.astype(np.float64) + 0.5

    center_x = (
        modis_transform.c
        + modis_transform.a * column_centers
        + modis_transform.b * row_centers
    )
    center_y = (
        modis_transform.f
        + modis_transform.d * column_centers
        + modis_transform.e * row_centers
    )

    xs, ys = warp_transform(
        modis_crs,
        fine_crs,
        center_x.ravel().tolist(),
        center_y.ravel().tolist(),
    )
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    rr, cc = rowcol(fine_transform, xs, ys)
    rr = np.asarray(rr, dtype=np.int64)
    cc = np.asarray(cc, dtype=np.int64)

    inside = (
        (rr >= 0)
        & (rr < fine_height)
        & (cc >= 0)
        & (cc < fine_width)
    )
    result = np.zeros(rr.size, dtype=bool)
    valid_positions = np.flatnonzero(inside)
    result[valid_positions] = domain[
        rr[valid_positions],
        cc[valid_positions],
    ]
    return result.reshape(modis_shape)


def build_overlap_edges(
    dataset: rasterio.io.DatasetReader,
    domain: np.ndarray,
    modis_et: np.ndarray,
    modis_transform,
    modis_crs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build overlap edges for MODIS cells fully represented in the raster."""

    center_candidate = candidate_center_mask(
        modis_shape=modis_et.shape,
        modis_transform=modis_transform,
        modis_crs=modis_crs,
        fine_crs=dataset.crs,
        fine_transform=dataset.transform,
        fine_height=dataset.height,
        fine_width=dataset.width,
        domain=domain,
    )

    candidate_rows, candidate_cols = np.where(
        center_candidate & np.isfinite(modis_et)
    )

    coarse_ids: list[np.ndarray] = []
    fine_ids: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    represented = np.zeros(modis_et.size, dtype=bool)

    fine_flat_domain = domain.ravel()
    fine_transform = dataset.transform
    pixel_width = float(fine_transform.a)
    pixel_height = float(-fine_transform.e)

    if not (
        fine_transform.b == 0
        and fine_transform.d == 0
        and pixel_width > 0
        and pixel_height > 0
    ):
        raise RuntimeError(
            "Diagnostic expects a north-up regular 20 m raster grid."
        )

    print(
        f"Candidate MODIS cells with centers in raster domain: "
        f"{candidate_rows.size:,}"
    )

    for position, (coarse_row, coarse_col) in enumerate(
        zip(candidate_rows, candidate_cols, strict=True),
        start=1,
    ):
        polygon = native_pixel_polygon(
            transform=modis_transform,
            row=int(coarse_row),
            col=int(coarse_col),
            source_crs=modis_crs,
            destination_crs=dataset.crs,
        )
        if polygon.is_empty or not polygon.is_valid:
            continue

        minx, miny, maxx, maxy = polygon.bounds
        raw_window = rasterio.windows.from_bounds(
            minx,
            miny,
            maxx,
            maxy,
            transform=fine_transform,
        )
        row0 = max(0, int(math.floor(raw_window.row_off)) - 1)
        col0 = max(0, int(math.floor(raw_window.col_off)) - 1)
        row1 = min(
            dataset.height,
            int(math.ceil(raw_window.row_off + raw_window.height)) + 1,
        )
        col1 = min(
            dataset.width,
            int(math.ceil(raw_window.col_off + raw_window.width)) + 1,
        )
        if row1 <= row0 or col1 <= col0:
            continue

        rows = np.arange(row0, row1, dtype=np.int64)
        cols = np.arange(col0, col1, dtype=np.int64)
        column_grid, row_grid = np.meshgrid(cols, rows)

        left = fine_transform.c + column_grid.ravel() * pixel_width
        right = left + pixel_width
        top = fine_transform.f - row_grid.ravel() * pixel_height
        bottom = top - pixel_height

        pixels = shapely.box(left, bottom, right, top)
        overlap_area = shapely.area(
            shapely.intersection(pixels, polygon)
        )
        positive = overlap_area > OVERLAP_AREA_EPSILON_M2
        if not positive.any():
            continue

        global_fine = (
            row_grid.ravel()[positive] * dataset.width
            + column_grid.ravel()[positive]
        ).astype(np.int64)
        overlap_area = np.asarray(overlap_area[positive], dtype=np.float64)

        # The old basin-masked raster cannot represent complete MODIS parents
        # that touch the basin edge. Keep only parents whose every intersecting
        # 20 m cell exists in the raster domain.
        if not fine_flat_domain[global_fine].all():
            continue

        coarse_flat = int(
            coarse_row * modis_et.shape[1] + coarse_col
        )
        represented[coarse_flat] = True
        coarse_ids.append(
            np.full(global_fine.size, coarse_flat, dtype=np.int32)
        )
        fine_ids.append(global_fine.astype(np.int32, copy=False))
        areas.append(overlap_area.astype(np.float32, copy=False))

        if position % 1000 == 0:
            print(
                f"  overlap geometry: {position:,}/{candidate_rows.size:,} "
                f"candidates; fully represented={represented.sum():,}"
            )

    if not coarse_ids:
        raise RuntimeError("No fully represented MODIS parents were found.")

    return (
        np.concatenate(coarse_ids),
        np.concatenate(fine_ids),
        np.concatenate(areas).astype(np.float64),
        represented.reshape(modis_et.shape),
    )


def solve_overlap_reconciliation(
    kc_raw: np.ndarray,
    usable: np.ndarray,
    modis_et: np.ndarray,
    coarse_edge: np.ndarray,
    fine_edge: np.ndarray,
    overlap_area: np.ndarray,
    represented_coarse: np.ndarray,
) -> dict[str, object]:
    coarse_count_total = modis_et.size

    coarse_area = np.bincount(
        coarse_edge,
        weights=overlap_area,
        minlength=coarse_count_total,
    )
    usable_area = np.bincount(
        coarse_edge,
        weights=overlap_area * usable.ravel()[fine_edge].astype(float),
        minlength=coarse_count_total,
    )
    usable_fraction = np.full(coarse_count_total, np.nan, dtype=float)
    positive_area = coarse_area > 0
    usable_fraction[positive_area] = (
        usable_area[positive_area] / coarse_area[positive_area]
    )

    valid_kc_edge = usable.ravel()[fine_edge] & np.isfinite(
        kc_raw.ravel()[fine_edge]
    )
    kc_weighted_sum = np.bincount(
        coarse_edge[valid_kc_edge],
        weights=(
            overlap_area[valid_kc_edge]
            * kc_raw.ravel()[fine_edge[valid_kc_edge]]
        ),
        minlength=coarse_count_total,
    )
    kc_valid_mean = np.full(coarse_count_total, np.nan, dtype=float)
    positive_usable_area = usable_area > 0
    kc_valid_mean[positive_usable_area] = (
        kc_weighted_sum[positive_usable_area]
        / usable_area[positive_usable_area]
    )

    represented_flat = represented_coarse.ravel()
    modis_flat = modis_et.ravel()
    eligible_flat = (
        represented_flat
        & np.isfinite(modis_flat)
        & np.isfinite(kc_valid_mean)
        & (usable_fraction >= USABLE_SUPPORT_FRACTION)
    )
    eligible_coarse = np.flatnonzero(eligible_flat).astype(np.int32)
    if eligible_coarse.size == 0:
        raise RuntimeError("No MODIS parents pass the 90% support rule.")

    coarse_to_row = np.full(coarse_count_total, -1, dtype=np.int32)
    coarse_to_row[eligible_coarse] = np.arange(
        eligible_coarse.size,
        dtype=np.int32,
    )

    edge_eligible = eligible_flat[coarse_edge]
    coarse_edge_eligible = coarse_edge[edge_eligible]
    fine_edge_eligible = fine_edge[edge_eligible]
    area_edge_eligible = overlap_area[edge_eligible]
    row_edge = coarse_to_row[coarse_edge_eligible]

    active_fine = np.unique(fine_edge_eligible).astype(np.int32)
    fine_to_column = np.full(kc_raw.size, -1, dtype=np.int32)
    fine_to_column[active_fine] = np.arange(
        active_fine.size,
        dtype=np.int32,
    )
    column_edge = fine_to_column[fine_edge_eligible]

    row_area = coarse_area[eligible_coarse]
    coefficients = area_edge_eligible / row_area[row_edge]

    c_matrix = coo_matrix(
        (
            coefficients,
            (row_edge, column_edge),
        ),
        shape=(eligible_coarse.size, active_fine.size),
    ).tocsr()
    c_matrix.sum_duplicates()

    # Neutral internal fill: for non-usable cells, use the area-weighted mean
    # of the valid Kc means of every eligible MODIS cell it actually overlaps.
    fine_overlap_sum = np.bincount(
        column_edge,
        weights=area_edge_eligible,
        minlength=active_fine.size,
    )
    fine_kc_mean_sum = np.bincount(
        column_edge,
        weights=(
            area_edge_eligible
            * kc_valid_mean[coarse_edge_eligible]
        ),
        minlength=active_fine.size,
    )
    neutral_kc = fine_kc_mean_sum / fine_overlap_sum

    active_usable = usable.ravel()[active_fine]
    active_kc_raw = kc_raw.ravel()[active_fine]
    kc_filled = np.where(
        active_usable & np.isfinite(active_kc_raw),
        active_kc_raw,
        neutral_kc,
    )

    kc_filled_mean = np.asarray(c_matrix @ kc_filled).ravel()
    if np.any(~np.isfinite(kc_filled_mean)) or np.any(kc_filled_mean <= 0):
        raise RuntimeError("Non-positive Kc mean reached an eligible parent.")

    target = modis_flat[eligible_coarse]
    parent_scale = target / kc_filled_mean

    # Initial ET field: each 20 m cell keeps one value. If it straddles a
    # MODIS boundary, its initial scale is the overlap-area-weighted mean of
    # the scales of the parents it actually intersects.
    fine_scale_sum = np.bincount(
        column_edge,
        weights=area_edge_eligible * parent_scale[row_edge],
        minlength=active_fine.size,
    )
    fine_scale = fine_scale_sum / fine_overlap_sum
    et_initial = kc_filled * fine_scale

    initial_error = np.asarray(c_matrix @ et_initial).ravel() - target

    gram = (c_matrix @ c_matrix.T).tocsc()
    rhs = -initial_error

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        lagrange = spsolve(gram, rhs)

    if np.any(~np.isfinite(lagrange)):
        raise RuntimeError("Overlap constraint system is singular or non-finite.")

    et_final = et_initial + np.asarray(c_matrix.T @ lagrange).ravel()
    final_error = np.asarray(c_matrix @ et_final).ravel() - target

    # A fine cell is publication-eligible only if it is originally usable and
    # every represented MODIS parent it overlaps is itself eligible. This
    # avoids assigning a crossing 20 m cell to one parent by nearest-neighbour.
    all_edge_coarse = coarse_edge
    all_edge_fine = fine_edge
    all_fine_active_positions = fine_to_column[all_edge_fine]
    touches_active = all_fine_active_positions >= 0
    publication_bad = np.zeros(active_fine.size, dtype=bool)
    if touches_active.any():
        bad_edge = ~eligible_flat[all_edge_coarse[touches_active]]
        if bad_edge.any():
            publication_bad[
                all_fine_active_positions[touches_active][bad_edge]
            ] = True
    publishable = active_usable & ~publication_bad

    return {
        "eligible_coarse": eligible_coarse,
        "eligible_flat": eligible_flat,
        "usable_fraction": usable_fraction,
        "kc_valid_mean": kc_valid_mean,
        "active_fine": active_fine,
        "publishable": publishable,
        "et_initial": et_initial,
        "et_final": et_final,
        "initial_error": initial_error,
        "final_error": final_error,
        "target": target,
        "c_matrix": c_matrix,
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def station_rows(
    root: Path,
    dataset: rasterio.io.DatasetReader,
    stations_requested: list[str],
    active_fine: np.ndarray,
    publishable: np.ndarray,
    et_initial: np.ndarray,
    et_final: np.ndarray,
) -> pd.DataFrame:
    stations = load_stations(root)
    fine_to_active = np.full(dataset.width * dataset.height, -1, dtype=np.int32)
    fine_to_active[active_fine] = np.arange(active_fine.size, dtype=np.int32)

    kc_band = dataset.read(band_index(dataset, "Kc_raw"), masked=True)
    old_et_band = dataset.read(band_index(dataset, "ET_mm_period"), masked=True)

    rows_output: list[dict[str, object]] = []
    for station_id in stations_requested:
        meta = stations[station_id]
        xs, ys = warp_transform(
            "EPSG:4326",
            dataset.crs,
            [meta["longitude"]],
            [meta["latitude"]],
        )
        rr, cc = rowcol(dataset.transform, xs[0], ys[0])
        fine_index = int(rr * dataset.width + cc)
        active_index = int(fine_to_active[fine_index])

        candidate_initial = np.nan
        candidate_final = np.nan
        candidate_published = False
        if active_index >= 0:
            candidate_initial = float(et_initial[active_index])
            candidate_final = float(et_final[active_index])
            candidate_published = bool(publishable[active_index])

        old_et = (
            np.nan
            if np.ma.is_masked(old_et_band[rr, cc])
            else float(old_et_band[rr, cc])
        )
        kc = (
            np.nan
            if np.ma.is_masked(kc_band[rr, cc])
            else float(kc_band[rr, cc])
        )

        rows_output.append(
            {
                "station_id": station_id,
                "station": meta["station"],
                "Kc_raw": kc,
                "ET_old_iterative": old_et,
                "ET_overlap_initial": candidate_initial,
                "ET_overlap_final": candidate_final,
                "candidate_publishable": candidate_published,
                "final_over_initial": (
                    candidate_final / candidate_initial
                    if np.isfinite(candidate_initial)
                    and abs(candidate_initial) > 1e-12
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows_output)


def main() -> None:
    args = parse_arguments()
    root = project_root()
    raster_path = find_raster(root, args.date, args.raster)

    print("Initializing Earth Engine...")
    ee.Initialize(project=args.project)

    with rasterio.open(raster_path) as dataset:
        print("Raster:", raster_path)
        print("Shape:", dataset.height, "x", dataset.width)
        print("CRS:", dataset.crs)

        if dataset.crs is None:
            raise RuntimeError("Raster has no CRS.")

        stack_band = dataset.read(
            band_index(dataset, "stack_valid"),
            masked=True,
        )
        domain = ~np.ma.getmaskarray(stack_band)

        kc_raw = dataset.read(
            band_index(dataset, "Kc_raw"),
            masked=True,
        ).filled(np.nan).astype(np.float64)
        usable = (
            dataset.read(
                band_index(dataset, "usable"),
                masked=True,
            ).filled(0.0)
            >= 0.5
        ) & domain

        bounds = dataset.bounds
        processing_geometry = ee.Geometry.Rectangle(
            [bounds.left, bounds.bottom, bounds.right, bounds.top],
            proj=str(dataset.crs),
            geodesic=False,
        )
        context = build_modis_period_context(
            args.date,
            processing_geometry,
        )
        context_for_grid = {
            "modis_projection": context["modis_projection"]
        }
        (
            modis_ee_crs,
            modis_local_crs,
            modis_transform,
            modis_shape,
        ) = build_native_modis_grid(
            context_for_grid,
            (bounds.left, bounds.bottom, bounds.right, bounds.top),
        )
        modis_height, modis_width = modis_shape

        print("Earth Engine MODIS CRS:", modis_ee_crs)
        print("Local MODIS CRS:", modis_local_crs.to_string())
        print("Corrected MODIS transform:", modis_transform)
        print("Corrected MODIS shape:", modis_shape)

        parameters = {
            "bands": ["ET_mm_period"],
            "crs": modis_ee_crs,
            "crs_transform": [
                modis_transform.a,
                modis_transform.b,
                modis_transform.c,
                modis_transform.d,
                modis_transform.e,
                modis_transform.f,
            ],
            "dimensions": [modis_width, modis_height],
            "format": "GEO_TIFF",
        }
        payload = _download_ee_bytes(
            ee.Image(context["modis_et"]),
            parameters,
            args.timeout_seconds,
        )
        modis_bands, downloaded_transform, downloaded_crs = (
            _read_downloaded_array(payload)
        )
        if not downloaded_transform.almost_equals(modis_transform):
            raise RuntimeError("Downloaded MODIS transform changed.")
        if downloaded_crs is None:
            raise RuntimeError("Downloaded MODIS raster has no CRS.")
        modis_et = modis_bands[0].filled(np.nan).astype(np.float64)

        print("Building exact overlap operator...")
        coarse_edge, fine_edge, overlap_area, represented = build_overlap_edges(
            dataset=dataset,
            domain=domain,
            modis_et=modis_et,
            modis_transform=modis_transform,
            modis_crs=modis_local_crs,
        )
        print(f"Overlap edges: {coarse_edge.size:,}")
        print(f"Fully represented MODIS parents: {represented.sum():,}")

        print("Solving one constrained global projection...")
        result = solve_overlap_reconciliation(
            kc_raw=kc_raw,
            usable=usable,
            modis_et=modis_et,
            coarse_edge=coarse_edge,
            fine_edge=fine_edge,
            overlap_area=overlap_area,
            represented_coarse=represented,
        )

        et_initial = result["et_initial"]
        et_final = result["et_final"]
        final_error = result["final_error"]
        initial_error = result["initial_error"]
        active_fine = result["active_fine"]
        publishable = result["publishable"]
        eligible_coarse = result["eligible_coarse"]

        delta = et_final - et_initial
        negative_count = int((et_final < 0).sum())
        negative_publishable_count = int(
            ((et_final < 0) & publishable).sum()
        )
        minimum_publishable_et = (
            float(np.min(et_final[publishable]))
            if publishable.any()
            else np.nan
        )

        print()
        print("=" * 100)
        print("OVERLAP RECONCILIATION DIAGNOSTIC")
        print("=" * 100)
        print(f"Eligible MODIS parents: {eligible_coarse.size:,}")
        print(f"Active 20 m cells: {active_fine.size:,}")
        print(f"Publishable 20 m cells: {publishable.sum():,}")
        print(
            "Initial max |MODIS error|: "
            f"{np.max(np.abs(initial_error)):.6f} mm"
        )
        print(
            "Final max |MODIS error|:   "
            f"{np.max(np.abs(final_error)):.12g} mm"
        )
        print(f"Negative final ET cells: {negative_count:,}")
        print(
            "Negative publishable ET cells: "
            f"{negative_publishable_count:,}"
        )
        print(f"Minimum final ET: {np.min(et_final):.6f} mm")
        print(
            "Minimum publishable ET: "
            f"{minimum_publishable_et:.6f} mm"
        )
        print(f"MAE adjustment vs initial: {np.mean(np.abs(delta)):.6f} mm")
        print(
            "RMSE adjustment vs initial: "
            f"{np.sqrt(np.mean(delta ** 2)):.6f} mm"
        )
        print(
            "Pearson(final, initial): "
            f"{safe_correlation(et_final, et_initial):.6f}"
        )

        # Independent GDAL/rasterio aggregation QA after placing the solution
        # back on the regular 20 m grid.
        candidate_full = np.full(dataset.width * dataset.height, np.nan)
        candidate_full[active_fine] = et_final
        candidate_full = candidate_full.reshape(dataset.height, dataset.width)
        candidate_prepared = np.where(
            np.isfinite(candidate_full),
            candidate_full,
            -9999.0,
        )
        gdal_aggregate = aggregate_average_to_grid(
            source_array=candidate_prepared,
            source_transform=dataset.transform,
            source_crs=dataset.crs,
            destination_shape=modis_shape,
            destination_transform=modis_transform,
            destination_crs=modis_local_crs,
            source_nodata=-9999.0,
        )
        eligible_mask = np.zeros(modis_et.size, dtype=bool)
        eligible_mask[eligible_coarse] = True
        eligible_mask = eligible_mask.reshape(modis_shape)
        gdal_error = gdal_aggregate - modis_et
        gdal_valid = eligible_mask & np.isfinite(gdal_error)
        gdal_abs_error = np.abs(gdal_error[gdal_valid])
        gdal_error_median = float(np.median(gdal_abs_error))
        gdal_error_p95 = float(np.quantile(gdal_abs_error, 0.95))
        gdal_error_p99 = float(np.quantile(gdal_abs_error, 0.99))
        gdal_error_max = float(np.max(gdal_abs_error))

        valid_flat = np.flatnonzero(gdal_valid.ravel())
        worst_flat = int(valid_flat[int(np.argmax(gdal_abs_error))])
        worst_row, worst_col = np.unravel_index(
            worst_flat,
            modis_shape,
        )

        print(
            "Independent GDAL max |MODIS error|: "
            f"{gdal_error_max:.6f} mm"
        )
        print(
            "GDAL |error| median / P95 / P99: "
            f"{gdal_error_median:.6f} / "
            f"{gdal_error_p95:.6f} / "
            f"{gdal_error_p99:.6f} mm"
        )
        print(
            "Worst GDAL parent: "
            f"row={worst_row}, col={worst_col}, "
            f"MODIS={modis_et[worst_row, worst_col]:.6f}, "
            f"GDAL={gdal_aggregate[worst_row, worst_col]:.6f}, "
            f"error={gdal_error[worst_row, worst_col]:+.6f} mm"
        )

        stations = station_rows(
            root=root,
            dataset=dataset,
            stations_requested=args.stations,
            active_fine=active_fine,
            publishable=publishable,
            et_initial=et_initial,
            et_final=et_final,
        )
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 220)
        print()
        print("STATIONS")
        print(stations.to_string(index=False))

        diagnostics_directory = (
            root.parent
            / "ET_fundacion_workspace"
            / "current"
            / "diagnostics"
            / "overlap_reconciliation"
        )
        diagnostics_directory.mkdir(parents=True, exist_ok=True)

        summary = {
            "date": args.date,
            "method": "global_overlap_constrained_projection_diagnostic",
            "source_raster": str(raster_path),
            "usable_support_fraction": USABLE_SUPPORT_FRACTION,
            "represented_parent_rule": (
                "every intersecting 20 m cell must exist in the basin-masked raster"
            ),
            "eligible_modis_parents": int(eligible_coarse.size),
            "active_fine_cells": int(active_fine.size),
            "publishable_fine_cells": int(publishable.sum()),
            "initial_max_abs_modis_error_mm": float(
                np.max(np.abs(initial_error))
            ),
            "final_max_abs_modis_error_mm": float(
                np.max(np.abs(final_error))
            ),
            "independent_gdal_max_abs_modis_error_mm": gdal_error_max,
            "independent_gdal_median_abs_modis_error_mm": gdal_error_median,
            "independent_gdal_p95_abs_modis_error_mm": gdal_error_p95,
            "independent_gdal_p99_abs_modis_error_mm": gdal_error_p99,
            "negative_final_et_cells": negative_count,
            "negative_publishable_et_cells": negative_publishable_count,
            "minimum_final_et_mm": float(np.min(et_final)),
            "minimum_publishable_et_mm": minimum_publishable_et,
            "mae_adjustment_mm": float(np.mean(np.abs(delta))),
            "rmse_adjustment_mm": float(np.sqrt(np.mean(delta ** 2))),
            "pearson_final_vs_initial": safe_correlation(et_final, et_initial),
        }
        summary_path = diagnostics_directory / f"summary_{args.date}.json"
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        stations_path = diagnostics_directory / f"stations_{args.date}.csv"
        stations.to_csv(stations_path, index=False)

        candidate_path = diagnostics_directory / f"ET_overlap_candidate_{args.date}_20m.tif"
        profile = dataset.profile.copy()
        profile.update(count=1, dtype="float32", nodata=-9999.0)
        with rasterio.open(candidate_path, "w", **profile) as destination:
            destination.set_band_description(1, "ET_overlap_candidate_mm_period")
            published_full = np.full(dataset.width * dataset.height, np.nan)
            published_full[active_fine[publishable]] = et_final[publishable]
            published_full = published_full.reshape(dataset.height, dataset.width)
            destination.write(
                np.where(
                    np.isfinite(published_full),
                    published_full,
                    -9999.0,
                ).astype(np.float32),
                1,
            )

        print()
        print("Saved:", summary_path)
        print("Saved:", stations_path)
        print("Saved:", candidate_path)


if __name__ == "__main__":
    main()
