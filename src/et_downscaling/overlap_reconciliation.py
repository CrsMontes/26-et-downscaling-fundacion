"""Exact-overlap reconciliation for non-nested MODIS and 20 m grids.

This module contains the parsimonious conservative correction selected after
reconciling the 2022-04-07 diagnostic. It deliberately does not use raster
``average`` followed by coarse-to-fine ``nearest`` corrections. Instead it
builds the real area-overlap operator between the regular 20 m UTM grid and
the native MODIS sinusoidal grid and solves one global least-squares
projection subject to exact coarse-support conservation.

The Ridge-25 prediction supplies the fine spatial pattern. The role of this
module is only to reconcile that pattern to MODIS support while changing it as
little as possible under the chosen Euclidean criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine, rowcol
from rasterio.warp import transform as warp_transform, transform_bounds
import rasterio.windows
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import shapely

from .config import ANALYSIS_CRS
from .local_reconciliation import (
    RIDGE25_RECONCILIATION_TOLERANCE_MM,
    RIDGE25_USABLE_SUPPORT_FRACTION,
)


MODIS_SINUSOIDAL_LOCAL_CRS = CRS.from_string(
    "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext +no_defs"
)

OVERLAP_AREA_EPSILON_M2 = 1e-6


@dataclass(frozen=True)
class NativeModisGrid:
    """Native MODIS subset definition for Earth Engine and local geometry."""

    earth_engine_crs: str
    local_crs: CRS
    transform: Affine
    shape: tuple[int, int]


@dataclass(frozen=True)
class OverlapEdges:
    """Sparse area-intersection representation between coarse and fine cells."""

    coarse_index: np.ndarray
    fine_index: np.ndarray
    overlap_area_m2: np.ndarray
    represented_coarse: np.ndarray


@dataclass(frozen=True)
class OverlapReconciliationResult:
    """Global exact-overlap reconciliation result."""

    eligible_coarse: np.ndarray
    eligible_coarse_mask: np.ndarray
    usable_fraction: np.ndarray
    kc_valid_mean: np.ndarray
    active_fine: np.ndarray
    publishable_active: np.ndarray
    et_initial: np.ndarray
    et_final: np.ndarray
    et_final_nonnegative: np.ndarray
    initial_error: np.ndarray
    final_error: np.ndarray
    final_error_after_nonnegative: np.ndarray
    target: np.ndarray
    constraint_matrix: csr_matrix
    max_abs_final_error_mm: float
    max_abs_error_after_nonnegative_mm: float
    negative_active_cells: int
    negative_publishable_cells: int


def build_native_modis_grid(
    projection_info: dict[str, object],
    processing_bounds: tuple[float, float, float, float],
    analysis_crs: str = ANALYSIS_CRS,
    margin_pixels: int = 1,
) -> NativeModisGrid:
    """Build a native MODIS subset while preserving its original grid phase.

    ``projection_info`` must be the result of Earth Engine
    ``image.projection().getInfo()`` (or an equivalent mapping containing
    ``crs`` and ``transform``). Earth Engine receives the exact native CRS
    identifier, while local Rasterio/PROJ geometry uses the explicit MODIS
    spherical sinusoidal definition. This distinction avoids the grid shift
    found when the WKT representation was used locally.
    """

    if margin_pixels < 0:
        raise ValueError("MODIS grid margin must be non-negative.")

    earth_engine_crs = str(projection_info["crs"])
    native_transform = Affine(
        *[float(value) for value in projection_info["transform"]]
    )

    xmin, ymin, xmax, ymax = processing_bounds
    left, bottom, right, top = transform_bounds(
        analysis_crs,
        MODIS_SINUSOIDAL_LOCAL_CRS,
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

    col0 = math.floor(min(columns)) - margin_pixels
    col1 = math.ceil(max(columns)) + margin_pixels
    row0 = math.floor(min(rows)) - margin_pixels
    row1 = math.ceil(max(rows)) + margin_pixels

    transform = native_transform * Affine.translation(col0, row0)

    return NativeModisGrid(
        earth_engine_crs=earth_engine_crs,
        local_crs=MODIS_SINUSOIDAL_LOCAL_CRS,
        transform=transform,
        shape=(row1 - row0, col1 - col0),
    )


def _native_pixel_polygon(
    transform: Affine,
    row: int,
    col: int,
    source_crs,
    destination_crs,
    segments_per_edge: int = 8,
):
    """Return one densified native MODIS pixel in the fine-grid CRS."""

    if segments_per_edge < 1:
        raise ValueError("segments_per_edge must be >= 1.")

    fractions = np.linspace(0.0, 1.0, segments_per_edge + 1)

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


def _candidate_center_mask(
    modis_shape: tuple[int, int],
    modis_transform: Affine,
    modis_crs,
    fine_crs,
    fine_transform: Affine,
    fine_shape: tuple[int, int],
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

    rr, cc = rowcol(
        fine_transform,
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
    )
    rr = np.asarray(rr, dtype=np.int64)
    cc = np.asarray(cc, dtype=np.int64)

    fine_height, fine_width = fine_shape
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
    domain: np.ndarray,
    fine_transform: Affine,
    fine_crs,
    modis_et: np.ndarray,
    modis_transform: Affine,
    modis_crs=MODIS_SINUSOIDAL_LOCAL_CRS,
    progress_every: int = 0,
) -> OverlapEdges:
    """Build exact overlap edges for fully represented MODIS parents.

    A coarse parent is marked represented only when every fine cell that
    intersects its transformed footprint belongs to ``domain``. Final basin
    production should therefore retain an external buffer through this step
    and mask to the basin only after the global reconciliation.
    """

    domain = np.asarray(domain, dtype=bool)
    modis_et = np.asarray(modis_et, dtype=float)

    if domain.ndim != 2:
        raise ValueError("Fine-grid domain must be two-dimensional.")
    if modis_et.ndim != 2:
        raise ValueError("MODIS ET must be two-dimensional.")

    fine_height, fine_width = domain.shape
    pixel_width = float(fine_transform.a)
    pixel_height = float(-fine_transform.e)

    if not (
        fine_transform.b == 0
        and fine_transform.d == 0
        and pixel_width > 0
        and pixel_height > 0
    ):
        raise ValueError("Fine grid must be regular, north-up and axis-aligned.")

    center_candidate = _candidate_center_mask(
        modis_shape=modis_et.shape,
        modis_transform=modis_transform,
        modis_crs=modis_crs,
        fine_crs=fine_crs,
        fine_transform=fine_transform,
        fine_shape=domain.shape,
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

    for position, (coarse_row, coarse_col) in enumerate(
        zip(candidate_rows, candidate_cols, strict=True),
        start=1,
    ):
        polygon = _native_pixel_polygon(
            transform=modis_transform,
            row=int(coarse_row),
            col=int(coarse_col),
            source_crs=modis_crs,
            destination_crs=fine_crs,
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
            fine_height,
            int(math.ceil(raw_window.row_off + raw_window.height)) + 1,
        )
        col1 = min(
            fine_width,
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
        overlap_area = shapely.area(shapely.intersection(pixels, polygon))
        positive = overlap_area > OVERLAP_AREA_EPSILON_M2

        if not positive.any():
            continue

        global_fine = (
            row_grid.ravel()[positive] * fine_width
            + column_grid.ravel()[positive]
        ).astype(np.int64)
        overlap_area = np.asarray(overlap_area[positive], dtype=np.float64)

        if not fine_flat_domain[global_fine].all():
            continue

        coarse_flat = int(coarse_row * modis_et.shape[1] + coarse_col)
        represented[coarse_flat] = True
        coarse_ids.append(
            np.full(global_fine.size, coarse_flat, dtype=np.int32)
        )
        fine_ids.append(global_fine.astype(np.int32, copy=False))
        areas.append(overlap_area)

        if progress_every and position % progress_every == 0:
            print(
                f"  overlap geometry: {position:,}/{candidate_rows.size:,}; "
                f"fully represented={represented.sum():,}"
            )

    if not coarse_ids:
        raise RuntimeError("No fully represented MODIS parents were found.")

    return OverlapEdges(
        coarse_index=np.concatenate(coarse_ids),
        fine_index=np.concatenate(fine_ids),
        overlap_area_m2=np.concatenate(areas),
        represented_coarse=represented.reshape(modis_et.shape),
    )


def solve_overlap_reconciliation(
    kc_raw: np.ndarray,
    usable: np.ndarray,
    modis_et: np.ndarray,
    edges: OverlapEdges,
    usable_support_fraction: float = RIDGE25_USABLE_SUPPORT_FRACTION,
    tolerance_mm: float = RIDGE25_RECONCILIATION_TOLERANCE_MM,
) -> OverlapReconciliationResult:
    """Reconcile Ridge-25 pattern to MODIS using one global projection.

    Non-usable fine cells are retained only as neutral internal support using
    an overlap-weighted mean Kc. Publication is restricted to originally
    usable cells whose every represented MODIS parent is eligible.

    Small negative ET values can arise from the unconstrained Euclidean
    projection. They are set to zero exactly once and the coarse conservation
    error is recomputed. The result is accepted only if that post-floor error
    remains within ``tolerance_mm``; otherwise the function fails rather than
    silently introducing another correction scheme.
    """

    kc_raw = np.asarray(kc_raw, dtype=float)
    usable = np.asarray(usable, dtype=bool)
    modis_et = np.asarray(modis_et, dtype=float)

    if kc_raw.shape != usable.shape:
        raise ValueError("Kc and usable-mask shapes differ.")
    if kc_raw.ndim != 2:
        raise ValueError("Kc must be two-dimensional.")
    if modis_et.ndim != 2:
        raise ValueError("MODIS ET must be two-dimensional.")
    if not (0 < usable_support_fraction <= 1):
        raise ValueError("Usable-support fraction must be in (0, 1].")
    if tolerance_mm <= 0:
        raise ValueError("Conservation tolerance must be positive.")

    coarse_edge = np.asarray(edges.coarse_index, dtype=np.int64)
    fine_edge = np.asarray(edges.fine_index, dtype=np.int64)
    overlap_area = np.asarray(edges.overlap_area_m2, dtype=float)
    represented_coarse = np.asarray(edges.represented_coarse, dtype=bool)

    if not (
        coarse_edge.size == fine_edge.size == overlap_area.size
        and coarse_edge.size > 0
    ):
        raise ValueError("Overlap-edge arrays must have equal non-zero length.")
    if represented_coarse.shape != modis_et.shape:
        raise ValueError("Represented-coarse mask shape differs from MODIS ET.")

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
        & (usable_fraction >= usable_support_fraction)
    )
    eligible_coarse = np.flatnonzero(eligible_flat).astype(np.int32)

    if eligible_coarse.size == 0:
        raise RuntimeError("No MODIS parents pass the support rule.")

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
    fine_to_column[active_fine] = np.arange(active_fine.size, dtype=np.int32)
    column_edge = fine_to_column[fine_edge_eligible]

    row_area = coarse_area[eligible_coarse]
    coefficients = area_edge_eligible / row_area[row_edge]

    constraint_matrix = coo_matrix(
        (
            coefficients,
            (row_edge, column_edge),
        ),
        shape=(eligible_coarse.size, active_fine.size),
    ).tocsr()
    constraint_matrix.sum_duplicates()

    fine_overlap_sum = np.bincount(
        column_edge,
        weights=area_edge_eligible,
        minlength=active_fine.size,
    )
    fine_kc_mean_sum = np.bincount(
        column_edge,
        weights=(
            area_edge_eligible * kc_valid_mean[coarse_edge_eligible]
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

    kc_filled_mean = np.asarray(constraint_matrix @ kc_filled).ravel()
    if np.any(~np.isfinite(kc_filled_mean)) or np.any(kc_filled_mean <= 0):
        raise RuntimeError("Non-positive Kc mean reached an eligible parent.")

    target = modis_flat[eligible_coarse]
    parent_scale = target / kc_filled_mean

    fine_scale_sum = np.bincount(
        column_edge,
        weights=area_edge_eligible * parent_scale[row_edge],
        minlength=active_fine.size,
    )
    fine_scale = fine_scale_sum / fine_overlap_sum
    et_initial = kc_filled * fine_scale

    initial_error = np.asarray(constraint_matrix @ et_initial).ravel() - target

    gram = (constraint_matrix @ constraint_matrix.T).tocsc()
    rhs = -initial_error

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        lagrange = spsolve(gram, rhs)

    if np.any(~np.isfinite(lagrange)):
        raise RuntimeError("Overlap constraint system is singular or non-finite.")

    et_final = et_initial + np.asarray(constraint_matrix.T @ lagrange).ravel()
    final_error = np.asarray(constraint_matrix @ et_final).ravel() - target
    max_abs_final_error_mm = float(np.max(np.abs(final_error)))

    all_fine_active_positions = fine_to_column[fine_edge]
    touches_active = all_fine_active_positions >= 0
    publication_bad = np.zeros(active_fine.size, dtype=bool)
    if touches_active.any():
        bad_edge = ~eligible_flat[coarse_edge[touches_active]]
        if bad_edge.any():
            publication_bad[
                all_fine_active_positions[touches_active][bad_edge]
            ] = True
    publishable_active = active_usable & ~publication_bad

    negative_active = et_final < 0
    negative_publishable = negative_active & publishable_active
    et_final_nonnegative = np.maximum(et_final, 0.0)
    final_error_after_nonnegative = (
        np.asarray(constraint_matrix @ et_final_nonnegative).ravel() - target
    )
    max_abs_error_after_nonnegative_mm = float(
        np.max(np.abs(final_error_after_nonnegative))
    )

    if max_abs_error_after_nonnegative_mm > tolerance_mm:
        raise RuntimeError(
            "Setting negative ET to zero violates the MODIS conservation "
            f"tolerance: {max_abs_error_after_nonnegative_mm:.6f} mm > "
            f"{tolerance_mm:.6f} mm."
        )

    return OverlapReconciliationResult(
        eligible_coarse=eligible_coarse,
        eligible_coarse_mask=eligible_flat.reshape(modis_et.shape),
        usable_fraction=usable_fraction.reshape(modis_et.shape),
        kc_valid_mean=kc_valid_mean.reshape(modis_et.shape),
        active_fine=active_fine,
        publishable_active=publishable_active,
        et_initial=et_initial,
        et_final=et_final,
        et_final_nonnegative=et_final_nonnegative,
        initial_error=initial_error,
        final_error=final_error,
        final_error_after_nonnegative=final_error_after_nonnegative,
        target=target,
        constraint_matrix=constraint_matrix,
        max_abs_final_error_mm=max_abs_final_error_mm,
        max_abs_error_after_nonnegative_mm=max_abs_error_after_nonnegative_mm,
        negative_active_cells=int(negative_active.sum()),
        negative_publishable_cells=int(negative_publishable.sum()),
    )


def materialize_active_values(
    fine_shape: tuple[int, int],
    active_fine: np.ndarray,
    values: np.ndarray,
    selected_active: np.ndarray | None = None,
) -> np.ndarray:
    """Map active-vector values back to the regular fine grid."""

    active_fine = np.asarray(active_fine, dtype=np.int64)
    values = np.asarray(values, dtype=float)

    if active_fine.shape != values.shape:
        raise ValueError("Active indices and values must have matching shapes.")

    if selected_active is None:
        selected_active = np.ones(active_fine.size, dtype=bool)
    else:
        selected_active = np.asarray(selected_active, dtype=bool)
        if selected_active.shape != values.shape:
            raise ValueError("Active selection shape differs from values.")

    output = np.full(int(np.prod(fine_shape)), np.nan, dtype=np.float64)
    output[active_fine[selected_active]] = values[selected_active]
    return output.reshape(fine_shape)
