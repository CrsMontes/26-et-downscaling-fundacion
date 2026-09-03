"""Local tiled production for the final Ridge-25 ET workflow.

Earth Engine provides the 25 Ridge predictors and native MODIS ET data.
Ridge prediction, AOA scoring, physical quality control, support gating
and conservative MODIS reconciliation are performed locally.

Only originally usable fine cells are published. Non-usable cells may be
used only as neutral internal fill when their MODIS parent satisfies the
minimum usable-support criterion.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import ee
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import Affine, from_origin
from rasterio.windows import Window
from rasterio.warp import (
    transform as warp_transform,
    transform_bounds,
)

from .config import ANALYSIS_CRS
from .local_reconciliation import (
    RIDGE25_RECONCILIATION_MAX_ITERATIONS,
    RIDGE25_RECONCILIATION_TOLERANCE_MM,
    RIDGE25_USABLE_SUPPORT_FRACTION,
    aggregate_average_to_grid,
    coarse_to_fine_nearest,
    reconcile_local_ridge25,
    score_local_ridge25,
)
from .local_tiles import (
    CompletedTile,
    Tile,
    _analysis_geometry,
    _normalize_tile_size,
    build_initial_tiles,
)
from .production import (
    PREDICTION_SCALE_M,
    PROCESSING_BUFFER_M,
)
from .ridge25 import RIDGE25_MODEL_FEATURES
from .ridge25_production import build_ridge25_production_stack
from .workspace import get_workspace_paths


RIDGE25_LOCAL_PRODUCTION_VERSION = (
    "ridge25_local_aoa_support90_tol001_v1"
)

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

OUTPUT_NODATA = -9999.0

DIRECT_DOWNLOAD_MAX_ATTEMPTS = 6

TRANSIENT_HTTP_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


def _is_transient_http_status(
    code: int,
) -> bool:
    return int(code) in TRANSIENT_HTTP_STATUS_CODES


def _download_retry_delay_seconds(
    attempt: int,
) -> float:
    base = min(
        30.0,
        2.0 ** attempt,
    )

    return base + random.uniform(
        0.0,
        min(
            1.0,
            base * 0.25,
        ),
    )


def _processing_grid(
    tile: Tile,
) -> tuple[
    float,
    float,
    float,
    float,
    int,
    int,
    Affine,
]:
    scale = float(
        PREDICTION_SCALE_M
    )

    buffer_m = float(
        PROCESSING_BUFFER_M
    )

    xmin = tile.xmin - buffer_m
    xmax = tile.xmax + buffer_m
    ymin = tile.ymin - buffer_m
    ymax = tile.ymax + buffer_m

    width = int(
        round(
            (xmax - xmin)
            / scale
        )
    )

    height = int(
        round(
            (ymax - ymin)
            / scale
        )
    )

    transform = from_origin(
        xmin,
        ymax,
        scale,
        scale,
    )

    return (
        xmin,
        ymin,
        xmax,
        ymax,
        width,
        height,
        transform,
    )


def _build_modis_grid(
    context: dict[str, object],
    processing_bounds: tuple[
        float,
        float,
        float,
        float,
    ],
) -> tuple[
    str,
    Affine,
    tuple[int, int],
]:
    projection = context[
        "modis_projection"
    ]

    projection_info = (
        projection.getInfo()
    )

    modis_crs = (
        projection
        .wkt()
        .getInfo()
    )

    native_transform = Affine(
        *[
            float(value)
            for value
            in projection_info[
                "transform"
            ]
        ]
    )

    xmin, ymin, xmax, ymax = (
        processing_bounds
    )

    left, bottom, right, top = (
        transform_bounds(
            ANALYSIS_CRS,
            modis_crs,
            xmin,
            ymin,
            xmax,
            ymax,
            densify_pts=21,
        )
    )

    inverse = (
        ~native_transform
    )

    corners = [
        inverse * (x, y)
        for x in (
            left,
            right,
        )
        for y in (
            bottom,
            top,
        )
    ]

    columns = [
        item[0]
        for item in corners
    ]

    rows = [
        item[1]
        for item in corners
    ]

    col0 = (
        math.floor(
            min(columns)
        )
        - 1
    )

    col1 = (
        math.ceil(
            max(columns)
        )
        + 1
    )

    row0 = (
        math.floor(
            min(rows)
        )
        - 1
    )

    row1 = (
        math.ceil(
            max(rows)
        )
        + 1
    )

    width = (
        col1 - col0
    )

    height = (
        row1 - row0
    )

    destination_transform = (
        native_transform
        * Affine.translation(
            col0,
            row0,
        )
    )

    return (
        modis_crs,
        destination_transform,
        (
            height,
            width,
        ),
    )


def _modis_parent_owned_by_tile(
    tile: Tile,
    modis_shape: tuple[int, int],
    modis_transform: Affine,
    modis_crs,
) -> np.ndarray:
    """Return MODIS parents whose pixel centers belong to the core tile.

    Ownership is used only for non-duplicated accounting. Reconciliation
    still includes every eligible MODIS parent intersecting the core.
    """

    rows, columns = np.indices(
        modis_shape
    )

    column_centers = (
        columns.astype(
            np.float64
        )
        + 0.5
    )

    row_centers = (
        rows.astype(
            np.float64
        )
        + 0.5
    )

    center_x = (
        modis_transform.c
        + (
            modis_transform.a
            * column_centers
        )
        + (
            modis_transform.b
            * row_centers
        )
    )

    center_y = (
        modis_transform.f
        + (
            modis_transform.d
            * column_centers
        )
        + (
            modis_transform.e
            * row_centers
        )
    )

    transformed_x, transformed_y = (
        warp_transform(
            modis_crs,
            ANALYSIS_CRS,
            center_x.ravel().tolist(),
            center_y.ravel().tolist(),
        )
    )

    analysis_x = np.asarray(
        transformed_x,
        dtype=np.float64,
    ).reshape(
        modis_shape
    )

    analysis_y = np.asarray(
        transformed_y,
        dtype=np.float64,
    ).reshape(
        modis_shape
    )

    return (
        (
            analysis_x
            >= tile.xmin
        )
        & (
            analysis_x
            < tile.xmax
        )
        & (
            analysis_y
            >= tile.ymin
        )
        & (
            analysis_y
            < tile.ymax
        )
    )


def _download_ee_bytes(
    image: ee.Image,
    parameters: dict,
    timeout_seconds: int,
) -> bytes:
    for attempt in range(
        1,
        DIRECT_DOWNLOAD_MAX_ATTEMPTS
        + 1,
    ):
        try:
            url = image.getDownloadURL(
                parameters
            )

            with urllib.request.urlopen(
                url,
                timeout=timeout_seconds,
            ) as response:
                return response.read()

        except HTTPError as error:
            try:
                body = error.read().decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                body = ""

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

                time.sleep(
                    delay
                )
                continue

            raise RuntimeError(
                "Earth Engine direct download failed "
                f"(HTTP {error.code}): "
                f"{body or error.reason}"
            ) from error

        except (
            URLError,
            TimeoutError,
        ) as error:
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

                time.sleep(
                    delay
                )
                continue

            raise RuntimeError(
                "Earth Engine direct download failed "
                "after repeated network errors: "
                f"{error}"
            ) from error

    raise RuntimeError(
        "Earth Engine direct download attempts exhausted."
    )


def _read_downloaded_array(
    payload: bytes,
):
    with MemoryFile(
        payload
    ) as memory_file:
        with memory_file.open() as source:
            data = source.read(
                masked=True
            )

            transform = (
                source.transform
            )

            crs = source.crs

    return (
        data,
        transform,
        crs,
    )


def _build_local_tile_product(
    project_root: Path,
    period_start: str,
    model,
    aoa_parameters,
    tile: Tile,
    timeout_seconds: int,
) -> tuple[
    np.ndarray,
    dict[str, object],
]:
    core = ee.Geometry.Rectangle(
        [
            tile.xmin,
            tile.ymin,
            tile.xmax,
            tile.ymax,
        ],
        proj=ANALYSIS_CRS,
        geodesic=False,
    )

    context = (
        build_ridge25_production_stack(
            period_start_text=period_start,
            basin_geometry=core,
        )
    )

    (
        source_xmin,
        source_ymin,
        source_xmax,
        source_ymax,
        source_width,
        source_height,
        requested_fine_transform,
    ) = _processing_grid(
        tile
    )

    predictor_image = (
        context["stack"]
        .select(
            RIDGE25_MODEL_FEATURES
        )
        .toFloat()
    )

    predictor_parameters = {
        "bands":
            RIDGE25_MODEL_FEATURES,
        "crs":
            ANALYSIS_CRS,
        "crs_transform": [
            PREDICTION_SCALE_M,
            0,
            source_xmin,
            0,
            -PREDICTION_SCALE_M,
            source_ymax,
        ],
        "dimensions": [
            source_width,
            source_height,
        ],
        "format":
            "GEO_TIFF",
    }

    predictor_payload = (
        _download_ee_bytes(
            predictor_image,
            predictor_parameters,
            timeout_seconds,
        )
    )

    (
        predictor_bands,
        fine_transform,
        fine_crs,
    ) = _read_downloaded_array(
        predictor_payload
    )

    expected_shape = (
        len(
            RIDGE25_MODEL_FEATURES
        ),
        source_height,
        source_width,
    )

    if (
        predictor_bands.shape
        != expected_shape
    ):
        raise RuntimeError(
            "Downloaded predictor stack has "
            f"shape {predictor_bands.shape}; "
            f"expected {expected_shape}."
        )

    if fine_crs is None:
        raise RuntimeError(
            "Downloaded predictor stack has no CRS."
        )

    if not fine_transform.almost_equals(
        requested_fine_transform
    ):
        raise RuntimeError(
            "Downloaded predictor transform differs "
            "from the requested 20 m grid."
        )

    predictor_cube = np.moveaxis(
        predictor_bands.filled(
            np.nan
        ),
        0,
        -1,
    ).astype(
        np.float64
    )

    state = score_local_ridge25(
        predictor_cube=(
            predictor_cube
        ),
        model=model,
        aoa_parameters=(
            aoa_parameters
        ),
    )

    (
        modis_crs,
        modis_transform,
        modis_shape,
    ) = _build_modis_grid(
        context,
        (
            source_xmin,
            source_ymin,
            source_xmax,
            source_ymax,
        ),
    )

    (
        modis_height,
        modis_width,
    ) = modis_shape

    modis_image = (
        context["modis_et"]
        .rename(
            "ET_MODIS_mm_period"
        )
        .toFloat()
    )

    modis_parameters = {
        "bands": [
            "ET_MODIS_mm_period"
        ],
        "crs":
            modis_crs,
        "crs_transform": [
            modis_transform.a,
            modis_transform.b,
            modis_transform.c,
            modis_transform.d,
            modis_transform.e,
            modis_transform.f,
        ],
        "dimensions": [
            modis_width,
            modis_height,
        ],
        "format":
            "GEO_TIFF",
    }

    modis_payload = (
        _download_ee_bytes(
            modis_image,
            modis_parameters,
            timeout_seconds,
        )
    )

    (
        modis_bands,
        downloaded_modis_transform,
        downloaded_modis_crs,
    ) = _read_downloaded_array(
        modis_payload
    )

    if (
        modis_bands.shape
        != (
            1,
            modis_height,
            modis_width,
        )
    ):
        raise RuntimeError(
            "Downloaded MODIS ET grid has "
            "an unexpected shape."
        )

    if (
        downloaded_modis_crs
        is None
    ):
        raise RuntimeError(
            "Downloaded MODIS ET has no CRS."
        )

    if not (
        downloaded_modis_transform
        .almost_equals(
            modis_transform
        )
    ):
        raise RuntimeError(
            "Downloaded MODIS transform differs "
            "from the requested native grid."
        )

    modis_et = (
        modis_bands[0]
        .astype(
            np.float64
        )
        .filled(
            np.nan
        )
    )

    basin_geometry = (
        _analysis_geometry(
            project_root
        )
    )

    basin_mask = rasterize(
        [
            (
                basin_geometry,
                1,
            )
        ],
        out_shape=(
            source_height,
            source_width,
        ),
        transform=(
            fine_transform
        ),
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(
        bool
    )

    buffer_pixels_float = (
        float(
            PROCESSING_BUFFER_M
        )
        / float(
            PREDICTION_SCALE_M
        )
    )

    buffer_pixels = int(
        round(
            buffer_pixels_float
        )
    )

    if not math.isclose(
        buffer_pixels_float,
        buffer_pixels,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "Production buffer is not aligned "
            "with the 20 m grid."
        )

    row_slice = slice(
        buffer_pixels,
        buffer_pixels
        + tile.height_px,
    )

    column_slice = slice(
        buffer_pixels,
        buffer_pixels
        + tile.width_px,
    )

    core_domain = np.zeros(
        (
            source_height,
            source_width,
        ),
        dtype=bool,
    )

    core_domain[
        row_slice,
        column_slice,
    ] = True

    basin_core_domain = (
        core_domain
        & basin_mask
    )

    core_fraction = (
        aggregate_average_to_grid(
            source_array=(
                basin_core_domain.astype(
                    np.float64
                )
            ),
            source_transform=(
                fine_transform
            ),
            source_crs=(
                fine_crs
            ),
            destination_shape=(
                modis_shape
            ),
            destination_transform=(
                modis_transform
            ),
            destination_crs=(
                modis_crs
            ),
        )
    )

    convergence_check_mask = (
        np.isfinite(
            core_fraction
        )
        & (
            core_fraction > 0
        )
    )

    result = reconcile_local_ridge25(
        kc_raw=state.kc_raw,
        usable=state.usable,
        modis_et=modis_et,
        fine_transform=(
            fine_transform
        ),
        fine_crs=fine_crs,
        modis_transform=(
            modis_transform
        ),
        modis_crs=modis_crs,
        usable_support_fraction=(
            RIDGE25_USABLE_SUPPORT_FRACTION
        ),
        tolerance_mm=(
            RIDGE25_RECONCILIATION_TOLERANCE_MM
        ),
        max_iterations=(
            RIDGE25_RECONCILIATION_MAX_ITERATIONS
        ),
        convergence_check_mask=(
            convergence_check_mask
        ),
    )

    if not result.converged:
        raise RuntimeError(
            "Local MODIS reconciliation did not "
            "converge within "
            f"{RIDGE25_RECONCILIATION_MAX_ITERATIONS} "
            "iterations; maximum error="
            f"{result.max_abs_conservation_error:.6f} mm."
        )

    usable_fraction_fine = (
        coarse_to_fine_nearest(
            coarse_array=(
                result.usable_fraction
            ),
            coarse_transform=(
                modis_transform
            ),
            coarse_crs=(
                modis_crs
            ),
            fine_shape=(
                source_height,
                source_width,
            ),
            fine_transform=(
                fine_transform
            ),
            fine_crs=fine_crs,
        )
    )

    conservation_error_fine = (
        coarse_to_fine_nearest(
            coarse_array=(
                result.conservation_error
            ),
            coarse_transform=(
                modis_transform
            ),
            coarse_crs=(
                modis_crs
            ),
            fine_shape=(
                source_height,
                source_width,
            ),
            fine_transform=(
                fine_transform
            ),
            fine_crs=fine_crs,
        )
    )

    full_arrays = [
        result.et_published,
        state.kc_raw,
        state.dissimilarity_index,
        state.stack_valid.astype(
            np.float64
        ),
        state.aoa_inside.astype(
            np.float64
        ),
        state.usable.astype(
            np.float64
        ),
        usable_fraction_fine,
        result.eligible_fine.astype(
            np.float64
        ),
        conservation_error_fine,
    ]

    core_basin = basin_mask[
        row_slice,
        column_slice,
    ]

    core_arrays = []

    for array in full_arrays:
        core_array = np.asarray(
            array[
                row_slice,
                column_slice,
            ],
            dtype=np.float64,
        )

        core_array = np.where(
            core_basin,
            core_array,
            np.nan,
        )

        core_arrays.append(
            core_array
        )

    output = np.stack(
        core_arrays,
        axis=0,
    )

    expected_output_shape = (
        len(
            OUTPUT_BANDS
        ),
        tile.height_px,
        tile.width_px,
    )

    if (
        output.shape
        != expected_output_shape
    ):
        raise RuntimeError(
            "Local output shape differs "
            "from the tile contract."
        )

    intersecting_eligible_parents = (
        convergence_check_mask
        & result.eligible_coarse
        & np.isfinite(
            result.conservation_error
        )
    )

    parent_owned_by_tile = (
        _modis_parent_owned_by_tile(
            tile=tile,
            modis_shape=modis_shape,
            modis_transform=(
                modis_transform
            ),
            modis_crs=modis_crs,
        )
    )

    owned_eligible_parents = (
        intersecting_eligible_parents
        & parent_owned_by_tile
    )

    neighbouring_eligible_parents = (
        intersecting_eligible_parents
        & ~parent_owned_by_tile
    )

    fill_values = (
        1.0
        - result.usable_fraction[
            intersecting_eligible_parents
        ]
    )

    published_values = (
        output[0][
            np.isfinite(
                output[0]
            )
        ]
    )

    negative_published_et = int(
        (
            published_values < 0
        ).sum()
    )

    if (
        negative_published_et
        != 0
    ):
        raise RuntimeError(
            "Negative ET reached the "
            "published product."
        )

    metadata = {
        "production_method_version":
            RIDGE25_LOCAL_PRODUCTION_VERSION,
        "period_start":
            period_start,
        "tile_id":
            tile.tile_id,
        "level":
            tile.level,
        "usable_support_fraction":
            RIDGE25_USABLE_SUPPORT_FRACTION,
        "conservation_tolerance_mm":
            RIDGE25_RECONCILIATION_TOLERANCE_MM,
        "maximum_iterations_allowed":
            RIDGE25_RECONCILIATION_MAX_ITERATIONS,
        "iterations_used":
            int(
                result.iterations_used
            ),
        "converged":
            bool(
                result.converged
            ),
        "max_abs_conservation_error_mm":
            float(
                result.max_abs_conservation_error
            ),
        "intersecting_eligible_parents":
            int(
                intersecting_eligible_parents.sum()
            ),
        "owned_eligible_parents":
            int(
                owned_eligible_parents.sum()
            ),
        "intersecting_eligible_parents_owned_by_neighbour":
            int(
                neighbouring_eligible_parents.sum()
            ),
        "published_pixels":
            int(
                published_values.size
            ),
        "usable_core_pixels":
            int(
                (
                    core_basin
                    & state.usable[
                        row_slice,
                        column_slice,
                    ]
                ).sum()
            ),
        "mean_internal_fill_fraction":
            (
                float(
                    np.mean(
                        fill_values
                    )
                )
                if fill_values.size
                else None
            ),
        "max_internal_fill_fraction":
            (
                float(
                    np.max(
                        fill_values
                    )
                )
                if fill_values.size
                else None
            ),
        "negative_published_et":
            negative_published_et,
    }

    return (
        output,
        metadata,
    )


def _expected_tile_transform(
    tile: Tile,
) -> Affine:
    return from_origin(
        tile.xmin,
        tile.ymax,
        PREDICTION_SCALE_M,
        PREDICTION_SCALE_M,
    )


def _validate_tile_file(
    path: Path,
    tile: Tile,
) -> None:
    with rasterio.open(
        path
    ) as dataset:
        if (
            dataset.count
            != len(
                OUTPUT_BANDS
            )
        ):
            raise RuntimeError(
                f"{path} band-count mismatch."
            )

        if (
            dataset.width
            != tile.width_px
        ):
            raise RuntimeError(
                f"{path} width mismatch."
            )

        if (
            dataset.height
            != tile.height_px
        ):
            raise RuntimeError(
                f"{path} height mismatch."
            )

        if tuple(
            dataset.descriptions
        ) != tuple(
            OUTPUT_BANDS
        ):
            raise RuntimeError(
                f"{path} band descriptions "
                "do not match the current contract."
            )

        if (
            dataset.crs is None
            or dataset.crs.to_string()
            != ANALYSIS_CRS
        ):
            raise RuntimeError(
                f"{path} CRS mismatch."
            )

        if not (
            dataset.transform
            .almost_equals(
                _expected_tile_transform(
                    tile
                )
            )
        ):
            raise RuntimeError(
                f"{path} transform mismatch."
            )


def _write_tile(
    path: Path,
    tile: Tile,
    data: np.ndarray,
) -> None:
    profile = {
        "driver":
            "GTiff",
        "width":
            tile.width_px,
        "height":
            tile.height_px,
        "count":
            len(
                OUTPUT_BANDS
            ),
        "dtype":
            "float32",
        "crs":
            ANALYSIS_CRS,
        "transform":
            _expected_tile_transform(
                tile
            ),
        "nodata":
            OUTPUT_NODATA,
        "compress":
            "deflate",
        "tiled":
            True,
        "blockxsize":
            256,
        "blockysize":
            256,
    }

    prepared = np.where(
        np.isfinite(
            data
        ),
        data,
        OUTPUT_NODATA,
    ).astype(
        np.float32
    )

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as destination:
        destination.write(
            prepared
        )

        for (
            index,
            band_name,
        ) in enumerate(
            OUTPUT_BANDS,
            start=1,
        ):
            destination.set_band_description(
                index,
                band_name,
            )


def download_tile(
    project_root: Path,
    period_start: str,
    model,
    aoa_parameters,
    tile: Tile,
    tile_directory: Path,
    timeout_seconds: int = 600,
) -> CompletedTile:
    tile_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        tile_directory
        / f"{tile.tile_id}.tif"
    )

    metadata_path = (
        tile_directory
        / f"{tile.tile_id}.json"
    )

    if (
        path.is_file()
        and metadata_path.is_file()
    ):
        try:
            _validate_tile_file(
                path,
                tile,
            )

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                metadata.get(
                    "production_method_version"
                )
                != RIDGE25_LOCAL_PRODUCTION_VERSION
            ):
                raise RuntimeError(
                    "Cached tile method version mismatch."
                )

            if not bool(
                metadata.get(
                    "converged",
                    False,
                )
            ):
                raise RuntimeError(
                    "Cached tile is not converged."
                )

            return CompletedTile(
                tile,
                path,
            )

        except Exception:
            path.unlink(
                missing_ok=True
            )

            metadata_path.unlink(
                missing_ok=True
            )

    data, metadata = (
        _build_local_tile_product(
            project_root=(
                project_root
            ),
            period_start=(
                period_start
            ),
            model=model,
            aoa_parameters=(
                aoa_parameters
            ),
            tile=tile,
            timeout_seconds=(
                timeout_seconds
            ),
        )
    )

    temporary_path = (
        path.with_suffix(
            ".part.tif"
        )
    )

    temporary_metadata_path = (
        metadata_path.with_suffix(
            ".part.json"
        )
    )

    temporary_path.unlink(
        missing_ok=True
    )

    temporary_metadata_path.unlink(
        missing_ok=True
    )

    _write_tile(
        temporary_path,
        tile,
        data,
    )

    _validate_tile_file(
        temporary_path,
        tile,
    )

    temporary_metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )

    temporary_metadata_path.replace(
        metadata_path
    )

    return CompletedTile(
        tile,
        path,
    )


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
                    "tile_id":
                        tile.tile_id,
                    "level":
                        tile.level,
                    "xmin":
                        tile.xmin,
                    "ymin":
                        tile.ymin,
                    "xmax":
                        tile.xmax,
                    "ymax":
                        tile.ymax,
                    "width_px":
                        tile.width_px,
                    "height_px":
                        tile.height_px,
                    "path":
                        str(
                            item.path
                        ),
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
    if not completed:
        raise ValueError(
            "No tiles are available to mosaic."
        )

    xmin, ymin, xmax, ymax = (
        grid_bounds
    )

    scale = float(
        PREDICTION_SCALE_M
    )

    width = int(
        round(
            (xmax - xmin)
            / scale
        )
    )

    height = int(
        round(
            (ymax - ymin)
            / scale
        )
    )

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
        "driver":
            "GTiff",
        "width":
            width,
        "height":
            height,
        "count":
            len(
                OUTPUT_BANDS
            ),
        "dtype":
            "float32",
        "crs":
            ANALYSIS_CRS,
        "transform":
            transform,
        "nodata":
            OUTPUT_NODATA,
        "compress":
            "deflate",
        "tiled":
            True,
        "blockxsize":
            256,
        "blockysize":
            256,
        "BIGTIFF":
            "IF_SAFER",
    }

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as destination:
        for (
            index,
            band_name,
        ) in enumerate(
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
                    (
                        tile.xmin
                        - xmin
                    )
                    / scale
                )
            )

            row_off = int(
                round(
                    (
                        ymax
                        - tile.ymax
                    )
                    / scale
                )
            )

            with rasterio.open(
                item.path
            ) as source:
                data = (
                    source.read(
                        masked=True
                    )
                    .filled(
                        OUTPUT_NODATA
                    )
                )

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
    aoa_parameters,
    tile_size_m: int = 4000,
    min_tile_size_m: int = 500,
) -> dict[str, object]:
    project_root = Path(
        project_root
    ).resolve()

    workspace = (
        get_workspace_paths(
            project_root
        )
        .ensure()
    )

    period_directory = (
        workspace.rasters
        / period_start
    )

    period_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tile_directory = (
        period_directory
        / (
            "tiles_"
            + RIDGE25_LOCAL_PRODUCTION_VERSION
        )
    )

    initial_tiles, grid_bounds = (
        build_initial_tiles(
            project_root,
            tile_size_m=(
                tile_size_m
            ),
        )
    )

    completed = []

    for index, tile in enumerate(
        initial_tiles,
        start=1,
    ):
        print(
            f"[{index}/{len(initial_tiles)}] "
            f"{tile.tile_id} "
            f"({int(tile.width_m)} m core)"
        )

        completed.append(
            download_tile(
                project_root=(
                    project_root
                ),
                period_start=(
                    period_start
                ),
                model=model,
                aoa_parameters=(
                    aoa_parameters
                ),
                tile=tile,
                tile_directory=(
                    tile_directory
                ),
            )
        )

    completed = sorted(
        completed,
        key=lambda item:
            item.tile.tile_id,
    )

    manifest_path = (
        period_directory
        / (
            "tile_manifest_"
            + RIDGE25_LOCAL_PRODUCTION_VERSION
            + ".csv"
        )
    )

    _write_manifest(
        completed,
        manifest_path,
    )

    raster_path = (
        period_directory
        / (
            "ET_"
            + RIDGE25_LOCAL_PRODUCTION_VERSION
            + f"_{period_start}_20m.tif"
        )
    )

    mosaic_tiles(
        completed,
        raster_path,
        grid_bounds,
    )

    tile_metadata = []

    for item in completed:
        metadata_path = (
            item.path.with_suffix(
                ".json"
            )
        )

        tile_metadata.append(
            json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
        )

    metadata = {
        "period_start":
            period_start,
        "production_method_version":
            RIDGE25_LOCAL_PRODUCTION_VERSION,
        "analysis_crs":
            ANALYSIS_CRS,
        "prediction_scale_m":
            PREDICTION_SCALE_M,
        "initial_tile_size_m":
            _normalize_tile_size(
                tile_size_m
            ),
        "minimum_tile_size_m":
            _normalize_tile_size(
                min_tile_size_m
            ),
        "initial_tiles":
            len(
                initial_tiles
            ),
        "adaptive_subdivision_used":
            False,
        "tile_failure_policy":
            "fail_after_download_retries",
        "completed_tiles":
            len(
                completed
            ),
        "intersecting_eligible_parent_records":
            sum(
                item[
                    "intersecting_eligible_parents"
                ]
                for item
                in tile_metadata
            ),
        "owned_eligible_parents":
            sum(
                item[
                    "owned_eligible_parents"
                ]
                for item
                in tile_metadata
            ),
        "neighbour_parent_records":
            sum(
                item[
                    "intersecting_eligible_parents_owned_by_neighbour"
                ]
                for item
                in tile_metadata
            ),
        "parent_accounting_rule":
            (
                "MODIS parent counted once by "
                "pixel-center ownership"
            ),
        "output_bands":
            OUTPUT_BANDS,
        "usable_support_fraction":
            RIDGE25_USABLE_SUPPORT_FRACTION,
        "conservation_tolerance_mm":
            RIDGE25_RECONCILIATION_TOLERANCE_MM,
        "maximum_reconciliation_iterations":
            RIDGE25_RECONCILIATION_MAX_ITERATIONS,
        "maximum_iterations_used":
            max(
                (
                    item[
                        "iterations_used"
                    ]
                    for item
                    in tile_metadata
                ),
                default=0,
            ),
        "maximum_abs_conservation_error_mm":
            max(
                (
                    item[
                        "max_abs_conservation_error_mm"
                    ]
                    for item
                    in tile_metadata
                ),
                default=0.0,
            ),
        "all_tiles_converged":
            all(
                item[
                    "converged"
                ]
                for item
                in tile_metadata
            ),
        "negative_published_et":
            sum(
                item[
                    "negative_published_et"
                ]
                for item
                in tile_metadata
            ),
        "applicability_rule":
            (
                "complete_stack AND "
                "AOA_inside AND Kc_raw >= 0"
            ),
        "internal_fill_published":
            False,
        "reconciliation_location":
            "local_python",
        "earth_engine_role":
            (
                "predictor_and_native_modis_"
                "data_provider"
            ),
        "google_drive_used":
            False,
        "earth_engine_asset_created":
            False,
        "model_source":
            "fitted_in_current_run",
        "raster":
            str(
                raster_path
            ),
        "tile_manifest":
            str(
                manifest_path
            ),
    }

    metadata_path = (
        period_directory
        / (
            "production_metadata_"
            + RIDGE25_LOCAL_PRODUCTION_VERSION
            + ".json"
        )
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "raster":
            raster_path,
        "manifest":
            manifest_path,
        "metadata":
            metadata_path,
        "completed_tiles":
            completed,
        "initial_tiles":
            initial_tiles,
    }
