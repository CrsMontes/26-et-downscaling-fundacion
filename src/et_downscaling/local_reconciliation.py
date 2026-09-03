"""Local Ridge-25 applicability and MODIS reconciliation utilities.

Earth Engine provides the predictor and MODIS source data. Ridge prediction,
AOA scoring, fine-support quality control and MODIS reconciliation are
performed locally to keep the production graph reproducible and memory-safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rasterio.warp import Resampling, reproject

from .aoa_ridge25 import (
    AOAParameters,
    score_unweighted_aoa,
)
from .ridge25 import RIDGE25_MODEL_FEATURES


RIDGE25_USABLE_SUPPORT_FRACTION = 0.90
RIDGE25_RECONCILIATION_TOLERANCE_MM = 0.01
RIDGE25_RECONCILIATION_MAX_ITERATIONS = 30


@dataclass(frozen=True)
class LocalRidge25State:
    """Fine-grid Ridge prediction and applicability masks."""

    kc_raw: np.ndarray
    dissimilarity_index: np.ndarray
    stack_valid: np.ndarray
    aoa_inside: np.ndarray
    usable: np.ndarray


@dataclass(frozen=True)
class LocalReconciliationResult:
    """Local conservative reconciliation result on fine and MODIS grids."""

    et_full_support: np.ndarray
    et_published: np.ndarray
    usable_fraction: np.ndarray
    eligible_coarse: np.ndarray
    eligible_fine: np.ndarray
    kc_valid_mean: np.ndarray
    mass_scale: np.ndarray
    et_reaggregated: np.ndarray
    conservation_error: np.ndarray
    converged: bool
    iterations_used: int
    max_abs_conservation_error: float


def score_local_ridge25(
    predictor_cube: np.ndarray,
    model,
    aoa_parameters: AOAParameters,
) -> LocalRidge25State:
    """Predict Kc and derive the final usable fine-grid support."""

    cube = np.asarray(
        predictor_cube,
        dtype=float,
    )

    if cube.ndim != 3:
        raise ValueError(
            "Ridge-25 predictor cube must be three-dimensional."
        )

    feature_count = len(
        RIDGE25_MODEL_FEATURES
    )

    if cube.shape[2] != feature_count:
        raise ValueError(
            "Ridge-25 predictor cube feature count differs "
            "from the model specification."
        )

    if tuple(
        aoa_parameters.feature_names
    ) != tuple(
        RIDGE25_MODEL_FEATURES
    ):
        raise ValueError(
            "AOA predictor schema differs from Ridge-25."
        )

    rows, columns, _ = cube.shape

    flat = cube.reshape(
        -1,
        feature_count,
    )

    stack_valid_flat = np.isfinite(
        flat
    ).all(
        axis=1
    )

    kc_flat = np.full(
        flat.shape[0],
        np.nan,
        dtype=float,
    )

    di_flat = np.full(
        flat.shape[0],
        np.nan,
        dtype=float,
    )

    aoa_flat = np.zeros(
        flat.shape[0],
        dtype=bool,
    )

    if stack_valid_flat.any():
        valid_predictors = flat[
            stack_valid_flat
        ]

        frame = pd.DataFrame(
            valid_predictors,
            columns=RIDGE25_MODEL_FEATURES,
        )

        predictions = np.asarray(
            model.predict(frame),
            dtype=float,
        )

        if predictions.shape != (
            valid_predictors.shape[0],
        ):
            raise RuntimeError(
                "Ridge-25 prediction count differs from valid pixels."
            )

        di, inside = score_unweighted_aoa(
            valid_predictors,
            aoa_parameters,
        )

        kc_flat[
            stack_valid_flat
        ] = predictions

        di_flat[
            stack_valid_flat
        ] = di

        aoa_flat[
            stack_valid_flat
        ] = inside

    kc_raw = kc_flat.reshape(
        rows,
        columns,
    )

    dissimilarity_index = (
        di_flat.reshape(
            rows,
            columns,
        )
    )

    stack_valid = (
        stack_valid_flat.reshape(
            rows,
            columns,
        )
    )

    aoa_inside = (
        aoa_flat.reshape(
            rows,
            columns,
        )
    )

    usable = (
        stack_valid
        & aoa_inside
        & np.isfinite(kc_raw)
        & (kc_raw >= 0)
    )

    return LocalRidge25State(
        kc_raw=kc_raw,
        dissimilarity_index=dissimilarity_index,
        stack_valid=stack_valid,
        aoa_inside=aoa_inside,
        usable=usable,
    )


def aggregate_average_to_grid(
    source_array: np.ndarray,
    source_transform,
    source_crs,
    destination_shape: tuple[int, int],
    destination_transform,
    destination_crs,
    source_nodata: float | None = None,
) -> np.ndarray:
    """Area-average a fine array onto an explicit destination grid."""

    destination = np.full(
        destination_shape,
        np.nan,
        dtype=np.float64,
    )

    kwargs = {}

    if source_nodata is not None:
        kwargs["src_nodata"] = (
            source_nodata
        )

    reproject(
        source=np.asarray(
            source_array,
            dtype=np.float64,
        ),
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
        init_dest_nodata=True,
        num_threads=1,
        **kwargs,
    )

    return destination


def coarse_to_fine_nearest(
    coarse_array: np.ndarray,
    coarse_transform,
    coarse_crs,
    fine_shape: tuple[int, int],
    fine_transform,
    fine_crs,
) -> np.ndarray:
    """Map native MODIS values to the fine grid without interpolation."""

    nodata = -9999.0

    source = np.where(
        np.isfinite(
            coarse_array
        ),
        coarse_array,
        nodata,
    ).astype(
        np.float64
    )

    destination = np.full(
        fine_shape,
        np.nan,
        dtype=np.float64,
    )

    reproject(
        source=source,
        destination=destination,
        src_transform=coarse_transform,
        src_crs=coarse_crs,
        src_nodata=nodata,
        dst_transform=fine_transform,
        dst_crs=fine_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
        init_dest_nodata=True,
        num_threads=1,
    )

    return destination


def reconcile_local_ridge25(
    kc_raw: np.ndarray,
    usable: np.ndarray,
    modis_et: np.ndarray,
    fine_transform,
    fine_crs,
    modis_transform,
    modis_crs,
    usable_support_fraction: float = (
        RIDGE25_USABLE_SUPPORT_FRACTION
    ),
    tolerance_mm: float = (
        RIDGE25_RECONCILIATION_TOLERANCE_MM
    ),
    max_iterations: int = (
        RIDGE25_RECONCILIATION_MAX_ITERATIONS
    ),
    convergence_check_mask: np.ndarray | None = None,
) -> LocalReconciliationResult:
    """Reconcile usable Ridge-25 Kc to native MODIS ET support.

    Non-usable fine cells are filled only internally with the mean Kc of
    usable cells in the same MODIS support. They are never included in
    ``et_published``.
    """

    kc_raw = np.asarray(
        kc_raw,
        dtype=float,
    )

    usable = np.asarray(
        usable,
        dtype=bool,
    )

    modis_et = np.asarray(
        modis_et,
        dtype=float,
    )

    if kc_raw.shape != usable.shape:
        raise ValueError(
            "Kc and usable-mask shapes differ."
        )

    if modis_et.ndim != 2:
        raise ValueError(
            "MODIS ET must be two-dimensional."
        )

    if not (
        0 < usable_support_fraction <= 1
    ):
        raise ValueError(
            "Usable-support fraction must be in (0, 1]."
        )

    if tolerance_mm <= 0:
        raise ValueError(
            "Conservation tolerance must be positive."
        )

    if max_iterations < 1:
        raise ValueError(
            "Maximum reconciliation iterations must be >= 1."
        )

    fine_shape = kc_raw.shape
    modis_shape = modis_et.shape

    def aggregate(
        array: np.ndarray,
        nodata: float | None = None,
    ) -> np.ndarray:
        return aggregate_average_to_grid(
            source_array=array,
            source_transform=fine_transform,
            source_crs=fine_crs,
            destination_shape=modis_shape,
            destination_transform=modis_transform,
            destination_crs=modis_crs,
            source_nodata=nodata,
        )

    def aggregate_finite(
        array: np.ndarray,
    ) -> np.ndarray:
        nodata = -9999.0

        prepared = np.where(
            np.isfinite(array),
            array,
            nodata,
        )

        return aggregate(
            prepared,
            nodata=nodata,
        )

    def to_fine(
        array: np.ndarray,
    ) -> np.ndarray:
        return coarse_to_fine_nearest(
            coarse_array=array,
            coarse_transform=modis_transform,
            coarse_crs=modis_crs,
            fine_shape=fine_shape,
            fine_transform=fine_transform,
            fine_crs=fine_crs,
        )

    usable_fraction = aggregate(
        usable.astype(
            np.float64
        )
    )

    kc_valid_mean = aggregate(
        np.where(
            usable,
            kc_raw,
            -9999.0,
        ),
        nodata=-9999.0,
    )

    eligible_coarse = (
        np.isfinite(
            usable_fraction
        )
        & (
            usable_fraction
            >= usable_support_fraction
        )
        & np.isfinite(
            kc_valid_mean
        )
        & np.isfinite(
            modis_et
        )
    )

    eligible_fine = (
        to_fine(
            eligible_coarse.astype(
                np.float64
            )
        )
        >= 0.5
    )

    kc_mean_fine = to_fine(
        kc_valid_mean
    )

    kc_filled = np.where(
        eligible_fine & usable,
        kc_raw,
        np.where(
            eligible_fine,
            kc_mean_fine,
            np.nan,
        ),
    )

    kc_filled_mean = aggregate_finite(
        kc_filled
    )

    mass_scale = np.full(
        modis_shape,
        np.nan,
        dtype=np.float64,
    )

    scale_mask = (
        eligible_coarse
        & np.isfinite(
            kc_filled_mean
        )
        & (
            np.abs(
                kc_filled_mean
            )
            > 1e-9
        )
    )

    mass_scale[
        scale_mask
    ] = (
        modis_et[
            scale_mask
        ]
        / kc_filled_mean[
            scale_mask
        ]
    )

    et_full_support = (
        kc_filled
        * to_fine(
            mass_scale
        )
    )

    if convergence_check_mask is None:
        check_base = (
            eligible_coarse.copy()
        )
    else:
        check_base = np.asarray(
            convergence_check_mask,
            dtype=bool,
        )

        if check_base.shape != modis_shape:
            raise ValueError(
                "Convergence-check mask shape differs from MODIS grid."
            )

        check_base = (
            check_base
            & eligible_coarse
        )

    converged = False
    iterations_used = 0
    max_abs_error = np.nan

    for iteration in range(
        max_iterations + 1
    ):
        et_reaggregated = (
            aggregate_finite(
                et_full_support
            )
        )

        conservation_error = (
            et_reaggregated
            - modis_et
        )

        check_mask = (
            check_base
            & np.isfinite(
                conservation_error
            )
        )

        if not check_mask.any():
            converged = True
            iterations_used = (
                iteration
            )
            max_abs_error = 0.0
            break

        max_abs_error = float(
            np.max(
                np.abs(
                    conservation_error[
                        check_mask
                    ]
                )
            )
        )

        if max_abs_error <= tolerance_mm:
            converged = True
            iterations_used = (
                iteration
            )
            break

        if iteration == max_iterations:
            iterations_used = (
                max_iterations
            )
            break

        correction = np.full(
            modis_shape,
            np.nan,
            dtype=np.float64,
        )

        correction_mask = (
            eligible_coarse
            & np.isfinite(
                et_reaggregated
            )
            & (
                np.abs(
                    et_reaggregated
                )
                > 1e-9
            )
        )

        correction[
            correction_mask
        ] = (
            modis_et[
                correction_mask
            ]
            / et_reaggregated[
                correction_mask
            ]
        )

        et_full_support = (
            et_full_support
            * to_fine(
                correction
            )
        )

    et_reaggregated = aggregate_finite(
        et_full_support
    )

    conservation_error = (
        et_reaggregated
        - modis_et
    )

    et_published = np.where(
        usable
        & eligible_fine
        & np.isfinite(
            et_full_support
        ),
        et_full_support,
        np.nan,
    )

    return LocalReconciliationResult(
        et_full_support=et_full_support,
        et_published=et_published,
        usable_fraction=usable_fraction,
        eligible_coarse=eligible_coarse,
        eligible_fine=eligible_fine,
        kc_valid_mean=kc_valid_mean,
        mass_scale=mass_scale,
        et_reaggregated=et_reaggregated,
        conservation_error=conservation_error,
        converged=converged,
        iterations_used=iterations_used,
        max_abs_conservation_error=(
            max_abs_error
        ),
    )
