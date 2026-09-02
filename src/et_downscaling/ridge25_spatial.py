"""Fine-resolution production for the accepted in-memory Ridge-25 model.

This module is deliberately separate from the legacy RF production path.
Sentinel-1 and CHIRPS are not production requirements for Ridge-25.

Reconciliation is applied only after the fitted Ridge model has generated the
fine Kc pattern. It never enters model training or out-of-fold validation.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import ee
from sklearn.pipeline import Pipeline

from .modis import get_modis_collection
from .production import (
    MODIS_RECONCILIATION_PASSES,
    OPTICAL_MIN_COVERAGE_FRACTION,
    _mean_at_modis_support,
    _valid_fraction_at_modis_support,
    load_basin_geometry,
)
from .ridge25 import (
    RIDGE25_MODEL_FEATURES,
    build_ee_ridge25_prediction,
)
from .ridge25_production import build_ridge25_production_stack


def build_ridge25_constrained_et(
    kc_raw: ee.Image,
    optical_predictors: ee.Image,
    model_stack: ee.Image,
    modis_et: ee.Image,
    modis_projection: ee.Projection,
    fine_projection: ee.Projection,
    basin_geometry: ee.Geometry,
) -> dict[str, ee.Image]:
    """Reconcile Ridge-25 fine predictions to native MODIS ET support.

    Eligibility reflects the accepted Ridge-25 training gate:
    - Sentinel-2 valid fraction >= 90%;
    - complete 25-band Ridge stack over the same threshold;
    - valid MODIS ET.

    Sentinel-1 is intentionally absent because it is not a Ridge-25 predictor
    and must not reduce the production domain.
    """
    kc_raw = ee.Image(kc_raw).rename("Kc_raw").toFloat()
    modis_et = (
        ee.Image(modis_et)
        .rename("ET_MODIS_mm_period")
        .toFloat()
    )

    optical_fraction = _valid_fraction_at_modis_support(
        optical_predictors,
        modis_projection,
        "optical_valid_fraction",
    )
    stack_fraction = _valid_fraction_at_modis_support(
        model_stack,
        modis_projection,
        "model_stack_valid_fraction",
    )

    eligible = (
        optical_fraction.gte(OPTICAL_MIN_COVERAGE_FRACTION)
        .And(
            stack_fraction.gte(
                OPTICAL_MIN_COVERAGE_FRACTION
            )
        )
        .And(modis_et.mask().gt(0))
        .rename("coarse_eligible")
    )

    kc_valid_mean = _mean_at_modis_support(
        kc_raw,
        modis_projection,
        "Kc_valid_mean",
    )

    kc_filled = (
        kc_raw
        .unmask(kc_valid_mean)
        .updateMask(eligible)
        .reproject(fine_projection)
        .rename("Kc_filled")
        .toFloat()
    )

    kc_filled_mean = _mean_at_modis_support(
        kc_filled,
        modis_projection,
        "Kc_filled_mean",
    )

    mass_scale = (
        modis_et
        .divide(kc_filled_mean)
        .updateMask(kc_filled_mean.abs().gt(1e-9))
        .updateMask(eligible)
        .rename("mass_scale")
        .toFloat()
    )

    et_full_support = (
        kc_filled
        .multiply(mass_scale)
        .rename("ET_mm_period")
        .reproject(fine_projection)
        .toFloat()
    )

    for correction_pass in range(
        MODIS_RECONCILIATION_PASSES
    ):
        pass_mean = _mean_at_modis_support(
            et_full_support,
            modis_projection,
            (
                "ET_reaggregated_pass_"
                f"{correction_pass + 1}"
            ),
        )
        pass_factor = (
            modis_et
            .divide(pass_mean)
            .updateMask(pass_mean.abs().gt(1e-9))
            .updateMask(eligible)
            .rename("reconciliation_factor")
            .toFloat()
        )
        et_full_support = (
            et_full_support
            .multiply(pass_factor)
            .rename("ET_mm_period")
            .reproject(fine_projection)
            .toFloat()
        )

    et_final = (
        et_full_support
        .clip(basin_geometry)
        .rename("ET_mm_period")
        .toFloat()
    )

    et_reaggregated = _mean_at_modis_support(
        et_full_support,
        modis_projection,
        "ET_reaggregated_mm_period",
    )
    conservation_error = (
        et_reaggregated
        .subtract(modis_et)
        .updateMask(eligible)
        .rename("ET_conservation_error_mm")
        .toFloat()
    )

    fill_fraction_raw = (
        ee.Image.constant(1)
        .subtract(stack_fraction)
    )
    fill_fraction = (
        fill_fraction_raw
        .where(fill_fraction_raw.lt(0), 0)
        .rename("fine_fill_fraction")
        .updateMask(eligible)
        .toFloat()
    )

    return {
        "et_final": et_final,
        "kc_raw": kc_raw.clip(basin_geometry),
        "kc_filled": kc_filled.clip(basin_geometry),
        "mass_scale": mass_scale,
        "eligible": eligible,
        "optical_valid_fraction": optical_fraction,
        "model_stack_valid_fraction": stack_fraction,
        "fine_fill_fraction": fill_fraction,
        "et_reaggregated": et_reaggregated,
        "conservation_error": conservation_error,
    }


def build_ridge25_product(
    project_root: Path,
    period_start_text: str,
    model: Pipeline,
) -> tuple[
    dict[str, ee.Image],
    dict[str, object],
    ee.Geometry,
    dict[str, object],
]:
    """Build one Ridge-25 ET product using the model fitted in the same run."""
    requested_date = date.fromisoformat(
        period_start_text
    )
    requested_start = ee.Date(period_start_text)

    modis_count = int(
        get_modis_collection()
        .filterDate(
            requested_start,
            requested_start.advance(1, "day"),
        )
        .size()
        .getInfo()
    )
    if modis_count != 1:
        raise ValueError(
            f"{period_start_text} is not an available "
            "MODIS period start."
        )

    basin = load_basin_geometry(project_root)
    context = build_ridge25_production_stack(
        period_start_text=period_start_text,
        basin_geometry=basin,
    )

    s2_count = int(
        context["optical_period"].size().getInfo()
    )
    if s2_count == 0:
        raise RuntimeError(
            "No Sentinel-2 observations are available "
            "for the requested MODIS period."
        )

    band_names = context["stack"].bandNames().getInfo()
    if band_names != list(RIDGE25_MODEL_FEATURES):
        raise RuntimeError(
            "Production predictor order differs from "
            "the fitted Ridge-25 schema."
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
        basin_geometry=basin,
    )

    number_days = int(
        round(
            float(
                context["number_days"].getInfo()
            )
        )
    )

    metadata = {
        "period_start": requested_date.isoformat(),
        "number_days": number_days,
        "modis_count": modis_count,
        "s2_count": s2_count,
        "predictors": len(RIDGE25_MODEL_FEATURES),
        "reconciliation_passes": (
            MODIS_RECONCILIATION_PASSES
        ),
        "model_source": "fitted_in_current_run",
        "sentinel1_required": False,
        "chirps_required": False,
    }

    return outputs, context, basin, metadata
