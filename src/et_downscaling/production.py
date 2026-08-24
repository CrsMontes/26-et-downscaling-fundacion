"""Spatial production utilities for MODIS-constrained Sentinel-2 ET.

The fine grid is a prediction support, not an independent ET observation
support. Sentinel-2 and Sentinel-1 provide the subpixel spatial pattern;
meteorological predictors retain their native coarse spatial support.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import ee
import numpy as np

from .config import (
    ANALYSIS_CRS,
    S1_FULL_COVERAGE,
)
from .modis import (
    get_modis_collection,
    get_modis_period_end,
    prepare_modis_et,
)
from .model_spec import (
    COMMON_MODEL_FEATURES,
    HARMONIC_FEATURES,
)
from .optical import (
    build_optical_predictors,
    filter_optical_period,
)
from .sentinel1 import (
    build_s1_median,
    get_sentinel1_collection,
)
from .sentinel2 import get_sentinel2_collection


PREDICTION_SCALE_M = 20
PROCESSING_BUFFER_M = 1000
OPTICAL_MIN_COVERAGE_FRACTION = 0.90
S1_MIN_COVERAGE_FRACTION = S1_FULL_COVERAGE
MODIS_RECONCILIATION_PASSES = 3
MODIS_CONSERVATION_TOLERANCE_MM = 0.01

ERA5_COLLECTION_ID = "ECMWF/ERA5_LAND/HOURLY"
CHIRPS_COLLECTION_ID = "UCSB-CHG/CHIRPS/DAILY"

OPTICAL_SOURCE_BANDS = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
    "NDVI",
    "EVI",
    "SAVI",
    "NDWI",
    "NDMI",
]

OPTICAL_MODEL_BANDS = [
    "Blue_mean",
    "Green_mean",
    "Red_mean",
    "NIR_mean",
    "SWIR1_mean",
    "SWIR2_mean",
    "NDVI_mean",
    "EVI_mean",
    "SAVI_mean",
    "NDWI_mean",
    "NDMI_mean",
]

S1_SOURCE_BANDS = [
    "VV_dB",
    "VH_dB",
    "VV_minus_VH_dB",
]

S1_MODEL_BANDS = [
    "VV_dB_mean",
    "VH_dB_mean",
    "VV_minus_VH_dB_mean",
]

METEOROLOGICAL_MODEL_BANDS = [
    "Tair_mean_C",
    "Tair_max_C",
    "VPD_mean_kPa",
    "SolarRad_MJ_m2_day",
    "Wind_mean_ms",
    "Precip_period_mm",
    "Precip_prev30d_mm",
]


def load_basin_geometry(
    project_root: Path,
) -> ee.Geometry:
    """Load the versioned Fundación basin GeoJSON as an Earth Engine geometry."""
    path = (
        Path(project_root)
        / "data"
        / "boundaries"
        / "fundacion_basin.geojson"
    )

    if not path.is_file():
        raise FileNotFoundError(f"Basin boundary not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    geojson_type = data.get("type")
    if geojson_type == "FeatureCollection":
        features = data.get("features", [])
        if len(features) != 1:
            raise ValueError(
                "Fundación basin GeoJSON must contain exactly one feature."
            )
        geometry = features[0].get("geometry")
    elif geojson_type == "Feature":
        geometry = data.get("geometry")
    else:
        geometry = data

    if not isinstance(geometry, dict) or "type" not in geometry:
        raise ValueError("Invalid basin geometry in GeoJSON.")

    return ee.Geometry(geometry)


def build_study_feature_collection(
    geometry: ee.Geometry,
) -> ee.FeatureCollection:
    """Wrap a geometry for collection functions used by the training pipeline."""
    return ee.FeatureCollection([ee.Feature(geometry)])


def get_fine_projection() -> ee.Projection:
    """Return the explicit 20 m UTM grid used for Sentinel-2 production."""
    return ee.Projection(ANALYSIS_CRS).atScale(PREDICTION_SCALE_M)


def build_modis_period_context(
    period_start_text: str,
    processing_geometry: ee.Geometry,
) -> dict[str, object]:
    """Build the native MODIS target image and exact temporal support."""
    requested_start = ee.Date(period_start_text)
    collection = (
        get_modis_collection()
        .filterDate(
            requested_start,
            requested_start.advance(1, "day"),
        )
    )

    image = ee.Image(collection.first())
    period_start = image.date()
    period_end = get_modis_period_end(period_start)
    number_days = period_end.difference(period_start, "day")

    prepared = prepare_modis_et(image, number_days)
    modis_et = (
        prepared
        .select("ET_mm_period")
        .toFloat()
    )

    return {
        "collection": collection,
        "source_image": image,
        "period_start": period_start,
        "period_end": period_end,
        "number_days": number_days,
        "modis_et": modis_et,
        "modis_projection": image.select("ET").projection(),
    }


def build_s2_common_predictors(
    study_features: ee.FeatureCollection,
    period_start: ee.Date,
    period_end: ee.Date,
    processing_geometry: ee.Geometry,
    fine_projection: ee.Projection,
) -> tuple[ee.Image, ee.ImageCollection]:
    """Build the same common Sentinel-2 predictors used during training."""
    collection = get_sentinel2_collection(study_features)
    period_collection = filter_optical_period(
        collection=collection,
        period_start=period_start,
        period_end=period_end,
        geometry=processing_geometry,
        source="S2",
    )

    predictors = build_optical_predictors(
        period_collection=period_collection,
        geometry=processing_geometry,
        source="S2",
    )

    common = (
        predictors
        .select(OPTICAL_SOURCE_BANDS, OPTICAL_MODEL_BANDS)
        .reproject(fine_projection)
        .toFloat()
    )

    return common, period_collection


def build_s1_common_predictors(
    study_features: ee.FeatureCollection,
    period_start: ee.Date,
    period_end: ee.Date,
    processing_geometry: ee.Geometry,
    fine_projection: ee.Projection,
) -> tuple[ee.Image, ee.ImageCollection]:
    """Build R077 ascending S1 median predictors and aggregate them to 20 m."""
    collection = get_sentinel1_collection(study_features)
    period_collection = (
        collection
        .filterDate(period_start, period_end)
        .filterBounds(processing_geometry)
    )

    median = build_s1_median(
        period_collection,
        processing_geometry,
    )

    predictors = (
        median
        .select(S1_SOURCE_BANDS)
        .reproject(
            crs=ANALYSIS_CRS,
            scale=10,
        )
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=4,
        )
        .reproject(fine_projection)
        .rename(S1_MODEL_BANDS)
        .toFloat()
    )

    return predictors, period_collection


def _saturation_vapor_pressure_kpa(
    temperature_c: ee.Image,
) -> ee.Image:
    """Saturation vapor pressure used by the local meteorology workflow."""
    temperature_c = ee.Image(temperature_c)
    return (
        temperature_c
        .multiply(17.27)
        .divide(temperature_c.add(237.3))
        .exp()
        .multiply(0.6108)
    )


def _prepare_era5_hourly_predictors(
    image: ee.Image,
) -> ee.Image:
    """Derive the hourly quantities that were aggregated for model training."""
    image = ee.Image(image)

    tair_c = image.select("temperature_2m").subtract(273.15).rename("Tair_C")
    tdew_c = (
        image.select("dewpoint_temperature_2m")
        .subtract(273.15)
        .rename("Tdew_C")
    )

    vapor_pressure_deficit_raw = (
        _saturation_vapor_pressure_kpa(tair_c)
        .subtract(_saturation_vapor_pressure_kpa(tdew_c))
    )
    vapor_pressure_deficit = (
        vapor_pressure_deficit_raw
        .where(vapor_pressure_deficit_raw.lt(0), 0)
        .rename("VPD_kPa")
    )

    wind = (
        image.select("u_component_of_wind_10m").pow(2)
        .add(image.select("v_component_of_wind_10m").pow(2))
        .sqrt()
        .rename("Wind_ms")
    )

    solar_j = image.select("surface_solar_radiation_downwards_hourly")
    solar_nonnegative = solar_j.where(solar_j.lt(0), 0)
    solar = solar_nonnegative.multiply(1e-6).rename("SolarRad_MJ_m2_hour")

    prepared = (
        tair_c
        .addBands(vapor_pressure_deficit)
        .addBands(wind)
        .addBands(solar)
    )

    prepared = ee.Image(
        prepared.copyProperties(
            image,
            ["system:time_start"],
        )
    )

    return (
        prepared
        .toFloat()
    )


def build_period_meteorological_predictors(
    period_start: ee.Date,
    period_end: ee.Date,
    number_days: ee.Number,
    processing_geometry: ee.Geometry,
) -> ee.Image:
    """Build coarse ERA5-Land/CHIRPS fields with training-compatible timing."""
    era5_source_bands = [
        "temperature_2m",
        "dewpoint_temperature_2m",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "surface_solar_radiation_downwards_hourly",
    ]

    era5 = (
        ee.ImageCollection(ERA5_COLLECTION_ID)
        .filterDate(period_start, period_end)
        .select(era5_source_bands)
        .map(_prepare_era5_hourly_predictors)
    )

    tair_mean = era5.select("Tair_C").mean().rename("Tair_mean_C")
    tair_max = era5.select("Tair_C").max().rename("Tair_max_C")
    vpd_mean = era5.select("VPD_kPa").mean().rename("VPD_mean_kPa")
    wind_mean = era5.select("Wind_ms").mean().rename("Wind_mean_ms")
    solar_mean_daily = (
        era5.select("SolarRad_MJ_m2_hour")
        .sum()
        .divide(number_days)
        .rename("SolarRad_MJ_m2_day")
    )

    chirps = ee.ImageCollection(CHIRPS_COLLECTION_ID).select("precipitation")
    precip_period = (
        chirps
        .filterDate(period_start, period_end)
        .sum()
        .rename("Precip_period_mm")
    )
    precip_previous = (
        chirps
        .filterDate(period_start.advance(-30, "day"), period_start)
        .sum()
        .rename("Precip_prev30d_mm")
    )

    return (
        tair_mean
        .addBands(tair_max)
        .addBands(vpd_mean)
        .addBands(solar_mean_daily)
        .addBands(wind_mean)
        .addBands(precip_period)
        .addBands(precip_previous)
        .select(METEOROLOGICAL_MODEL_BANDS)
        .clip(processing_geometry)
        .toFloat()
    )


def build_harmonic_predictors(
    period_start_text: str,
    processing_geometry: ee.Geometry,
) -> ee.Image:
    """Build constant annual harmonics using the exact local model formula."""
    day_of_year = date.fromisoformat(period_start_text).timetuple().tm_yday

    values = []
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * day_of_year / 365.25
        values.extend([np.sin(angle), np.cos(angle)])

    return (
        ee.Image.constant(values)
        .rename(HARMONIC_FEATURES)
        .clip(processing_geometry)
        .toFloat()
    )


def build_production_stack(
    period_start_text: str,
    basin_geometry: ee.Geometry,
) -> dict[str, object]:
    """Build all 25 model predictors without inventing fine meteorology."""
    processing_geometry = basin_geometry.buffer(PROCESSING_BUFFER_M)
    study_features = build_study_feature_collection(processing_geometry)
    fine_projection = get_fine_projection()

    modis = build_modis_period_context(
        period_start_text,
        processing_geometry,
    )

    optical, optical_period = build_s2_common_predictors(
        study_features=study_features,
        period_start=modis["period_start"],
        period_end=modis["period_end"],
        processing_geometry=processing_geometry,
        fine_projection=fine_projection,
    )

    s1, s1_period = build_s1_common_predictors(
        study_features=study_features,
        period_start=modis["period_start"],
        period_end=modis["period_end"],
        processing_geometry=processing_geometry,
        fine_projection=fine_projection,
    )

    meteorology = build_period_meteorological_predictors(
        period_start=modis["period_start"],
        period_end=modis["period_end"],
        number_days=modis["number_days"],
        processing_geometry=processing_geometry,
    )

    harmonics = build_harmonic_predictors(
        period_start_text,
        processing_geometry,
    )

    stack = (
        optical
        .addBands(s1)
        .addBands(meteorology)
        .addBands(harmonics)
        .select(COMMON_MODEL_FEATURES)
        .reproject(fine_projection)
        .toFloat()
    )

    return {
        "stack": stack,
        "optical": optical,
        "s1": s1,
        "meteorology": meteorology,
        "harmonics": harmonics,
        "optical_period": optical_period,
        "s1_period": s1_period,
        "fine_projection": fine_projection,
        "processing_geometry": processing_geometry,
        **modis,
    }


def _valid_fraction_at_modis_support(
    image: ee.Image,
    modis_projection: ee.Projection,
    band_name: str,
) -> ee.Image:
    """Area-weighted fraction of fine cells with valid data per MODIS pixel."""
    valid = (
        ee.Image(image)
        .mask()
        .reduce(ee.Reducer.min())
        .unmask(0)
        .rename("valid")
        .toFloat()
    )

    return (
        valid
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=2048,
        )
        .reproject(modis_projection)
        .rename(band_name)
        .toFloat()
    )


def _mean_at_modis_support(
    image: ee.Image,
    modis_projection: ee.Projection,
    band_name: str,
) -> ee.Image:
    """Area-weighted fine-image mean aligned exactly to native MODIS pixels."""
    return (
        ee.Image(image)
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=2048,
        )
        .reproject(modis_projection)
        .rename(band_name)
        .toFloat()
    )


def build_modis_constrained_et(
    kc_raw: ee.Image,
    optical_predictors: ee.Image,
    s1_predictors: ee.Image,
    model_stack: ee.Image,
    modis_et: ee.Image,
    modis_projection: ee.Projection,
    fine_projection: ee.Projection,
    basin_geometry: ee.Geometry,
) -> dict[str, ee.Image]:
    """Reconcile fine Kc spatial structure to the parent MODIS ET mean.

    Eligibility matches training support as closely as possible:
    optical valid fraction >= 90% and Sentinel-1 valid fraction >= 99.9%.
    Fine Kc gaps up to the accepted optical allowance are filled with the
    valid-pixel Kc mean of the same MODIS pixel. This fill is neutral with
    respect to subpixel spatial contrast and introduces no artificial texture.
    """
    kc_raw = ee.Image(kc_raw).rename("Kc_raw").toFloat()
    modis_et = ee.Image(modis_et).rename("ET_MODIS_mm_period").toFloat()

    optical_fraction = _valid_fraction_at_modis_support(
        optical_predictors,
        modis_projection,
        "optical_valid_fraction",
    )
    s1_fraction = _valid_fraction_at_modis_support(
        s1_predictors,
        modis_projection,
        "s1_valid_fraction",
    )
    stack_fraction = _valid_fraction_at_modis_support(
        model_stack,
        modis_projection,
        "model_stack_valid_fraction",
    )

    eligible = (
        optical_fraction.gte(OPTICAL_MIN_COVERAGE_FRACTION)
        .And(s1_fraction.gte(S1_MIN_COVERAGE_FRACTION))
        .And(stack_fraction.gte(OPTICAL_MIN_COVERAGE_FRACTION))
        .And(modis_et.mask().gt(0))
        .rename("coarse_eligible")
    )

    kc_valid_mean = _mean_at_modis_support(
        kc_raw,
        modis_projection,
        "Kc_valid_mean",
    )

    # Fill only masked fine cells; valid model predictions are never altered here.
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

    # The MODIS sinusoidal grid and the 20 m UTM prediction grid are not
    # nested. A single coarse-to-fine scaling step therefore does not
    # reaggregate exactly because boundary fine pixels overlap adjacent
    # MODIS cells. A small fixed number of proportional correction passes
    # brings the parent-pixel means within the accepted numerical tolerance.
    for correction_pass in range(MODIS_RECONCILIATION_PASSES):
        pass_mean = _mean_at_modis_support(
            et_full_support,
            modis_projection,
            f"ET_reaggregated_pass_{correction_pass + 1}",
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

    # Conservation is evaluated before clipping to the basin so boundary
    # MODIS pixels are checked over their full parent-pixel support.
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

    fill_fraction_raw = ee.Image.constant(1).subtract(stack_fraction)
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
        "s1_valid_fraction": s1_fraction,
        "model_stack_valid_fraction": stack_fraction,
        "fine_fill_fraction": fill_fraction,
        "et_reaggregated": et_reaggregated,
        "conservation_error": conservation_error,
    }
