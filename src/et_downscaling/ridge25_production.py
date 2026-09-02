"""Production-stack builder for the accepted Ridge-25 Kc model.

This module builds only the predictors required by Ridge-25:
16 Sentinel-2 optical variables, five ERA5-Land variables, and four temporal
harmonics. Sentinel-1 and CHIRPS are deliberately not queried because they are
not predictors of the accepted model.
"""

from __future__ import annotations

import ee

from .optical import build_optical_predictors, filter_optical_period
from .production import (
    ERA5_COLLECTION_ID,
    PROCESSING_BUFFER_M,
    _prepare_era5_hourly_predictors,
    build_harmonic_predictors,
    build_modis_period_context,
    build_study_feature_collection,
    get_fine_projection,
)
from .ridge25 import (
    RIDGE25_HARMONIC_FEATURES,
    RIDGE25_METEOROLOGICAL_FEATURES,
    RIDGE25_MODEL_FEATURES,
    RIDGE25_OPTICAL_FEATURES,
)
from .sentinel2 import get_sentinel2_collection


RIDGE25_OPTICAL_SOURCE_BANDS = [
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
    "RedEdge1",
    "RedEdge2",
    "RedEdge3",
    "NIR_Broad",
    "NDRE",
]


def build_s2_ridge25_predictors(
    study_features: ee.FeatureCollection,
    period_start: ee.Date,
    period_end: ee.Date,
    processing_geometry: ee.Geometry,
    fine_projection: ee.Projection,
) -> tuple[ee.Image, ee.ImageCollection]:
    """Build the 16 Sentinel-2 predictors used by Ridge-25."""
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

    optical = (
        predictors
        .select(
            RIDGE25_OPTICAL_SOURCE_BANDS,
            RIDGE25_OPTICAL_FEATURES,
        )
        .reproject(fine_projection)
        .toFloat()
    )
    return optical, period_collection


def build_ridge25_meteorological_predictors(
    period_start: ee.Date,
    period_end: ee.Date,
    number_days: ee.Number,
    processing_geometry: ee.Geometry,
) -> ee.Image:
    """Build the five ERA5-Land fields required by Ridge-25."""
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

    return (
        tair_mean
        .addBands(tair_max)
        .addBands(vpd_mean)
        .addBands(solar_mean_daily)
        .addBands(wind_mean)
        .select(RIDGE25_METEOROLOGICAL_FEATURES)
        .clip(processing_geometry)
        .toFloat()
    )


def build_ridge25_production_stack(
    period_start_text: str,
    basin_geometry: ee.Geometry,
) -> dict[str, object]:
    """Build the exact 25-band production stack for Ridge-25."""
    processing_geometry = basin_geometry.buffer(PROCESSING_BUFFER_M)
    study_features = build_study_feature_collection(processing_geometry)
    fine_projection = get_fine_projection()

    modis = build_modis_period_context(
        period_start_text,
        processing_geometry,
    )
    optical, optical_period = build_s2_ridge25_predictors(
        study_features=study_features,
        period_start=modis["period_start"],
        period_end=modis["period_end"],
        processing_geometry=processing_geometry,
        fine_projection=fine_projection,
    )
    meteorology = build_ridge25_meteorological_predictors(
        period_start=modis["period_start"],
        period_end=modis["period_end"],
        number_days=modis["number_days"],
        processing_geometry=processing_geometry,
    )
    harmonics = build_harmonic_predictors(
        period_start_text,
        processing_geometry,
    ).select(RIDGE25_HARMONIC_FEATURES)

    stack = (
        optical
        .addBands(meteorology)
        .addBands(harmonics)
        .select(RIDGE25_MODEL_FEATURES)
        .reproject(fine_projection)
        .toFloat()
    )

    return {
        "stack": stack,
        "optical": optical,
        "meteorology": meteorology,
        "harmonics": harmonics,
        "optical_period": optical_period,
        "fine_projection": fine_projection,
        "processing_geometry": processing_geometry,
        **modis,
    }
