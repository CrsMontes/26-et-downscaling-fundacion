"""Shared 20 m production utilities for the final RF-25 workflow."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import ee
import numpy as np

from .config import ANALYSIS_CRS
from .modis import get_modis_collection, get_modis_period_end, prepare_modis_et
from .rf25 import RF25_HARMONIC_FEATURES


PREDICTION_SCALE_M = 20
PROCESSING_BUFFER_M = 1000
MODIS_CONSERVATION_TOLERANCE_MM = 0.01
ERA5_COLLECTION_ID = "ECMWF/ERA5_LAND/HOURLY"


def load_basin_geometry(project_root: Path) -> ee.Geometry:
    """Load the versioned Fundación basin GeoJSON as an Earth Engine geometry."""
    path = Path(project_root) / "data" / "boundaries" / "fundacion_basin.geojson"
    if not path.is_file():
        raise FileNotFoundError(f"Basin boundary not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    geojson_type = data.get("type")
    if geojson_type == "FeatureCollection":
        features = data.get("features", [])
        if len(features) != 1:
            raise ValueError("Fundación basin GeoJSON must contain exactly one feature.")
        geometry = features[0].get("geometry")
    elif geojson_type == "Feature":
        geometry = data.get("geometry")
    else:
        geometry = data
    if not isinstance(geometry, dict) or "type" not in geometry:
        raise ValueError("Invalid basin geometry in GeoJSON.")
    return ee.Geometry(geometry)


def build_study_feature_collection(geometry: ee.Geometry) -> ee.FeatureCollection:
    """Wrap a geometry for collection functions reused from the extraction pipeline."""
    return ee.FeatureCollection([ee.Feature(geometry)])


def get_fine_projection() -> ee.Projection:
    """Return the explicit 20 m UTM prediction grid."""
    return ee.Projection(ANALYSIS_CRS).atScale(PREDICTION_SCALE_M)


def build_modis_period_context(
    period_start_text: str,
    processing_geometry: ee.Geometry,
) -> dict[str, object]:
    """Build the native MODIS target image and exact 8-day temporal support."""
    requested_start = ee.Date(period_start_text)
    collection = get_modis_collection().filterDate(
        requested_start,
        requested_start.advance(1, "day"),
    )
    image = ee.Image(collection.first())
    period_start = image.date()
    period_end = get_modis_period_end(period_start)
    number_days = period_end.difference(period_start, "day")
    prepared = prepare_modis_et(image, number_days)
    modis_et = prepared.select("ET_mm_period").toFloat()
    return {
        "collection": collection,
        "source_image": image,
        "period_start": period_start,
        "period_end": period_end,
        "number_days": number_days,
        "modis_et": modis_et,
        "modis_projection": image.select("ET").projection(),
        "processing_geometry": processing_geometry,
    }


def _saturation_vapor_pressure_kpa(temperature_c: ee.Image) -> ee.Image:
    temperature_c = ee.Image(temperature_c)
    return (
        temperature_c.multiply(17.27)
        .divide(temperature_c.add(237.3))
        .exp()
        .multiply(0.6108)
    )


def _prepare_era5_hourly_predictors(image: ee.Image) -> ee.Image:
    """Derive hourly ERA5-Land fields used by the final meteorological predictors."""
    image = ee.Image(image)
    tair_c = image.select("temperature_2m").subtract(273.15).rename("Tair_C")
    tdew_c = image.select("dewpoint_temperature_2m").subtract(273.15).rename("Tdew_C")
    vpd_raw = _saturation_vapor_pressure_kpa(tair_c).subtract(
        _saturation_vapor_pressure_kpa(tdew_c)
    )
    vpd = vpd_raw.where(vpd_raw.lt(0), 0).rename("VPD_kPa")
    wind = (
        image.select("u_component_of_wind_10m").pow(2)
        .add(image.select("v_component_of_wind_10m").pow(2))
        .sqrt()
        .rename("Wind_ms")
    )
    solar_j = image.select("surface_solar_radiation_downwards_hourly")
    solar = solar_j.where(solar_j.lt(0), 0).multiply(1e-6).rename("SolarRad_MJ_m2_hour")
    prepared = tair_c.addBands(vpd).addBands(wind).addBands(solar)
    return ee.Image(prepared.copyProperties(image, ["system:time_start"])).toFloat()


def build_harmonic_predictors(
    period_start_text: str,
    processing_geometry: ee.Geometry,
) -> ee.Image:
    """Build the two annual harmonic pairs used by RF-25."""
    day_of_year = date.fromisoformat(period_start_text).timetuple().tm_yday
    values = []
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * day_of_year / 365.25
        values.extend([np.sin(angle), np.cos(angle)])
    return (
        ee.Image.constant(values)
        .rename(RF25_HARMONIC_FEATURES)
        .clip(processing_geometry)
        .toFloat()
    )
