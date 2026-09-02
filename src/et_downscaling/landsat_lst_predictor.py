"""Landsat L8/L9 surface-temperature predictor extraction.

The source thermal signal is distributed on the Landsat 30 m grid and has an
approximately 100 m effective thermal support. LST is bilinearly sampled onto
the established 20 m UTM modelling grid only to align grids; this operation
does not create independent 20 m thermal information.
"""

from __future__ import annotations

import ee

from .config import ANALYSIS_CRS
from .modis import get_modis_period_end
from .thermal_availability import (
    DISTRIBUTED_GRID_M,
    NATIVE_THERMAL_SUPPORT_M_APPROX,
    _st_qa_stats,
    build_temporal_medoid,
)


FINE_GRID_M = 20
COMPOSITE_METHOD = "pixelwise_univariate_temporal_medoid"
RESAMPLING_METHOD = "bilinear"
AGGREGATION_METHOD = "valid_area_weighted_mean_on_20m_grid"

EXPORT_SELECTORS = (
    "station", "station_id", "modis_pixel_id", "period_start", "period_end",
    "period_end_exclusive", "period_days", "footprint_area_m2",
    "LST_parent_mean_K", "LST_valid_count_20m", "LST_valid_area_m2",
    "LST_valid_coverage_pct", "landsat_products", "landsat_unique_dates",
    "landsat_acquisition_dates", "landsat_dates_with_valid_lst",
    "l8_products", "l9_products", "l8_unique_dates", "l9_unique_dates",
    "sensors_present", "ST_QA_count_30m", "ST_QA_mean_K_30m",
    "ST_QA_min_K_30m", "ST_QA_max_K_30m", "ST_QA_stddev_K_30m",
    "working_grid_crs", "working_grid_m", "distributed_grid_m",
    "native_thermal_support_m_approx", "resampling_method",
    "composite_method", "footprint_aggregation_method",
)


def get_fine_projection():
    """Return the same explicit 20 m UTM grid used by production."""
    return ee.Projection(ANALYSIS_CRS).atScale(FINE_GRID_M)


def regrid_lst_to_fine_grid(medoid):
    """Bilinearly align continuous LST to the fixed 20 m modelling grid."""
    return (
        ee.Image(medoid).select("LST").resample(RESAMPLING_METHOD)
        .reproject(get_fine_projection()).rename("LST_K").toFloat()
    )


def _safe_number(value, fallback=0):
    return ee.Number(
        ee.Algorithms.If(ee.Algorithms.IsEqual(value, None), fallback, value)
    )


def area_weighted_parent_stats(lst_20m, geometry, footprint_area):
    """Return valid-area-weighted LST statistics at MODIS-parent support."""
    lst_20m = ee.Image(lst_20m)
    projection = get_fine_projection()
    valid_area = ee.Image.pixelArea().updateMask(lst_20m.mask()).rename("area")
    weighted = lst_20m.multiply(valid_area).rename("weighted_lst")
    sums = ee.Dictionary(
        weighted.addBands(valid_area).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geometry, crs=projection,
            scale=FINE_GRID_M, maxPixels=1e7, tileScale=4,
        )
    )
    area = _safe_number(sums.get("area"))
    weighted_sum = _safe_number(sums.get("weighted_lst"))
    count = _safe_number(lst_20m.reduceRegion(
        reducer=ee.Reducer.count(), geometry=geometry, crs=projection,
        scale=FINE_GRID_M, maxPixels=1e7, tileScale=4,
    ).get("LST_K"))
    return {
        "LST_parent_mean_K": ee.Algorithms.If(
            area.gt(0), weighted_sum.divide(area), None
        ),
        "LST_valid_count_20m": count,
        "LST_valid_area_m2": area,
        "LST_valid_coverage_pct": area.divide(footprint_area).multiply(100),
    }


def _period_properties(collection, medoid, geometry, footprint_area):
    collection = ee.ImageCollection(collection)
    dates = ee.List(collection.aggregate_array("date_key")).distinct().sort()
    l8 = collection.filter(ee.Filter.eq("sensor", "L8"))
    l9 = collection.filter(ee.Filter.eq("sensor", "L9"))
    qa = _st_qa_stats(medoid, geometry)
    properties = area_weighted_parent_stats(
        regrid_lst_to_fine_grid(medoid), geometry, footprint_area
    )
    properties.update({
        "landsat_products": collection.size(),
        "landsat_unique_dates": dates.size(),
        "landsat_acquisition_dates": dates.join(";"),
        "landsat_dates_with_valid_lst": medoid.get("dates_with_valid_lst"),
        "l8_products": l8.size(),
        "l9_products": l9.size(),
        "l8_unique_dates": ee.List(l8.aggregate_array("date_key")).distinct().size(),
        "l9_unique_dates": ee.List(l9.aggregate_array("date_key")).distinct().size(),
        "sensors_present": ee.List(
            collection.aggregate_array("sensor")
        ).distinct().sort().join(";"),
        "ST_QA_count_30m": qa.get("ST_QA_K_count"),
        "ST_QA_mean_K_30m": qa.get("ST_QA_K_mean"),
        "ST_QA_min_K_30m": qa.get("ST_QA_K_min"),
        "ST_QA_max_K_30m": qa.get("ST_QA_K_max"),
        "ST_QA_stddev_K_30m": qa.get("ST_QA_K_stdDev"),
    })
    return properties


def build_landsat_lst_predictor(
    modis_inputs, landsat_collection, partition_start, partition_end
):
    """Build one uniquely keyed LST row per station and MODIS period."""
    modis = ee.ImageCollection(modis_inputs["collection"]).filterDate(
        partition_start, partition_end
    )
    images = modis.toList(modis.size())
    image_indexes = ee.List.sequence(0, modis.size().subtract(1))
    footprints = ee.FeatureCollection(modis_inputs["station_footprints"])
    footprint_list = footprints.toList(footprints.size())
    footprint_indexes = ee.List.sequence(0, footprints.size().subtract(1))

    def process_period(image_index):
        image = ee.Image(images.get(image_index))
        period_start = image.date()
        period_end = get_modis_period_end(period_start)
        period_days = period_end.difference(period_start, "day")

        def process_footprint(footprint_index):
            footprint = ee.Feature(footprint_list.get(footprint_index))
            geometry = footprint.geometry()
            footprint_area = ee.Number(footprint.get("footprint_area_m2"))
            period = (
                ee.ImageCollection(landsat_collection)
                .filterDate(period_start, period_end).filterBounds(geometry)
            )
            medoid = build_temporal_medoid(period, geometry)
            properties = {
                "station": footprint.get("station"),
                "station_id": footprint.get("station_id"),
                "modis_pixel_id": footprint.get("modis_pixel_id"),
                "period_start": period_start.format("yyyy-MM-dd"),
                "period_end": period_end.advance(-1, "day").format("yyyy-MM-dd"),
                "period_end_exclusive": period_end.format("yyyy-MM-dd"),
                "period_days": period_days,
                "footprint_area_m2": footprint_area,
                "working_grid_crs": ANALYSIS_CRS,
                "working_grid_m": FINE_GRID_M,
                "distributed_grid_m": DISTRIBUTED_GRID_M,
                "native_thermal_support_m_approx": NATIVE_THERMAL_SUPPORT_M_APPROX,
                "resampling_method": RESAMPLING_METHOD,
                "composite_method": COMPOSITE_METHOD,
                "footprint_aggregation_method": AGGREGATION_METHOD,
            }
            properties.update(
                _period_properties(period, medoid, geometry, footprint_area)
            )
            return ee.Feature(None, properties)

        return footprint_indexes.map(process_footprint)

    return ee.FeatureCollection(image_indexes.map(process_period).flatten())


def configuration_manifest():
    return {
        "predictor": "LST_parent_mean_K",
        "fine_grid_crs": ANALYSIS_CRS,
        "fine_grid_m": FINE_GRID_M,
        "source_distributed_grid_m": DISTRIBUTED_GRID_M,
        "native_thermal_support_m_approx": NATIVE_THERMAL_SUPPORT_M_APPROX,
        "resampling_method": RESAMPLING_METHOD,
        "composite_method": COMPOSITE_METHOD,
        "footprint_aggregation_method": AGGREGATION_METHOD,
        "fine_scale_interpretation": (
            "Grid alignment only; thermal information retains approximately "
            "100 m effective support."
        ),
    }
