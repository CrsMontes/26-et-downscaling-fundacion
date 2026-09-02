"""Landsat LST availability and descriptive-QA diagnostics.

The distributed product grid is 30 m. Landsat 8 TIRS and Landsat 9 TIRS-2
thermal information has approximately 100 m native support. Nothing in this
module reprojects or represents LST as independent 20 m information.
"""

from __future__ import annotations

from datetime import date, timedelta

import ee

from .availability_diagnostic import get_dynamic_modis_inputs, validate_period
from .modis import get_modis_period_end


L8_COLLECTION_ID = "LANDSAT/LC08/C02/T1_L2"
L9_COLLECTION_ID = "LANDSAT/LC09/C02/T1_L2"
PROCESSING_LEVEL = "L2SP"
LST_SOURCE_BAND = "ST_B10"
ST_QA_BAND = "ST_QA"
LST_SCALE_FACTOR = 0.00341802
LST_OFFSET = 149.0
HISTORICAL_MIN_DN = 293
MAX_UINT16_DN = 65535
DISTRIBUTED_GRID_M = 30
NATIVE_THERMAL_SUPPORT_M_APPROX = 100
THRESHOLDS_PCT = (80.0, 90.0, 99.0)
VIEWS = ("l8_only", "l8_l9_combined")

QA_FILL_BIT = 0
QA_DILATED_CLOUD_BIT = 1
QA_CIRRUS_BIT = 2
QA_CLOUD_BIT = 3
QA_CLOUD_SHADOW_BIT = 4
QA_SNOW_BIT = 5


COMMON_COLUMNS = (
    "station", "station_id", "period_start", "period_end",
    "period_end_exclusive", "period_days", "footprint_area_m2",
    "distributed_grid_m", "native_thermal_support_m_approx",
    "reduction_scale_m", "reduction_crs",
)


def _view_columns(prefix):
    return (
        f"{prefix}_products", f"{prefix}_unique_dates",
        f"{prefix}_acquisition_dates", f"{prefix}_dates_with_valid_lst",
        f"{prefix}_acquisition_present", f"{prefix}_any_valid_lst",
        f"{prefix}_l8_products", f"{prefix}_l9_products",
        f"{prefix}_l8_unique_dates", f"{prefix}_l9_unique_dates",
        f"{prefix}_sensors_present", f"{prefix}_valid_area_m2",
        f"{prefix}_valid_coverage_pct", f"{prefix}_ge_80",
        f"{prefix}_ge_90", f"{prefix}_ge_99",
        f"{prefix}_historical_dn_ge_293_valid_area_m2",
        f"{prefix}_historical_dn_ge_293_coverage_pct",
        f"{prefix}_historical_dn_ge_293_any_valid_lst",
        f"{prefix}_historical_dn_ge_293_ge_80",
        f"{prefix}_historical_dn_ge_293_ge_90",
        f"{prefix}_historical_dn_ge_293_ge_99",
        f"{prefix}_st_qa_count", f"{prefix}_st_qa_mean_k",
        f"{prefix}_st_qa_min_k", f"{prefix}_st_qa_max_k",
        f"{prefix}_st_qa_stddev_k", f"{prefix}_selected_l8_area_m2",
        f"{prefix}_selected_l9_area_m2",
    )


EXPORT_SELECTORS = COMMON_COLUMNS + _view_columns("l8_only") + _view_columns(
    "l8_l9_combined"
)


def annual_partitions(start_date, end_date_exclusive):
    start, end = validate_period(start_date, end_date_exclusive)
    return [
        (max(start, date(year, 1, 1)).isoformat(),
         min(end, date(year + 1, 1, 1)).isoformat())
        for year in range(start.year, (end - timedelta(days=1)).year + 1)
    ]


def split_partition(start_date, end_date_exclusive):
    start, end = validate_period(start_date, end_date_exclusive)
    days = (end - start).days
    if days < 2:
        raise ValueError("A one-day partition cannot be split")
    middle = start + timedelta(days=days // 2)
    return ((start.isoformat(), middle.isoformat()),
            (middle.isoformat(), end.isoformat()))


def _set_metadata(image, sensor):
    image = ee.Image(image)
    return image.set({
        "sensor": sensor,
        "date_key": image.date().format("yyyy-MM-dd"),
    })


def get_landsat_collection(footprints, start_date, end_date_exclusive):
    """Return L8/L9 L2SP scenes, retaining explicit sensor provenance."""
    validate_period(start_date, end_date_exclusive)
    geometry = ee.FeatureCollection(footprints).geometry()

    def collection(collection_id, sensor):
        return (
            ee.ImageCollection(collection_id)
            .filterBounds(geometry)
            .filterDate(start_date, end_date_exclusive)
            .filter(ee.Filter.eq("PROCESSING_LEVEL", PROCESSING_LEVEL))
            .map(lambda image: _set_metadata(image, sensor))
        )

    return collection(L8_COLLECTION_ID, "L8").merge(
        collection(L9_COLLECTION_ID, "L9")
    ).sort("system:time_start")


def _basic_qa_mask(image):
    qa = ee.Image(image).select("QA_PIXEL")
    mask = ee.Image.constant(1)
    for bit in (
        QA_FILL_BIT, QA_DILATED_CLOUD_BIT, QA_CIRRUS_BIT,
        QA_CLOUD_BIT, QA_CLOUD_SHADOW_BIT, QA_SNOW_BIT,
    ):
        mask = mask.And(qa.bitwiseAnd(1 << bit).eq(0))
    saturation_free = ee.Image(image).select("QA_RADSAT").eq(0)
    return mask.And(saturation_free).rename("basic_valid")


def prepare_landsat_lst(image):
    """Prepare primary QA and the documented historical DN sensitivity."""
    image = ee.Image(image)
    dn = image.select(LST_SOURCE_BAND)
    basic = _basic_qa_mask(image).And(dn.mask())
    historical = basic.And(dn.gte(HISTORICAL_MIN_DN)).And(
        dn.lte(MAX_UINT16_DN)
    )
    lst = dn.multiply(LST_SCALE_FACTOR).add(LST_OFFSET).rename("LST")
    st_qa = image.select(ST_QA_BAND).multiply(0.01).rename("ST_QA_K")
    sensor_code = ee.Image.constant(
        ee.Number(ee.Algorithms.If(
            ee.String(image.get("sensor")).compareTo("L8").eq(0), 8, 9
        ))
    ).rename("sensor_code")
    result = (
        lst.updateMask(basic)
        .addBands(st_qa.updateMask(basic))
        .addBands(sensor_code.updateMask(basic))
        .addBands(historical.rename("historical_dn_ge_293_valid").uint8())
        .toFloat()
    )
    return result.copyProperties(
        image, ["system:time_start", "system:index", "sensor", "date_key",
                "PROCESSING_LEVEL", "WRS_PATH", "WRS_ROW"]
    )


def _empty_image():
    return (
        ee.Image.constant([0, 0, 0, 0])
        .rename(["LST", "ST_QA_K", "sensor_code", "historical_dn_ge_293_valid"])
        .updateMask(ee.Image.constant(0)).toFloat()
    )


def build_daily_collection(period_collection, geometry):
    """Mosaic all same-day scenes before temporal medoid selection."""
    period_collection = ee.ImageCollection(period_collection)
    dates = ee.List(period_collection.aggregate_array("date_key")).distinct().sort()

    def daily(date_key):
        date_key = ee.String(date_key)
        same_day = (
            period_collection.filter(ee.Filter.eq("date_key", date_key))
            .sort("system:time_start").map(prepare_landsat_lst)
        )
        mosaic = same_day.mosaic().clip(geometry)
        any_valid = ee.Number(
            mosaic.select("LST").mask().unmask(0).reduceRegion(
                reducer=ee.Reducer.max(), geometry=geometry,
                scale=DISTRIBUTED_GRID_M, maxPixels=1e7, tileScale=4,
            ).get("LST")
        )
        return mosaic.set({
            "date_key": date_key,
            "has_any_valid_lst": any_valid,
            "system:time_start": ee.Image(same_day.first()).get("system:time_start"),
        })

    return ee.ImageCollection.fromImages(dates.map(daily)).sort("system:time_start")


def build_temporal_medoid(period_collection, geometry):
    """Select the observed daily LST nearest the per-pixel temporal median."""
    daily = build_daily_collection(period_collection, geometry)
    safe = daily.merge(ee.ImageCollection([_empty_image()]))
    median = safe.select("LST").median()

    def score(image):
        image = ee.Image(image)
        return image.addBands(
            image.select("LST").subtract(median).abs().multiply(-1).rename("score")
        )

    fallback_source = (
        ee.ImageCollection(L8_COLLECTION_ID)
        .filterBounds(geometry)
        .filter(ee.Filter.eq("PROCESSING_LEVEL", PROCESSING_LEVEL))
        .first()
    )
    reference_source = ee.Image(ee.Algorithms.If(
        ee.ImageCollection(period_collection).size().gt(0),
        ee.ImageCollection(period_collection).first(),
        fallback_source,
    ))
    reference_projection = reference_source.select(LST_SOURCE_BAND).projection()
    return (
        safe.map(score).qualityMosaic("score")
        .select(["LST", "ST_QA_K", "sensor_code", "historical_dn_ge_293_valid"])
        .setDefaultProjection(reference_projection)
        .clip(geometry).toFloat()
        .set({
            "scene_count": ee.ImageCollection(period_collection).size(),
            "day_count": daily.size(),
            "dates_with_valid_lst": daily.aggregate_sum("has_any_valid_lst"),
            "composite_method": "pixelwise_univariate_temporal_medoid",
        })
    )


def _area_and_coverage(mask, geometry, footprint_area):
    area_raw = ee.Image.pixelArea().updateMask(mask).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geometry,
        scale=DISTRIBUTED_GRID_M, maxPixels=1e7, tileScale=4,
    ).get("area")
    area = ee.Number(ee.Algorithms.If(
        ee.Algorithms.IsEqual(area_raw, None), 0, area_raw
    ))
    return area, area.divide(footprint_area).multiply(100)


def _flags(prefix, coverage):
    return {
        f"{prefix}_ge_{int(threshold)}": ee.Number(coverage).gte(threshold).int()
        for threshold in THRESHOLDS_PCT
    }


def _st_qa_stats(medoid, geometry):
    reducer = (
        ee.Reducer.count().combine(ee.Reducer.mean(), sharedInputs=True)
        .combine(ee.Reducer.minMax(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
    )
    return ee.Dictionary(medoid.select("ST_QA_K").reduceRegion(
        reducer=reducer, geometry=geometry, scale=DISTRIBUTED_GRID_M,
        maxPixels=1e7, tileScale=4,
    ))


def _view_properties(collection, geometry, footprint_area, prefix):
    collection = ee.ImageCollection(collection)
    dates = ee.List(collection.aggregate_array("date_key")).distinct().sort()
    l8 = collection.filter(ee.Filter.eq("sensor", "L8"))
    l9 = collection.filter(ee.Filter.eq("sensor", "L9"))
    l8_dates = ee.List(l8.aggregate_array("date_key")).distinct().sort()
    l9_dates = ee.List(l9.aggregate_array("date_key")).distinct().sort()
    sensors = ee.List(collection.aggregate_array("sensor")).distinct().sort()
    medoid = build_temporal_medoid(collection, geometry)
    valid_area, coverage = _area_and_coverage(
        medoid.select("LST").mask(), geometry, footprint_area
    )
    historical_area, historical_coverage = _area_and_coverage(
        medoid.select("historical_dn_ge_293_valid").eq(1), geometry, footprint_area
    )
    selected_l8_area, _ = _area_and_coverage(
        medoid.select("sensor_code").eq(8), geometry, footprint_area
    )
    selected_l9_area, _ = _area_and_coverage(
        medoid.select("sensor_code").eq(9), geometry, footprint_area
    )
    qa = _st_qa_stats(medoid, geometry)
    properties = {
        f"{prefix}_products": collection.size(),
        f"{prefix}_unique_dates": dates.size(),
        f"{prefix}_acquisition_dates": dates.join(";"),
        f"{prefix}_dates_with_valid_lst": medoid.get("dates_with_valid_lst"),
        f"{prefix}_acquisition_present": collection.size().gt(0).int(),
        f"{prefix}_any_valid_lst": coverage.gt(0).int(),
        f"{prefix}_l8_products": l8.size(),
        f"{prefix}_l9_products": l9.size(),
        f"{prefix}_l8_unique_dates": l8_dates.size(),
        f"{prefix}_l9_unique_dates": l9_dates.size(),
        f"{prefix}_sensors_present": sensors.join(";"),
        f"{prefix}_valid_area_m2": valid_area,
        f"{prefix}_valid_coverage_pct": coverage,
        f"{prefix}_historical_dn_ge_293_valid_area_m2": historical_area,
        f"{prefix}_historical_dn_ge_293_coverage_pct": historical_coverage,
        f"{prefix}_historical_dn_ge_293_any_valid_lst": historical_coverage.gt(0).int(),
        f"{prefix}_st_qa_count": qa.get("ST_QA_K_count"),
        f"{prefix}_st_qa_mean_k": qa.get("ST_QA_K_mean"),
        f"{prefix}_st_qa_min_k": qa.get("ST_QA_K_min"),
        f"{prefix}_st_qa_max_k": qa.get("ST_QA_K_max"),
        f"{prefix}_st_qa_stddev_k": qa.get("ST_QA_K_stdDev"),
        f"{prefix}_selected_l8_area_m2": selected_l8_area,
        f"{prefix}_selected_l9_area_m2": selected_l9_area,
    }
    properties.update(_flags(prefix, coverage))
    properties.update(_flags(f"{prefix}_historical_dn_ge_293", historical_coverage))
    return properties, medoid


def build_thermal_availability(
    modis_inputs, landsat_collection, partition_start, partition_end
):
    """Build one row containing both thermal views per station-period."""
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

        def process_station(footprint_index):
            footprint = ee.Feature(footprint_list.get(footprint_index))
            geometry = footprint.geometry()
            footprint_area = ee.Number(footprint.get("footprint_area_m2"))
            period = (
                ee.ImageCollection(landsat_collection)
                .filterDate(period_start, period_end).filterBounds(geometry)
            )
            l8_only = period.filter(ee.Filter.eq("sensor", "L8"))
            l8_values, l8_medoid = _view_properties(
                l8_only, geometry, footprint_area, "l8_only"
            )
            combined_values, _ = _view_properties(
                period, geometry, footprint_area, "l8_l9_combined"
            )
            properties = {
                "station": footprint.get("station"),
                "station_id": footprint.get("station_id"),
                "period_start": period_start.format("yyyy-MM-dd"),
                "period_end": period_end.advance(-1, "day").format("yyyy-MM-dd"),
                "period_end_exclusive": period_end.format("yyyy-MM-dd"),
                "period_days": period_days,
                "footprint_area_m2": footprint_area,
                "distributed_grid_m": DISTRIBUTED_GRID_M,
                "native_thermal_support_m_approx": NATIVE_THERMAL_SUPPORT_M_APPROX,
                "reduction_scale_m": DISTRIBUTED_GRID_M,
                "reduction_crs": l8_medoid.select("LST").projection().crs(),
            }
            properties.update(l8_values)
            properties.update(combined_values)
            return ee.Feature(None, properties)

        return footprint_indexes.map(process_station)

    return ee.FeatureCollection(image_indexes.map(process_period).flatten())


def configuration_manifest():
    return {
        "collections": {"L8": L8_COLLECTION_ID, "L9": L9_COLLECTION_ID},
        "processing_level": PROCESSING_LEVEL,
        "band": LST_SOURCE_BAND,
        "scale_factor": LST_SCALE_FACTOR,
        "offset": LST_OFFSET,
        "units": "K",
        "basic_qa_pixel_bits_excluded": [0, 1, 2, 3, 4, 5],
        "qa_radsat_required_zero": True,
        "water_retained": True,
        "primary_dn_minimum": None,
        "historical_dn_sensitivity_minimum": HISTORICAL_MIN_DN,
        "historical_dn_minimum_is_final_methodology": False,
        "st_qa_used_as_filter": False,
        "st_qa_exported_descriptively": True,
        "distributed_grid_m": DISTRIBUTED_GRID_M,
        "native_thermal_support_m_approx": NATIVE_THERMAL_SUPPORT_M_APPROX,
        "reprojected_to_20m": False,
        "views": ["L8_ONLY", "L8_L9_COMBINED"],
        "training_performed": False,
    }
