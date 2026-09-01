"""Isolated builders for the 2020-2024 paired S2/HLS experiment.

Earth Engine exposes HLS reflectance bands already unpacked.  This module
therefore never applies an additional reflectance scale or offset.
"""

from __future__ import annotations

from datetime import date

import ee

from .availability_diagnostic import (
    _base_properties,
    _period_context,
    _period_values,
    filter_dynamic_hls_to_geometry,
)
from .hls import (
    HLS_MEDOID_SCORE_BANDS,
    build_empty_hls_image,
    build_hls_daily_collection,
)
from .schema import COMMON_OPTICAL_MODEL_BANDS
from .sentinel2 import build_s2_medoid


START_DATE = "2020-01-01"
END_DATE_EXCLUSIVE = "2025-01-01"
PERIOD_LABEL = "2020_2024"
THRESHOLDS = (80, 90, 99)
COMMON_REFLECTANCE_BANDS = ("Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2")
COMMON_INDEX_BANDS = ("NDVI", "EVI", "SAVI", "NDWI", "NDMI")
COMMON_PREDICTORS = tuple(COMMON_OPTICAL_MODEL_BANDS)
HLS_SOURCE_CODES = {"S30": 30, "L30": 20}


def validate_context(start_date, end_date_exclusive, period_label):
    if (start_date, end_date_exclusive, period_label) != (
        START_DATE, END_DATE_EXCLUSIVE, PERIOD_LABEL
    ):
        raise ValueError("Phase 3A is confined to the approved 2020_2024 context")


def _safe_ratio(numerator, denominator, name):
    return (
        numerator.divide(denominator)
        .updateMask(denominator.abs().gt(1e-6))
        .rename(name)
        .toFloat()
    )


def add_common_indices(image):
    """Add the five identically defined indices used for both sources."""
    image = ee.Image(image)
    blue = image.select("Blue")
    green = image.select("Green")
    red = image.select("Red")
    nir = image.select("NIR")
    swir1 = image.select("SWIR1")
    indices = [
        _safe_ratio(nir.subtract(red), nir.add(red), "NDVI"),
        _safe_ratio(
            nir.subtract(red).multiply(2.5),
            nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1),
            "EVI",
        ),
        _safe_ratio(
            nir.subtract(red).multiply(1.5), nir.add(red).add(0.5), "SAVI"
        ),
        _safe_ratio(green.subtract(nir), green.add(nir), "NDWI"),
        _safe_ratio(nir.subtract(swir1), nir.add(swir1), "NDMI"),
    ]
    return image.addBands(indices).select(COMMON_PREDICTORS).toFloat()


def build_hls_medoid_with_provenance(period_collection, geometry):
    """Reproduce the HLS medoid while retaining the selected sensor per pixel."""
    daily = build_hls_daily_collection(period_collection, geometry)

    def add_source_band(image):
        image = ee.Image(image)
        source_code = ee.Number(
            ee.Algorithms.If(
                ee.String(image.get("sensor")).compareTo("S30").eq(0),
                HLS_SOURCE_CODES["S30"], HLS_SOURCE_CODES["L30"],
            )
        )
        band = (
            ee.Image.constant(source_code)
            .rename("hls_source_code")
            .updateMask(image.select("Blue").mask())
            .uint8()
        )
        return image.addBands(band)

    prepared = daily.map(add_source_band)
    empty = build_empty_hls_image().addBands(
        ee.Image.constant(0).rename("hls_source_code").updateMask(ee.Image.constant(0))
    )
    safe = prepared.merge(ee.ImageCollection([empty]))
    spectral_median = safe.select(HLS_MEDOID_SCORE_BANDS).median()

    def score(image):
        image = ee.Image(image)
        distance = (
            image.select(HLS_MEDOID_SCORE_BANDS)
            .subtract(spectral_median).pow(2).reduce(ee.Reducer.sum())
        )
        return image.addBands(distance.multiply(-1).rename("medoid_score"))

    return (
        safe.map(score).qualityMosaic("medoid_score")
        .select([*COMMON_REFLECTANCE_BANDS, "hls_source_code"])
        .toFloat()
    )


def _mean_properties(image, geometry, scale, prefix):
    defaults = ee.Dictionary.fromLists(
        [f"{prefix}_{name}_mean" for name in COMMON_PREDICTORS],
        ee.List.repeat(-9999, len(COMMON_PREDICTORS)),
    )
    values = ee.Dictionary(
        ee.Image(image).select(COMMON_PREDICTORS).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry, scale=scale,
            maxPixels=1_000_000, tileScale=4,
        )
    )
    renamed = ee.Dictionary.fromLists(
        [f"{prefix}_{name}_mean" for name in COMMON_PREDICTORS],
        [values.get(name, -9999) for name in COMMON_PREDICTORS],
    )
    return defaults.combine(renamed, True)


def _selected_area(source_band, source_code, geometry):
    value = (
        ee.Image.pixelArea().updateMask(source_band.eq(source_code))
        .reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geometry, scale=30,
            maxPixels=1_000_000, tileScale=4,
        ).get("area")
    )
    return ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(value, None), 0, value))


def build_paired_optical_table(
    modis_inputs, s2_collection, hls_collection, partition_start, partition_end
):
    """Build one wide S2/HLS predictor row per station and MODIS period."""
    images, image_indexes, footprints, footprint_indexes = _period_context(
        modis_inputs, partition_start, partition_end
    )

    def process_image(image_index):
        period_start, period_end, period_days = _period_values(images.get(image_index))

        def process_footprint(footprint_index):
            footprint = ee.Feature(footprints.get(footprint_index))
            geometry = footprint.geometry()
            s2_period = (
                ee.ImageCollection(s2_collection)
                .filterDate(period_start, period_end).filterBounds(geometry)
            )
            local_hls = filter_dynamic_hls_to_geometry(hls_collection, geometry)
            hls_period = local_hls.filterDate(period_start, period_end)

            s2_predictors = add_common_indices(
                build_s2_medoid(s2_period, geometry).select(COMMON_REFLECTANCE_BANDS)
            )
            hls_medoid = build_hls_medoid_with_provenance(hls_period, geometry)
            hls_predictors = add_common_indices(hls_medoid.select(COMMON_REFLECTANCE_BANDS))
            source_band = hls_medoid.select("hls_source_code")
            footprint_area = ee.Number(geometry.area(maxError=1))
            s30_area = _selected_area(source_band, HLS_SOURCE_CODES["S30"], geometry)
            l30_area = _selected_area(source_band, HLS_SOURCE_CODES["L30"], geometry)

            properties = _base_properties(
                footprint, period_start, period_end, period_days
            )
            properties.update({
                "modis_pixel_id": footprint.get("modis_pixel_id"),
                "footprint_area_m2": footprint_area,
                "s2_products": s2_period.size(),
                "s2_unique_dates": ee.List(s2_period.aggregate_array("date_key")).distinct().size(),
                "hls_s30_products": hls_period.filter(ee.Filter.eq("sensor", "S30")).size(),
                "hls_l30_products": hls_period.filter(ee.Filter.eq("sensor", "L30")).size(),
                "hls_s30_unique_dates": ee.List(hls_period.filter(ee.Filter.eq("sensor", "S30")).aggregate_array("date_key")).distinct().size(),
                "hls_l30_unique_dates": ee.List(hls_period.filter(ee.Filter.eq("sensor", "L30")).aggregate_array("date_key")).distinct().size(),
                "hls_local_mgrs_tiles": ee.List(local_hls.get("local_mgrs_tiles")).join(";"),
                "hls_selected_s30_area_m2": s30_area,
                "hls_selected_l30_area_m2": l30_area,
                "hls_selected_s30_pct": s30_area.divide(footprint_area).multiply(100),
                "hls_selected_l30_pct": l30_area.divide(footprint_area).multiply(100),
                "hls_reflectance_already_scaled_by_ee": True,
                "additional_hls_scaling_applied": False,
            })
            return (
                ee.Feature(None, properties)
                .set(_mean_properties(s2_predictors, geometry, 20, "s2"))
                .set(_mean_properties(hls_predictors, geometry, 30, "hls"))
            )

        return footprint_indexes.map(process_footprint)

    return ee.FeatureCollection(image_indexes.map(process_image).flatten())


BASE_SELECTORS = (
    "station", "station_id", "modis_pixel_id", "period_start", "period_end",
    "period_end_exclusive", "period_days", "footprint_area_m2",
    "s2_products", "s2_unique_dates", "hls_s30_products", "hls_l30_products",
    "hls_s30_unique_dates", "hls_l30_unique_dates", "hls_local_mgrs_tiles",
    "hls_selected_s30_area_m2", "hls_selected_l30_area_m2",
    "hls_selected_s30_pct", "hls_selected_l30_pct",
    "hls_reflectance_already_scaled_by_ee", "additional_hls_scaling_applied",
)
EXPORT_SELECTORS = BASE_SELECTORS + tuple(
    f"{prefix}_{name}_mean" for prefix in ("s2", "hls") for name in COMMON_PREDICTORS
)


def expected_rows():
    return 5 * 230


def experiment_configuration():
    return {
        "start_date": START_DATE,
        "end_date_exclusive": END_DATE_EXCLUSIVE,
        "period_label": PERIOD_LABEL,
        "common_predictors": list(COMMON_PREDICTORS),
        "thresholds_pct": list(THRESHOLDS),
        "s2_reduction_scale_m": 20,
        "hls_reduction_scale_m": 30,
        "hls_values_already_scaled_by_earth_engine": True,
        "additional_hls_reflectance_scaling_applied": False,
        "training_performed": False,
        "aoa_di_performed": False,
    }
