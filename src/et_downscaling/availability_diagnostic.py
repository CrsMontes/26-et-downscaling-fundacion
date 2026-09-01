"""Earth Engine builders for source-availability diagnostics.

This module deliberately produces metadata and continuous coverage only.  It
does not build predictor tables, calibrate dataset-derived values, or train a
model.  All analysis dates are explicit arguments so the production defaults
remain untouched.
"""

from __future__ import annotations

from datetime import date, timedelta

import ee

from .config import (
    ANALYSIS_CRS,
    MODIS_ET_MAX_VALID_DN,
    MODIS_ET_MIN_VALID_DN,
    MODIS_ET_SCALE_FACTOR,
    MODIS_REQUIRE_STRICT_QC,
    MODIS_STRICT_SCF_MAX,
    S1_FOOTPRINT_REDUCTION_SCALE_M,
    S2_CLEAR_THRESHOLD,
    S2_QA_BAND,
)
from .hls import (
    HLS_L30_COLLECTION_ID,
    HLS_S30_COLLECTION_ID,
    _add_hls_mgrs_tile,
    _set_hls_metadata,
    build_hls_daily_collection,
)
from .modis import (
    MODIS_COLLECTION_ID,
    assign_station_footprints,
    build_modis_grid,
    build_modis_pixel_id,
    get_modis_period_end,
    get_modis_projection,
    get_modis_scale,
    prepare_modis_et,
)
from .sentinel1 import S1_COLLECTION_ID, build_s1_median, get_s1_coverage
from .sentinel2 import (
    CLOUD_SCORE_COLLECTION_ID,
    S2_COLLECTION_ID,
    build_s2_daily_collection,
)
from .spatial import get_coverage_fraction
from .stations import get_station_collection


OPTICAL_THRESHOLDS_PCT = (80.0, 90.0, 99.0)
HLS_VARIANTS = ("S30", "L30", "COMBINED")


MODIS_SELECTORS = (
    "station", "station_id", "modis_pixel_id", "period_start",
    "period_end", "period_end_exclusive", "period_days", "ET_mm_period",
    "ET_mm_day", "modis_value_valid", "modis_good", "ET_QC",
    "modis_qc_present", "modis_qc_good", "modis_modland_qc",
    "modis_sensor", "modis_dead_detector", "modis_cloud_state",
    "modis_scf_qc",
)

S2_SELECTORS = (
    "station", "station_id", "period_start", "period_end",
    "period_end_exclusive", "period_days", "source", "working_scale_m",
    "products", "unique_dates", "acquisition_dates", "tiles",
    "continuous_valid_coverage_pct", "ge_80", "ge_90", "ge_99",
)

HLS_SELECTORS = (
    "station", "station_id", "period_start", "period_end",
    "period_end_exclusive", "period_days", "local_mgrs_tiles",
    "s30_products", "s30_unique_dates", "s30_acquisition_dates",
    "s30_tiles", "s30_continuous_valid_coverage_pct", "s30_ge_80",
    "s30_ge_90", "s30_ge_99", "l30_products", "l30_unique_dates",
    "l30_acquisition_dates", "l30_tiles",
    "l30_continuous_valid_coverage_pct", "l30_ge_80", "l30_ge_90",
    "l30_ge_99", "combined_products", "combined_unique_dates",
    "combined_acquisition_dates", "combined_tiles",
    "combined_continuous_valid_coverage_pct", "combined_ge_80",
    "combined_ge_90", "combined_ge_99",
)

S1_INVENTORY_SELECTORS = (
    "pass", "relative_orbit", "scenes", "unique_dates", "first_date",
    "last_date", "stations_intersected", "covers_all_stations",
)

S1_PERIOD_SELECTORS = (
    "station", "station_id", "period_start", "period_end",
    "period_end_exclusive", "period_days", "pass", "relative_orbit",
    "products", "unique_dates", "acquisition_dates", "has_acquisition",
    "continuous_valid_vv_vh_coverage_pct", "has_valid_vv_vh_coverage",
    "angle_count", "angle_mean_deg", "angle_min_deg", "angle_max_deg",
    "angle_stddev_deg",
)


def validate_period(start_date: str, end_date_exclusive: str) -> tuple[date, date]:
    """Validate and return an inclusive/exclusive ISO analysis period."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date_exclusive)
    if end <= start:
        raise ValueError("end_date_exclusive must be later than start_date")
    return start, end


def annual_partitions(start_date: str, end_date_exclusive: str) -> list[tuple[str, str]]:
    """Return clipped annual partitions for a period."""
    start, end = validate_period(start_date, end_date_exclusive)
    result = []
    for year in range(start.year, (end - timedelta(days=1)).year + 1):
        lower = max(start, date(year, 1, 1))
        upper = min(end, date(year + 1, 1, 1))
        if lower < upper:
            result.append((lower.isoformat(), upper.isoformat()))
    return result


def split_partition(start_date: str, end_date_exclusive: str) -> tuple[tuple[str, str], tuple[str, str]]:
    """Bisect a failed partition deterministically on a calendar-day boundary."""
    start, end = validate_period(start_date, end_date_exclusive)
    days = (end - start).days
    if days < 2:
        raise ValueError("A one-day partition cannot be split further")
    midpoint = start + timedelta(days=days // 2)
    return ((start.isoformat(), midpoint.isoformat()), (midpoint.isoformat(), end.isoformat()))


def threshold_flags(coverage_pct):
    """Return EE threshold flags without selecting a threshold."""
    coverage_pct = ee.Number(coverage_pct)
    return {
        f"ge_{int(threshold)}": coverage_pct.gte(threshold).int()
        for threshold in OPTICAL_THRESHOLDS_PCT
    }


def get_dynamic_modis_inputs(start_date: str, end_date_exclusive: str):
    """Build the existing MODIS target and station footprints for explicit dates."""
    validate_period(start_date, end_date_exclusive)
    collection = (
        ee.ImageCollection(MODIS_COLLECTION_ID)
        .filterDate(start_date, end_date_exclusive)
        .select(["ET", "ET_QC"])
        .sort("system:time_start")
    )
    projection = get_modis_projection(collection)
    scale = get_modis_scale(projection)
    pixel_id = build_modis_pixel_id(projection)
    samples = get_station_collection()
    grid = build_modis_grid(samples, pixel_id, projection, scale)
    footprints = assign_station_footprints(
        samples, grid, pixel_id, projection, scale
    )
    return {
        "collection": collection,
        "projection": projection,
        "scale": scale,
        "station_footprints": footprints,
    }


def _period_context(modis_inputs, partition_start, partition_end):
    collection = ee.ImageCollection(modis_inputs["collection"]).filterDate(
        partition_start, partition_end
    )
    images = collection.toList(collection.size())
    image_indexes = ee.List.sequence(0, collection.size().subtract(1))
    footprints = ee.FeatureCollection(modis_inputs["station_footprints"])
    footprint_list = footprints.toList(footprints.size())
    footprint_indexes = ee.List.sequence(0, footprints.size().subtract(1))
    return images, image_indexes, footprint_list, footprint_indexes


def _period_values(modis_image):
    period_start = ee.Image(modis_image).date()
    period_end = get_modis_period_end(period_start)
    period_days = period_end.difference(period_start, "day")
    return period_start, period_end, period_days


def _base_properties(footprint, period_start, period_end, period_days):
    return {
        "station": footprint.get("station"),
        "station_id": footprint.get("station_id"),
        "period_start": period_start.format("yyyy-MM-dd"),
        "period_end": period_end.advance(-1, "day").format("yyyy-MM-dd"),
        "period_end_exclusive": period_end.format("yyyy-MM-dd"),
        "period_days": period_days,
        "system:time_start": period_start.millis(),
    }


def build_modis_availability(modis_inputs, partition_start, partition_end):
    """Build one row per station and MODIS period using existing target QC."""
    images, image_indexes, footprints, footprint_indexes = _period_context(
        modis_inputs, partition_start, partition_end
    )
    projection = modis_inputs["projection"]
    scale = modis_inputs["scale"]

    def process_image(image_index):
        image = ee.Image(images.get(image_index))
        period_start, period_end, period_days = _period_values(image)
        prepared = prepare_modis_et(image, period_days)

        def process_footprint(footprint_index):
            footprint = ee.Feature(footprints.get(footprint_index))
            point = ee.Geometry.Point(
                [footprint.get("longitude"), footprint.get("latitude")]
            )
            target = prepared.reduceRegion(
                reducer=ee.Reducer.first(), geometry=point,
                crs=projection, scale=scale, maxPixels=100,
            )
            properties = _base_properties(
                footprint, period_start, period_end, period_days
            )
            properties["modis_pixel_id"] = footprint.get("modis_pixel_id")
            return ee.Feature(None, properties).set(target)

        return footprint_indexes.map(process_footprint)

    return ee.FeatureCollection(image_indexes.map(process_image).flatten())


def get_dynamic_s2_collection(footprints, start_date, end_date_exclusive):
    geometry = ee.FeatureCollection(footprints).geometry()
    source = (
        ee.ImageCollection(S2_COLLECTION_ID)
        .filterBounds(geometry).filterDate(start_date, end_date_exclusive)
    )
    cloud_score = (
        ee.ImageCollection(CLOUD_SCORE_COLLECTION_ID)
        .filterBounds(geometry).filterDate(start_date, end_date_exclusive)
    )
    return source.linkCollection(cloud_score, [S2_QA_BAND]).map(
        lambda image: ee.Image(image).set(
            "date_key", ee.Image(image).date().format("yyyy-MM-dd")
        )
    )


def _union_coverage(daily_collection, geometry, scale):
    zero = ee.Image.constant(0).rename("valid").uint8()

    def valid_mask(image):
        return (
            ee.Image(image).select(["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"])
            .mask().reduce(ee.Reducer.min()).rename("valid").uint8()
        )

    union = ee.ImageCollection(daily_collection).map(valid_mask).merge(
        ee.ImageCollection([zero])
    ).max()
    return get_coverage_fraction(union, geometry, scale=scale).multiply(100)


def build_s2_availability(
    modis_inputs, s2_collection, partition_start, partition_end
):
    """Build station-period S2 acquisition and continuous coverage metadata."""
    images, image_indexes, footprints, footprint_indexes = _period_context(
        modis_inputs, partition_start, partition_end
    )

    def process_image(image_index):
        period_start, period_end, period_days = _period_values(images.get(image_index))

        def process_footprint(footprint_index):
            footprint = ee.Feature(footprints.get(footprint_index))
            geometry = footprint.geometry()
            period = (
                ee.ImageCollection(s2_collection)
                .filterDate(period_start, period_end).filterBounds(geometry)
            )
            dates = ee.List(period.aggregate_array("date_key")).distinct().sort()
            tiles = ee.List(period.aggregate_array("MGRS_TILE")).distinct().sort()
            daily = build_s2_daily_collection(period, geometry)
            coverage = _union_coverage(daily, geometry, 20)
            properties = _base_properties(
                footprint, period_start, period_end, period_days
            )
            properties.update({
                "source": "S2", "working_scale_m": 20,
                "products": period.size(), "unique_dates": dates.size(),
                "acquisition_dates": dates.join(";"), "tiles": tiles.join(";"),
                "continuous_valid_coverage_pct": coverage,
            })
            properties.update(threshold_flags(coverage))
            return ee.Feature(None, properties)

        return footprint_indexes.map(process_footprint)

    return ee.FeatureCollection(image_indexes.map(process_image).flatten())


def get_dynamic_hls_collection(footprints, start_date, end_date_exclusive):
    geometry = ee.FeatureCollection(footprints).geometry()
    s30 = (
        ee.ImageCollection(HLS_S30_COLLECTION_ID)
        .filterBounds(geometry).filterDate(start_date, end_date_exclusive)
        .map(lambda image: _set_hls_metadata(image, "S30"))
        .map(_add_hls_mgrs_tile)
    )
    l30 = (
        ee.ImageCollection(HLS_L30_COLLECTION_ID)
        .filterBounds(geometry).filterDate(start_date, end_date_exclusive)
        .map(lambda image: _set_hls_metadata(image, "L30"))
        .map(_add_hls_mgrs_tile)
    )
    return (
        s30.merge(l30).sort("system:time_start")
        .set("analysis_start_date", start_date)
        .set("analysis_end_date_exclusive", end_date_exclusive)
    )


def filter_dynamic_hls_to_geometry(collection, geometry):
    """Apply the existing verified-MGRS rule with explicit diagnostic dates."""
    collection = ee.ImageCollection(collection)
    geometry = ee.Geometry(geometry)
    local_tiles = (
        ee.ImageCollection(S2_COLLECTION_ID)
        .filterBounds(geometry)
        .filterDate(
            collection.get("analysis_start_date"),
            collection.get("analysis_end_date_exclusive"),
        )
        .aggregate_array("MGRS_TILE")
    )
    local_tiles = ee.List(local_tiles).distinct().sort()
    return (
        collection
        .filter(ee.Filter.inList("hls_mgrs_tile", local_tiles))
        .filterBounds(geometry)
        .set("local_mgrs_tiles", local_tiles)
    )


def _hls_variant_properties(collection, geometry, variant):
    if variant == "COMBINED":
        selected = ee.ImageCollection(collection)
    else:
        selected = ee.ImageCollection(collection).filter(ee.Filter.eq("sensor", variant))
    dates = ee.List(selected.aggregate_array("date_key")).distinct().sort()
    tiles = ee.List(selected.aggregate_array("hls_mgrs_tile")).distinct().sort()
    daily = build_hls_daily_collection(selected, geometry)
    coverage = _union_coverage(daily, geometry, 30)
    prefix = variant.lower()
    values = {
        f"{prefix}_products": selected.size(),
        f"{prefix}_unique_dates": dates.size(),
        f"{prefix}_acquisition_dates": dates.join(";"),
        f"{prefix}_tiles": tiles.join(";"),
        f"{prefix}_continuous_valid_coverage_pct": coverage,
    }
    values.update({f"{prefix}_{key}": value for key, value in threshold_flags(coverage).items()})
    return values


def build_hls_availability(
    modis_inputs, hls_collection, partition_start, partition_end
):
    """Build S30, L30, and combined HLS metrics in one station-period row."""
    images, image_indexes, footprints, footprint_indexes = _period_context(
        modis_inputs, partition_start, partition_end
    )

    def process_image(image_index):
        period_start, period_end, period_days = _period_values(images.get(image_index))

        def process_footprint(footprint_index):
            footprint = ee.Feature(footprints.get(footprint_index))
            geometry = footprint.geometry()
            local = filter_dynamic_hls_to_geometry(hls_collection, geometry)
            period = local.filterDate(period_start, period_end)
            properties = _base_properties(
                footprint, period_start, period_end, period_days
            )
            local_tiles = ee.List(local.get("local_mgrs_tiles"))
            properties["local_mgrs_tiles"] = local_tiles.join(";")
            for variant in HLS_VARIANTS:
                properties.update(_hls_variant_properties(period, geometry, variant))
            return ee.Feature(None, properties)

        return footprint_indexes.map(process_footprint)

    return ee.FeatureCollection(image_indexes.map(process_image).flatten())


def get_dynamic_s1_collection(footprints, start_date, end_date_exclusive):
    """Return dual-polarization IW S1 with no pass/orbit restriction."""
    geometry = ee.FeatureCollection(footprints).geometry()
    return (
        ee.ImageCollection(S1_COLLECTION_ID)
        .filterBounds(geometry).filterDate(start_date, end_date_exclusive)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH", "angle"])
        .map(lambda image: ee.Image(image).set(
            "date_key", ee.Image(image).date().format("yyyy-MM-dd")
        ))
    )


def _s1_geometries(collection):
    return ee.ImageCollection(collection).distinct(
        ["orbitProperties_pass", "relativeOrbitNumber_start"]
    )


def build_s1_geometry_inventory(s1_collection, footprints):
    """Summarize every observed pass/orbit pair without selecting one."""
    stations = ee.FeatureCollection(footprints)
    geometries = _s1_geometries(s1_collection)
    geometry_list = geometries.toList(geometries.size())
    geometry_indexes = ee.List.sequence(0, geometries.size().subtract(1))

    def summarize(geometry_index):
        reference = ee.Image(geometry_list.get(geometry_index))
        orbit_pass = reference.get("orbitProperties_pass")
        orbit = reference.get("relativeOrbitNumber_start")
        subset = (
            ee.ImageCollection(s1_collection)
            .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
            .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
        )
        dates = ee.List(subset.aggregate_array("date_key")).distinct().sort()

        def station_has_scene(station):
            station = ee.Feature(station)
            count = subset.filterBounds(station.geometry()).size()
            return station.set("geometry_scene_count", count)

        station_support = stations.map(station_has_scene)
        supported = station_support.filter(ee.Filter.gt("geometry_scene_count", 0)).size()
        return ee.Feature(None, {
            "pass": orbit_pass, "relative_orbit": orbit,
            "scenes": subset.size(), "unique_dates": dates.size(),
            "first_date": ee.Algorithms.If(dates.size().gt(0), dates.get(0), ""),
            "last_date": ee.Algorithms.If(dates.size().gt(0), dates.get(-1), ""),
            "stations_intersected": supported,
            "covers_all_stations": supported.eq(stations.size()).int(),
        })

    return ee.FeatureCollection(geometry_indexes.map(summarize))


def _angle_statistics(s1_median, geometry):
    reducer = (
        ee.Reducer.count().combine(ee.Reducer.mean(), sharedInputs=True)
        .combine(ee.Reducer.minMax(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
    )
    raw = ee.Dictionary(
        ee.Image(s1_median).select("Angle_deg").reduceRegion(
            reducer=reducer, geometry=geometry, crs=ANALYSIS_CRS,
            scale=S1_FOOTPRINT_REDUCTION_SCALE_M, maxPixels=1e7, tileScale=4,
        )
    )
    return {
        "angle_count": raw.get("Angle_deg_count"),
        "angle_mean_deg": raw.get("Angle_deg_mean"),
        "angle_min_deg": raw.get("Angle_deg_min"),
        "angle_max_deg": raw.get("Angle_deg_max"),
        "angle_stddev_deg": raw.get("Angle_deg_stdDev"),
    }


def build_s1_period_availability(
    modis_inputs, s1_collection, partition_start, partition_end
):
    """Build long-form station-period rows for every observed S1 geometry."""
    images, image_indexes, footprints, footprint_indexes = _period_context(
        modis_inputs, partition_start, partition_end
    )
    geometries = _s1_geometries(s1_collection)
    geometry_list = geometries.toList(geometries.size())
    geometry_indexes = ee.List.sequence(0, geometries.size().subtract(1))

    def process_image(image_index):
        period_start, period_end, period_days = _period_values(images.get(image_index))

        def process_footprint(footprint_index):
            footprint = ee.Feature(footprints.get(footprint_index))
            footprint_geometry = footprint.geometry()

            def process_geometry(geometry_index):
                reference = ee.Image(geometry_list.get(geometry_index))
                orbit_pass = reference.get("orbitProperties_pass")
                orbit = reference.get("relativeOrbitNumber_start")
                period = (
                    ee.ImageCollection(s1_collection)
                    .filterDate(period_start, period_end)
                    .filterBounds(footprint_geometry)
                    .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
                    .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
                )
                dates = ee.List(period.aggregate_array("date_key")).distinct().sort()
                median = build_s1_median(period, footprint_geometry)
                coverage = get_s1_coverage(median, footprint_geometry).multiply(100)
                properties = _base_properties(
                    footprint, period_start, period_end, period_days
                )
                properties.update({
                    "pass": orbit_pass, "relative_orbit": orbit,
                    "products": period.size(), "unique_dates": dates.size(),
                    "acquisition_dates": dates.join(";"),
                    "has_acquisition": period.size().gt(0).int(),
                    "continuous_valid_vv_vh_coverage_pct": coverage,
                    "has_valid_vv_vh_coverage": coverage.gt(0).int(),
                })
                properties.update(_angle_statistics(median, footprint_geometry))
                return ee.Feature(None, properties)

            return geometry_indexes.map(process_geometry)

        return footprint_indexes.map(process_footprint).flatten()

    return ee.FeatureCollection(image_indexes.map(process_image).flatten())


def scientific_configuration():
    """Return the auditable fixed rules used by this diagnostic."""
    return {
        "modis": {
            "product": MODIS_COLLECTION_ID,
            "et_scale_factor": MODIS_ET_SCALE_FACTOR,
            "valid_dn_min": MODIS_ET_MIN_VALID_DN,
            "valid_dn_max": MODIS_ET_MAX_VALID_DN,
            "strict_qc_required": MODIS_REQUIRE_STRICT_QC,
            "strict_scf_max": MODIS_STRICT_SCF_MAX,
        },
        "sentinel2": {
            "product": S2_COLLECTION_ID,
            "cloud_score_product": CLOUD_SCORE_COLLECTION_ID,
            "qa_band": S2_QA_BAND,
            "clear_threshold": S2_CLEAR_THRESHOLD,
            "working_scale_m": 20,
        },
        "hls": {
            "s30_product": HLS_S30_COLLECTION_ID,
            "l30_product": HLS_L30_COLLECTION_ID,
            "working_scale_m": 30,
            "variants": list(HLS_VARIANTS),
            "local_mgrs_verification": True,
        },
        "sentinel1": {
            "product": S1_COLLECTION_ID,
            "instrument_mode": "IW",
            "polarizations": ["VV", "VH"],
            "pass_filter": None,
            "relative_orbit_filter": None,
            "reduction_scale_m": S1_FOOTPRINT_REDUCTION_SCALE_M,
        },
        "diagnostic_optical_thresholds_pct": list(OPTICAL_THRESHOLDS_PCT),
        "training_performed": False,
    }
