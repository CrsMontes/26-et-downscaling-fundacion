"""Earth Engine extraction of raw meteorological inputs.

This module intentionally performs only operations that require Earth Engine:
resolving spatial support and sampling raw ERA5-Land, CHIRPS, and NASADEM.
Temporal aggregation, derived meteorological variables, reference ET, and Kc
are calculated locally in pandas/numpy.
"""

import ee

from .config import ERA5_SEARCH_RADIUS_M

ERA5_COLLECTION_ID = "ECMWF/ERA5_LAND/HOURLY"
CHIRPS_COLLECTION_ID = "UCSB-CHG/CHIRPS/DAILY"
NASADEM_IMAGE_ID = "NASA/NASADEM_HGT/001"

ERA5_SOURCE_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_hourly",
]

ERA5_EXPORT_BANDS = [
    "temperature_2m_K",
    "dewpoint_temperature_2m_K",
    "u_wind_10m_ms",
    "v_wind_10m_ms",
    "surface_solar_radiation_downwards_hourly_J_m2",
]

ERA5_EXPORT_SELECTORS = [
    "station",
    "station_id",
    "timestamp_utc",
    "system_time_start",
    "era5_sampling_longitude",
    "era5_sampling_latitude",
    *ERA5_EXPORT_BANDS,
]

CHIRPS_EXPORT_SELECTORS = [
    "station",
    "station_id",
    "date",
    "system_time_start",
    "station_longitude",
    "station_latitude",
    "precipitation_mm",
]

STATION_SUPPORT_SELECTORS = [
    "station",
    "station_id",
    "station_longitude",
    "station_latitude",
    "modis_pixel_id",
    "footprint_area_m2",
    "footprint_centroid_longitude",
    "footprint_centroid_latitude",
    "footprint_mean_elevation_m",
    "elevation_sampling_method",
    "elevation_grid_scale_m",
    "era5_support_m",
    "era5_sampling_method",
    "era5_sampling_longitude",
    "era5_sampling_latitude",
    "era5_sampling_distance_m",
    "chirps_support_m",
]


def get_era5_collection():
    return ee.ImageCollection(ERA5_COLLECTION_ID).select(ERA5_SOURCE_BANDS)


def get_chirps_collection():
    return ee.ImageCollection(CHIRPS_COLLECTION_ID).select("precipitation")


def get_era5_projection():
    return ee.Image(get_era5_collection().first()).select("temperature_2m").projection()


def get_chirps_projection():
    return ee.Image(get_chirps_collection().first()).select("precipitation").projection()


def get_nearest_valid_era5_point(
    station_point,
    era5_reference,
    era5_projection,
    era5_scale,
):
    station_point = ee.Geometry(station_point)
    valid_pixels = ee.Image(era5_reference).sample(
        region=station_point.buffer(ERA5_SEARCH_RADIUS_M),
        projection=era5_projection,
        scale=era5_scale,
        geometries=True,
        dropNulls=True,
    )

    with_distance = valid_pixels.map(
        lambda feature: ee.Feature(feature).set(
            "era5_distance_m",
            ee.Feature(feature).geometry().distance(station_point),
        )
    )

    return ee.Feature(with_distance.sort("era5_distance_m").first())


def build_era5_station_supports(station_footprints):
    station_footprints = ee.FeatureCollection(station_footprints)
    era5_collection = get_era5_collection()
    era5_reference = ee.Image(era5_collection.first()).select("temperature_2m")
    era5_projection = era5_reference.projection()
    era5_scale = era5_projection.nominalScale()

    def resolve_station(feature):
        feature = ee.Feature(feature)
        station_point = ee.Geometry.Point(
            [feature.get("longitude"), feature.get("latitude")]
        )
        support = get_nearest_valid_era5_point(
            station_point,
            era5_reference,
            era5_projection,
            era5_scale,
        )
        coordinates = support.geometry().coordinates()
        return ee.Feature(
            station_point,
            {
                "station": feature.get("station"),
                "station_id": feature.get("station_id"),
                "station_longitude": feature.get("longitude"),
                "station_latitude": feature.get("latitude"),
                "era5_sampling_method": "nearest_valid_land_pixel",
                "era5_sampling_longitude": coordinates.get(0),
                "era5_sampling_latitude": coordinates.get(1),
                "era5_sampling_distance_m": support.get("era5_distance_m"),
                "era5_support_m": era5_scale,
            },
        )

    resolved = station_footprints.map(resolve_station)

    # Materialize only this five-row lookup. Doing it once prevents the
    # nearest-land-pixel search from being repeated for every hour/date.
    info = resolved.getInfo()
    features = []
    for feature_info in info.get("features", []):
        properties = feature_info.get("properties", {})
        if properties.get("era5_sampling_longitude") is None:
            raise RuntimeError("No valid ERA5-Land support pixel was resolved.")
        features.append(
            ee.Feature(
                ee.Geometry.Point(
                    [
                        properties["station_longitude"],
                        properties["station_latitude"],
                    ]
                ),
                properties,
            )
        )
    if not features:
        raise RuntimeError("ERA5-Land station support lookup is empty.")
    return ee.FeatureCollection(features)


def build_station_support_table(station_footprints, era5_station_supports):
    station_footprints = ee.FeatureCollection(station_footprints)
    era5_station_supports = ee.FeatureCollection(era5_station_supports)

    dem = ee.Image(NASADEM_IMAGE_ID).select("elevation")
    dem_projection = dem.projection()
    dem_scale = dem_projection.nominalScale()
    chirps_scale = get_chirps_projection().nominalScale()

    def add_support(feature):
        feature = ee.Feature(feature)
        station_id = feature.get("station_id")
        geometry = feature.geometry()
        centroid = geometry.centroid(maxError=1)
        centroid_coordinates = centroid.coordinates()
        era5_support = ee.Feature(
            era5_station_supports.filter(
                ee.Filter.eq("station_id", station_id)
            ).first()
        )
        mean_elevation = dem.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            crs=dem_projection,
            scale=dem_scale,
            maxPixels=1e6,
            tileScale=4,
        ).get("elevation")

        return ee.Feature(
            None,
            {
                "station": feature.get("station"),
                "station_id": station_id,
                "station_longitude": feature.get("longitude"),
                "station_latitude": feature.get("latitude"),
                "modis_pixel_id": feature.get("modis_pixel_id"),
                "footprint_area_m2": feature.get("footprint_area_m2"),
                "footprint_centroid_longitude": centroid_coordinates.get(0),
                "footprint_centroid_latitude": centroid_coordinates.get(1),
                "footprint_mean_elevation_m": mean_elevation,
                "elevation_sampling_method": "MODIS_footprint_mean",
                "elevation_grid_scale_m": dem_scale,
                "era5_support_m": era5_support.get("era5_support_m"),
                "era5_sampling_method": era5_support.get("era5_sampling_method"),
                "era5_sampling_longitude": era5_support.get("era5_sampling_longitude"),
                "era5_sampling_latitude": era5_support.get("era5_sampling_latitude"),
                "era5_sampling_distance_m": era5_support.get("era5_sampling_distance_m"),
                "chirps_support_m": chirps_scale,
            },
        )

    return station_footprints.map(add_support)


def get_station_support(station_support_table, station_id):
    return ee.Feature(
        ee.FeatureCollection(station_support_table)
        .filter(ee.Filter.eq("station_id", station_id))
        .first()
    )


def build_era5_hourly_table(station_support, utc_start, utc_end):
    station_support = ee.Feature(station_support)
    support_point = ee.Geometry.Point(
        [
            station_support.get("era5_sampling_longitude"),
            station_support.get("era5_sampling_latitude"),
        ]
    )
    projection = get_era5_projection()
    scale = projection.nominalScale()
    collection = (
        get_era5_collection()
        .filterDate(utc_start, utc_end)
        .map(
            lambda image: ee.Image(image)
            .select(ERA5_SOURCE_BANDS, ERA5_EXPORT_BANDS)
            .toFloat()
        )
    )

    defaults = ee.Dictionary.fromLists(
        ERA5_EXPORT_BANDS,
        ee.List.repeat(-9999, len(ERA5_EXPORT_BANDS)),
    )

    def sample_image(image):
        image = ee.Image(image)
        values = defaults.combine(
            ee.Dictionary(
                image.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=support_point,
                    crs=projection,
                    scale=scale,
                    maxPixels=100,
                )
            ),
            True,
        )
        return ee.Feature(
            None,
            {
                "station": station_support.get("station"),
                "station_id": station_support.get("station_id"),
                "timestamp_utc": image.date().format("yyyy-MM-dd'T'HH:mm:ss'Z'"),
                "system_time_start": image.get("system:time_start"),
                "era5_sampling_longitude": station_support.get("era5_sampling_longitude"),
                "era5_sampling_latitude": station_support.get("era5_sampling_latitude"),
            },
        ).set(values)

    image_list = collection.toList(collection.size())
    return ee.FeatureCollection(image_list.map(sample_image))


def build_chirps_daily_table(station_support, start_date, end_date):
    station_support = ee.Feature(station_support)
    station_point = ee.Geometry.Point(
        [
            station_support.get("station_longitude"),
            station_support.get("station_latitude"),
        ]
    )
    projection = get_chirps_projection()
    scale = projection.nominalScale()
    collection = get_chirps_collection().filterDate(start_date, end_date)

    def sample_image(image):
        image = ee.Image(image)
        raw_value = image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=station_point,
            crs=projection,
            scale=scale,
            maxPixels=100,
        ).get("precipitation")
        precipitation = ee.Number(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(raw_value, None),
                -9999,
                raw_value,
            )
        )
        return ee.Feature(
            None,
            {
                "station": station_support.get("station"),
                "station_id": station_support.get("station_id"),
                "date": image.date().format("yyyy-MM-dd"),
                "system_time_start": image.get("system:time_start"),
                "station_longitude": station_support.get("station_longitude"),
                "station_latitude": station_support.get("station_latitude"),
                "precipitation_mm": precipitation,
            },
        )

    image_list = collection.toList(collection.size())
    return ee.FeatureCollection(image_list.map(sample_image))
