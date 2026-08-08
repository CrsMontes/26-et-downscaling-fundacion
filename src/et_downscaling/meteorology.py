import ee

from .schema import (
    EXPECTED_METEOROLOGICAL_KEYS,
)

from .config import (
    END_DATE,
    START_DATE,
    ERA5_SEARCH_RADIUS_M,
)


ERA5_COLLECTION_ID = (
    "ECMWF/ERA5_LAND/HOURLY"
)

CHIRPS_COLLECTION_ID = (
    "UCSB-CHG/CHIRPS/DAILY"
)


ERA5_SOURCE_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    (
        "surface_solar_radiation_"
        "downwards_hourly"
    ),
]


# ============================================================
# ERA5-Land collection
# ============================================================

def get_era5_land_collection():
    return (
        ee.ImageCollection(
            ERA5_COLLECTION_ID
        )
        .filterDate(
            START_DATE,
            END_DATE,
        )
        .select(
            ERA5_SOURCE_BANDS
        )
    )


# ============================================================
# CHIRPS collection
# ============================================================

def get_chirps_collection():
    return (
        ee.ImageCollection(
            CHIRPS_COLLECTION_ID
        )
        .filterDate(
            ee.Date(
                START_DATE
            ).advance(
                -30,
                "day",
            ),
            END_DATE,
        )
        .select(
            "precipitation"
        )
    )


# ============================================================
# Native projections and scales
# ============================================================

def get_era5_projection(
    era5_collection,
):
    reference_image = ee.Image(
        era5_collection.first()
    )

    return (
        reference_image
        .select(
            "temperature_2m"
        )
        .projection()
    )


def get_chirps_projection(
    chirps_collection,
):
    reference_image = ee.Image(
        chirps_collection.first()
    )

    return (
        reference_image
        .select(
            "precipitation"
        )
        .projection()
    )


# ============================================================
# Saturation vapor pressure
# ============================================================

def saturation_vapor_pressure_kpa(
    temperature_c,
):
    temperature_c = ee.Image(
        temperature_c
    )

    return temperature_c.expression(
        (
            "0.6108 * exp("
            "(17.27 * T) / "
            "(T + 237.3)"
            ")"
        ),
        {
            "T": temperature_c,
        },
    )


# ============================================================
# Prepare hourly ERA5-Land variables
# ============================================================

def prepare_era5_hourly(
    image,
):
    image = ee.Image(
        image
    )

    tair_c = (
        image
        .select(
            "temperature_2m"
        )
        .subtract(
            273.15
        )
        .rename(
            "Tair_C"
        )
    )

    tdew_c = (
        image
        .select(
            "dewpoint_temperature_2m"
        )
        .subtract(
            273.15
        )
        .rename(
            "Tdew_C"
        )
    )

    saturation_pressure = (
        saturation_vapor_pressure_kpa(
            tair_c
        )
    )

    actual_vapor_pressure = (
        saturation_vapor_pressure_kpa(
            tdew_c
        )
    )

    vpd = (
        saturation_pressure
        .subtract(
            actual_vapor_pressure
        )
        .max(
            0
        )
        .rename(
            "VPD_kPa"
        )
    )

    u_wind = (
        image.select(
            "u_component_of_wind_10m"
        )
    )

    v_wind = (
        image.select(
            "v_component_of_wind_10m"
        )
    )

    wind = (
        u_wind
        .pow(
            2
        )
        .add(
            v_wind.pow(
                2
            )
        )
        .sqrt()
        .rename(
            "Wind_ms"
        )
    )

    solar = (
        image
        .select(
            (
                "surface_solar_radiation_"
                "downwards_hourly"
            )
        )
        .divide(
            1e6
        )
        .rename(
            "SolarRad_MJ_m2_hour"
        )
    )

    return ee.Image(
        tair_c
        .addBands(
            tdew_c
        )
        .addBands(
            vpd
        )
        .addBands(
            wind
        )
        .addBands(
            solar
        )
        .copyProperties(
            image,
            [
                "system:time_start",
            ],
        )
    )


# ============================================================
# Find nearest valid ERA5-Land pixel
# ============================================================

def get_nearest_valid_era5_point(
    station_point,
    era5_reference,
    era5_projection,
    era5_scale,
):
    station_point = ee.Geometry(
        station_point
    )

    valid_pixels = (
        ee.Image(
            era5_reference
        )
        .sample(
            region=(
                station_point.buffer(
                    ERA5_SEARCH_RADIUS_M
                )
            ),
            projection=(
                era5_projection
            ),
            scale=(
                era5_scale
            ),
            geometries=True,
            dropNulls=True,
        )
    )

    with_distance = (
        valid_pixels.map(
            lambda feature: (
                ee.Feature(
                    feature
                ).set(
                    "era5_distance_m",
                    ee.Feature(
                        feature
                    )
                    .geometry()
                    .distance(
                        station_point
                    ),
                )
            )
        )
    )

    nearest = ee.Feature(
        with_distance
        .sort(
            "era5_distance_m"
        )
        .first()
    )

    return nearest


# ============================================================
# Resolve ERA5-Land support once per station
# ============================================================

def build_era5_station_supports(
    station_footprints,
    era5_reference,
    era5_projection,
    era5_scale,
):
    station_footprints = (
        ee.FeatureCollection(
            station_footprints
        )
    )

    def resolve_station(
        feature,
    ):
        feature = ee.Feature(
            feature
        )

        station_point = (
            ee.Geometry.Point(
                [
                    feature.get(
                        "longitude"
                    ),
                    feature.get(
                        "latitude"
                    ),
                ]
            )
        )

        support_feature = (
            get_nearest_valid_era5_point(
                station_point,
                era5_reference,
                era5_projection,
                era5_scale,
            )
        )

        support_coordinates = (
            support_feature
            .geometry()
            .coordinates()
        )

        return ee.Feature(
            station_point,
            {
                "station": (
                    feature.get(
                        "station"
                    )
                ),
                "station_id": (
                    feature.get(
                        "station_id"
                    )
                ),
                "station_longitude": (
                    feature.get(
                        "longitude"
                    )
                ),
                "station_latitude": (
                    feature.get(
                        "latitude"
                    )
                ),
                "era5_sampling_method": (
                    "nearest_valid_land_pixel"
                ),
                "era5_sampling_longitude": (
                    support_coordinates.get(
                        0
                    )
                ),
                "era5_sampling_latitude": (
                    support_coordinates.get(
                        1
                    )
                ),
                "era5_sampling_distance_m": (
                    support_feature.get(
                        "era5_distance_m"
                    )
                ),
            },
        )

    resolved_collection = (
        station_footprints.map(
            resolve_station
        )
    )

    # Materialize the small station-level lookup once.
    # Later observations reuse these literal values and do not
    # repeat the raster search for the nearest ERA5-Land cell.
    resolved_info = (
        resolved_collection.getInfo()
    )

    resolved_features = []

    for feature_info in resolved_info.get(
        "features",
        [],
    ):
        properties = feature_info.get(
            "properties",
            {},
        )

        station_longitude = properties.get(
            "station_longitude"
        )

        station_latitude = properties.get(
            "station_latitude"
        )

        if (
            station_longitude is None
            or station_latitude is None
        ):
            raise RuntimeError(
                "Station coordinates are missing while "
                "building ERA5-Land support lookup."
            )

        if (
            properties.get(
                "era5_sampling_longitude"
            )
            is None
            or properties.get(
                "era5_sampling_latitude"
            )
            is None
        ):
            raise RuntimeError(
                "No valid ERA5-Land support pixel was "
                "resolved for a station."
            )

        resolved_features.append(
            ee.Feature(
                ee.Geometry.Point(
                    [
                        station_longitude,
                        station_latitude,
                    ]
                ),
                properties,
            )
        )

    if not resolved_features:
        raise RuntimeError(
            "ERA5-Land station support lookup is empty."
        )

    return ee.FeatureCollection(
        resolved_features
    )


# ============================================================
# Select precomputed ERA5-Land support for a station point
# ============================================================

def get_precomputed_era5_support(
    station_point,
    era5_station_supports,
):
    station_point = ee.Geometry(
        station_point
    )

    era5_station_supports = (
        ee.FeatureCollection(
            era5_station_supports
        )
    )

    def add_station_distance(
        feature,
    ):
        feature = ee.Feature(
            feature
        )

        return feature.set(
            "lookup_distance_m",
            feature
            .geometry()
            .distance(
                station_point
            ),
        )

    return ee.Feature(
        era5_station_supports
        .map(
            add_station_distance
        )
        .sort(
            "lookup_distance_m"
        )
        .first()
    )


# ============================================================
# Build meteorological inputs
# ============================================================

def build_meteorology_inputs(
    station_footprints,
):
    era5_collection = (
        get_era5_land_collection()
    )

    chirps_collection = (
        get_chirps_collection()
    )

    era5_projection = (
        get_era5_projection(
            era5_collection
        )
    )

    era5_scale = (
        era5_projection.nominalScale()
    )

    chirps_projection = (
        get_chirps_projection(
            chirps_collection
        )
    )

    chirps_scale = (
        chirps_projection.nominalScale()
    )

    era5_derived = (
        era5_collection.map(
            prepare_era5_hourly
        )
    )

    # Static spatial reference used only once to identify
    # the valid ERA5-Land support cell for each station.
    era5_reference = (
        ee.Image(
            era5_collection.first()
        )
        .select(
            "temperature_2m"
        )
    )

    era5_station_supports = (
        build_era5_station_supports(
            station_footprints,
            era5_reference,
            era5_projection,
            era5_scale,
        )
    )

    return {
        "era5": (
            era5_collection
        ),
        "era5_derived": (
            era5_derived
        ),
        "era5_projection": (
            era5_projection
        ),
        "era5_scale": (
            era5_scale
        ),
        "era5_station_supports": (
            era5_station_supports
        ),
        "chirps": (
            chirps_collection
        ),
        "chirps_projection": (
            chirps_projection
        ),
        "chirps_scale": (
            chirps_scale
        ),
    }


# ============================================================
# Meteorology by station and MODIS period
# ============================================================

def get_meteorological_properties(
    period_start,
    period_end,
    number_days,
    station_point,
    meteorology_inputs,
):
    period_start = ee.Date(
        period_start
    )

    period_end = ee.Date(
        period_end
    )

    number_days = ee.Number(
        number_days
    )

    station_point = ee.Geometry(
        station_point
    )

    era5_derived = (
        meteorology_inputs[
            "era5_derived"
        ]
    )

    era5_projection = (
        meteorology_inputs[
            "era5_projection"
        ]
    )

    era5_scale = (
        meteorology_inputs[
            "era5_scale"
        ]
    )

    era5_station_supports = (
        meteorology_inputs[
            "era5_station_supports"
        ]
    )

    chirps = (
        meteorology_inputs[
            "chirps"
        ]
    )

    chirps_projection = (
        meteorology_inputs[
            "chirps_projection"
        ]
    )

    chirps_scale = (
        meteorology_inputs[
            "chirps_scale"
        ]
    )

    # ========================================================
    # ERA5-Land spatial support
    # ========================================================

    era5_support_feature = (
        get_precomputed_era5_support(
            station_point,
            era5_station_supports,
        )
    )

    era5_sampling_longitude = (
        ee.Number(
            era5_support_feature.get(
                "era5_sampling_longitude"
            )
        )
    )

    era5_sampling_latitude = (
        ee.Number(
            era5_support_feature.get(
                "era5_sampling_latitude"
            )
        )
    )

    era5_sampling_distance_m = (
        ee.Number(
            era5_support_feature.get(
                "era5_sampling_distance_m"
            )
        )
    )

    era5_sampling_method = (
        era5_support_feature.get(
            "era5_sampling_method"
        )
    )

    era5_support_point = (
        ee.Geometry.Point(
            [
                era5_sampling_longitude,
                era5_sampling_latitude,
            ]
        )
    )

    # ========================================================
    # ERA5-Land
    # ========================================================

    era5_period = (
        era5_derived.filterDate(
            period_start,
            period_end,
        )
    )

    era5_hours_total = (
        era5_period.size()
    )

    era5_hours_expected = (
        number_days.multiply(
            24
        )
    )

    tair_mean = (
        era5_period
        .select(
            "Tair_C"
        )
        .mean()
        .rename(
            "Tair_mean_C"
        )
    )

    tair_max = (
        era5_period
        .select(
            "Tair_C"
        )
        .max()
        .rename(
            "Tair_max_C"
        )
    )

    vpd_mean = (
        era5_period
        .select(
            "VPD_kPa"
        )
        .mean()
        .rename(
            "VPD_mean_kPa"
        )
    )

    vpd_max = (
        era5_period
        .select(
            "VPD_kPa"
        )
        .max()
        .rename(
            "VPD_max_kPa"
        )
    )

    wind_mean = (
        era5_period
        .select(
            "Wind_ms"
        )
        .mean()
        .rename(
            "Wind_mean_ms"
        )
    )

    solar_daily = (
        era5_period
        .select(
            "SolarRad_MJ_m2_hour"
        )
        .sum()
        .divide(
            number_days
        )
        .rename(
            "SolarRad_MJ_m2_day"
        )
    )

    era5_summary = (
        tair_mean
        .addBands(
            tair_max
        )
        .addBands(
            vpd_mean
        )
        .addBands(
            vpd_max
        )
        .addBands(
            solar_daily
        )
        .addBands(
            wind_mean
        )
    )

    # ERA5-Land is sampled at the station-specific support
    # pixel that was resolved once during initialization.
    era5_dictionary = ee.Dictionary(
        era5_summary.reduceRegion(
            reducer=(
                ee.Reducer.first()
            ),
            geometry=(
                era5_support_point
            ),
            crs=(
                era5_projection
            ),
            scale=(
                era5_scale
            ),
            maxPixels=100,
        )
    )

    # ========================================================
    # CHIRPS - precipitation during MODIS period
    # ========================================================

    chirps_period = (
        chirps.filterDate(
            period_start,
            period_end,
        )
    )

    chirps_days_period = (
        chirps_period.size()
    )

    precip_period = (
        chirps_period
        .sum()
        .rename(
            "Precip_period_mm"
        )
    )

    # ========================================================
    # CHIRPS - previous 30 days
    # ========================================================

    previous_30_start = (
        period_start.advance(
            -30,
            "day",
        )
    )

    chirps_previous_30 = (
        chirps.filterDate(
            previous_30_start,
            period_start,
        )
    )

    chirps_days_prev30 = (
        chirps_previous_30.size()
    )

    precip_previous_30 = (
        chirps_previous_30
        .sum()
        .rename(
            "Precip_prev30d_mm"
        )
    )

    chirps_summary = (
        precip_period
        .addBands(
            precip_previous_30
        )
    )

    # CHIRPS remains sampled at the original station point.
    chirps_dictionary = ee.Dictionary(
        chirps_summary.reduceRegion(
            reducer=(
                ee.Reducer.first()
            ),
            geometry=(
                station_point
            ),
            crs=(
                chirps_projection
            ),
            scale=(
                chirps_scale
            ),
            maxPixels=100,
        )
    )

    # ========================================================
    # Meteorological integrity
    # ========================================================

    raw_meteorology = (
        era5_dictionary.combine(
            chirps_dictionary,
            True,
        )
    )

    expected_keys = ee.List(
        EXPECTED_METEOROLOGICAL_KEYS
    )

    missing_meteo_keys = (
        expected_keys.removeAll(
            raw_meteorology.keys()
        )
    )

    meteo_missing_count = (
        missing_meteo_keys.size()
    )

    temporal_complete_condition = (
        era5_hours_total
        .eq(
            era5_hours_expected
        )
        .And(
            chirps_days_period.eq(
                number_days
            )
        )
        .And(
            chirps_days_prev30.eq(
                30
            )
        )
    )

    meteo_temporal_complete = (
        ee.Number(
            ee.Algorithms.If(
                temporal_complete_condition,
                1,
                0,
            )
        )
    )

    meteo_complete = (
        ee.Number(
            ee.Algorithms.If(
                meteo_missing_count
                .eq(
                    0
                )
                .And(
                    meteo_temporal_complete
                    .eq(
                        1
                    )
                ),
                1,
                0,
            )
        )
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = ee.Dictionary(
        {
            "era5_hours_total":
                era5_hours_total,

            "era5_hours_expected":
                era5_hours_expected,

            "chirps_days_period":
                chirps_days_period,

            "chirps_days_expected":
                number_days,

            "chirps_days_prev30":
                chirps_days_prev30,

            "chirps_days_prev30_expected":
                30,

            "era5_support_m":
                era5_scale,

            "chirps_support_m":
                chirps_scale,

            "era5_sampling_method":
                era5_sampling_method,

            "era5_sampling_longitude":
                era5_sampling_longitude,

            "era5_sampling_latitude":
                era5_sampling_latitude,

            "era5_sampling_distance_m":
                era5_sampling_distance_m,

            "meteo_missing_count":
                meteo_missing_count,

            "meteo_temporal_complete":
                meteo_temporal_complete,

            "meteo_complete":
                meteo_complete,
        }
    )

    return raw_meteorology.combine(
        metadata,
        True,
    )