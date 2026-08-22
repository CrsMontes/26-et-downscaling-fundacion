"""
Standardized reference evapotranspiration.

This module calculates daily short-reference ETo and tall-reference
ETr using the ASCE-EWRI standardized Penman-Monteith equation.

Hourly ERA5-Land meteorology is aggregated to local calendar days
in Colombia (UTC-5). Daily ETo and ETr are then summed over the
exact temporal interval represented by each MODIS ET observation.

Elevation is obtained from NASADEM and averaged over each MODIS
station footprint. The DEM is used only to characterize elevation;
its 30 m grid does not define the spatial resolution of ETo/ETr.

Reference ET remains associated with the MODIS footprint and the
coarse ERA5-Land atmospheric support.

Two forms of relative solar radiation are retained:

    Rs_Rso_raw
        Unmodified Rs / Rso ratio retained for quality assessment.

    Rs_Rso_used
        Rs / Rso bounded to [0.30, 1.00] according to the
        standardized daily reference ET procedure and used only
        in the net longwave radiation cloudiness term.
"""

import math

import ee


# ============================================================
# Data sources
# ============================================================

NASADEM_IMAGE_ID = "NASA/NASADEM_HGT/001"
NASADEM_BAND = "elevation"


# ============================================================
# Temporal configuration
# ============================================================

# Colombia Standard Time.
LOCAL_UTC_OFFSET_HOURS = -5


# ============================================================
# ASCE-EWRI standardized reference parameters
# ============================================================

# Short reference:
# approximately 0.12 m clipped grass.
SHORT_REFERENCE_CN = 900.0
SHORT_REFERENCE_CD = 0.34

# Tall reference:
# approximately 0.50 m alfalfa.
TALL_REFERENCE_CN = 1600.0
TALL_REFERENCE_CD = 0.38

REFERENCE_ALBEDO = 0.23

STEFAN_BOLTZMANN_MJ_K4_M2_DAY = 4.903e-9
SOLAR_CONSTANT_MJ_M2_MIN = 0.0820

WIND_MEASUREMENT_HEIGHT_M = 10.0

MIN_RS_RSO = 0.30
MAX_RS_RSO = 1.00


# ============================================================
# NASADEM
# ============================================================

def get_nasadem():
    """
    Return NASADEM elevation.

    Returns
    -------
    ee.Image
        NASADEM elevation in meters.
    """

    return (
        ee.Image(
            NASADEM_IMAGE_ID
        )
        .select(
            NASADEM_BAND
        )
        .rename(
            "Elevation_m"
        )
    )


# ============================================================
# Saturation vapor pressure
# ============================================================

def saturation_vapor_pressure_kpa(
    temperature_c,
):
    """
    Calculate saturation vapor pressure.

    Parameters
    ----------
    temperature_c : ee.Image
        Air temperature in degrees Celsius.

    Returns
    -------
    ee.Image
        Saturation vapor pressure in kPa.
    """

    temperature_c = ee.Image(
        temperature_c
    )

    return (
        temperature_c
        .multiply(
            17.27
        )
        .divide(
            temperature_c
            .add(
                237.3
            )
        )
        .exp()
        .multiply(
            0.6108
        )
        .toFloat()
    )


# ============================================================
# Wind adjustment
# ============================================================

def wind_speed_to_2m(
    wind_speed,
    measurement_height_m=WIND_MEASUREMENT_HEIGHT_M,
):
    """
    Convert wind speed measured at height z to 2 m.

    Parameters
    ----------
    wind_speed : ee.Image
        Wind speed in m s-1.
    measurement_height_m : float
        Original wind measurement height.

    Returns
    -------
    ee.Image
        Wind speed adjusted to 2 m.
    """

    conversion_factor = (
        4.87
        / math.log(
            67.8
            * measurement_height_m
            - 5.42
        )
    )

    return (
        ee.Image(
            wind_speed
        )
        .multiply(
            conversion_factor
        )
        .rename(
            "Wind2m_mean_ms"
        )
        .toFloat()
    )


# ============================================================
# Atmospheric pressure
# ============================================================

def calculate_atmospheric_pressure_kpa(
    elevation_m,
):
    """
    Calculate atmospheric pressure from elevation.

    Parameters
    ----------
    elevation_m : ee.Number
        Elevation in meters above sea level.

    Returns
    -------
    ee.Number
        Atmospheric pressure in kPa.
    """

    elevation_m = ee.Number(
        elevation_m
    )

    return (
        ee.Number(
            293.0
        )
        .subtract(
            elevation_m
            .multiply(
                0.0065
            )
        )
        .divide(
            293.0
        )
        .pow(
            5.26
        )
        .multiply(
            101.3
        )
    )


# ============================================================
# Extraterrestrial radiation
# ============================================================

def calculate_extraterrestrial_radiation(
    latitude_deg,
    day_of_year,
):
    """
    Calculate daily extraterrestrial radiation.

    Parameters
    ----------
    latitude_deg : ee.Number
        Latitude in decimal degrees.
    day_of_year : ee.Number
        Day of year from 1 to 365/366.

    Returns
    -------
    ee.Number
        Extraterrestrial radiation in MJ m-2 day-1.
    """

    latitude_rad = (
        ee.Number(
            latitude_deg
        )
        .multiply(
            math.pi
            / 180.0
        )
    )

    day_of_year = ee.Number(
        day_of_year
    )

    annual_angle = (
        day_of_year
        .multiply(
            2.0
            * math.pi
            / 365.0
        )
    )

    inverse_relative_distance = (
        ee.Number(
            1.0
        )
        .add(
            annual_angle
            .cos()
            .multiply(
                0.033
            )
        )
    )

    solar_declination = (
        annual_angle
        .subtract(
            1.39
        )
        .sin()
        .multiply(
            0.409
        )
    )

    sunset_argument = (
        latitude_rad
        .tan()
        .multiply(
            solar_declination
            .tan()
        )
        .multiply(
            -1.0
        )
        .max(
            -1.0
        )
        .min(
            1.0
        )
    )

    sunset_hour_angle = (
        sunset_argument.acos()
    )

    radiation_term = (
        sunset_hour_angle
        .multiply(
            latitude_rad.sin()
        )
        .multiply(
            solar_declination.sin()
        )
        .add(
            latitude_rad
            .cos()
            .multiply(
                solar_declination.cos()
            )
            .multiply(
                sunset_hour_angle.sin()
            )
        )
    )

    return (
        radiation_term
        .multiply(
            inverse_relative_distance
        )
        .multiply(
            SOLAR_CONSTANT_MJ_M2_MIN
        )
        .multiply(
            24.0
            * 60.0
            / math.pi
        )
    )


# ============================================================
# Hourly actual vapor pressure
# ============================================================

def add_hourly_actual_vapor_pressure(
    image,
):
    """
    Calculate hourly actual vapor pressure from dewpoint.

    ERA5-Land dewpoint temperature provides the information
    needed to estimate actual vapor pressure.

    Parameters
    ----------
    image : ee.Image
        Hourly ERA5-derived image.

    Returns
    -------
    ee.Image
        Original image plus ea_kPa.
    """

    image = ee.Image(
        image
    )

    ea = (
        saturation_vapor_pressure_kpa(
            image.select(
                "Tdew_C"
            )
        )
        .rename(
            "ea_kPa"
        )
    )

    return (
        image
        .addBands(
            ea
        )
        .toFloat()
    )


# ============================================================
# Daily ERA5 meteorology
# ============================================================

def build_daily_reference_meteorology(
    era5_derived,
    start_date,
    end_date,
):
    """
    Aggregate hourly ERA5-Land data to local calendar days.

    Colombia uses UTC-5. Therefore a local calendar day is
    represented by:

        05:00 UTC current date
        to
        05:00 UTC next date

    Parameters
    ----------
    era5_derived : ee.ImageCollection
        Hourly ERA5-Land collection already containing Tair_C,
        Tdew_C, Wind_ms, and SolarRad_MJ_m2_hour.
    start_date : str or ee.Date
        First local calendar date.
    end_date : str or ee.Date
        Exclusive final local calendar date.

    Returns
    -------
    ee.ImageCollection
        One image per local day.
    """

    era5_derived = (
        ee.ImageCollection(
            era5_derived
        )
        .map(
            add_hourly_actual_vapor_pressure
        )
    )

    start_date = ee.Date(
        start_date
    )

    end_date = ee.Date(
        end_date
    )

    number_days = (
        end_date
        .difference(
            start_date,
            "day",
        )
        .round()
    )

    day_offsets = (
        ee.List.sequence(
            0,
            number_days.subtract(
                1
            ),
        )
    )

    def build_day(
        day_offset,
    ):
        day_offset = ee.Number(
            day_offset
        )

        local_date = (
            start_date
            .advance(
                day_offset,
                "day",
            )
        )

        utc_start = (
            local_date
            .advance(
                -LOCAL_UTC_OFFSET_HOURS,
                "hour",
            )
        )

        utc_end = (
            utc_start
            .advance(
                1,
                "day",
            )
        )

        daily_collection = (
            era5_derived
            .filterDate(
                utc_start,
                utc_end,
            )
        )

        hour_count = (
            daily_collection.size()
        )

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        tmin = (
            daily_collection
            .select(
                "Tair_C"
            )
            .min()
            .rename(
                "Tmin_day_C"
            )
        )

        tmax = (
            daily_collection
            .select(
                "Tair_C"
            )
            .max()
            .rename(
                "Tmax_day_C"
            )
        )

        tmean = (
            tmin
            .add(
                tmax
            )
            .divide(
                2.0
            )
            .rename(
                "Tmean_day_C"
            )
        )

        # ----------------------------------------------------
        # Actual vapor pressure
        # ----------------------------------------------------

        ea_mean = (
            daily_collection
            .select(
                "ea_kPa"
            )
            .mean()
            .rename(
                "ea_day_kPa"
            )
        )

        # ----------------------------------------------------
        # Wind
        # ----------------------------------------------------

        wind10_mean = (
            daily_collection
            .select(
                "Wind_ms"
            )
            .mean()
            .rename(
                "Wind10m_mean_ms"
            )
        )

        wind2_mean = (
            wind_speed_to_2m(
                wind10_mean
            )
        )

        # ----------------------------------------------------
        # Solar radiation
        # ----------------------------------------------------

        # SolarRad_MJ_m2_hour represents energy accumulated
        # during each hourly interval. Summing the intervals
        # produces daily solar radiation.
        rs_daily = (
            daily_collection
            .select(
                "SolarRad_MJ_m2_hour"
            )
            .sum()
            .rename(
                "Rs_day_MJ_m2"
            )
        )

        daily_image = (
            tmin
            .addBands(
                tmax
            )
            .addBands(
                tmean
            )
            .addBands(
                ea_mean
            )
            .addBands(
                wind10_mean
            )
            .addBands(
                wind2_mean
            )
            .addBands(
                rs_daily
            )
            .toFloat()
        )

        return ee.Image(
            daily_image
            .set(
                {
                    "system:time_start":
                        utc_start.millis(),

                    "local_date":
                        local_date.format(
                            "yyyy-MM-dd"
                        ),

                    "local_date_millis":
                        local_date.millis(),

                    "utc_window_start":
                        utc_start.format(
                            "yyyy-MM-dd HH:mm"
                        ),

                    "utc_window_end":
                        utc_end.format(
                            "yyyy-MM-dd HH:mm"
                        ),

                    "era5_hours_total":
                        hour_count,

                    "era5_hours_expected":
                        24,
                }
            )
        )

    return (
        ee.ImageCollection
        .fromImages(
            day_offsets.map(
                build_day
            )
        )
        .sort(
            "system:time_start"
        )
    )


# ============================================================
# Daily standardized reference ET
# ============================================================

def calculate_daily_reference_et(
    daily_meteorology,
    latitude_deg,
    elevation_m,
):
    """
    Calculate daily ETo and ETr.

    Uses the ASCE-EWRI standardized daily Penman-Monteith
    equation.

    Parameters
    ----------
    daily_meteorology : ee.Image
        Daily meteorological image.
    latitude_deg : ee.Number
        Representative footprint latitude.
    elevation_m : ee.Number
        Mean NASADEM elevation inside the footprint.

    Returns
    -------
    ee.Image
        Daily meteorology plus radiation diagnostics, ETo, ETr.
    """

    image = ee.Image(
        daily_meteorology
    )

    local_date = ee.Date(
        image.get(
            "local_date_millis"
        )
    )

    day_of_year = (
        ee.Number(
            local_date.getRelative(
                "day",
                "year",
            )
        )
        .add(
            1
        )
    )

    tmin = (
        image.select(
            "Tmin_day_C"
        )
    )

    tmax = (
        image.select(
            "Tmax_day_C"
        )
    )

    tmean = (
        image.select(
            "Tmean_day_C"
        )
    )

    ea = (
        image.select(
            "ea_day_kPa"
        )
    )

    wind2 = (
        image.select(
            "Wind2m_mean_ms"
        )
    )

    rs = (
        image.select(
            "Rs_day_MJ_m2"
        )
    )


    # ========================================================
    # Saturation vapor pressure and VPD
    # ========================================================

    es_tmin = (
        saturation_vapor_pressure_kpa(
            tmin
        )
    )

    es_tmax = (
        saturation_vapor_pressure_kpa(
            tmax
        )
    )

    es = (
        es_tmin
        .add(
            es_tmax
        )
        .divide(
            2.0
        )
        .rename(
            "es_day_kPa"
        )
    )

    vpd = (
        es
        .subtract(
            ea
        )
        .max(
            0
        )
        .rename(
            "VPD_day_kPa"
        )
    )


    # ========================================================
    # Saturation vapor-pressure curve slope
    # ========================================================

    es_tmean = (
        saturation_vapor_pressure_kpa(
            tmean
        )
    )

    delta = (
        es_tmean
        .multiply(
            4098.0
        )
        .divide(
            tmean
            .add(
                237.3
            )
            .pow(
                2
            )
        )
        .rename(
            "Delta_kPa_C"
        )
    )


    # ========================================================
    # Atmospheric pressure and psychrometric constant
    # ========================================================

    pressure = (
        calculate_atmospheric_pressure_kpa(
            elevation_m
        )
    )

    gamma = (
        pressure
        .multiply(
            0.000665
        )
    )

    pressure_image = (
        ee.Image.constant(
            pressure
        )
        .rename(
            "Pressure_kPa"
        )
        .toFloat()
    )

    gamma_image = (
        ee.Image.constant(
            gamma
        )
        .rename(
            "Gamma_kPa_C"
        )
        .toFloat()
    )


    # ========================================================
    # Extraterrestrial radiation
    # ========================================================

    ra = (
        calculate_extraterrestrial_radiation(
            latitude_deg,
            day_of_year,
        )
    )

    ra_image = (
        ee.Image.constant(
            ra
        )
        .rename(
            "Ra_day_MJ_m2"
        )
        .toFloat()
    )


    # ========================================================
    # Clear-sky solar radiation
    # ========================================================

    rso = (
        ee.Number(
            0.75
        )
        .add(
            ee.Number(
                elevation_m
            )
            .multiply(
                2e-5
            )
        )
        .multiply(
            ra
        )
    )

    rso_image = (
        ee.Image.constant(
            rso
        )
        .rename(
            "Rso_day_MJ_m2"
        )
        .toFloat()
    )


    # ========================================================
    # Net shortwave radiation
    # ========================================================

    rns = (
        rs
        .multiply(
            1.0
            - REFERENCE_ALBEDO
        )
        .rename(
            "Rns_day_MJ_m2"
        )
    )


    # ========================================================
    # Relative solar radiation
    # ========================================================

    # Raw Rs/Rso is retained unchanged for quality assessment.
    rs_rso_raw = (
        rs
        .divide(
            rso_image
        )
        .rename(
            "Rs_Rso_raw"
        )
        .toFloat()
    )

    # The standardized daily procedure bounds Rs/Rso before
    # using it in the cloudiness term of net longwave radiation.
    rs_rso_used = (
        rs_rso_raw
        .max(
            MIN_RS_RSO
        )
        .min(
            MAX_RS_RSO
        )
        .rename(
            "Rs_Rso_used"
        )
        .toFloat()
    )


    # ========================================================
    # Net longwave radiation
    # ========================================================

    cloudiness_factor = (
        rs_rso_used
        .multiply(
            1.35
        )
        .subtract(
            0.35
        )
        .rename(
            "Cloudiness_factor"
        )
    )

    humidity_factor = (
        ea
        .sqrt()
        .multiply(
            -0.14
        )
        .add(
            0.34
        )
        .rename(
            "Humidity_factor"
        )
    )

    tmax_kelvin = (
        tmax.add(
            273.16
        )
    )

    tmin_kelvin = (
        tmin.add(
            273.16
        )
    )

    temperature_factor = (
        tmax_kelvin
        .pow(
            4
        )
        .add(
            tmin_kelvin.pow(
                4
            )
        )
        .divide(
            2.0
        )
        .multiply(
            STEFAN_BOLTZMANN_MJ_K4_M2_DAY
        )
        .rename(
            "Longwave_temperature_factor"
        )
    )

    rnl = (
        temperature_factor
        .multiply(
            humidity_factor
        )
        .multiply(
            cloudiness_factor
        )
        .rename(
            "Rnl_day_MJ_m2"
        )
    )


    # ========================================================
    # Net radiation
    # ========================================================

    rn = (
        rns
        .subtract(
            rnl
        )
        .rename(
            "Rn_day_MJ_m2"
        )
    )


    # ========================================================
    # Soil heat flux
    # ========================================================

    # Daily standardized reference ET assumes G approximately
    # equal to zero.
    soil_heat_flux = 0.0


    # ========================================================
    # Standardized Penman-Monteith
    # ========================================================

    def calculate_reference_et(
        cn,
        cd,
        output_name,
    ):
        """
        Calculate one standardized reference ET.

        All spatial terms are handled as ee.Image objects.
        """

        energy_term = (
            delta
            .multiply(
                rn.subtract(
                    soil_heat_flux
                )
            )
            .multiply(
                0.408
            )
        )

        aerodynamic_term = (
            gamma_image
            .multiply(
                cn
            )
            .multiply(
                wind2
            )
            .multiply(
                vpd
            )
            .divide(
                tmean.add(
                    273.0
                )
            )
        )

        denominator = (
            delta
            .add(
                gamma_image
                .multiply(
                    wind2
                    .multiply(
                        cd
                    )
                    .add(
                        1.0
                    )
                )
            )
        )

        return (
            energy_term
            .add(
                aerodynamic_term
            )
            .divide(
                denominator
            )
            .rename(
                output_name
            )
            .toFloat()
        )

    eto = (
        calculate_reference_et(
            SHORT_REFERENCE_CN,
            SHORT_REFERENCE_CD,
            "ETo_mm_day",
        )
    )

    etr = (
        calculate_reference_et(
            TALL_REFERENCE_CN,
            TALL_REFERENCE_CD,
            "ETr_mm_day",
        )
    )


    # ========================================================
    # Output
    # ========================================================

    output = (
        image
        .addBands(
            es
        )
        .addBands(
            vpd
        )
        .addBands(
            delta
        )
        .addBands(
            pressure_image
        )
        .addBands(
            gamma_image
        )
        .addBands(
            ra_image
        )
        .addBands(
            rso_image
        )
        .addBands(
            rs_rso_raw
        )
        .addBands(
            rs_rso_used
        )
        .addBands(
            cloudiness_factor
        )
        .addBands(
            humidity_factor
        )
        .addBands(
            temperature_factor
        )
        .addBands(
            rns
        )
        .addBands(
            rnl
        )
        .addBands(
            rn
        )
        .addBands(
            eto
        )
        .addBands(
            etr
        )
        .toFloat()
    )

    return ee.Image(
        output
        .copyProperties(
            image,
            [
                "system:time_start",
                "local_date",
                "local_date_millis",
                "utc_window_start",
                "utc_window_end",
                "era5_hours_total",
                "era5_hours_expected",
            ],
        )
        .set(
            {
                "reference_elevation_m":
                    elevation_m,

                "reference_latitude_deg":
                    latitude_deg,

                "reference_albedo":
                    REFERENCE_ALBEDO,

                "rs_rso_min_used":
                    MIN_RS_RSO,

                "rs_rso_max_used":
                    MAX_RS_RSO,

                "eto_cn":
                    SHORT_REFERENCE_CN,

                "eto_cd":
                    SHORT_REFERENCE_CD,

                "etr_cn":
                    TALL_REFERENCE_CN,

                "etr_cd":
                    TALL_REFERENCE_CD,
            }
        )
    )


# ============================================================
# Static support by MODIS footprint
# ============================================================

def build_reference_et_station_supports(
    station_footprints,
    meteorology_inputs,
):
    """
    Resolve static reference-ET support information.

    For every station footprint this function calculates once:

    - mean NASADEM elevation inside the footprint;
    - footprint centroid latitude and longitude;
    - ERA5-Land support pixel already resolved by meteorology.py.

    The NASADEM 30 m grid is used only to calculate a
    representative mean footprint elevation. It does not make
    ETo or ETr a 30 m product.
    """

    station_footprints = (
        ee.FeatureCollection(
            station_footprints
        )
    )

    era5_station_supports = (
        ee.FeatureCollection(
            meteorology_inputs[
                "era5_station_supports"
            ]
        )
    )

    dem = (
        get_nasadem()
    )

    dem_projection = (
        dem.projection()
    )

    dem_scale = (
        dem_projection.nominalScale()
    )

    def resolve_support(
        feature,
    ):
        feature = ee.Feature(
            feature
        )

        geometry = (
            feature.geometry()
        )

        station_id = (
            feature.get(
                "station_id"
            )
        )

        station_name = (
            feature.get(
                "station"
            )
        )

        centroid = (
            geometry.centroid(
                maxError=1
            )
        )

        centroid_coordinates = (
            centroid.coordinates()
        )

        mean_elevation = (
            dem
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                crs=dem_projection,
                scale=dem_scale,
                maxPixels=1e6,
                tileScale=4,
            )
            .get(
                "Elevation_m"
            )
        )

        era5_support = ee.Feature(
            era5_station_supports
            .filter(
                ee.Filter.eq(
                    "station_id",
                    station_id,
                )
            )
            .first()
        )

        return ee.Feature(
            centroid,
            {
                "station":
                    station_name,

                "station_id":
                    station_id,

                "footprint_centroid_longitude":
                    centroid_coordinates.get(
                        0
                    ),

                "footprint_centroid_latitude":
                    centroid_coordinates.get(
                        1
                    ),

                "footprint_mean_elevation_m":
                    mean_elevation,

                "elevation_source":
                    NASADEM_IMAGE_ID,

                "elevation_method":
                    "footprint_mean",

                "elevation_grid_scale_m":
                    dem_scale,

                "era5_sampling_method":
                    era5_support.get(
                        "era5_sampling_method"
                    ),

                "era5_sampling_longitude":
                    era5_support.get(
                        "era5_sampling_longitude"
                    ),

                "era5_sampling_latitude":
                    era5_support.get(
                        "era5_sampling_latitude"
                    ),

                "era5_sampling_distance_m":
                    era5_support.get(
                        "era5_sampling_distance_m"
                    ),
            },
        )

    support_collection = (
        station_footprints.map(
            resolve_support
        )
    )

    # Only a few footprints exist. Materialize once so the DEM
    # and ERA5 support searches are not repeated for every date.
    support_info = (
        support_collection.getInfo()
    )

    support_features = []

    for feature_info in support_info.get(
        "features",
        [],
    ):
        properties = (
            feature_info.get(
                "properties",
                {}
            )
        )

        longitude = properties.get(
            "footprint_centroid_longitude"
        )

        latitude = properties.get(
            "footprint_centroid_latitude"
        )

        elevation = properties.get(
            "footprint_mean_elevation_m"
        )

        if (
            longitude is None
            or latitude is None
            or elevation is None
        ):
            raise RuntimeError(
                "Reference ET support could not be "
                "resolved for a station footprint."
            )

        support_features.append(
            ee.Feature(
                ee.Geometry.Point(
                    [
                        longitude,
                        latitude,
                    ]
                ),
                properties,
            )
        )

    if not support_features:
        raise RuntimeError(
            "Reference ET station support lookup is empty."
        )

    return ee.FeatureCollection(
        support_features
    )


# ============================================================
# Build reusable reference-ET inputs
# ============================================================

def build_reference_et_inputs(
    station_footprints,
    meteorology_inputs,
    start_date,
    end_date,
):
    """
    Build reusable reference ET inputs.

    Daily ERA5 meteorology and static NASADEM/ERA5 station
    support information are prepared once.
    """

    daily_meteorology = (
        build_daily_reference_meteorology(
            meteorology_inputs[
                "era5_derived"
            ],
            start_date,
            end_date,
        )
    )

    station_supports = (
        build_reference_et_station_supports(
            station_footprints,
            meteorology_inputs,
        )
    )

    return {
        "daily_meteorology":
            daily_meteorology,

        "station_supports":
            station_supports,

        "era5_projection":
            meteorology_inputs[
                "era5_projection"
            ],

        "era5_scale":
            meteorology_inputs[
                "era5_scale"
            ],
    }


# ============================================================
# Get static station support
# ============================================================

def get_reference_et_station_support(
    station_id,
    reference_et_inputs,
):
    """
    Retrieve static reference-ET support for one station.
    """

    station_supports = (
        ee.FeatureCollection(
            reference_et_inputs[
                "station_supports"
            ]
        )
    )

    return ee.Feature(
        station_supports
        .filter(
            ee.Filter.eq(
                "station_id",
                station_id,
            )
        )
        .first()
    )


# ============================================================
# Reference ET for one MODIS period
# ============================================================

def get_reference_et_properties(
    period_start,
    period_end,
    station_id,
    reference_et_inputs,
):
    """
    Calculate ETo and ETr for one MODIS period and footprint.

    Daily ETo and ETr are calculated first and subsequently
    summed over exactly [period_start, period_end).

    Parameters
    ----------
    period_start : str or ee.Date
        Inclusive MODIS period start.
    period_end : str or ee.Date
        Exclusive MODIS period end.
    station_id
        Station identifier associated with the MODIS footprint.
    reference_et_inputs : dict
        Output of build_reference_et_inputs().

    Returns
    -------
    ee.Dictionary
        Period ETo/ETr and diagnostic metadata.
    """

    period_start = ee.Date(
        period_start
    )

    period_end = ee.Date(
        period_end
    )

    support = (
        get_reference_et_station_support(
            station_id,
            reference_et_inputs,
        )
    )

    latitude = (
        ee.Number(
            support.get(
                "footprint_centroid_latitude"
            )
        )
    )

    elevation = (
        ee.Number(
            support.get(
                "footprint_mean_elevation_m"
            )
        )
    )

    era5_longitude = (
        ee.Number(
            support.get(
                "era5_sampling_longitude"
            )
        )
    )

    era5_latitude = (
        ee.Number(
            support.get(
                "era5_sampling_latitude"
            )
        )
    )

    era5_support_point = (
        ee.Geometry.Point(
            [
                era5_longitude,
                era5_latitude,
            ]
        )
    )

    daily_meteorology = (
        ee.ImageCollection(
            reference_et_inputs[
                "daily_meteorology"
            ]
        )
    )

    era5_projection = (
        reference_et_inputs[
            "era5_projection"
        ]
    )

    era5_scale = (
        reference_et_inputs[
            "era5_scale"
        ]
    )

    daily_period = (
        daily_meteorology
        .filterDate(
            period_start,
            period_end,
        )
    )

    daily_reference_et = (
        daily_period.map(
            lambda image:
                calculate_daily_reference_et(
                    image,
                    latitude,
                    elevation,
                )
        )
    )

    expected_days = (
        period_end
        .difference(
            period_start,
            "day",
        )
        .round()
    )

    total_days = (
        daily_reference_et.size()
    )

    complete_days = (
        daily_reference_et
        .filter(
            ee.Filter.eq(
                "era5_hours_total",
                24,
            )
        )
        .size()
    )


    # ========================================================
    # Period sums
    # ========================================================

    eto_period = (
        daily_reference_et
        .select(
            "ETo_mm_day"
        )
        .sum()
        .rename(
            "ETo_period_mm"
        )
    )

    etr_period = (
        daily_reference_et
        .select(
            "ETr_mm_day"
        )
        .sum()
        .rename(
            "ETr_period_mm"
        )
    )


    # ========================================================
    # Mean daily values
    # ========================================================

    eto_mean = (
        eto_period
        .divide(
            expected_days
        )
        .rename(
            "ETo_mean_mm_day"
        )
    )

    etr_mean = (
        etr_period
        .divide(
            expected_days
        )
        .rename(
            "ETr_mean_mm_day"
        )
    )


    # ========================================================
    # Period summary
    # ========================================================

    summary_image = (
        eto_period
        .addBands(
            etr_period
        )
        .addBands(
            eto_mean
        )
        .addBands(
            etr_mean
        )
    )

    summary = ee.Dictionary(
        summary_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=era5_support_point,
            crs=era5_projection,
            scale=era5_scale,
            maxPixels=100,
        )
    )

    eto_period_value = (
        ee.Number(
            summary.get(
                "ETo_period_mm"
            )
        )
    )

    etr_period_value = (
        ee.Number(
            summary.get(
                "ETr_period_mm"
            )
        )
    )

    etr_eto_ratio = (
        etr_period_value
        .divide(
            eto_period_value
        )
    )


    # ========================================================
    # Daily Rs/Rso QA across the MODIS period
    # ========================================================

    rs_rso_raw_min_image = (
        daily_reference_et
        .select(
            "Rs_Rso_raw"
        )
        .min()
        .rename(
            "Rs_Rso_raw_min"
        )
    )

    rs_rso_raw_max_image = (
        daily_reference_et
        .select(
            "Rs_Rso_raw"
        )
        .max()
        .rename(
            "Rs_Rso_raw_max"
        )
    )

    rs_rso_raw_mean_image = (
        daily_reference_et
        .select(
            "Rs_Rso_raw"
        )
        .mean()
        .rename(
            "Rs_Rso_raw_mean"
        )
    )

    rs_rso_qa_image = (
        rs_rso_raw_min_image
        .addBands(
            rs_rso_raw_max_image
        )
        .addBands(
            rs_rso_raw_mean_image
        )
    )

    rs_rso_summary = ee.Dictionary(
        rs_rso_qa_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=era5_support_point,
            crs=era5_projection,
            scale=era5_scale,
            maxPixels=100,
        )
    )


    # ========================================================
    # Completeness
    # ========================================================

    complete_condition = (
        total_days
        .eq(
            expected_days
        )
        .And(
            complete_days.eq(
                expected_days
            )
        )
    )

    reference_et_complete = (
        ee.Number(
            ee.Algorithms.If(
                complete_condition,
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
            "reference_et_days_total":
                total_days,

            "reference_et_days_expected":
                expected_days,

            "reference_et_complete_days":
                complete_days,

            "reference_et_complete":
                reference_et_complete,

            "ETr_ETo_ratio":
                etr_eto_ratio,

            "Rs_Rso_raw_min":
                rs_rso_summary.get(
                    "Rs_Rso_raw_min"
                ),

            "Rs_Rso_raw_max":
                rs_rso_summary.get(
                    "Rs_Rso_raw_max"
                ),

            "Rs_Rso_raw_mean":
                rs_rso_summary.get(
                    "Rs_Rso_raw_mean"
                ),

            "reference_et_elevation_m":
                elevation,

            "reference_et_latitude_deg":
                latitude,

            "reference_et_elevation_source":
                support.get(
                    "elevation_source"
                ),

            "reference_et_elevation_method":
                support.get(
                    "elevation_method"
                ),

            "reference_et_timezone":
                "America/Bogota",

            "reference_et_utc_offset_hours":
                LOCAL_UTC_OFFSET_HOURS,

            "reference_et_rs_rso_min_used":
                MIN_RS_RSO,

            "reference_et_rs_rso_max_used":
                MAX_RS_RSO,

            "era5_sampling_longitude":
                era5_longitude,

            "era5_sampling_latitude":
                era5_latitude,

            "era5_sampling_distance_m":
                support.get(
                    "era5_sampling_distance_m"
                ),
        }
    )

    return (
        summary
        .combine(
            rs_rso_summary,
            True,
        )
        .combine(
            metadata,
            True,
        )
    )