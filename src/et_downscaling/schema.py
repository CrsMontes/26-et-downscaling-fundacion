# ============================================================
# Meteorological variables
# ============================================================

METEOROLOGICAL_COLUMNS = [
    "Tair_mean_C",
    "Tair_max_C",
    "VPD_mean_kPa",
    "VPD_max_kPa",
    "SolarRad_MJ_m2_day",
    "Wind_mean_ms",
    "Precip_period_mm",
    "Precip_prev30d_mm",
]


EXPECTED_METEOROLOGICAL_KEYS = (
    METEOROLOGICAL_COLUMNS.copy()
)


# ============================================================
# Sentinel-2 source bands
#
# Retained because sentinel2.py uses these names conceptually
# and diagnostic notebooks may import them.
# ============================================================

S2_SOURCE_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]


S2_BAND_NAMES = [
    "Blue",
    "Green",
    "Red",
    "RedEdge1",
    "RedEdge2",
    "RedEdge3",
    "NIR_Broad",
    "NIR",
    "SWIR1",
    "SWIR2",
]


# ============================================================
# Source-neutral optical model bands
# ============================================================

COMMON_OPTICAL_REFLECTANCE_BANDS = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
]


COMMON_OPTICAL_INDEX_BANDS = [
    "NDVI",
    "EVI",
    "SAVI",
    "NDWI",
    "NDMI",
]


COMMON_OPTICAL_PREDICTOR_BANDS = (
    COMMON_OPTICAL_REFLECTANCE_BANDS
    + COMMON_OPTICAL_INDEX_BANDS
)


# ============================================================
# Sentinel-1 model and QA bands
# ============================================================

S1_MODEL_BANDS = [
    "VV_dB",
    "VH_dB",
    "VV_minus_VH_dB",
]

# Incidence angle is retained for QA/geometry diagnostics but is
# deliberately excluded from the transferable model predictor set.
S1_QA_BANDS = [
    "Angle_deg",
]


# ============================================================
# Raster predictor bands used for the training table
# ============================================================

PREDICTOR_BANDS = (
    COMMON_OPTICAL_PREDICTOR_BANDS
    + S1_MODEL_BANDS
)


# ============================================================
# Scale-transferable footprint statistics
#
# Only means are exported. Within-footprint standard deviation,
# percentiles, and other heterogeneity statistics are excluded
# because they do not have an equivalent meaning for a single
# fine-grid prediction cell.
# ============================================================

STAT_COLUMNS = [
    f"{band_name}_mean"
    for band_name in PREDICTOR_BANDS
]

EXPECTED_STAT_KEYS = (
    STAT_COLUMNS.copy()
)


QA_STAT_COLUMNS = [
    "Angle_deg_mean",
]


# ============================================================
# Export columns
# ============================================================

BASE_EXPORT_COLUMNS = [
    "station",
    "station_id",
    "longitude",
    "latitude",

    "modis_pixel_id",
    "footprint_area_m2",

    "period_start",
    "period_end",
    "number_days",

    # --------------------------------------------------------
    # MODIS ET
    # --------------------------------------------------------

    "ET_mm_period",
    "ET_mm_day",

    "modis_value_valid",
    "modis_good",

    "ET_QC",
    "modis_qc_present",
    "modis_qc_good",
    "modis_modland_qc",
    "modis_sensor",
    "modis_dead_detector",
    "modis_cloud_state",
    "modis_scf_qc",

    # --------------------------------------------------------
    # Optical source and availability
    # --------------------------------------------------------

    "optical_source",
    "optical_scale_m",
    "optical_dates_total",
    "optical_dates",
    "optical_products_total",
    "optical_union_coverage_pct",
    "optical_valid",

    # --------------------------------------------------------
    # Sentinel-1 availability
    # --------------------------------------------------------

    "s1_dates_total",
    "s1_dates",
    "s1_products_total",
    "s1_union_coverage_pct",

    "s1_pass",
    "s1_relative_orbit",

    # --------------------------------------------------------
    # Meteorology
    # --------------------------------------------------------

    "Tair_mean_C",
    "Tair_max_C",
    "VPD_mean_kPa",
    "VPD_max_kPa",
    "SolarRad_MJ_m2_day",
    "Wind_mean_ms",
    "Precip_period_mm",
    "Precip_prev30d_mm",

    # --------------------------------------------------------
    # Meteorological temporal integrity
    # --------------------------------------------------------

    "era5_hours_total",
    "era5_hours_expected",

    "chirps_days_period",
    "chirps_days_expected",

    "chirps_days_prev30",
    "chirps_days_prev30_expected",

    # --------------------------------------------------------
    # Meteorological spatial support
    # --------------------------------------------------------

    "era5_support_m",
    "chirps_support_m",

    "era5_sampling_method",
    "era5_sampling_longitude",
    "era5_sampling_latitude",
    "era5_sampling_distance_m",

    # --------------------------------------------------------
    # Meteorological completeness
    # --------------------------------------------------------

    "meteo_missing_count",
    "meteo_temporal_complete",
    "meteo_complete",

    # --------------------------------------------------------
    # Spatial support
    # --------------------------------------------------------

    "scale",
    "predictor_support",
    "target_support",

    # --------------------------------------------------------
    # Predictor completeness
    # --------------------------------------------------------

    "missing_stats_count",
    "stats_complete",

    "system:time_start",
]


# ============================================================
# Final export selectors
# ============================================================

EXPORT_SELECTORS = (
    BASE_EXPORT_COLUMNS
    + STAT_COLUMNS
    + QA_STAT_COLUMNS
)


# ============================================================
# Base observation properties
# ============================================================

BASE_PROPERTY_NAMES = [
    "station",
    "station_id",
    "longitude",
    "latitude",

    "modis_pixel_id",
    "footprint_area_m2",

    "period_start",
    "period_end",
    "number_days",

    "ET_mm_period",
    "ET_mm_day",

    "modis_value_valid",
    "modis_good",

    "ET_QC",
    "modis_qc_present",
    "modis_qc_good",
    "modis_modland_qc",
    "modis_sensor",
    "modis_dead_detector",
    "modis_cloud_state",
    "modis_scf_qc",

    "optical_source",
    "optical_scale_m",
    "optical_dates_total",
    "optical_dates",
    "optical_products_total",
    "optical_union_coverage_pct",
    "optical_valid",

    "s1_dates_total",
    "s1_dates",
    "s1_products_total",
    "s1_union_coverage_pct",

    "Tair_mean_C",
    "Tair_max_C",
    "VPD_mean_kPa",
    "VPD_max_kPa",
    "SolarRad_MJ_m2_day",
    "Wind_mean_ms",
    "Precip_period_mm",
    "Precip_prev30d_mm",

    "era5_hours_total",
    "era5_hours_expected",

    "chirps_days_period",
    "chirps_days_expected",

    "chirps_days_prev30",
    "chirps_days_prev30_expected",

    "era5_support_m",
    "chirps_support_m",

    "era5_sampling_method",
    "era5_sampling_longitude",
    "era5_sampling_latitude",
    "era5_sampling_distance_m",

    "meteo_missing_count",
    "meteo_temporal_complete",
    "meteo_complete",

    "target_support",

    "system:time_start",
]
