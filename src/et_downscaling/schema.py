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
# Sentinel-2 bands
# ============================================================

S2_SOURCE_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
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
    "NIR",
    "SWIR1",
    "SWIR2",
]


# ============================================================
# Predictor bands
# ============================================================

PREDICTOR_BANDS = [
    "Blue",
    "Green",
    "Red",
    "RedEdge1",
    "RedEdge2",
    "RedEdge3",
    "NIR",
    "SWIR1",
    "SWIR2",
    "NDVI",
    "NDMI",
    "NDWI",
    "VV_dB",
    "VH_dB",
    "VV_minus_VH_dB",
    "Angle_deg",
]


# ============================================================
# Statistical columns
# ============================================================

STAT_COLUMNS = []

for band_name in PREDICTOR_BANDS:
    STAT_COLUMNS.append(
        f"{band_name}_mean"
    )

    STAT_COLUMNS.append(
        f"{band_name}_stdDev"
    )

EXPECTED_STAT_KEYS = (
    STAT_COLUMNS.copy()
)


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

    "ET_mm_period",
    "ET_mm_day",

    "modis_good",
    "modis_scf_qc",

    "s2_dates_total",
    "s2_dates",
    "s2_union_coverage_pct",

    "s1_dates_total",
    "s1_dates",
    "s1_products_total",
    "s1_union_coverage_pct",

    "s1_pass",
    "s1_relative_orbit",

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

    "scale",
    "predictor_support",
    "target_support",

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

    "modis_good",
    "modis_scf_qc",

    "s2_dates_total",
    "s2_dates",
    "s2_union_coverage_pct",

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