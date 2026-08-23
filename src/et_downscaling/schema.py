from .config import normalize_optical_source


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
# Optical features that are directly comparable between S2
# and combined HLS S30 + L30
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


COMMON_OPTICAL_MODEL_BANDS = (
    COMMON_OPTICAL_REFLECTANCE_BANDS
    + COMMON_OPTICAL_INDEX_BANDS
)


# ============================================================
# Source-specific optical extraction features
#
# Extraction is intentionally richer than the common matched
# model feature set. These variables are retained so their
# value can be tested locally without another Earth Engine run.
#
# They are NOT automatically model predictors.
# ============================================================

S2_SOURCE_SPECIFIC_EXTRACTION_BANDS = [
    "RedEdge1",
    "RedEdge2",
    "RedEdge3",
    "NIR_Broad",
    "NDRE",
    "Albedo",
    "FVC",
]


HLS_SOURCE_SPECIFIC_EXTRACTION_BANDS = [
    "Albedo",
    "FVC",
]


SOURCE_OPTICAL_EXTRACTION_BANDS = {
    "S2": (
        COMMON_OPTICAL_MODEL_BANDS
        + S2_SOURCE_SPECIFIC_EXTRACTION_BANDS
    ),
    "HLS": (
        COMMON_OPTICAL_MODEL_BANDS
        + HLS_SOURCE_SPECIFIC_EXTRACTION_BANDS
    ),
}


# ============================================================
# Sentinel-1 model and QA bands
# ============================================================

S1_MODEL_BANDS = [
    "VV_dB",
    "VH_dB",
    "VV_minus_VH_dB",
]


S1_QA_BANDS = [
    "Angle_deg",
]


# ============================================================
# Model candidate sets
#
# COMMON_* is the matched S2/HLS set used for controlled source
# comparisons.
#
# SOURCE_* adds source-specific variables that may be evaluated
# later. Presence in these lists does not mean final inclusion.
# ============================================================

COMMON_SATELLITE_MODEL_BANDS = (
    COMMON_OPTICAL_MODEL_BANDS
    + S1_MODEL_BANDS
)


SOURCE_SATELLITE_MODEL_CANDIDATE_BANDS = {
    "S2": (
        COMMON_OPTICAL_MODEL_BANDS
        + S2_SOURCE_SPECIFIC_EXTRACTION_BANDS
        + S1_MODEL_BANDS
    ),
    "HLS": (
        COMMON_OPTICAL_MODEL_BANDS
        + HLS_SOURCE_SPECIFIC_EXTRACTION_BANDS
        + S1_MODEL_BANDS
    ),
}


# ============================================================
# Source-specific extraction helpers
# ============================================================

def get_optical_extraction_bands(
    source,
):
    source = normalize_optical_source(
        source
    )

    return list(
        SOURCE_OPTICAL_EXTRACTION_BANDS[
            source
        ]
    )


def get_satellite_extraction_bands(
    source,
):
    return (
        get_optical_extraction_bands(
            source
        )
        + S1_MODEL_BANDS
    )


def get_source_model_candidate_bands(
    source,
):
    source = normalize_optical_source(
        source
    )

    return list(
        SOURCE_SATELLITE_MODEL_CANDIDATE_BANDS[
            source
        ]
    )


def get_stat_columns(
    band_names,
):
    return [
        f"{band_name}_mean"
        for band_name in band_names
    ]


def get_satellite_stat_columns(
    source,
):
    return get_stat_columns(
        get_satellite_extraction_bands(
            source
        )
    )


def get_source_model_candidate_stat_columns(
    source,
):
    return get_stat_columns(
        get_source_model_candidate_bands(
            source
        )
    )


COMMON_SATELLITE_MODEL_STAT_COLUMNS = (
    get_stat_columns(
        COMMON_SATELLITE_MODEL_BANDS
    )
)


QA_STAT_COLUMNS = [
    "Angle_deg_mean",
]


# ============================================================
# Backward-compatible aliases
#
# These point only to the common matched model set. New code
# should prefer the explicit helper functions above.
# ============================================================

PREDICTOR_BANDS = (
    COMMON_SATELLITE_MODEL_BANDS
)

STAT_COLUMNS = (
    COMMON_SATELLITE_MODEL_STAT_COLUMNS
)

EXPECTED_STAT_KEYS = (
    STAT_COLUMNS.copy()
)


# ============================================================
# Satellite export columns
# ============================================================

SATELLITE_BASE_EXPORT_COLUMNS = [
    "station",
    "station_id",
    "longitude",
    "latitude",

    "modis_pixel_id",
    "footprint_area_m2",

    "period_start",
    "period_end",
    "number_days",

    # MODIS target and QC
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

    # Optical availability
    "optical_source",
    "optical_scale_m",
    "optical_dates_total",
    "optical_dates",
    "optical_products_total",
    "optical_union_coverage_pct",
    "optical_valid",

    # Sentinel-1 availability and geometry
    "s1_dates_total",
    "s1_dates",
    "s1_products_total",
    "s1_union_coverage_pct",
    "s1_valid",
    "s1_pass",
    "s1_relative_orbit",

    # Spatial support
    "scale",
    "predictor_support",
    "target_support",

    # Extraction completeness
    "missing_stats_count",
    "stats_complete",

    "system:time_start",
]


def get_satellite_export_selectors(
    source,
):
    return (
        SATELLITE_BASE_EXPORT_COLUMNS
        + get_satellite_stat_columns(
            source
        )
        + QA_STAT_COLUMNS
    )


# Default alias retained for older diagnostic code.
SATELLITE_EXPORT_SELECTORS = (
    get_satellite_export_selectors(
        "S2"
    )
)

EXPORT_SELECTORS = (
    SATELLITE_EXPORT_SELECTORS
)


# ============================================================
# Base observation properties used inside Earth Engine
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
    "s1_valid",

    "target_support",
    "system:time_start",
]
