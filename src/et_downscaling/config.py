import os

from .period import AnalysisPeriod


# ============================================================
# Local station input
# ============================================================

STATIONS_GEOJSON_PATH = "data/stations/fundacion_stations.geojson"


# ============================================================
# Analysis period
# ============================================================

# START_DATE is inclusive.
# END_DATE is exclusive, following Earth Engine filterDate().
START_DATE = os.environ.get("ET_START_DATE", "2021-01-01")
END_DATE = os.environ.get("ET_END_DATE_EXCLUSIVE", "2024-01-01")
ANALYSIS_PERIOD = AnalysisPeriod.from_strings(START_DATE, END_DATE)


# ============================================================
# MODIS MOD16A2GF
# ============================================================

MODIS_ET_SCALE_FACTOR = 0.1

MODIS_ET_MIN_VALID_DN = 0
MODIS_ET_MAX_VALID_DN = 32700

MODIS_STRICT_SCF_MAX = 1

# False:
# MODIS validity is based on the ET value itself.
# ET_QC is retained for traceability and sensitivity analyses.
#
# True:
# Also require the legacy strict ET_QC criteria.
MODIS_REQUIRE_STRICT_QC = False


# ============================================================
# Analysis-period label
# ============================================================

def _build_period_label():
    return ANALYSIS_PERIOD.label


OUTPUT_PERIOD_LABEL = (
    _build_period_label()
)


# ============================================================
# Spatial analysis
# ============================================================

# Common projected CRS for footprint reductions in the
# Fundación study area.
ANALYSIS_CRS = "EPSG:32618"

# Legacy default retained for backward compatibility only.
# Production optical processing must use the source-specific
# scale returned by get_optical_scale().
ANALYSIS_SCALE = 20


# ============================================================
# Optical source
# ============================================================

DEFAULT_OPTICAL_SOURCE = "S2"

SUPPORTED_OPTICAL_SOURCES = (
    "S2",
    "HLS",
)

OPTICAL_SOURCE_ALIASES = {
    "S2": "S2",
    "SENTINEL2": "S2",
    "SENTINEL-2": "S2",
    "HLS": "HLS",
    "HLS_COMBINED": "HLS",
}

OPTICAL_OUTPUT_LABELS = {
    "S2": "S2",
    "HLS": "HLS_COMBINED",
}

OPTICAL_SCALES_M = {
    "S2": 20,
    "HLS": 30,
}

# Coverage is retained continuously in the raw satellite master.
# These thresholds are evaluated locally; they do not determine
# Earth Engine extraction eligibility.
OPTICAL_FULL_COVERAGE = 0.999
OPTICAL_QA_THRESHOLDS_PCT = (80.0, 90.0, 99.0)


def normalize_optical_source(
    source,
):
    normalized = (
        str(source)
        .strip()
        .upper()
    )

    if normalized not in OPTICAL_SOURCE_ALIASES:
        raise ValueError(
            "Unsupported optical source: "
            f"{source}. Expected one of: "
            "S2, HLS, HLS_COMBINED."
        )

    return OPTICAL_SOURCE_ALIASES[
        normalized
    ]


def get_optical_scale(
    source,
):
    source = normalize_optical_source(
        source
    )

    return OPTICAL_SCALES_M[
        source
    ]


def get_optical_output_label(
    source,
):
    source = normalize_optical_source(
        source
    )

    return OPTICAL_OUTPUT_LABELS[
        source
    ]


def build_satellite_output_filename(
    source,
):
    source_label = (
        get_optical_output_label(
            source
        )
    )

    return (
        f"ET_{source_label}_S1_SATELLITE_FOOTPRINT_"
        f"{OUTPUT_PERIOD_LABEL}.csv"
    )


def build_training_output_filename(
    source,
):
    source_label = (
        get_optical_output_label(
            source
        )
    )

    return (
        f"ET_{source_label}_S1_METEO_KC_FOOTPRINT_"
        f"{OUTPUT_PERIOD_LABEL}.csv"
    )


# ============================================================
# Sentinel-2
# ============================================================

S2_QA_BAND = "cs_cdf"
S2_CLEAR_THRESHOLD = 0.60

# Retained for notebooks and backward compatibility.
S2_FULL_COVERAGE = OPTICAL_FULL_COVERAGE


# ============================================================
# HLS
# ============================================================

HLS_FULL_COVERAGE = OPTICAL_FULL_COVERAGE


# ============================================================
# Sentinel-1
# ============================================================

S1_FULL_COVERAGE = 0.999
S1_ORBIT_PASS = "ASCENDING"
S1_RELATIVE_ORBIT = 77
S1_FOOTPRINT_REDUCTION_SCALE_M = 10


# ============================================================
# ERA5-Land
# ============================================================

ERA5_SEARCH_RADIUS_M = 50000


# ============================================================
# Output
# ============================================================

OUTPUT_DIRECTORY = "outputs"

# Backward-compatible default. The production script builds the
# filename dynamically from the selected optical source.
OUTPUT_FILENAME = (
    build_training_output_filename(
        DEFAULT_OPTICAL_SOURCE
    )
)
