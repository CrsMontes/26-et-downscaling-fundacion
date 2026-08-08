from datetime import date, timedelta


# ============================================================
# Samples
# ============================================================

SAMPLES_ASSET = "projects/ee-change/assets/ETP_samples"
STATION_FIELD = "Name"


# ============================================================
# Analysis period
# ============================================================

# START_DATE is inclusive.
# END_DATE is exclusive, following Earth Engine filterDate().
START_DATE = "2021-01-01"
END_DATE = "2024-01-01"


def _build_period_label():
    start_date = date.fromisoformat(
        START_DATE
    )

    end_date = date.fromisoformat(
        END_DATE
    )

    if end_date <= start_date:
        raise ValueError(
            "END_DATE must be later than START_DATE."
        )

    last_included_date = (
        end_date
        - timedelta(days=1)
    )

    # Use a compact label for complete calendar years.
    if (
        start_date.month == 1
        and start_date.day == 1
        and end_date.month == 1
        and end_date.day == 1
    ):
        last_year = end_date.year - 1

        if start_date.year == last_year:
            return str(
                start_date.year
            )

        return (
            f"{start_date.year}_"
            f"{last_year}"
        )

    # Use exact dates for partial-year periods.
    return (
        f"{start_date:%Y%m%d}_"
        f"{last_included_date:%Y%m%d}"
    )


OUTPUT_PERIOD_LABEL = (
    _build_period_label()
)


# ============================================================
# Spatial analysis
# ============================================================

ANALYSIS_CRS = "EPSG:32618"
ANALYSIS_SCALE = 20


# ============================================================
# Sentinel-2
# ============================================================

S2_QA_BAND = "cs_cdf"
S2_CLEAR_THRESHOLD = 0.60
S2_FULL_COVERAGE = 0.999


# ============================================================
# Sentinel-1
# ============================================================

S1_FULL_COVERAGE = 0.999
S1_ORBIT_PASS = "ASCENDING"
S1_RELATIVE_ORBIT = 77


# ============================================================
# ERA5-Land
# ============================================================

ERA5_SEARCH_RADIUS_M = 50000


# ============================================================
# Output
# ============================================================

OUTPUT_DIRECTORY = "outputs"

OUTPUT_FILENAME = (
    "ET_S2_S1_METEO_MULTISCALE_"
    f"{OUTPUT_PERIOD_LABEL}.csv"
)