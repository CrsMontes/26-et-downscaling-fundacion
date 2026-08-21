import json
import math
from pathlib import Path

import ee


# ============================================================
# FVC calibration configuration
# ============================================================

FVC_CONFIG_RELATIVE_PATH = (
    Path("config")
    / "fvc_endmembers.json"
)

SUPPORTED_FVC_SOURCES = {
    "HLS",
    "S2",
}


# ============================================================
# Locate calibration file
# ============================================================

def find_fvc_config_path() -> Path:
    module_path = Path(
        __file__
    ).resolve()

    search_roots = [
        module_path.parent,
        *module_path.parents,
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
    ]

    checked_paths = []

    for root in search_roots:
        candidate = (
            root
            / FVC_CONFIG_RELATIVE_PATH
        )

        checked_paths.append(
            candidate
        )

        if candidate.is_file():
            return candidate

    checked_text = "\n".join(
        str(path)
        for path in checked_paths
    )

    raise FileNotFoundError(
        "FVC calibration file was not found. "
        "The production pipeline requires "
        "'config/fvc_endmembers.json'.\n"
        f"Checked:\n{checked_text}"
    )


# ============================================================
# Load and validate source-specific endmembers
# ============================================================

def load_fvc_endmembers(
    source: str,
) -> tuple[float, float]:
    source = str(
        source
    ).upper()

    if source not in SUPPORTED_FVC_SOURCES:
        raise ValueError(
            "Unsupported FVC optical source: "
            f"{source}. "
            "Expected one of: "
            f"{sorted(SUPPORTED_FVC_SOURCES)}."
        )

    config_path = (
        find_fvc_config_path()
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        calibration = json.load(
            file
        )

    sources = calibration.get(
        "sources"
    )

    if not isinstance(
        sources,
        dict,
    ):
        raise ValueError(
            "Invalid FVC calibration file: "
            "'sources' dictionary is missing."
        )

    if source not in sources:
        raise ValueError(
            "FVC calibration does not contain "
            f"source '{source}'."
        )

    source_calibration = (
        sources[source]
    )

    try:
        ndvi_low = float(
            source_calibration[
                "ndvi_low_endmember"
            ]
        )

        ndvi_high = float(
            source_calibration[
                "ndvi_high_endmember"
            ]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Invalid FVC calibration for "
            f"source '{source}'. "
            "Expected numeric "
            "'ndvi_low_endmember' and "
            "'ndvi_high_endmember'."
        ) from error

    if not (
        math.isfinite(ndvi_low)
        and math.isfinite(ndvi_high)
    ):
        raise ValueError(
            "FVC NDVI endmembers must be finite."
        )

    if not (
        -1.0
        <= ndvi_low
        < ndvi_high
        <= 1.0
    ):
        raise ValueError(
            "Invalid FVC NDVI endmembers for "
            f"source '{source}': "
            f"low={ndvi_low}, "
            f"high={ndvi_high}. "
            "Expected "
            "-1 <= low < high <= 1."
        )

    return (
        ndvi_low,
        ndvi_high,
    )


# ============================================================
# Raw FVC
# ============================================================

def calculate_fvc_raw(
    image,
    source: str,
):
    image = ee.Image(
        image
    )

    (
        ndvi_low,
        ndvi_high,
    ) = load_fvc_endmembers(
        source
    )

    ndvi_range = (
        ndvi_high
        - ndvi_low
    )

    return (
        image
        .select("NDVI")
        .subtract(
            ndvi_low
        )
        .divide(
            ndvi_range
        )
        .rename(
            "FVC_raw"
        )
        .toFloat()
    )


# ============================================================
# Physical FVC
# ============================================================

def calculate_fvc(
    image,
    source: str,
):
    return (
        calculate_fvc_raw(
            image,
            source,
        )
        .clamp(
            0.0,
            1.0,
        )
        .rename(
            "FVC"
        )
        .toFloat()
    )


# ============================================================
# Add FVC predictor band
# ============================================================

def add_fvc_band(
    image,
    source: str,
):
    image = ee.Image(
        image
    )

    fvc = calculate_fvc(
        image,
        source,
    )

    return (
        image
        .addBands(
            fvc
        )
    )