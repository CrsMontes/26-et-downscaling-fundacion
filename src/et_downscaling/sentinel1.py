import ee

from .config import (
    END_DATE,
    S1_ORBIT_PASS,
    S1_RELATIVE_ORBIT,
    START_DATE,
)

from .spatial import get_coverage_fraction

S1_COLLECTION_ID = "COPERNICUS/S1_GRD"


# ============================================================
# Add date key
# ============================================================

def _add_date_key(image):
    image = ee.Image(image)

    return ee.Image(
        image.set(
            "date_key",
            image.date().format("yyyy-MM-dd"),
        )
    )


# ============================================================
# Sentinel-1 collection
# ============================================================

def get_sentinel1_collection(
    station_footprints,
):
    geometry = station_footprints.geometry()

    return (
        ee.ImageCollection(S1_COLLECTION_ID)
        .filterBounds(geometry)
        .filterDate(
            START_DATE,
            END_DATE,
        )
        .filter(
            ee.Filter.eq(
                "instrumentMode",
                "IW",
            )
        )
        .filter(
            ee.Filter.eq(
                "orbitProperties_pass",
                S1_ORBIT_PASS,
            )
        )
        .filter(
            ee.Filter.eq(
                "relativeOrbitNumber_start",
                S1_RELATIVE_ORBIT,
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VV",
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VH",
            )
        )
        .select(
            [
                "VV",
                "VH",
                "angle",
            ]
        )
        .map(_add_date_key)
    )


# ============================================================
# Empty Sentinel-1 image
# ============================================================

def build_empty_s1_image():
    return (
        ee.Image.constant(
            [
                0,
                0,
                0,
            ]
        )
        .rename(
            [
                "VV",
                "VH",
                "angle",
            ]
        )
        .updateMask(
            ee.Image.constant(0)
        )
        .toFloat()
    )


# ============================================================
# Sentinel-1 median predictors
# ============================================================

def build_s1_median(
    s1_period,
    geometry,
):
    s1_period = ee.ImageCollection(
        s1_period
    )

    empty_s1_image = (
        build_empty_s1_image()
    )

    safe_collection = (
        s1_period.merge(
            ee.ImageCollection(
                [
                    empty_s1_image,
                ]
            )
        )
    )

    median = (
        safe_collection
        .median()
        .clip(
            geometry.buffer(100)
        )
    )

    vv = (
        median
        .select("VV")
        .rename("VV_dB")
    )

    vh = (
        median
        .select("VH")
        .rename("VH_dB")
    )

    vv_minus_vh = (
        vv
        .subtract(vh)
        .rename(
            "VV_minus_VH_dB"
        )
    )

    angle = (
        median
        .select("angle")
        .rename("Angle_deg")
    )

    return (
        vv
        .addBands(vh)
        .addBands(vv_minus_vh)
        .addBands(angle)
        .toFloat()
    )

# ============================================================
# Sentinel-1 spatial coverage
# ============================================================

def get_s1_coverage(
    s1_predictors,
    geometry,
):
    s1_predictors = ee.Image(
        s1_predictors
    )

    joint_mask = (
        s1_predictors
        .select(
            [
                "VV_dB",
                "VH_dB",
            ]
        )
        .mask()
        .reduce(
            ee.Reducer.min()
        )
        .rename("valid")
    )

    return get_coverage_fraction(
        joint_mask,
        geometry,
    )