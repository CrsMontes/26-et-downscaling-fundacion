import ee

from .config import (
    END_DATE,
    START_DATE,
)


HLS_S30_COLLECTION_ID = "NASA/HLS/HLSS30/v002"
HLS_L30_COLLECTION_ID = "NASA/HLS/HLSL30/v002"


HLS_S30_SOURCE_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B8A",
    "B11",
    "B12",
    "Fmask",
]

HLS_L30_SOURCE_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "Fmask",
]

HLS_STANDARD_BANDS = [
    "Coastal",
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
    "Fmask",
]

HLS_REFLECTANCE_BANDS = [
    "Coastal",
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
]


def _prepare_source_image(
    image,
    source_bands,
    source_name,
):
    image = ee.Image(image)

    standardized = image.select(
        source_bands,
        HLS_STANDARD_BANDS,
    )

    standardized = standardized.set(
        {
            "optical_sensor": source_name,
            "date_key": image.date().format(
                "yyyy-MM-dd"
            ),
        }
    )

    return ee.Image(
        standardized.copyProperties(
            image,
            [
                "system:time_start",
            ],
        )
    )


def _prepare_s30_source(image):
    return _prepare_source_image(
        image,
        HLS_S30_SOURCE_BANDS,
        "HLS_S30",
    )


def _prepare_l30_source(image):
    return _prepare_source_image(
        image,
        HLS_L30_SOURCE_BANDS,
        "HLS_L30",
    )


def get_hls_collection(
    station_footprints,
):
    geometry = station_footprints.geometry()

    s30 = (
        ee.ImageCollection(
            HLS_S30_COLLECTION_ID
        )
        .filterBounds(geometry)
        .filterDate(
            START_DATE,
            END_DATE,
        )
        .map(_prepare_s30_source)
    )

    l30 = (
        ee.ImageCollection(
            HLS_L30_COLLECTION_ID
        )
        .filterBounds(geometry)
        .filterDate(
            START_DATE,
            END_DATE,
        )
        .map(_prepare_l30_source)
    )

    return (
        s30
        .merge(l30)
        .sort("system:time_start")
    )


def build_hls_clear_mask(image):
    image = ee.Image(image)

    fmask = image.select("Fmask")

    cloud = (
        fmask
        .bitwiseAnd(1 << 1)
        .neq(0)
    )

    adjacent = (
        fmask
        .bitwiseAnd(1 << 2)
        .neq(0)
    )

    cloud_shadow = (
        fmask
        .bitwiseAnd(1 << 3)
        .neq(0)
    )

    snow_ice = (
        fmask
        .bitwiseAnd(1 << 4)
        .neq(0)
    )

    invalid = (
        cloud
        .Or(adjacent)
        .Or(cloud_shadow)
        .Or(snow_ice)
    )

    return invalid.Not()


def prepare_hls(image):
    image = ee.Image(image)

    clear_mask = build_hls_clear_mask(
        image
    )

    spectral_mask = (
        image
        .select(
            HLS_REFLECTANCE_BANDS
        )
        .mask()
        .reduce(
            ee.Reducer.min()
        )
    )

    valid_mask = (
        spectral_mask
        .And(clear_mask)
    )

    reflectance = (
        image
        .select(
            HLS_REFLECTANCE_BANDS
        )
        .toFloat()
        .updateMask(
            valid_mask
        )
    )

    return ee.Image(
        reflectance.copyProperties(
            image,
            [
                "system:time_start",
                "date_key",
                "optical_sensor",
            ],
        )
    )