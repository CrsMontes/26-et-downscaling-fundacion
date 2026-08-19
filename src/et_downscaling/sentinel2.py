import ee

from .config import (
    END_DATE,
    S2_CLEAR_THRESHOLD,
    S2_QA_BAND,
    START_DATE,
)

from .schema import (
    S2_BAND_NAMES,
    S2_SOURCE_BANDS,
)


S2_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"

CLOUD_SCORE_COLLECTION_ID = (
    "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
)


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
# Sentinel-2 + Cloud Score+
# ============================================================

def get_sentinel2_collection(
    station_footprints,
):
    geometry = station_footprints.geometry()

    s2_raw = (
        ee.ImageCollection(S2_COLLECTION_ID)
        .filterBounds(geometry)
        .filterDate(
            START_DATE,
            END_DATE,
        )
    )

    cloud_score = (
        ee.ImageCollection(
            CLOUD_SCORE_COLLECTION_ID
        )
        .filterBounds(geometry)
        .filterDate(
            START_DATE,
            END_DATE,
        )
    )

    return (
        s2_raw
        .linkCollection(
            cloud_score,
            [S2_QA_BAND],
        )
        .map(_add_date_key)
    )


# ============================================================
# Prepare Sentinel-2
# ============================================================

def prepare_sentinel2(image):
    image = ee.Image(image)

    spectral_mask = (
        image
        .select(S2_SOURCE_BANDS)
        .mask()
        .reduce(
            ee.Reducer.min()
        )
    )

    clear_mask = (
        image
        .select(S2_QA_BAND)
        .gte(
            S2_CLEAR_THRESHOLD
        )
    )

    reflectance = (
        image
        .select(
            S2_SOURCE_BANDS,
            S2_BAND_NAMES,
        )
        .multiply(0.0001)
        .toFloat()
    )

    return ee.Image(
        reflectance
        .updateMask(
            spectral_mask.And(
                clear_mask
            )
        )
        .resample("bilinear")
        .copyProperties(
            image,
            [
                "system:time_start",
                "system:index",
                "date_key",
                "MGRS_TILE",
            ],
        )
    )


# ============================================================
# Empty Sentinel-2 image
# ============================================================

def build_empty_s2_image():
    return (
        ee.Image.constant(
            [0] * len(S2_BAND_NAMES)
        )
        .rename(
            S2_BAND_NAMES
        )
        .updateMask(
            ee.Image.constant(0)
        )
        .toFloat()
    )


# ============================================================
# Daily Sentinel-2 mosaics
# ============================================================

def build_s2_daily_collection(
    s2_period,
    geometry,
):
    s2_period = ee.ImageCollection(
        s2_period
    )

    date_keys = (
        ee.List(
            s2_period.aggregate_array(
                "date_key"
            )
        )
        .distinct()
        .sort()
    )

    def build_daily_image(date_key):
        date_key = ee.String(
            date_key
        )

        same_date = (
            s2_period
            .filter(
                ee.Filter.eq(
                    "date_key",
                    date_key,
                )
            )
            .map(
                prepare_sentinel2
            )
        )

        return (
            same_date
            .mosaic()
            .clip(
                geometry.buffer(100)
            )
            .set(
                "date_key",
                date_key,
            )
        )

    return ee.ImageCollection.fromImages(
        date_keys.map(
            build_daily_image
        )
    )


# ============================================================
# Sentinel-2 medoid
# ============================================================

def build_s2_medoid(
    s2_period,
    geometry,
):
    daily_images = (
        build_s2_daily_collection(
            s2_period,
            geometry,
        )
    )

    safe_daily_images = (
        daily_images.merge(
            ee.ImageCollection(
                [
                    build_empty_s2_image(),
                ]
            )
        )
    )

    spectral_median = (
        safe_daily_images
        .select(S2_BAND_NAMES)
        .median()
    )

    def score_image(image):
        image = ee.Image(image)

        squared_distance = (
            image
            .select(S2_BAND_NAMES)
            .subtract(
                spectral_median
            )
            .pow(2)
            .reduce(
                ee.Reducer.sum()
            )
        )

        medoid_score = (
            squared_distance
            .multiply(-1)
            .rename(
                "medoid_score"
            )
        )

        return image.addBands(
            medoid_score
        )

    scored_collection = (
        safe_daily_images.map(
            score_image
        )
    )

    return (
        scored_collection
        .qualityMosaic(
            "medoid_score"
        )
        .select(
            S2_BAND_NAMES
        )
        .toFloat()
    )


# ============================================================
# Sentinel-2 spectral indices
# ============================================================

def add_s2_indices(image):
    image = ee.Image(image)

    ndvi = (
        image
        .normalizedDifference(
            [
                "NIR",
                "Red",
            ]
        )
        .rename("NDVI")
    )

    ndmi = (
        image
        .normalizedDifference(
            [
                "NIR",
                "SWIR1",
            ]
        )
        .rename("NDMI")
    )

    ndwi = (
        image
        .normalizedDifference(
            [
                "Green",
                "NIR",
            ]
        )
        .rename("NDWI")
    )

    return image.addBands(
        [
            ndvi,
            ndmi,
            ndwi,
        ]
    )