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


S2_10M_SOURCE_BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
]

S2_10M_BAND_NAMES = [
    "Blue",
    "Green",
    "Red",
    "NIR_Broad",
]


S2_20M_SOURCE_BANDS = [
    "B5",
    "B6",
    "B7",
    "B8A",
    "B11",
    "B12",
]

S2_20M_BAND_NAMES = [
    "RedEdge1",
    "RedEdge2",
    "RedEdge3",
    "NIR",
    "SWIR1",
    "SWIR2",
]


S2_MEDOID_BANDS = [
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


S2_INDEX_BANDS = [
    "NDVI",
    "EVI",
    "SAVI",
    "NDWI",
    "NDMI",
    "NDRE",
]


S2_PREDICTOR_BANDS = (
    S2_BAND_NAMES
    + S2_INDEX_BANDS
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
        ee.ImageCollection(
            S2_COLLECTION_ID
        )
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
# Aggregate 10 m image to native 20 m grid
# ============================================================

def _aggregate_mean_to_20m(
    image,
    target_projection,
):
    return (
        ee.Image(image)
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            bestEffort=False,
            maxPixels=16,
        )
        .reproject(
            crs=target_projection
        )
    )


# ============================================================
# Require full 2 x 2 valid support
# ============================================================

def _aggregate_full_validity_to_20m(
    valid_10m,
    target_projection,
):
    return (
        ee.Image(valid_10m)
        .unmask(0)
        .reduceResolution(
            reducer=ee.Reducer.min(),
            bestEffort=False,
            maxPixels=16,
        )
        .reproject(
            crs=target_projection
        )
        .eq(1)
    )


# ============================================================
# Prepare Sentinel-2 at 20 m
# ============================================================

def prepare_sentinel2(image):
    image = ee.Image(image)

    target_projection = (
        image
        .select("B8A")
        .projection()
    )

    clear_10m = (
        image
        .select(S2_QA_BAND)
        .gte(
            S2_CLEAR_THRESHOLD
        )
    )

    spectral_valid_10m = (
        image
        .select(
            S2_10M_SOURCE_BANDS
        )
        .mask()
        .reduce(
            ee.Reducer.min()
        )
    )

    valid_10m = (
        spectral_valid_10m
        .And(clear_10m)
    )

    full_valid_20m = (
        _aggregate_full_validity_to_20m(
            valid_10m,
            target_projection,
        )
    )

    spectral_valid_20m = (
        image
        .select(
            S2_20M_SOURCE_BANDS
        )
        .mask()
        .reduce(
            ee.Reducer.min()
        )
        .unmask(0)
        .reproject(
            crs=target_projection
        )
        .eq(1)
    )

    final_valid_mask = (
        full_valid_20m
        .And(
            spectral_valid_20m
        )
    )

    reflectance_10m = (
        image
        .select(
            S2_10M_SOURCE_BANDS,
            S2_10M_BAND_NAMES,
        )
        .multiply(0.0001)
        .toFloat()
    )

    reflectance_10m_to_20m = (
        _aggregate_mean_to_20m(
            reflectance_10m,
            target_projection,
        )
    )

    reflectance_20m = (
        image
        .select(
            S2_20M_SOURCE_BANDS,
            S2_20M_BAND_NAMES,
        )
        .multiply(0.0001)
        .toFloat()
        .reproject(
            crs=target_projection
        )
    )

    reflectance = (
        reflectance_10m_to_20m
        .addBands(
            reflectance_20m
        )
        .select(
            S2_BAND_NAMES
        )
        .updateMask(
            final_valid_mask
        )
        .toFloat()
    )

    return ee.Image(
        reflectance.copyProperties(
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

    return (
        ee.ImageCollection.fromImages(
            date_keys.map(
                build_daily_image
            )
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
        .select(
            S2_MEDOID_BANDS
        )
        .median()
    )

    def score_image(image):
        image = ee.Image(image)

        squared_distance = (
            image
            .select(
                S2_MEDOID_BANDS
            )
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
# Safe ratio
# ============================================================

def _safe_ratio(
    numerator,
    denominator,
    band_name,
    epsilon=1e-6,
):
    numerator = ee.Image(
        numerator
    )

    denominator = ee.Image(
        denominator
    )

    valid_denominator = (
        denominator
        .abs()
        .gt(epsilon)
    )

    return (
        numerator
        .divide(
            denominator
        )
        .updateMask(
            valid_denominator
        )
        .rename(
            band_name
        )
        .toFloat()
    )


# ============================================================
# Sentinel-2 spectral indices
# ============================================================

def add_s2_indices(image):
    image = ee.Image(image)

    blue = image.select("Blue")
    green = image.select("Green")
    red = image.select("Red")
    red_edge_1 = image.select("RedEdge1")
    nir = image.select("NIR")
    swir1 = image.select("SWIR1")

    ndvi = _safe_ratio(
        nir.subtract(red),
        nir.add(red),
        "NDVI",
    )

    evi = _safe_ratio(
        nir
        .subtract(red)
        .multiply(2.5),
        (
            nir
            .add(
                red.multiply(6.0)
            )
            .subtract(
                blue.multiply(7.5)
            )
            .add(1.0)
        ),
        "EVI",
    )

    savi = _safe_ratio(
        nir
        .subtract(red)
        .multiply(1.5),
        (
            nir
            .add(red)
            .add(0.5)
        ),
        "SAVI",
    )

    ndwi = _safe_ratio(
        green.subtract(nir),
        green.add(nir),
        "NDWI",
    )

    ndmi = _safe_ratio(
        nir.subtract(swir1),
        nir.add(swir1),
        "NDMI",
    )

    ndre = _safe_ratio(
        nir.subtract(red_edge_1),
        nir.add(red_edge_1),
        "NDRE",
    )

    return (
        image
        .addBands(
            [
                ndvi,
                evi,
                savi,
                ndwi,
                ndmi,
                ndre,
            ]
        )
        .select(
            S2_PREDICTOR_BANDS
        )
        .toFloat()
    )


# ============================================================
# Sentinel-2 predictors
# ============================================================

def build_s2_predictors(
    s2_period,
    geometry,
):
    medoid = build_s2_medoid(
        s2_period,
        geometry,
    )

    return add_s2_indices(
        medoid
    )