import ee

from .config import (
    END_DATE,
    START_DATE,
)


# ============================================================
# HLS collections
# ============================================================

HLS_S30_COLLECTION_ID = "NASA/HLS/HLSS30/v002"
HLS_L30_COLLECTION_ID = "NASA/HLS/HLSL30/v002"

HLS_S30_SENSOR = "HLS_S30"
HLS_L30_SENSOR = "HLS_L30"

HLS_SENSOR_PROPERTY = "hls_sensor"
HLS_DATE_PROPERTY = "date_key"
HLS_ACQUISITION_PROPERTY = "acquisition_key"


# ============================================================
# Common HLS spectral space
# ============================================================

HLS_REFLECTANCE_BANDS = [
    "Coastal",
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
]


HLS_S30_SOURCE_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B8A",
    "B11",
    "B12",
]


HLS_L30_SOURCE_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
]


# These are the bands used by the HLS-derived indices.
# Coastal is intentionally excluded from the positivity QA.
HLS_INDEX_REFLECTANCE_BANDS = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
]


HLS_INDEX_BANDS = [
    "NDVI",
    "EVI",
    "SAVI",
    "NDWI",
    "NDMI",
]


HLS_PREDICTOR_BANDS = (
    HLS_REFLECTANCE_BANDS
    + HLS_INDEX_BANDS
)


# ============================================================
# Metadata
# ============================================================

def _add_hls_metadata(
    image,
    sensor_name,
):
    image = ee.Image(image)

    date_key = (
        image
        .date()
        .format("yyyy-MM-dd")
    )

    acquisition_key = (
        ee.String(sensor_name)
        .cat("_")
        .cat(date_key)
    )

    return ee.Image(
        image.set(
            {
                HLS_SENSOR_PROPERTY: sensor_name,
                HLS_DATE_PROPERTY: date_key,
                HLS_ACQUISITION_PROPERTY: acquisition_key,
            }
        )
    )


def _tag_hls_s30(image):
    return _add_hls_metadata(
        image,
        HLS_S30_SENSOR,
    )


def _tag_hls_l30(image):
    return _add_hls_metadata(
        image,
        HLS_L30_SENSOR,
    )


# ============================================================
# HLS collection
# ============================================================

def get_hls_collection(
    station_footprints,
):
    geometry = (
        station_footprints
        .geometry()
    )

    s30 = (
        ee.ImageCollection(
            HLS_S30_COLLECTION_ID
        )
        .filterBounds(
            geometry
        )
        .filterDate(
            START_DATE,
            END_DATE,
        )
        .map(
            _tag_hls_s30
        )
    )

    l30 = (
        ee.ImageCollection(
            HLS_L30_COLLECTION_ID
        )
        .filterBounds(
            geometry
        )
        .filterDate(
            START_DATE,
            END_DATE,
        )
        .map(
            _tag_hls_l30
        )
    )

    return (
        s30
        .merge(l30)
        .sort(
            "system:time_start"
        )
    )


# ============================================================
# HLS Fmask
# ============================================================

def build_hls_clear_mask(image):
    image = ee.Image(image)

    fmask = (
        image
        .select("Fmask")
    )

    cloud_clear = (
        fmask
        .bitwiseAnd(
            1 << 1
        )
        .eq(0)
    )

    adjacent_clear = (
        fmask
        .bitwiseAnd(
            1 << 2
        )
        .eq(0)
    )

    shadow_clear = (
        fmask
        .bitwiseAnd(
            1 << 3
        )
        .eq(0)
    )

    snow_clear = (
        fmask
        .bitwiseAnd(
            1 << 4
        )
        .eq(0)
    )

    return (
        cloud_clear
        .And(adjacent_clear)
        .And(shadow_clear)
        .And(snow_clear)
        .unmask(0)
        .rename(
            "hls_clear"
        )
    )


# ============================================================
# Positive-reflectance QA
# ============================================================

def build_hls_positive_reflectance_mask(
    reflectance,
):
    reflectance = ee.Image(
        reflectance
    )

    return (
        reflectance
        .select(
            HLS_INDEX_REFLECTANCE_BANDS
        )
        .gt(0)
        .reduce(
            ee.Reducer.min()
        )
        .rename(
            "hls_positive_reflectance"
        )
    )


# ============================================================
# Standardize one HLS product
# ============================================================

def prepare_hls(image):
    image = ee.Image(image)

    sensor = ee.String(
        image.get(
            HLS_SENSOR_PROPERTY
        )
    )

    is_s30 = (
        sensor
        .compareTo(
            HLS_S30_SENSOR
        )
        .eq(0)
    )

    reflectance = ee.Image(
        ee.Algorithms.If(
            is_s30,
            image.select(
                HLS_S30_SOURCE_BANDS,
                HLS_REFLECTANCE_BANDS,
            ),
            image.select(
                HLS_L30_SOURCE_BANDS,
                HLS_REFLECTANCE_BANDS,
            ),
        )
    ).toFloat()

    # Require all seven common reflectance bands to exist.
    spectral_valid_mask = (
        reflectance
        .mask()
        .unmask(0)
        .reduce(
            ee.Reducer.min()
        )
        .eq(1)
    )

    # Apply the HLS Fmask QA first.
    clear_mask = (
        build_hls_clear_mask(
            image
        )
    )

    reflectance = (
        reflectance
        .updateMask(
            spectral_valid_mask
        )
        .updateMask(
            clear_mask
        )
    )

    # Exclude acquisition pixels with non-positive reflectance
    # in any band required by the derived indices.
    #
    # Coastal is deliberately not part of this condition.
    positive_reflectance_mask = (
        build_hls_positive_reflectance_mask(
            reflectance
        )
    )

    reflectance = (
        reflectance
        .updateMask(
            positive_reflectance_mask
        )
        .toFloat()
    )

    return ee.Image(
        reflectance
        .copyProperties(
            image
        )
        .set(
            "system:time_start",
            image.get(
                "system:time_start"
            ),
        )
        .set(
            HLS_SENSOR_PROPERTY,
            image.get(
                HLS_SENSOR_PROPERTY
            ),
        )
        .set(
            HLS_DATE_PROPERTY,
            image.get(
                HLS_DATE_PROPERTY
            ),
        )
        .set(
            HLS_ACQUISITION_PROPERTY,
            image.get(
                HLS_ACQUISITION_PROPERTY
            ),
        )
    )


# ============================================================
# Empty HLS image
# ============================================================

def build_empty_hls_image():
    return (
        ee.Image.constant(
            [0]
            * len(
                HLS_REFLECTANCE_BANDS
            )
        )
        .rename(
            HLS_REFLECTANCE_BANDS
        )
        .updateMask(
            ee.Image.constant(0)
        )
        .toFloat()
    )


# ============================================================
# Build one mosaic per sensor and acquisition date
# ============================================================

def build_hls_acquisition_collection(
    hls_period,
    geometry,
):
    hls_period = ee.ImageCollection(
        hls_period
    )

    acquisition_keys = (
        ee.List(
            hls_period
            .aggregate_array(
                HLS_ACQUISITION_PROPERTY
            )
        )
        .distinct()
        .sort()
    )

    def build_acquisition(
        acquisition_key,
    ):
        acquisition_key = ee.String(
            acquisition_key
        )

        same_acquisition = (
            hls_period
            .filter(
                ee.Filter.eq(
                    HLS_ACQUISITION_PROPERTY,
                    acquisition_key,
                )
            )
            .map(
                prepare_hls
            )
        )

        first_image = ee.Image(
            same_acquisition
            .first()
        )

        reference_projection = (
            first_image
            .select("Coastal")
            .projection()
        )

        acquisition_mosaic = (
            same_acquisition
            .mosaic()
            .setDefaultProjection(
                reference_projection
            )
            .clip(
                geometry.buffer(100)
            )
            .set(
                HLS_ACQUISITION_PROPERTY,
                acquisition_key,
            )
            .set(
                HLS_DATE_PROPERTY,
                first_image.get(
                    HLS_DATE_PROPERTY
                ),
            )
            .set(
                HLS_SENSOR_PROPERTY,
                first_image.get(
                    HLS_SENSOR_PROPERTY
                ),
            )
            .set(
                "system:time_start",
                first_image.get(
                    "system:time_start"
                ),
            )
            .set(
                "hls_products_in_acquisition",
                same_acquisition.size(),
            )
        )

        return acquisition_mosaic

    return (
        ee.ImageCollection.fromImages(
            acquisition_keys.map(
                build_acquisition
            )
        )
        .sort(
            "system:time_start"
        )
    )


# ============================================================
# Union coverage mask
# ============================================================

def build_hls_union_mask(
    hls_period,
    geometry,
):
    acquisitions = (
        build_hls_acquisition_collection(
            hls_period,
            geometry,
        )
    )

    def acquisition_mask(image):
        image = ee.Image(image)

        return (
            image
            .select("Blue")
            .mask()
            .gt(0)
            .unmask(0)
            .rename(
                "hls_union_mask"
            )
            .toByte()
        )

    valid_masks = (
        acquisitions
        .map(
            acquisition_mask
        )
    )

    empty_mask = (
        ee.Image.constant(0)
        .rename(
            "hls_union_mask"
        )
        .toByte()
    )

    safe_masks = (
        valid_masks
        .merge(
            ee.ImageCollection.fromImages(
                [
                    empty_mask,
                ]
            )
        )
    )

    return (
        safe_masks
        .max()
        .rename(
            "hls_union_mask"
        )
        .clip(
            geometry.buffer(100)
        )
        .toByte()
    )


# ============================================================
# HLS medoid
# ============================================================

def build_hls_medoid(
    hls_period,
    geometry,
):
    acquisitions = (
        build_hls_acquisition_collection(
            hls_period,
            geometry,
        )
    )

    safe_acquisitions = (
        acquisitions
        .merge(
            ee.ImageCollection.fromImages(
                [
                    build_empty_hls_image(),
                ]
            )
        )
    )

    spectral_median = (
        safe_acquisitions
        .select(
            HLS_REFLECTANCE_BANDS
        )
        .median()
    )

    def score_image(image):
        image = ee.Image(image)

        squared_distance = (
            image
            .select(
                HLS_REFLECTANCE_BANDS
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

        return (
            image
            .addBands(
                medoid_score
            )
        )

    scored_collection = (
        safe_acquisitions
        .map(
            score_image
        )
    )

    return (
        scored_collection
        .qualityMosaic(
            "medoid_score"
        )
        .select(
            HLS_REFLECTANCE_BANDS
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
        .gt(
            epsilon
        )
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
# HLS indices
# ============================================================

def add_hls_indices(image):
    image = ee.Image(image)

    blue = (
        image
        .select("Blue")
    )

    green = (
        image
        .select("Green")
    )

    red = (
        image
        .select("Red")
    )

    nir = (
        image
        .select("NIR")
    )

    swir1 = (
        image
        .select("SWIR1")
    )

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

    # Homogeneous SAVI definition used in this project
    # for both HLS and Sentinel-2: L = 0.5.
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

    return (
        image
        .addBands(
            [
                ndvi,
                evi,
                savi,
                ndwi,
                ndmi,
            ]
        )
        .select(
            HLS_PREDICTOR_BANDS
        )
        .toFloat()
    )


# ============================================================
# HLS predictors
# ============================================================

def build_hls_predictors(
    hls_period,
    geometry,
):
    medoid = (
        build_hls_medoid(
            hls_period,
            geometry,
        )
    )

    return (
        add_hls_indices(
            medoid
        )
    )