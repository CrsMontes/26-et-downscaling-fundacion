import ee

from .albedo import (
    add_s2_albedo,
)

from .config import (
    END_DATE,
    S2_CLEAR_THRESHOLD,
    S2_QA_BAND,
    START_DATE,
)

from .fvc import (
    add_fvc_band,
)


# ============================================================
# Sentinel-2 collections
# ============================================================

S2_COLLECTION_ID = (
    "COPERNICUS/S2_SR_HARMONIZED"
)

CLOUD_SCORE_COLLECTION_ID = (
    "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
)


# ============================================================
# Sentinel-2 spectral configuration
# ============================================================

# Native 10 m bands.
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


# Native 20 m bands.
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


# Final reflectance stack at common 20 m support.
S2_REFLECTANCE_BANDS = [
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


# B8 and B8A both describe the NIR region.
# NIR_Broad (B8) is retained as a predictor and for albedo,
# but it is excluded from the medoid distance to avoid
# double-weighting NIR information.
S2_MEDOID_SCORE_BANDS = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
]


# ============================================================
# Add date key
# ============================================================

def _add_date_key(image):
    image = ee.Image(image)

    return ee.Image(
        image.set(
            "date_key",
            image.date().format(
                "yyyy-MM-dd"
            ),
        )
    )


# ============================================================
# Sentinel-2 + Cloud Score+
# ============================================================

def get_sentinel2_collection(
    station_footprints,
):
    geometry = (
        ee.FeatureCollection(
            station_footprints
        )
        .geometry()
    )

    s2_raw = (
        ee.ImageCollection(
            S2_COLLECTION_ID
        )
        .filterBounds(
            geometry
        )
        .filterDate(
            START_DATE,
            END_DATE,
        )
    )

    cloud_score = (
        ee.ImageCollection(
            CLOUD_SCORE_COLLECTION_ID
        )
        .filterBounds(
            geometry
        )
        .filterDate(
            START_DATE,
            END_DATE,
        )
    )

    return (
        s2_raw
        .linkCollection(
            cloud_score,
            [
                S2_QA_BAND,
            ],
        )
        .map(
            _add_date_key
        )
    )


# ============================================================
# Aggregate 10 m bands to the native 20 m grid
# ============================================================

def _aggregate_10m_reflectance_to_20m(
    image,
    reference_projection,
):
    image = ee.Image(
        image
    )

    reflectance_10m = (
        image
        .select(
            S2_10M_SOURCE_BANDS,
            S2_10M_BAND_NAMES,
        )
        .multiply(
            0.0001
        )
        .toFloat()
    )

    return (
        reflectance_10m
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=4,
        )
        .reproject(
            reference_projection
        )
        .toFloat()
    )


# ============================================================
# Build strict 20 m clear-sky mask
# ============================================================

def _build_20m_clear_mask(
    image,
    reference_projection,
):
    image = ee.Image(
        image
    )

    # All required 10 m spectral bands must exist.
    valid_10m_spectral = (
        image
        .select(
            S2_10M_SOURCE_BANDS
        )
        .mask()
        .reduce(
            ee.Reducer.min()
        )
    )

    # Cloud Score+ must also exist.
    valid_qa = (
        image
        .select(
            S2_QA_BAND
        )
        .mask()
    )

    clear_10m = (
        image
        .select(
            S2_QA_BAND
        )
        .gte(
            S2_CLEAR_THRESHOLD
        )
    )

    valid_clear_10m = (
        valid_10m_spectral
        .And(
            valid_qa
        )
        .And(
            clear_10m
        )
        .unmask(0)
        .rename(
            "valid_clear_10m"
        )
    )

    # A 20 m output pixel is accepted only when all
    # four nested 10 m subpixels are clear and valid.
    all_clear_20m = (
        valid_clear_10m
        .reduceResolution(
            reducer=ee.Reducer.min(),
            maxPixels=4,
        )
        .reproject(
            reference_projection
        )
        .eq(1)
    )

    # Native 20 m bands must also contain valid data.
    valid_20m_spectral = (
        image
        .select(
            S2_20M_SOURCE_BANDS
        )
        .mask()
        .reduce(
            ee.Reducer.min()
        )
        .reproject(
            reference_projection
        )
    )

    return (
        all_clear_20m
        .And(
            valid_20m_spectral
        )
        .rename(
            "valid_20m"
        )
    )


# ============================================================
# Prepare Sentinel-2
# ============================================================

def prepare_sentinel2(image):
    image = ee.Image(
        image
    )

    # B8A is native 20 m and defines the common grid.
    reference_projection = (
        image
        .select(
            "B8A"
        )
        .projection()
    )

    reflectance_from_10m = (
        _aggregate_10m_reflectance_to_20m(
            image,
            reference_projection,
        )
    )

    reflectance_native_20m = (
        image
        .select(
            S2_20M_SOURCE_BANDS,
            S2_20M_BAND_NAMES,
        )
        .multiply(
            0.0001
        )
        .reproject(
            reference_projection
        )
        .toFloat()
    )

    clear_mask_20m = (
        _build_20m_clear_mask(
            image,
            reference_projection,
        )
    )

    reflectance = (
        reflectance_from_10m
        .addBands(
            reflectance_native_20m
        )
        .select(
            S2_REFLECTANCE_BANDS
        )
        .updateMask(
            clear_mask_20m
        )
        .toFloat()
    )

    return ee.Image(
        reflectance
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
            [
                0,
            ]
            * len(
                S2_REFLECTANCE_BANDS
            )
        )
        .rename(
            S2_REFLECTANCE_BANDS
        )
        .updateMask(
            ee.Image.constant(
                0
            )
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
    s2_period = (
        ee.ImageCollection(
            s2_period
        )
    )

    date_keys = (
        ee.List(
            s2_period
            .aggregate_array(
                "date_key"
            )
        )
        .distinct()
        .sort()
    )

    def build_daily_image(
        date_key,
    ):
        date_key = (
            ee.String(
                date_key
            )
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
                ee.Geometry(
                    geometry
                )
                .buffer(
                    100
                )
            )
            .set(
                "date_key",
                date_key,
            )
        )

    return (
        ee.ImageCollection
        .fromImages(
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

    # Add a fully masked image so empty periods return
    # the expected band structure instead of failing.
    safe_daily_images = (
        daily_images
        .merge(
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
            S2_MEDOID_SCORE_BANDS
        )
        .median()
    )

    def score_image(
        image,
    ):
        image = ee.Image(
            image
        )

        squared_distance = (
            image
            .select(
                S2_MEDOID_SCORE_BANDS
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
            .multiply(
                -1
            )
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
        safe_daily_images
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
            S2_REFLECTANCE_BANDS
        )
        .toFloat()
    )


# ============================================================
# Safe ratio
# ============================================================

def _safe_ratio(
    numerator,
    denominator,
    output_name,
    epsilon=1e-6,
):
    numerator = ee.Image(
        numerator
    )

    denominator = ee.Image(
        denominator
    )

    return (
        numerator
        .divide(
            denominator
        )
        .updateMask(
            denominator
            .abs()
            .gt(
                epsilon
            )
        )
        .rename(
            output_name
        )
        .toFloat()
    )


# ============================================================
# Sentinel-2 spectral indices + FVC + albedo
# ============================================================

def add_s2_indices(image):
    image = ee.Image(
        image
    )

    blue = image.select(
        "Blue"
    )

    green = image.select(
        "Green"
    )

    red = image.select(
        "Red"
    )

    red_edge_1 = image.select(
        "RedEdge1"
    )

    nir = image.select(
        "NIR"
    )

    swir1 = image.select(
        "SWIR1"
    )


    # ========================================================
    # NDVI
    # ========================================================

    ndvi = _safe_ratio(
        nir.subtract(
            red
        ),
        nir.add(
            red
        ),
        "NDVI",
    )


    # ========================================================
    # EVI
    # ========================================================

    evi = _safe_ratio(
        nir
        .subtract(
            red
        )
        .multiply(
            2.5
        ),
        nir
        .add(
            red.multiply(
                6.0
            )
        )
        .subtract(
            blue.multiply(
                7.5
            )
        )
        .add(
            1.0
        ),
        "EVI",
    )


    # ========================================================
    # SAVI
    # L = 0.5
    # ========================================================

    savi = _safe_ratio(
        nir
        .subtract(
            red
        )
        .multiply(
            1.5
        ),
        nir
        .add(
            red
        )
        .add(
            0.5
        ),
        "SAVI",
    )


    # ========================================================
    # NDWI
    # McFeeters: Green - NIR
    # ========================================================

    ndwi = _safe_ratio(
        green.subtract(
            nir
        ),
        green.add(
            nir
        ),
        "NDWI",
    )


    # ========================================================
    # NDMI
    # ========================================================

    ndmi = _safe_ratio(
        nir.subtract(
            swir1
        ),
        nir.add(
            swir1
        ),
        "NDMI",
    )


    # ========================================================
    # NDRE
    # ========================================================

    ndre = _safe_ratio(
        nir.subtract(
            red_edge_1
        ),
        nir.add(
            red_edge_1
        ),
        "NDRE",
    )


    image_with_indices = (
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
        .toFloat()
    )


    # ========================================================
    # Fractional vegetation cover
    # ========================================================

    image_with_fvc = (
        add_fvc_band(
            image_with_indices,
            source="S2",
        )
    )


    # ========================================================
    # Shortwave broadband surface albedo
    # ========================================================

    return (
        add_s2_albedo(
            image_with_fvc
        )
        .toFloat()
    )