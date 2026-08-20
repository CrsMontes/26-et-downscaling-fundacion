import ee

from .config import (
    END_DATE,
    START_DATE,
)


HLS_S30_COLLECTION_ID = "NASA/HLS/HLSS30/v002"
HLS_L30_COLLECTION_ID = "NASA/HLS/HLSL30/v002"

HLS_NATIVE_SCALE = 30


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

HLS_MEDOID_BANDS = [
    "Coastal",
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


def _prepare_source_image(
    image,
    source_bands,
    source_name,
):
    image = ee.Image(image)

    date_key = (
        image.date()
        .format("yyyy-MM-dd")
    )

    acquisition_key = (
        ee.String(source_name)
        .cat("_")
        .cat(date_key)
    )

    standardized = (
        image
        .select(
            source_bands,
            HLS_STANDARD_BANDS,
        )
        .set(
            {
                "optical_sensor": source_name,
                "date_key": date_key,
                "acquisition_key": acquisition_key,
            }
        )
    )

    return ee.Image(
        standardized.copyProperties(
            image,
            [
                "system:time_start",
                "system:index",
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
    geometry = (
        station_footprints.geometry()
    )

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

    clear_mask = (
        build_hls_clear_mask(
            image
        )
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
                "system:index",
                "date_key",
                "acquisition_key",
                "optical_sensor",
            ],
        )
    )


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


def build_hls_acquisition_collection(
    hls_period,
    geometry,
):
    hls_period = ee.ImageCollection(
        hls_period
    )

    acquisition_keys = (
        ee.List(
            hls_period.aggregate_array(
                "acquisition_key"
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

        acquisition = (
            hls_period
            .filter(
                ee.Filter.eq(
                    "acquisition_key",
                    acquisition_key,
                )
            )
            .map(
                prepare_hls
            )
        )

        reference = ee.Image(
            acquisition.first()
        )

        return ee.Image(
            acquisition
            .mosaic()
            .clip(
                geometry.buffer(100)
            )
            .set(
                {
                    "acquisition_key":
                        acquisition_key,

                    "date_key":
                        reference.get(
                            "date_key"
                        ),

                    "optical_sensor":
                        reference.get(
                            "optical_sensor"
                        ),

                    "system:time_start":
                        reference.get(
                            "system:time_start"
                        ),
                }
            )
        )

    return (
        ee.ImageCollection.fromImages(
            acquisition_keys.map(
                build_acquisition
            )
        )
    )


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

    def get_valid_mask(image):
        image = ee.Image(image)

        return (
            image
            .select(
                HLS_REFLECTANCE_BANDS
            )
            .mask()
            .reduce(
                ee.Reducer.min()
            )
            .rename("valid")
            .uint8()
        )

    masks = acquisitions.map(
        get_valid_mask
    )

    safe_masks = (
        masks.merge(
            ee.ImageCollection(
                [
                    ee.Image.constant(0)
                    .rename("valid")
                    .uint8(),
                ]
            )
        )
    )

    return (
        safe_masks
        .max()
        .rename("valid")
    )


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
        acquisitions.merge(
            ee.ImageCollection(
                [
                    build_empty_hls_image(),
                ]
            )
        )
    )

    spectral_median = (
        safe_acquisitions
        .select(
            HLS_MEDOID_BANDS
        )
        .median()
    )

    def score_image(image):
        image = ee.Image(image)

        squared_distance = (
            image
            .select(
                HLS_MEDOID_BANDS
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
        safe_acquisitions.map(
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
        .divide(denominator)
        .updateMask(
            valid_denominator
        )
        .rename(
            band_name
        )
        .toFloat()
    )


def add_hls_indices(image):
    image = ee.Image(image)

    blue = image.select("Blue")
    green = image.select("Green")
    red = image.select("Red")
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
        nir
        .add(red)
        .add(0.5),
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


def build_hls_predictors(
    hls_period,
    geometry,
):
    medoid = build_hls_medoid(
        hls_period,
        geometry,
    )

    return add_hls_indices(
        medoid
    )