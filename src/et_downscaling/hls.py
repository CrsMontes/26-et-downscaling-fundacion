import ee

from .albedo import (
    add_hls_albedo,
)

from .config import (
    END_DATE,
    START_DATE,
)

from .fvc import (
    add_fvc_band,
)


# ============================================================
# HLS collections
# ============================================================

HLS_S30_COLLECTION_ID = (
    "NASA/HLS/HLSS30/v002"
)

HLS_L30_COLLECTION_ID = (
    "NASA/HLS/HLSL30/v002"
)


# ============================================================
# HLS standardized spectral configuration
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


# Sentinel-2-derived HLS S30 bands.
HLS_S30_SOURCE_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B8A",
    "B11",
    "B12",
]


# Landsat-derived HLS L30 bands.
HLS_L30_SOURCE_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
]


# Coastal is deliberately excluded from the positivity test.
# Small negative Coastal reflectances can be valid in HLS.
HLS_POSITIVE_BANDS = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
]


# All harmonized reflectance bands participate in the medoid
# distance.
HLS_MEDOID_SCORE_BANDS = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
]


# ============================================================
# HLS Fmask configuration
# ============================================================

# HLS Fmask:
#
# Bit 1: cloud
# Bit 2: adjacent to cloud/shadow
# Bit 3: cloud shadow
# Bit 4: snow/ice
#
# Bit 5: water -> retained
# Bits 6-7: aerosol -> not used as hard filters

HLS_FMASK_CLOUD_BIT = 1
HLS_FMASK_ADJACENT_BIT = 2
HLS_FMASK_SHADOW_BIT = 3
HLS_FMASK_SNOW_BIT = 4


# ============================================================
# Add HLS metadata
# ============================================================

def _set_hls_metadata(
    image,
    sensor,
):
    image = ee.Image(
        image
    )

    return ee.Image(
        image
        .set(
            "sensor",
            sensor,
        )
        .set(
            "hls_sensor",
            sensor,
        )
        .set(
            "date_key",
            image.date().format(
                "yyyy-MM-dd"
            ),
        )
    )


# ============================================================
# Standardize HLS MGRS identifier
# ============================================================

def _add_hls_mgrs_tile(
    image,
):
    """
    Parse the MGRS tile from the HLS system:index.

    Examples
    --------
    1_T18PWS_20210103T152641 -> 18PWS
    2_T18PWS_20210111T151708 -> 18PWS
    """
    image = ee.Image(
        image
    )

    system_index = ee.String(
        image.get(
            "system:index"
        )
    )

    tile_token = ee.String(
        system_index
        .split("_")
        .get(1)
    )

    mgrs_tile = (
        tile_token
        .slice(1)
    )

    return (
        image.set(
            "hls_mgrs_tile",
            mgrs_tile,
        )
    )


# ============================================================
# Local HLS MGRS support
# ============================================================

def get_local_hls_mgrs_tiles(
    geometry,
):
    """
    Identify the MGRS tiles intersecting one local footprint.

    Sentinel-2 MGRS metadata is used as an independent spatial
    reference because the diagnostic reproduction showed that
    HLS filterBounds() alone could admit non-local HLS assets.
    """
    geometry = ee.Geometry(
        geometry
    )

    local_s2 = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
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
        ee.List(
            local_s2.aggregate_array(
                "MGRS_TILE"
            )
        )
        .distinct()
        .sort()
    )


def filter_hls_collection_to_geometry(
    collection,
    geometry,
):
    """
    Restrict HLS to the verified local MGRS tile set and geometry.
    """
    geometry = ee.Geometry(
        geometry
    )

    local_tiles = (
        get_local_hls_mgrs_tiles(
            geometry
        )
    )

    return (
        ee.ImageCollection(
            collection
        )
        .filter(
            ee.Filter.inList(
                "hls_mgrs_tile",
                local_tiles,
            )
        )
        .filterBounds(
            geometry
        )
    )


# ============================================================
# HLS collection
# ============================================================

def get_hls_collection(
    station_footprints,
):
    geometry = (
        ee.FeatureCollection(
            station_footprints
        )
        .geometry()
    )

    s30_collection = (
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
            lambda image:
                _set_hls_metadata(
                    image,
                    "S30",
                )
        )
        .map(
            _add_hls_mgrs_tile
        )
    )

    l30_collection = (
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
            lambda image:
                _set_hls_metadata(
                    image,
                    "L30",
                )
        )
        .map(
            _add_hls_mgrs_tile
        )
    )

    return (
        s30_collection
        .merge(
            l30_collection
        )
        .sort(
            "system:time_start"
        )
    )


# ============================================================
# HLS clear-sky mask
# ============================================================

def _build_hls_fmask_clear_mask(
    image,
    exclude_high_aerosol=False,
):
    image = ee.Image(
        image
    )

    fmask = (
        image
        .select(
            "Fmask"
        )
    )

    no_cloud = (
        fmask
        .bitwiseAnd(
            1
            << HLS_FMASK_CLOUD_BIT
        )
        .eq(0)
    )

    no_adjacent = (
        fmask
        .bitwiseAnd(
            1
            << HLS_FMASK_ADJACENT_BIT
        )
        .eq(0)
    )

    no_shadow = (
        fmask
        .bitwiseAnd(
            1
            << HLS_FMASK_SHADOW_BIT
        )
        .eq(0)
    )

    no_snow = (
        fmask
        .bitwiseAnd(
            1
            << HLS_FMASK_SNOW_BIT
        )
        .eq(0)
    )

    clear_mask = (
        no_cloud
        .And(
            no_adjacent
        )
        .And(
            no_shadow
        )
        .And(
            no_snow
        )
    )

    if exclude_high_aerosol:
        aerosol_level = (
            fmask
            .rightShift(6)
            .bitwiseAnd(3)
        )

        clear_mask = (
            clear_mask
            .And(
                aerosol_level.neq(3)
            )
        )

    return (
        clear_mask
        .rename(
            "hls_clear"
        )
    )


# ============================================================
# Prepare standardized HLS image
# ============================================================

def _prepare_hls_image(
    image,
    source_bands,
    exclude_high_aerosol=False,
):
    image = ee.Image(
        image
    )

    reflectance = (
        image
        .select(
            source_bands,
            HLS_REFLECTANCE_BANDS,
        )
        .toFloat()
    )

    # All seven source bands must contain valid data.
    spectral_valid = (
        reflectance
        .mask()
        .reduce(
            ee.Reducer.min()
        )
    )

    # Require positive reflectance only in bands used by the
    # vegetation/water indices and albedo.
    #
    # Coastal is deliberately excluded.
    positive_reflectance = (
        reflectance
        .select(
            HLS_POSITIVE_BANDS
        )
        .gt(0)
        .reduce(
            ee.Reducer.min()
        )
    )

    clear_mask = (
        _build_hls_fmask_clear_mask(
            image,
            exclude_high_aerosol=(
                exclude_high_aerosol
            ),
        )
    )

    valid_mask = (
        spectral_valid
        .And(
            positive_reflectance
        )
        .And(
            clear_mask
        )
    )

    prepared = (
        reflectance
        .updateMask(
            valid_mask
        )
    )

    # copyProperties returns a generic Earth Engine Element.
    # Explicitly cast it back to ee.Image before using image
    # methods such as toFloat().
    prepared = ee.Image(
        prepared.copyProperties(
            image,
            [
                "system:time_start",
                "system:index",
                "sensor",
                "hls_sensor",
                "date_key",
                "hls_mgrs_tile",
            ],
        )
    )

    return (
        prepared
        .toFloat()
    )


# ============================================================
# Prepare HLS S30
# ============================================================

def prepare_hls_s30(
    image,
):
    return (
        _prepare_hls_image(
            image,
            HLS_S30_SOURCE_BANDS,
            exclude_high_aerosol=True,
        )
    )


# ============================================================
# Prepare HLS L30
# ============================================================

def prepare_hls_l30(
    image,
):
    return (
        _prepare_hls_image(
            image,
            HLS_L30_SOURCE_BANDS,
            exclude_high_aerosol=False,
        )
    )


# ============================================================
# Empty HLS image
# ============================================================

def build_empty_hls_image():
    return (
        ee.Image.constant(
            [
                0,
            ]
            * len(
                HLS_REFLECTANCE_BANDS
            )
        )
        .rename(
            HLS_REFLECTANCE_BANDS
        )
        .updateMask(
            ee.Image.constant(
                0
            )
        )
        .toFloat()
    )


# ============================================================
# Daily HLS mosaics for one sensor
# ============================================================

def _build_hls_sensor_daily_collection(
    hls_period,
    geometry,
    sensor,
):
    hls_period = (
        ee.ImageCollection(
            hls_period
        )
        .filter(
            ee.Filter.eq(
                "sensor",
                sensor,
            )
        )
    )

    date_keys = (
        ee.List(
            hls_period
            .aggregate_array(
                "date_key"
            )
        )
        .distinct()
        .sort()
    )

    if sensor == "S30":
        preparation_function = (
            prepare_hls_s30
        )

    elif sensor == "L30":
        preparation_function = (
            prepare_hls_l30
        )

    else:
        raise ValueError(
            "Unsupported HLS sensor: "
            f"{sensor}"
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
            hls_period
            .filter(
                ee.Filter.eq(
                    "date_key",
                    date_key,
                )
            )
            .map(
                preparation_function
            )
        )

        daily_mosaic = (
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
        )

        # set() also returns a generic EE object.
        # Explicitly cast the result back to ee.Image.
        return ee.Image(
            daily_mosaic.set(
                {
                    "date_key":
                        date_key,

                    "sensor":
                        sensor,

                    "hls_sensor":
                        sensor,
                }
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
# Combined daily HLS collection
# ============================================================

def build_hls_daily_collection(
    hls_period,
    geometry,
):
    hls_period = (
        ee.ImageCollection(
            hls_period
        )
    )

    s30_daily = (
        _build_hls_sensor_daily_collection(
            hls_period,
            geometry,
            "S30",
        )
    )

    l30_daily = (
        _build_hls_sensor_daily_collection(
            hls_period,
            geometry,
            "L30",
        )
    )

    # S30 and L30 remain separate daily observations.
    # Both compete jointly in the temporal medoid.
    return (
        s30_daily
        .merge(
            l30_daily
        )
        .sort(
            "date_key"
        )
    )


# ============================================================
# HLS temporal medoid
# ============================================================

def build_hls_medoid(
    hls_period,
    geometry,
):
    daily_images = (
        build_hls_daily_collection(
            hls_period,
            geometry,
        )
    )

    # A fully masked fallback guarantees a stable output band
    # structure when a period contains no valid observations.
    safe_daily_images = (
        daily_images
        .merge(
            ee.ImageCollection(
                [
                    build_empty_hls_image(),
                ]
            )
        )
    )

    spectral_median = (
        safe_daily_images
        .select(
            HLS_MEDOID_SCORE_BANDS
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
                HLS_MEDOID_SCORE_BANDS
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
# HLS spectral indices + FVC + albedo
# ============================================================

def add_hls_indices(
    image,
):
    image = ee.Image(
        image
    )

    blue = (
        image.select(
            "Blue"
        )
    )

    green = (
        image.select(
            "Green"
        )
    )

    red = (
        image.select(
            "Red"
        )
    )

    nir = (
        image.select(
            "NIR"
        )
    )

    swir1 = (
        image.select(
            "SWIR1"
        )
    )


    # ========================================================
    # NDVI
    # ========================================================

    ndvi = (
        _safe_ratio(
            nir.subtract(
                red
            ),
            nir.add(
                red
            ),
            "NDVI",
        )
    )


    # ========================================================
    # EVI
    # ========================================================

    evi = (
        _safe_ratio(
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
    )


    # ========================================================
    # SAVI
    # L = 0.5
    # ========================================================

    savi = (
        _safe_ratio(
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
    )


    # ========================================================
    # NDWI
    # McFeeters: Green - NIR
    # ========================================================

    ndwi = (
        _safe_ratio(
            green.subtract(
                nir
            ),
            green.add(
                nir
            ),
            "NDWI",
        )
    )


    # ========================================================
    # NDMI
    # ========================================================

    ndmi = (
        _safe_ratio(
            nir.subtract(
                swir1
            ),
            nir.add(
                swir1
            ),
            "NDMI",
        )
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
            source="HLS",
        )
    )


    # ========================================================
    # Shortwave broadband surface albedo
    # ========================================================

    return (
        add_hls_albedo(
            image_with_fvc
        )
        .toFloat()
    )
    