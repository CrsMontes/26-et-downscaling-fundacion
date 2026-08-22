import ee

from .config import (
    get_optical_output_label,
    get_optical_scale,
    normalize_optical_source,
)

from .hls import (
    build_hls_daily_collection,
    build_hls_medoid,
    filter_hls_collection_to_geometry,
    get_hls_collection,
)

from .schema import (
    COMMON_OPTICAL_PREDICTOR_BANDS,
    COMMON_OPTICAL_REFLECTANCE_BANDS,
)

from .sentinel2 import (
    build_s2_daily_collection,
    build_s2_medoid,
    get_sentinel2_collection,
)

from .spatial import (
    get_coverage_fraction,
)


# ============================================================
# Source collection
# ============================================================

def get_optical_collection(
    station_footprints,
    source,
):
    source = normalize_optical_source(
        source
    )

    if source == "S2":
        return (
            get_sentinel2_collection(
                station_footprints
            )
        )

    if source == "HLS":
        return (
            get_hls_collection(
                station_footprints
            )
        )

    raise ValueError(
        f"Unsupported optical source: {source}"
    )


# ============================================================
# Period and local spatial filtering
# ============================================================

def filter_optical_period(
    collection,
    period_start,
    period_end,
    geometry,
    source,
):
    source = normalize_optical_source(
        source
    )

    geometry = ee.Geometry(
        geometry
    )

    period_collection = (
        ee.ImageCollection(
            collection
        )
        .filterDate(
            period_start,
            period_end,
        )
    )

    if source == "S2":
        return (
            period_collection
            .filterBounds(
                geometry
            )
        )

    if source == "HLS":
        return (
            filter_hls_collection_to_geometry(
                period_collection,
                geometry,
            )
        )

    raise ValueError(
        f"Unsupported optical source: {source}"
    )


# ============================================================
# Daily observations
# ============================================================

def build_optical_daily_collection(
    period_collection,
    geometry,
    source,
):
    source = normalize_optical_source(
        source
    )

    if source == "S2":
        return (
            build_s2_daily_collection(
                period_collection,
                geometry,
            )
        )

    if source == "HLS":
        return (
            build_hls_daily_collection(
                period_collection,
                geometry,
            )
        )

    raise ValueError(
        f"Unsupported optical source: {source}"
    )


# ============================================================
# Temporal medoid
# ============================================================

def build_optical_medoid(
    period_collection,
    geometry,
    source,
):
    source = normalize_optical_source(
        source
    )

    if source == "S2":
        return (
            build_s2_medoid(
                period_collection,
                geometry,
            )
        )

    if source == "HLS":
        return (
            build_hls_medoid(
                period_collection,
                geometry,
            )
        )

    raise ValueError(
        f"Unsupported optical source: {source}"
    )


# ============================================================
# Common optical indices
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


def add_common_optical_indices(
    image,
):
    """
    Add only indices with equivalent definitions for S2 and HLS.
    """
    image = (
        ee.Image(
            image
        )
        .select(
            COMMON_OPTICAL_REFLECTANCE_BANDS
        )
        .toFloat()
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

    nir = image.select(
        "NIR"
    )

    swir1 = image.select(
        "SWIR1"
    )

    ndvi = _safe_ratio(
        nir.subtract(
            red
        ),
        nir.add(
            red
        ),
        "NDVI",
    )

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

    ndwi = _safe_ratio(
        green.subtract(
            nir
        ),
        green.add(
            nir
        ),
        "NDWI",
    )

    ndmi = _safe_ratio(
        nir.subtract(
            swir1
        ),
        nir.add(
            swir1
        ),
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
            COMMON_OPTICAL_PREDICTOR_BANDS
        )
        .toFloat()
    )


def build_optical_predictors(
    period_collection,
    geometry,
    source,
):
    medoid = (
        build_optical_medoid(
            period_collection,
            geometry,
            source,
        )
    )

    return (
        add_common_optical_indices(
            medoid
        )
    )


# ============================================================
# Availability and coverage
# ============================================================

def get_optical_date_keys(
    period_collection,
):
    return (
        ee.List(
            ee.ImageCollection(
                period_collection
            )
            .aggregate_array(
                "date_key"
            )
        )
        .distinct()
        .sort()
    )


def build_optical_union_mask(
    period_collection,
    geometry,
    source,
):
    daily_collection = (
        build_optical_daily_collection(
            period_collection,
            geometry,
            source,
        )
    )

    zero_mask = (
        ee.Image.constant(0)
        .rename(
            "valid"
        )
        .uint8()
    )

    def get_date_mask(
        image,
    ):
        image = ee.Image(
            image
        )

        return (
            image
            .select(
                "Blue"
            )
            .mask()
            .rename(
                "valid"
            )
            .uint8()
        )

    date_masks = (
        daily_collection.map(
            get_date_mask
        )
    )

    safe_date_masks = (
        date_masks.merge(
            ee.ImageCollection(
                [
                    zero_mask,
                ]
            )
        )
    )

    return (
        safe_date_masks
        .max()
        .rename(
            "valid"
        )
    )


def get_optical_coverage(
    period_collection,
    geometry,
    source,
):
    source = normalize_optical_source(
        source
    )

    source_scale = (
        get_optical_scale(
            source
        )
    )

    union_mask = (
        build_optical_union_mask(
            period_collection,
            geometry,
            source,
        )
    )

    return (
        get_coverage_fraction(
            union_mask,
            geometry,
            scale=source_scale,
        )
    )


def get_optical_metadata(
    source,
):
    source = normalize_optical_source(
        source
    )

    return {
        "source":
            source,

        "output_label":
            get_optical_output_label(
                source
            ),

        "scale_m":
            get_optical_scale(
                source
            ),
    }
