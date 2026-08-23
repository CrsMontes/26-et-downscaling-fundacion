import ee

from .config import (
    normalize_optical_source,
)

from .hls import (
    add_hls_indices,
    build_hls_daily_collection,
    build_hls_medoid,
    filter_hls_collection_to_geometry,
    get_hls_collection,
)

from .schema import (
    get_optical_extraction_bands,
)

from .sentinel2 import (
    add_s2_indices,
    build_s2_daily_collection,
    build_s2_medoid,
    get_sentinel2_collection,
)

from .spatial import (
    get_coverage_fraction,
)

from .config import (
    get_optical_scale,
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
# Rich source-specific optical extraction stack
# ============================================================

def build_optical_predictors(
    period_collection,
    geometry,
    source,
):
    """
    Build the optical extraction stack for one source.

    The exported stack is intentionally richer than the common
    S2/HLS model feature set. Final feature selection is local.

    HLS FVC must only be used after the source-specific FVC
    calibration has been regenerated with the corrected HLS
    spatial selection.
    """
    source = normalize_optical_source(
        source
    )

    medoid = (
        build_optical_medoid(
            period_collection,
            geometry,
            source,
        )
    )

    if source == "S2":
        predictors = (
            add_s2_indices(
                medoid
            )
        )

    elif source == "HLS":
        predictors = (
            add_hls_indices(
                medoid
            )
        )

    else:
        raise ValueError(
            f"Unsupported optical source: {source}"
        )

    extraction_bands = (
        get_optical_extraction_bands(
            source
        )
    )

    return (
        predictors
        .select(
            extraction_bands
        )
        .toFloat()
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
