"""
Landsat 8/9 surface temperature processing.

This module builds pixel-level Landsat surface temperature (LST)
medoid composites for the same temporal periods used by the ET
downscaling workflow.

The module does not spatially aggregate LST values. Spatial
statistics over MODIS footprints or prediction grids are handled
later in the workflow.

Landsat Collection 2 Level-2 ST_B10 is distributed on a 30 m grid,
although the native thermal information from TIRS has a coarser
spatial support.

Surface temperature is converted to Kelvin using the official
Collection 2 Level-2 scale and offset:

    LST = ST_B10 * 0.00341802 + 149.0

Only L2SP products are used because L2SR products do not contain
valid surface temperature data.
"""

import ee

from .config import (
    END_DATE,
    START_DATE,
)


# ============================================================
# Landsat collections
# ============================================================

LANDSAT_8_COLLECTION_ID = (
    "LANDSAT/LC08/C02/T1_L2"
)

LANDSAT_9_COLLECTION_ID = (
    "LANDSAT/LC09/C02/T1_L2"
)


# ============================================================
# Surface temperature configuration
# ============================================================

LST_SOURCE_BAND = "ST_B10"
LST_OUTPUT_BAND = "LST"

LST_SCALE_FACTOR = 0.00341802
LST_OFFSET = 149.0

# Official Collection 2 Level-2 valid ST digital-number range.
# DN = 0 is the fill value.
LST_MIN_VALID_DN = 293
LST_MAX_VALID_DN = 65535


# ============================================================
# QA_PIXEL bit configuration
# ============================================================

# Landsat Collection 2 QA_PIXEL:
#
# Bit 0: Fill
# Bit 1: Dilated cloud
# Bit 2: Cirrus
# Bit 3: Cloud
# Bit 4: Cloud shadow
# Bit 5: Snow
#
# Water is deliberately retained.

QA_FILL_BIT = 0
QA_DILATED_CLOUD_BIT = 1
QA_CIRRUS_BIT = 2
QA_CLOUD_BIT = 3
QA_CLOUD_SHADOW_BIT = 4
QA_SNOW_BIT = 5


# ============================================================
# Metadata
# ============================================================

def _set_landsat_metadata(
    image,
    sensor,
):
    """
    Add standardized sensor and acquisition-date metadata.

    Parameters
    ----------
    image : ee.Image
        Original Landsat Collection 2 Level-2 image.
    sensor : str
        Sensor identifier: "L8" or "L9".

    Returns
    -------
    ee.Image
        Image with standardized metadata.
    """

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
            "landsat_sensor",
            sensor,
        )
        .set(
            "date_key",
            image
            .date()
            .format(
                "yyyy-MM-dd"
            ),
        )
    )


# ============================================================
# Landsat LST collection
# ============================================================

def get_landsat_lst_collection(
    geometry,
):
    """
    Return Landsat 8 and Landsat 9 Level-2 surface temperature
    observations intersecting the analysis geometry.

    Only L2SP products are retained because ST_B10 is fully
    masked in L2SR-only products.

    Parameters
    ----------
    geometry : ee.Geometry or ee.FeatureCollection
        Analysis geometry.

    Returns
    -------
    ee.ImageCollection
        Merged Landsat 8/9 L2SP collection.
    """

    if isinstance(
        geometry,
        ee.featurecollection.FeatureCollection,
    ):
        geometry = (
            ee.FeatureCollection(
                geometry
            )
            .geometry()
        )

    geometry = ee.Geometry(
        geometry
    )

    landsat_8 = (
        ee.ImageCollection(
            LANDSAT_8_COLLECTION_ID
        )
        .filterBounds(
            geometry
        )
        .filterDate(
            START_DATE,
            END_DATE,
        )
        .filter(
            ee.Filter.eq(
                "PROCESSING_LEVEL",
                "L2SP",
            )
        )
        .map(
            lambda image:
                _set_landsat_metadata(
                    image,
                    "L8",
                )
        )
    )

    landsat_9 = (
        ee.ImageCollection(
            LANDSAT_9_COLLECTION_ID
        )
        .filterBounds(
            geometry
        )
        .filterDate(
            START_DATE,
            END_DATE,
        )
        .filter(
            ee.Filter.eq(
                "PROCESSING_LEVEL",
                "L2SP",
            )
        )
        .map(
            lambda image:
                _set_landsat_metadata(
                    image,
                    "L9",
                )
        )
    )

    return (
        landsat_8
        .merge(
            landsat_9
        )
        .sort(
            "system:time_start"
        )
    )


# ============================================================
# QA mask
# ============================================================

def _build_landsat_lst_valid_mask(
    image,
):
    """
    Build the technical validity mask for Landsat LST.

    Pixels flagged as fill, dilated cloud, cirrus, cloud,
    cloud shadow, or snow are excluded.

    Water is retained.

    The official ST_B10 digital-number valid range is also
    enforced.

    No local temperature plausibility threshold is applied here.

    Parameters
    ----------
    image : ee.Image
        Landsat Collection 2 Level-2 image.

    Returns
    -------
    ee.Image
        Boolean validity mask.
    """

    image = ee.Image(
        image
    )

    qa_pixel = (
        image
        .select(
            "QA_PIXEL"
        )
    )

    no_fill = (
        qa_pixel
        .bitwiseAnd(
            1 << QA_FILL_BIT
        )
        .eq(0)
    )

    no_dilated_cloud = (
        qa_pixel
        .bitwiseAnd(
            1
            << QA_DILATED_CLOUD_BIT
        )
        .eq(0)
    )

    no_cirrus = (
        qa_pixel
        .bitwiseAnd(
            1 << QA_CIRRUS_BIT
        )
        .eq(0)
    )

    no_cloud = (
        qa_pixel
        .bitwiseAnd(
            1 << QA_CLOUD_BIT
        )
        .eq(0)
    )

    no_cloud_shadow = (
        qa_pixel
        .bitwiseAnd(
            1
            << QA_CLOUD_SHADOW_BIT
        )
        .eq(0)
    )

    no_snow = (
        qa_pixel
        .bitwiseAnd(
            1 << QA_SNOW_BIT
        )
        .eq(0)
    )

    st_dn = (
        image
        .select(
            LST_SOURCE_BAND
        )
    )

    valid_st_dn = (
        st_dn
        .gte(
            LST_MIN_VALID_DN
        )
        .And(
            st_dn
            .lte(
                LST_MAX_VALID_DN
            )
        )
    )

    return (
        no_fill
        .And(
            no_dilated_cloud
        )
        .And(
            no_cirrus
        )
        .And(
            no_cloud
        )
        .And(
            no_cloud_shadow
        )
        .And(
            no_snow
        )
        .And(
            valid_st_dn
        )
        .rename(
            "LST_VALID"
        )
    )


# ============================================================
# Prepare one Landsat LST observation
# ============================================================

def prepare_landsat_lst(
    image,
):
    """
    Convert ST_B10 to surface temperature in Kelvin and apply
    technical quality-control masking.

    Parameters
    ----------
    image : ee.Image
        Landsat Collection 2 Level-2 L2SP image.

    Returns
    -------
    ee.Image
        Single-band LST image in Kelvin.
    """

    image = ee.Image(
        image
    )

    valid_mask = (
        _build_landsat_lst_valid_mask(
            image
        )
    )

    lst = (
        image
        .select(
            LST_SOURCE_BAND
        )
        .multiply(
            LST_SCALE_FACTOR
        )
        .add(
            LST_OFFSET
        )
        .rename(
            LST_OUTPUT_BAND
        )
        .updateMask(
            valid_mask
        )
        .toFloat()
    )

    lst = ee.Image(
        lst.copyProperties(
            image,
            [
                "system:time_start",
                "system:index",
                "sensor",
                "landsat_sensor",
                "date_key",
                "PROCESSING_LEVEL",
            ],
        )
    )

    return (
        lst
        .toFloat()
    )


# ============================================================
# Empty LST image
# ============================================================

def build_empty_landsat_lst_image():
    """
    Build a fully masked LST image.

    This provides a stable band structure for periods with no
    valid Landsat surface-temperature observations.

    Returns
    -------
    ee.Image
        Fully masked single-band LST image.
    """

    return (
        ee.Image
        .constant(
            0
        )
        .rename(
            LST_OUTPUT_BAND
        )
        .updateMask(
            ee.Image.constant(
                0
            )
        )
        .toFloat()
    )


# ============================================================
# Daily Landsat LST mosaics
# ============================================================

def build_landsat_lst_daily_collection(
    lst_period,
    geometry,
):
    """
    Build one Landsat LST observation per acquisition date.

    Multiple Landsat scenes acquired on the same date are
    mosaicked before temporal medoid selection. This avoids
    treating overlapping path/row scenes from the same day as
    independent temporal observations.

    Parameters
    ----------
    lst_period : ee.ImageCollection
        Landsat LST collection already filtered to the target
        temporal period.
    geometry : ee.Geometry
        Analysis geometry.

    Returns
    -------
    ee.ImageCollection
        Daily LST mosaics.
    """

    lst_period = ee.ImageCollection(
        lst_period
    )

    geometry = ee.Geometry(
        geometry
    )

    date_keys = (
        ee.List(
            lst_period
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
        date_key = ee.String(
            date_key
        )

        same_date = (
            lst_period
            .filter(
                ee.Filter.eq(
                    "date_key",
                    date_key,
                )
            )
            .sort(
                "system:time_start"
            )
            .map(
                prepare_landsat_lst
            )
        )

        daily_mosaic = (
            same_date
            .mosaic()
            .clip(
                geometry
            )
            .rename(
                LST_OUTPUT_BAND
            )
            .toFloat()
        )

        return ee.Image(
            daily_mosaic
            .set(
                {
                    "date_key":
                        date_key,

                    "system:time_start":
                        ee.Image(
                            same_date.first()
                        )
                        .get(
                            "system:time_start"
                        ),
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
        .sort(
            "system:time_start"
        )
    )


# ============================================================
# Pixelwise temporal LST medoid
# ============================================================

def build_landsat_lst_medoid(
    lst_period,
    geometry,
):
    """
    Build the pixelwise temporal medoid of Landsat LST.

    For each pixel:

    1. Compute the temporal median LST.
    2. Compute the absolute distance between each actual
       observation and the temporal median.
    3. Select the observed LST value with the minimum distance.

    Because only one variable is used, this is a one-dimensional
    medoid. The selected output value always comes from an actual
    Landsat observation rather than from an interpolated temporal
    statistic.

    Parameters
    ----------
    lst_period : ee.ImageCollection
        Landsat L2SP observations filtered to exactly the same
        temporal period used by MODIS/S2/HLS.
    geometry : ee.Geometry
        Analysis geometry.

    Returns
    -------
    ee.Image
        Pixel-level LST medoid in Kelvin.
    """

    lst_period = ee.ImageCollection(
        lst_period
    )

    geometry = ee.Geometry(
        geometry
    )

    daily_collection = (
        build_landsat_lst_daily_collection(
            lst_period,
            geometry,
        )
    )

    empty_image = (
        build_empty_landsat_lst_image()
    )

    # The masked fallback guarantees a stable band structure
    # when the period contains no valid Landsat observations.
    safe_collection = (
        daily_collection
        .merge(
            ee.ImageCollection(
                [
                    empty_image,
                ]
            )
        )
    )

    temporal_median = (
        safe_collection
        .select(
            LST_OUTPUT_BAND
        )
        .median()
        .rename(
            "LST_MEDIAN"
        )
    )

    def score_observation(
        image,
    ):
        image = ee.Image(
            image
        )

        distance = (
            image
            .select(
                LST_OUTPUT_BAND
            )
            .subtract(
                temporal_median
            )
            .abs()
            .rename(
                "LST_MEDOID_DISTANCE"
            )
        )

        # qualityMosaic selects the maximum score.
        # Therefore, the negative absolute distance selects the
        # observation closest to the temporal median.
        medoid_score = (
            distance
            .multiply(
                -1
            )
            .rename(
                "LST_MEDOID_SCORE"
            )
        )

        return (
            image
            .addBands(
                medoid_score
            )
        )

    scored_collection = (
        safe_collection
        .map(
            score_observation
        )
    )

    medoid = (
        scored_collection
        .qualityMosaic(
            "LST_MEDOID_SCORE"
        )
        .select(
            LST_OUTPUT_BAND
        )
        .rename(
            LST_OUTPUT_BAND
        )
        .clip(
            geometry
        )
        .toFloat()
    )

    return (
        medoid
        .set(
            {
                "lst_scene_count":
                    lst_period.size(),

                "lst_day_count":
                    daily_collection.size(),

                "composite_method":
                    "pixelwise_temporal_medoid",

                "temperature_units":
                    "K",
            }
        )
    )


# ============================================================
# Build medoid for an exact ET period
# ============================================================

def build_landsat_lst_period_medoid(
    lst_collection,
    period_start,
    period_end,
    geometry,
):
    """
    Build a Landsat LST medoid for an exact ET analysis period.

    The temporal boundaries are supplied by the caller so that
    this module uses exactly the same periods as MODIS ET,
    Sentinel-2, and HLS.

    Earth Engine filterDate uses an inclusive start and exclusive
    end date.

    Parameters
    ----------
    lst_collection : ee.ImageCollection
        Landsat 8/9 L2SP collection.
    period_start : str or ee.Date
        Inclusive period start.
    period_end : str or ee.Date
        Exclusive period end.
    geometry : ee.Geometry
        Analysis geometry.

    Returns
    -------
    ee.Image
        Pixel-level Landsat LST medoid for the specified period.
    """

    period_start = ee.Date(
        period_start
    )

    period_end = ee.Date(
        period_end
    )

    lst_period = (
        ee.ImageCollection(
            lst_collection
        )
        .filterDate(
            period_start,
            period_end,
        )
    )

    medoid = (
        build_landsat_lst_medoid(
            lst_period,
            geometry,
        )
    )

    return (
        medoid
        .set(
            {
                "period_start":
                    period_start
                    .format(
                        "yyyy-MM-dd"
                    ),

                "period_end":
                    period_end
                    .format(
                        "yyyy-MM-dd"
                    ),
            }
        )
    )


# ============================================================
# LST valid coverage
# ============================================================

def calculate_lst_coverage_pct(
    lst_image,
    geometry,
    scale=30,
):
    """
    Calculate the percentage of the footprint covered by valid
    LST pixels.

    This function evaluates only data availability. It does not
    spatially summarize the LST values themselves.

    The 30 m scale corresponds to the distributed ST_B10 grid.
    It must not be interpreted as independent 30 m thermal
    information.

    Parameters
    ----------
    lst_image : ee.Image
        Pixel-level LST medoid.
    geometry : ee.Geometry
        Target footprint.
    scale : int or float, optional
        Operational scale of the distributed ST_B10 grid.

    Returns
    -------
    ee.Number
        Percentage of the footprint containing valid LST.
    """

    lst_image = ee.Image(
        lst_image
    )

    geometry = ee.Geometry(
        geometry
    )

    valid_area_image = (
        ee.Image
        .pixelArea()
        .updateMask(
            lst_image
            .select(
                LST_OUTPUT_BAND
            )
            .mask()
        )
        .rename(
            "valid_area_m2"
        )
    )

    valid_area_raw = (
        valid_area_image
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e7,
            tileScale=4,
        )
        .get(
            "valid_area_m2"
        )
    )

    valid_area = ee.Number(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(
                valid_area_raw,
                None,
            ),
            0,
            valid_area_raw,
        )
    )

    footprint_area = (
        geometry.area(
            maxError=1
        )
    )

    return (
        valid_area
        .divide(
            footprint_area
        )
        .multiply(
            100
        )
    )