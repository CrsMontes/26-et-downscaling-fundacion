import ee

from .config import (
    ANALYSIS_CRS,
    ANALYSIS_SCALE,
)


# ============================================================
# Local 60 x 60 m geometry
# ============================================================

def make_local_geometry(point_geometry):
    point_geometry = ee.Geometry(
        point_geometry
    )

    return (
        point_geometry
        .transform(
            ANALYSIS_CRS,
            1,
        )
        .buffer(30)
        .bounds(
            1,
            ANALYSIS_CRS,
        )
    )


# ============================================================
# Spatial coverage fraction
# ============================================================

def get_coverage_fraction(
    mask_image,
    geometry,
):
    mask_image = (
        ee.Image(mask_image)
        .unmask(0)
        .rename("valid")
    )

    result = mask_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        crs=ANALYSIS_CRS,
        scale=ANALYSIS_SCALE,
        maxPixels=1e7,
        tileScale=4,
    )

    raw_value = ee.Dictionary(
        result
    ).get("valid")

    return ee.Number(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(
                raw_value,
                None,
            ),
            0,
            raw_value,
        )
    )