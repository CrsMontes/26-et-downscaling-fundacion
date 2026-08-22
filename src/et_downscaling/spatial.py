import ee

from .config import (
    ANALYSIS_CRS,
    ANALYSIS_SCALE,
)


# ============================================================
# Spatial coverage fraction
# ============================================================

def get_coverage_fraction(
    mask_image,
    geometry,
    scale=None,
    crs=ANALYSIS_CRS,
):
    """
    Calculate the fraction of an analysis geometry covered by
    valid pixels.

    Parameters
    ----------
    mask_image : ee.Image
        Single-band validity mask where valid pixels are 1.
    geometry : ee.Geometry
        Geometry over which coverage is calculated.
    scale : float or int, optional
        Processing scale in metres. Production optical workflows
        should pass the source-specific scale explicitly.
        ANALYSIS_SCALE is retained only as a backward-compatible
        fallback.
    crs : str
        Projected CRS used for the reduction.
    """
    if scale is None:
        scale = ANALYSIS_SCALE

    mask_image = (
        ee.Image(
            mask_image
        )
        .unmask(0)
        .rename(
            "valid"
        )
    )

    result = (
        mask_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            crs=crs,
            scale=scale,
            maxPixels=1e7,
            tileScale=4,
        )
    )

    raw_value = (
        ee.Dictionary(
            result
        )
        .get(
            "valid"
        )
    )

    return (
        ee.Number(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(
                    raw_value,
                    None,
                ),
                0,
                raw_value,
            )
        )
    )
