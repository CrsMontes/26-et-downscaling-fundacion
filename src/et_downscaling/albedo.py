import ee


# ============================================================
# Sentinel-2 broadband albedo coefficients
# ============================================================

# Bonafoni and Sekertekin (2020)
#
# Shortwave broadband surface albedo estimated from
# Sentinel-2 MSI surface reflectance:
#
# Albedo =
#     0.2266 * B2
#   + 0.1236 * B3
#   + 0.1573 * B4
#   + 0.3417 * B8
#   + 0.1170 * B11
#   + 0.0338 * B12
#
# In the project:
#     B2  -> Blue
#     B3  -> Green
#     B4  -> Red
#     B8  -> NIR_Broad
#     B11 -> SWIR1
#     B12 -> SWIR2

S2_ALBEDO_COEFFICIENTS = {
    "Blue": 0.2266,
    "Green": 0.1236,
    "Red": 0.1573,
    "NIR_Broad": 0.3417,
    "SWIR1": 0.1170,
    "SWIR2": 0.0338,
}


# ============================================================
# HLS broadband albedo coefficients
# ============================================================

# Landsat-8 OLI narrow-to-broadband formulation.
#
# Shortwave broadband surface albedo:
#
# Albedo =
#     0.2453 * Blue
#   + 0.0508 * Green
#   + 0.1804 * Red
#   + 0.3081 * NIR
#   + 0.1332 * SWIR1
#   + 0.0521 * SWIR2
#   + 0.0011
#
# HLS V2 common bands are harmonized using OLI as the
# spectral reference. Therefore, this OLI formulation is used
# here as an approximation for HLS OLI-like NBAR.
#
# Important:
# The original OLI formulation was not specifically developed
# for HLS NBAR. This assumption must therefore be evaluated
# empirically against Sentinel-2 albedo in coincident
# footprint-period observations.

HLS_ALBEDO_COEFFICIENTS = {
    "Blue": 0.2453,
    "Green": 0.0508,
    "Red": 0.1804,
    "NIR": 0.3081,
    "SWIR1": 0.1332,
    "SWIR2": 0.0521,
}

HLS_ALBEDO_INTERCEPT = 0.0011


# ============================================================
# Sentinel-2 albedo
# ============================================================

def calculate_s2_albedo(image):
    """
    Estimate Sentinel-2 shortwave broadband surface albedo.

    Parameters
    ----------
    image : ee.Image
        Sentinel-2 image containing surface reflectance in
        decimal units and the following bands:

        - Blue
        - Green
        - Red
        - NIR_Broad
        - SWIR1
        - SWIR2

    Returns
    -------
    ee.Image
        Single-band image named "Albedo".

    Notes
    -----
    NIR_Broad corresponds to Sentinel-2 B8 and must not be
    replaced by NIR, which corresponds to B8A in this project.

    No clipping is applied. Values outside the physically
    expected range should be identified during quality
    assessment rather than silently constrained.
    """
    image = ee.Image(
        image
    )

    albedo = (
        image
        .select("Blue")
        .multiply(
            S2_ALBEDO_COEFFICIENTS[
                "Blue"
            ]
        )
        .add(
            image
            .select("Green")
            .multiply(
                S2_ALBEDO_COEFFICIENTS[
                    "Green"
                ]
            )
        )
        .add(
            image
            .select("Red")
            .multiply(
                S2_ALBEDO_COEFFICIENTS[
                    "Red"
                ]
            )
        )
        .add(
            image
            .select("NIR_Broad")
            .multiply(
                S2_ALBEDO_COEFFICIENTS[
                    "NIR_Broad"
                ]
            )
        )
        .add(
            image
            .select("SWIR1")
            .multiply(
                S2_ALBEDO_COEFFICIENTS[
                    "SWIR1"
                ]
            )
        )
        .add(
            image
            .select("SWIR2")
            .multiply(
                S2_ALBEDO_COEFFICIENTS[
                    "SWIR2"
                ]
            )
        )
        .rename(
            "Albedo"
        )
        .toFloat()
    )

    return albedo


def add_s2_albedo(image):
    """
    Add Sentinel-2 broadband albedo as an image band.
    """
    image = ee.Image(
        image
    )

    albedo = calculate_s2_albedo(
        image
    )

    return (
        image
        .addBands(
            albedo
        )
    )


# ============================================================
# HLS albedo
# ============================================================

def calculate_hls_albedo(image):
    """
    Estimate HLS shortwave broadband surface albedo.

    Parameters
    ----------
    image : ee.Image
        HLS medoid containing reflectance in decimal units
        and the following harmonized bands:

        - Blue
        - Green
        - Red
        - NIR
        - SWIR1
        - SWIR2

    Returns
    -------
    ee.Image
        Single-band image named "Albedo".

    Notes
    -----
    The formulation is based on Landsat-8 OLI
    narrow-to-broadband coefficients.

    HLS common bands are harmonized toward the OLI spectral
    reference, but HLS provides Nadir BRDF-Adjusted
    Reflectance (NBAR). Therefore, this calculation is treated
    as an OLI-based approximation for HLS rather than as a
    formulation specifically calibrated for HLS NBAR.

    No clipping is applied. Physical plausibility and
    consistency with Sentinel-2 should be evaluated during
    quality assessment.
    """
    image = ee.Image(
        image
    )

    albedo = (
        image
        .select("Blue")
        .multiply(
            HLS_ALBEDO_COEFFICIENTS[
                "Blue"
            ]
        )
        .add(
            image
            .select("Green")
            .multiply(
                HLS_ALBEDO_COEFFICIENTS[
                    "Green"
                ]
            )
        )
        .add(
            image
            .select("Red")
            .multiply(
                HLS_ALBEDO_COEFFICIENTS[
                    "Red"
                ]
            )
        )
        .add(
            image
            .select("NIR")
            .multiply(
                HLS_ALBEDO_COEFFICIENTS[
                    "NIR"
                ]
            )
        )
        .add(
            image
            .select("SWIR1")
            .multiply(
                HLS_ALBEDO_COEFFICIENTS[
                    "SWIR1"
                ]
            )
        )
        .add(
            image
            .select("SWIR2")
            .multiply(
                HLS_ALBEDO_COEFFICIENTS[
                    "SWIR2"
                ]
            )
        )
        .add(
            HLS_ALBEDO_INTERCEPT
        )
        .rename(
            "Albedo"
        )
        .toFloat()
    )

    return albedo


def add_hls_albedo(image):
    """
    Add HLS broadband albedo as an image band.
    """
    image = ee.Image(
        image
    )

    albedo = calculate_hls_albedo(
        image
    )

    return (
        image
        .addBands(
            albedo
        )
    )