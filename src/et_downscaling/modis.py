import ee

from .config import (
    END_DATE,
    MODIS_ET_MAX_VALID_DN,
    MODIS_ET_MIN_VALID_DN,
    MODIS_ET_SCALE_FACTOR,
    MODIS_REQUIRE_STRICT_QC,
    MODIS_STRICT_SCF_MAX,
    SAMPLES_ASSET,
    START_DATE,
    STATION_FIELD,
)


MODIS_COLLECTION_ID = "MODIS/061/MOD16A2GF"


# ============================================================
# MODIS collection
# ============================================================

def get_modis_collection():
    return (
        ee.ImageCollection(
            MODIS_COLLECTION_ID
        )
        .filterDate(
            START_DATE,
            END_DATE,
        )
        .select(
            [
                "ET",
                "ET_QC",
            ]
        )
        .sort(
            "system:time_start"
        )
    )


# ============================================================
# MODIS native projection and scale
# ============================================================

def get_modis_projection(
    modis_collection,
):
    reference_image = ee.Image(
        modis_collection.first()
    )

    return (
        reference_image
        .select("ET")
        .projection()
    )


def get_modis_scale(
    modis_projection,
):
    return (
        modis_projection
        .nominalScale()
    )


# ============================================================
# MODIS pixel identifier
# ============================================================

def build_modis_pixel_id(
    modis_projection,
):
    pixel_coordinates = (
        ee.Image.pixelCoordinates(
            modis_projection
        )
    )

    pixel_x = (
        pixel_coordinates
        .select("x")
        .toInt64()
    )

    pixel_y = (
        pixel_coordinates
        .select("y")
        .toInt64()
    )

    return (
        pixel_x
        .add(1_000_000)
        .multiply(10_000_000)
        .add(
            pixel_y.add(
                1_000_000
            )
        )
        .rename(
            "modis_pixel_id"
        )
        .toInt64()
    )


# ============================================================
# MODIS pixel grid
# ============================================================

def build_modis_grid(
    samples,
    modis_pixel_id,
    modis_projection,
    modis_scale,
):
    grid_region = (
        samples
        .geometry()
        .buffer(
            modis_scale.multiply(2)
        )
    )

    return (
        modis_pixel_id
        .reduceToVectors(
            geometry=grid_region,
            crs=modis_projection,
            scale=modis_scale,
            geometryType="polygon",
            eightConnected=False,
            labelProperty="modis_pixel_id",
            reducer=ee.Reducer.countEvery(),
            geometryInNativeProjection=True,
            maxPixels=1e9,
            tileScale=4,
        )
    )


# ============================================================
# Assign MODIS footprint to each station
# ============================================================

def assign_station_footprints(
    samples,
    modis_grid,
    modis_pixel_id,
    modis_projection,
    modis_scale,
):
    def assign_footprint(
        point,
    ):
        point = ee.Feature(
            point
        )

        point_geometry = (
            point.geometry()
        )

        coordinates = (
            point_geometry
            .coordinates()
        )

        id_dictionary = (
            modis_pixel_id
            .reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point_geometry,
                crs=modis_projection,
                scale=modis_scale,
                maxPixels=100,
            )
        )

        pixel_id_value = (
            ee.Number(
                id_dictionary.get(
                    "modis_pixel_id"
                )
            )
        )

        footprint = (
            ee.Feature(
                modis_grid
                .filter(
                    ee.Filter.eq(
                        "modis_pixel_id",
                        pixel_id_value,
                    )
                )
                .first()
            )
        )

        return (
            ee.Feature(
                footprint.geometry(),
                {
                    "station":
                        point.get(
                            STATION_FIELD
                        ),

                    "station_id":
                        point.get(
                            "system:index"
                        ),

                    "longitude":
                        coordinates.get(0),

                    "latitude":
                        coordinates.get(1),

                    "modis_pixel_id":
                        pixel_id_value,

                    "footprint_area_m2":
                        footprint
                        .geometry()
                        .area(1),
                },
            )
        )

    return (
        samples.map(
            assign_footprint
        )
    )


# ============================================================
# MODIS period end
# ============================================================

def get_modis_period_end(
    start_date,
):
    start_date = ee.Date(
        start_date
    )

    regular_end = (
        start_date.advance(
            8,
            "day",
        )
    )

    next_year_start = (
        ee.Date.fromYMD(
            ee.Number(
                start_date.get(
                    "year"
                )
            ).add(1),
            1,
            1,
        )
    )

    return (
        ee.Date(
            ee.Algorithms.If(
                regular_end
                .millis()
                .lte(
                    next_year_start
                    .millis()
                ),
                regular_end,
                next_year_start,
            )
        )
    )


# ============================================================
# Prepare MODIS evapotranspiration
# ============================================================

def prepare_modis_et(
    image,
    number_days,
):
    image = ee.Image(
        image
    )

    number_days = ee.Number(
        number_days
    )

    et_dn = (
        image
        .select("ET")
    )

    qc = (
        image
        .select("ET_QC")
    )

    # ========================================================
    # ET value validity
    # ========================================================

    value_valid_boolean = (
        et_dn
        .gte(
            MODIS_ET_MIN_VALID_DN
        )
        .And(
            et_dn.lte(
                MODIS_ET_MAX_VALID_DN
            )
        )
    )

    modis_value_valid = (
        value_valid_boolean
        .rename(
            "modis_value_valid"
        )
        .unmask(0)
        .uint8()
    )

    # ========================================================
    # QC availability
    # ========================================================

    modis_qc_present = (
        qc
        .mask()
        .gt(0)
        .rename(
            "modis_qc_present"
        )
        .unmask(0)
        .uint8()
    )

    # Use 255 only as an explicit missing-QC sentinel.
    # modis_qc_present distinguishes it from real QC values.
    qc_safe = (
        qc
        .unmask(255)
        .toUint8()
    )

    raw_qc = (
        qc_safe
        .rename("ET_QC")
    )

    # ========================================================
    # ET_QC bit fields
    # ========================================================

    # Bit 0:
    # 0 = good quality / main algorithm
    # 1 = other quality
    modis_modland_qc = (
        qc_safe
        .bitwiseAnd(1)
        .rename(
            "modis_modland_qc"
        )
        .uint8()
    )

    # Bit 1:
    # 0 = Terra
    # 1 = Aqua
    modis_sensor = (
        qc_safe
        .rightShift(1)
        .bitwiseAnd(1)
        .rename(
            "modis_sensor"
        )
        .uint8()
    )

    # Bit 2:
    # 0 = detectors apparently fine
    # 1 = dead-detector impact
    modis_dead_detector = (
        qc_safe
        .rightShift(2)
        .bitwiseAnd(1)
        .rename(
            "modis_dead_detector"
        )
        .uint8()
    )

    # Bits 3-4:
    # 0 = clear
    # 1 = significant clouds present
    # 2 = mixed cloud
    # 3 = cloud state not defined
    modis_cloud_state = (
        qc_safe
        .rightShift(3)
        .bitwiseAnd(3)
        .rename(
            "modis_cloud_state"
        )
        .uint8()
    )

    # Bits 5-7:
    # 0 = main method, best result
    # 1 = main method with saturation
    # 2 = empirical algorithm, geometry issue
    # 3 = empirical algorithm, other issue
    # 4 = pixel not produced
    #
    # Value 7 can only appear here when ET_QC was missing
    # and qc_safe was set to 255. Check modis_qc_present.
    modis_scf_qc = (
        qc_safe
        .rightShift(5)
        .bitwiseAnd(7)
        .rename(
            "modis_scf_qc"
        )
        .uint8()
    )

    # ========================================================
    # Legacy strict QC criterion
    # ========================================================

    strict_qc_boolean = (
        modis_qc_present
        .eq(1)
        .And(
            modis_modland_qc
            .eq(0)
        )
        .And(
            modis_dead_detector
            .eq(0)
        )
        .And(
            modis_scf_qc
            .lte(
                MODIS_STRICT_SCF_MAX
            )
        )
    )

    modis_qc_good = (
        strict_qc_boolean
        .rename(
            "modis_qc_good"
        )
        .unmask(0)
        .uint8()
    )

    # ========================================================
    # Active MODIS validity
    # ========================================================

    if MODIS_REQUIRE_STRICT_QC:
        modis_good_boolean = (
            value_valid_boolean
            .And(
                strict_qc_boolean
            )
        )

    else:
        modis_good_boolean = (
            value_valid_boolean
        )

    # Keep the historical name for compatibility with the
    # existing availability and extraction workflow.
    modis_good = (
        modis_good_boolean
        .rename(
            "modis_good"
        )
        .unmask(0)
        .uint8()
    )

    # ========================================================
    # Scale valid ET values
    # ========================================================

    et_period = (
        et_dn
        .multiply(
            MODIS_ET_SCALE_FACTOR
        )
        .rename(
            "ET_mm_period"
        )
        .updateMask(
            value_valid_boolean
        )
        .toFloat()
    )

    et_daily = (
        et_period
        .divide(
            number_days
        )
        .rename(
            "ET_mm_day"
        )
        .toFloat()
    )

    # ========================================================
    # Output
    # ========================================================

    return (
        et_period
        .addBands(
            et_daily
        )
        .addBands(
            modis_value_valid
        )
        .addBands(
            modis_good
        )
        .addBands(
            raw_qc
        )
        .addBands(
            modis_qc_present
        )
        .addBands(
            modis_qc_good
        )
        .addBands(
            modis_modland_qc
        )
        .addBands(
            modis_sensor
        )
        .addBands(
            modis_dead_detector
        )
        .addBands(
            modis_cloud_state
        )
        .addBands(
            modis_scf_qc
        )
    )


# ============================================================
# Build MODIS inputs
# ============================================================

def build_modis_inputs():
    samples = (
        ee.FeatureCollection(
            SAMPLES_ASSET
        )
    )

    modis_collection = (
        get_modis_collection()
    )

    modis_projection = (
        get_modis_projection(
            modis_collection
        )
    )

    modis_scale = (
        get_modis_scale(
            modis_projection
        )
    )

    modis_pixel_id = (
        build_modis_pixel_id(
            modis_projection
        )
    )

    modis_grid = (
        build_modis_grid(
            samples,
            modis_pixel_id,
            modis_projection,
            modis_scale,
        )
    )

    station_footprints = (
        assign_station_footprints(
            samples,
            modis_grid,
            modis_pixel_id,
            modis_projection,
            modis_scale,
        )
    )

    return {
        "samples":
            samples,

        "collection":
            modis_collection,

        "projection":
            modis_projection,

        "scale":
            modis_scale,

        "pixel_id":
            modis_pixel_id,

        "grid":
            modis_grid,

        "station_footprints":
            station_footprints,
    }