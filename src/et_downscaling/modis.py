import ee

from .config import (
    END_DATE,
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
        ee.ImageCollection(MODIS_COLLECTION_ID)
        .filterDate(START_DATE, END_DATE)
        .select(["ET", "ET_QC"])
        .sort("system:time_start")
    )


# ============================================================
# MODIS native projection and scale
# ============================================================

def get_modis_projection(modis_collection):
    reference_image = ee.Image(
        modis_collection.first()
    )

    return (
        reference_image
        .select("ET")
        .projection()
    )


def get_modis_scale(modis_projection):
    return modis_projection.nominalScale()


# ============================================================
# MODIS pixel identifier
# ============================================================

def build_modis_pixel_id(modis_projection):
    pixel_coordinates = ee.Image.pixelCoordinates(
        modis_projection
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
            pixel_y.add(1_000_000)
        )
        .rename("modis_pixel_id")
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

    return modis_pixel_id.reduceToVectors(
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
    def assign_footprint(point):
        point = ee.Feature(point)

        point_geometry = point.geometry()
        coordinates = point_geometry.coordinates()

        id_dictionary = modis_pixel_id.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point_geometry,
            crs=modis_projection,
            scale=modis_scale,
            maxPixels=100,
        )

        pixel_id_value = ee.Number(
            id_dictionary.get(
                "modis_pixel_id"
            )
        )

        footprint = ee.Feature(
            modis_grid
            .filter(
                ee.Filter.eq(
                    "modis_pixel_id",
                    pixel_id_value,
                )
            )
            .first()
        )

        return ee.Feature(
            footprint.geometry(),
            {
                "station": point.get(
                    STATION_FIELD
                ),
                "station_id": point.get(
                    "system:index"
                ),
                "longitude": coordinates.get(0),
                "latitude": coordinates.get(1),
                "modis_pixel_id": pixel_id_value,
                "footprint_area_m2": (
                    footprint
                    .geometry()
                    .area(1)
                ),
            },
        )

    return samples.map(assign_footprint)


# ============================================================
# MODIS period end
# ============================================================

def get_modis_period_end(start_date):
    start_date = ee.Date(start_date)

    regular_end = start_date.advance(
        8,
        "day",
    )

    next_year_start = ee.Date.fromYMD(
        ee.Number(
            start_date.get("year")
        ).add(1),
        1,
        1,
    )

    return ee.Date(
        ee.Algorithms.If(
            regular_end
            .millis()
            .lte(
                next_year_start.millis()
            ),
            regular_end,
            next_year_start,
        )
    )


# ============================================================
# Prepare MODIS evapotranspiration
# ============================================================

def prepare_modis_et(
    image,
    number_days,
):
    image = ee.Image(image)
    number_days = ee.Number(number_days)

    et_dn = image.select("ET")
    qc = image.select("ET_QC")

    # Bit 0: 0 = main algorithm
    good_modland = (
        qc
        .bitwiseAnd(1)
        .eq(0)
    )

    # Bit 2: 0 = good detector
    good_detector = (
        qc
        .rightShift(2)
        .bitwiseAnd(1)
        .eq(0)
    )

    # Bits 5-7: SCF quality
    scf_quality = (
        qc
        .rightShift(5)
        .bitwiseAnd(7)
    )

    valid_value = (
        et_dn
        .gte(0)
        .And(
            et_dn.lte(32700)
        )
    )

    good_quality = (
        good_modland
        .And(good_detector)
        .And(
            scf_quality.lte(1)
        )
        .And(valid_value)
        .rename("modis_good")
        .unmask(0)
        .uint8()
    )

    et_period = (
        et_dn
        .multiply(0.1)
        .rename("ET_mm_period")
    )

    et_daily = (
        et_period
        .divide(number_days)
        .rename("ET_mm_day")
    )

    return (
        et_period
        .addBands(et_daily)
        .addBands(good_quality)
        .addBands(
            scf_quality
            .unmask(255)
            .rename("modis_scf_qc")
        )
    )


# ============================================================
# Build MODIS inputs
# ============================================================

def build_modis_inputs():
    samples = ee.FeatureCollection(
        SAMPLES_ASSET
    )

    modis_collection = get_modis_collection()

    modis_projection = get_modis_projection(
        modis_collection
    )

    modis_scale = get_modis_scale(
        modis_projection
    )

    modis_pixel_id = build_modis_pixel_id(
        modis_projection
    )

    modis_grid = build_modis_grid(
        samples,
        modis_pixel_id,
        modis_projection,
        modis_scale,
    )

    station_footprints = assign_station_footprints(
        samples,
        modis_grid,
        modis_pixel_id,
        modis_projection,
        modis_scale,
    )

    return {
        "samples": samples,
        "collection": modis_collection,
        "projection": modis_projection,
        "scale": modis_scale,
        "pixel_id": modis_pixel_id,
        "grid": modis_grid,
        "station_footprints": station_footprints,
    }