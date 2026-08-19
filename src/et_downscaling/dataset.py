import ee

from .config import (
    ANALYSIS_CRS,
    ANALYSIS_SCALE,
    END_DATE,
    S1_FULL_COVERAGE,
    S2_FULL_COVERAGE,
    S1_ORBIT_PASS,
    S1_RELATIVE_ORBIT,
)

from .meteorology import (
    get_meteorological_properties,
)

from .schema import (
    EXPECTED_STAT_KEYS,
    PREDICTOR_BANDS,
    BASE_PROPERTY_NAMES,
)

from .modis import (
    get_modis_period_end,
    prepare_modis_et,
)

from .sentinel1 import (
    build_s1_median,
    get_s1_coverage,
)

from .sentinel2 import (
    add_s2_indices,
    build_s2_daily_collection,
    build_s2_medoid,
)

from .spatial import (
    get_coverage_fraction,
    make_local_geometry,
)


# ============================================================
# Build availability table
# ============================================================

def build_availability_table(
    modis_inputs,
    s2_collection,
    s1_collection,
):
    modis_collection = (
        modis_inputs["collection"]
    )

    modis_projection = (
        modis_inputs["projection"]
    )

    modis_scale = (
        modis_inputs["scale"]
    )

    station_footprints = (
        modis_inputs[
            "station_footprints"
        ]
    )

    modis_list = (
        modis_collection.toList(
            modis_collection.size()
        )
    )

    modis_sequence = (
        ee.List.sequence(
            0,
            modis_collection
            .size()
            .subtract(1),
        )
    )

    footprint_list = (
        station_footprints.toList(
            station_footprints.size()
        )
    )

    footprint_sequence = (
        ee.List.sequence(
            0,
            station_footprints
            .size()
            .subtract(1),
        )
    )

    def process_modis_image(
        modis_index,
    ):
        modis_index = ee.Number(
            modis_index
        )

        modis_image = ee.Image(
            modis_list.get(
                modis_index
            )
        )

        period_start = (
            modis_image.date()
        )

        period_end = (
            get_modis_period_end(
                period_start
            )
        )

        number_days = (
            period_end.difference(
                period_start,
                "day",
            )
        )

        prepared_et = (
            prepare_modis_et(
                modis_image,
                number_days,
            )
        )

        def process_station(
            station_index,
        ):
            station_index = ee.Number(
                station_index
            )

            footprint = ee.Feature(
                footprint_list.get(
                    station_index
                )
            )

            footprint_geometry = (
                footprint.geometry()
            )

            station_point = (
                ee.Geometry.Point(
                    [
                        footprint.get(
                            "longitude"
                        ),
                        footprint.get(
                            "latitude"
                        ),
                    ]
                )
            )

            # ================================================
            # MODIS
            # ================================================

            et_stats = (
                prepared_et.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=station_point,
                    crs=modis_projection,
                    scale=modis_scale,
                    maxPixels=10,
                )
            )

            et_dictionary = (
                ee.Dictionary(
                    et_stats
                )
            )

            modis_good_raw = (
                et_dictionary.get(
                    "modis_good"
                )
            )

            modis_good = (
                ee.Number(
                    ee.Algorithms.If(
                        ee.Algorithms.IsEqual(
                            modis_good_raw,
                            None,
                        ),
                        0,
                        modis_good_raw,
                    )
                )
            )

            # ================================================
            # Sentinel-2
            # ================================================

            s2_period = (
                s2_collection
                .filterDate(
                    period_start,
                    period_end,
                )
                .filterBounds(
                    footprint_geometry
                )
            )

            s2_date_keys = (
                ee.List(
                    s2_period
                    .aggregate_array(
                        "date_key"
                    )
                )
                .distinct()
                .sort()
            )

            daily_s2 = (
                build_s2_daily_collection(
                    s2_period,
                    footprint_geometry,
                )
            )

            zero_mask = (
                ee.Image.constant(0)
                .rename("valid")
                .uint8()
            )

            def get_s2_date_mask(
                image,
            ):
                image = ee.Image(
                    image
                )

                return (
                    image
                    .select("Blue")
                    .mask()
                    .rename("valid")
                    .uint8()
                )

            date_masks = (
                daily_s2.map(
                    get_s2_date_mask
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

            s2_union_mask = (
                safe_date_masks
                .max()
                .rename("valid")
            )

            s2_coverage = (
                get_coverage_fraction(
                    s2_union_mask,
                    footprint_geometry,
                )
            )

            s2_valid = (
                ee.Number(
                    ee.Algorithms.If(
                        s2_coverage.gte(
                            S2_FULL_COVERAGE
                        ),
                        1,
                        0,
                    )
                )
            )

            # ================================================
            # Sentinel-1
            # ================================================

            s1_period = (
                s1_collection
                .filterDate(
                    period_start,
                    period_end,
                )
                .filterBounds(
                    footprint_geometry
                )
            )

            s1_date_keys = (
                ee.List(
                    s1_period
                    .aggregate_array(
                        "date_key"
                    )
                )
                .distinct()
                .sort()
            )

            s1_predictors = (
                build_s1_median(
                    s1_period,
                    footprint_geometry,
                )
            )

            s1_coverage = (
                get_s1_coverage(
                    s1_predictors,
                    footprint_geometry,
                )
            )

            s1_has_products = (
                s1_period
                .size()
                .gt(0)
            )

            s1_has_coverage = (
                s1_coverage.gte(
                    S1_FULL_COVERAGE
                )
            )

            s1_valid = (
                ee.Number(
                    ee.Algorithms.If(
                        s1_has_products.And(
                            s1_has_coverage
                        ),
                        1,
                        0,
                    )
                )
            )

            # ================================================
            # Analysis period
            # ================================================

            period_within_analysis = (
                period_end
                .millis()
                .lte(
                    ee.Date(
                        END_DATE
                    ).millis()
                )
            )

            # ================================================
            # Joint validity - legacy baseline
            # ================================================

            valid_condition = (
                modis_good
                .eq(1)
                .And(
                    s2_valid.eq(1)
                )
                .And(
                    s1_valid.eq(1)
                )
                .And(
                    period_within_analysis
                )
            )

            valid_observation = (
                ee.Number(
                    ee.Algorithms.If(
                        valid_condition,
                        1,
                        0,
                    )
                )
            )

            # ================================================
            # Availability feature
            # ================================================

            feature = ee.Feature(
                footprint_geometry,
                {
                    "station":
                        footprint.get(
                            "station"
                        ),

                    "station_id":
                        footprint.get(
                            "station_id"
                        ),

                    "longitude":
                        footprint.get(
                            "longitude"
                        ),

                    "latitude":
                        footprint.get(
                            "latitude"
                        ),

                    "modis_pixel_id":
                        footprint.get(
                            "modis_pixel_id"
                        ),

                    "footprint_area_m2":
                        footprint.get(
                            "footprint_area_m2"
                        ),

                    "period_start":
                        period_start.format(
                            "yyyy-MM-dd"
                        ),

                    "period_end":
                        period_end
                        .advance(
                            -1,
                            "day",
                        )
                        .format(
                            "yyyy-MM-dd"
                        ),

                    "number_days":
                        number_days,

                    "s2_dates_total":
                        s2_date_keys.size(),

                    "s2_dates":
                        s2_date_keys.join(
                            ";"
                        ),

                    "s2_products_total":
                        s2_period.size(),

                    "s2_union_coverage_pct":
                        s2_coverage
                        .multiply(100),

                    "s1_dates_total":
                        s1_date_keys.size(),

                    "s1_dates":
                        s1_date_keys.join(
                            ";"
                        ),

                    "s1_products_total":
                        s1_period.size(),

                    "s1_union_coverage_pct":
                        s1_coverage
                        .multiply(100),

                    "modis_good":
                        modis_good,

                    "s2_valid":
                        s2_valid,

                    "s1_valid":
                        s1_valid,

                    "period_within_analysis":
                        ee.Number(
                            ee.Algorithms.If(
                                period_within_analysis,
                                1,
                                0,
                            )
                        ),

                    "valid_observation":
                        valid_observation,

                    "target_support":
                        "MODIS_footprint",

                    "system:time_start":
                        period_start.millis(),
                },
            )

            return ee.Feature(
                feature.set(
                    et_dictionary
                )
            )

        return footprint_sequence.map(
            process_station
        )

    availability_list = (
        modis_sequence.map(
            process_modis_image
        )
    )

    return ee.FeatureCollection(
        ee.List(
            availability_list
        ).flatten()
    )


# ============================================================
# Valid observations - legacy baseline
# ============================================================

def get_valid_observations(
    availability_table,
):
    return (
        ee.FeatureCollection(
            availability_table
        )
        .filter(
            ee.Filter.eq(
                "valid_observation",
                1,
            )
        )
    )


# ============================================================
# Extraction observations
# ============================================================

def get_extraction_observations(
    availability_table,
):
    return (
        ee.FeatureCollection(
            availability_table
        )
        .filter(
            ee.Filter.eq(
                "period_within_analysis",
                1,
            )
        )
        .filter(
            ee.Filter.eq(
                "modis_good",
                1,
            )
        )
    )


# ============================================================
# Statistical reducer
# ============================================================

def get_statistics_reducer():
    return (
        ee.Reducer.mean()
        .combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True,
        )
    )


# ============================================================
# Statistical integrity
# ============================================================

def get_empty_stats_dictionary():
    expected_keys = ee.List(
        EXPECTED_STAT_KEYS
    )

    return ee.Dictionary.fromLists(
        expected_keys,
        ee.List.repeat(
            -9999,
            expected_keys.size(),
        ),
    )


def complete_stats_dictionary(
    raw_dictionary,
):
    raw_dictionary = (
        ee.Dictionary(
            raw_dictionary
        )
    )

    return (
        get_empty_stats_dictionary()
        .combine(
            raw_dictionary,
            True,
        )
    )


def get_missing_stat_keys(
    raw_dictionary,
):
    raw_dictionary = (
        ee.Dictionary(
            raw_dictionary
        )
    )

    return (
        ee.List(
            EXPECTED_STAT_KEYS
        )
        .removeAll(
            raw_dictionary.keys()
        )
    )


# ============================================================
# Calculate statistics for one observation
# ============================================================

def calculate_observation(
    observation,
    s2_collection,
    s1_collection,
    meteorology_inputs,
):
    observation = ee.Feature(
        observation
    )

    footprint_geometry = (
        observation.geometry()
    )

    period_start = (
        ee.Date(
            observation.get(
                "system:time_start"
            )
        )
    )

    period_end = (
        get_modis_period_end(
            period_start
        )
    )

    number_days = (
        ee.Number(
            observation.get(
                "number_days"
            )
        )
    )

    station_point = (
        ee.Geometry.Point(
            [
                observation.get(
                    "longitude"
                ),
                observation.get(
                    "latitude"
                ),
            ]
        )
    )

    local_geometry = (
        make_local_geometry(
            station_point
        )
    )

    # ========================================================
    # Meteorology
    # ========================================================

    meteorology = (
        get_meteorological_properties(
            period_start,
            period_end,
            number_days,
            station_point,
            meteorology_inputs,
        )
    )

    # ========================================================
    # Sentinel-2
    # ========================================================

    s2_period = (
        s2_collection
        .filterDate(
            period_start,
            period_end,
        )
        .filterBounds(
            footprint_geometry
        )
    )

    s2_medoid = (
        build_s2_medoid(
            s2_period,
            footprint_geometry,
        )
    )

    s2_predictors = (
        add_s2_indices(
            s2_medoid
        )
    )

    # ========================================================
    # Sentinel-1
    # ========================================================

    s1_period = (
        s1_collection
        .filterDate(
            period_start,
            period_end,
        )
        .filterBounds(
            footprint_geometry
        )
    )

    s1_predictors = (
        build_s1_median(
            s1_period,
            footprint_geometry,
        )
    )

    # ========================================================
    # Combined predictors
    # ========================================================

    predictors = (
        s2_predictors
        .addBands(
            s1_predictors
        )
        .select(
            PREDICTOR_BANDS
        )
        .toFloat()
    )

    statistics_reducer = (
        get_statistics_reducer()
    )

    # ========================================================
    # MODIS footprint statistics
    # ========================================================

    footprint_stats_raw = (
        ee.Dictionary(
            predictors.reduceRegion(
                reducer=statistics_reducer,
                geometry=footprint_geometry,
                crs=ANALYSIS_CRS,
                scale=ANALYSIS_SCALE,
                maxPixels=1e7,
                tileScale=8,
            )
        )
    )

    footprint_missing_keys = (
        get_missing_stat_keys(
            footprint_stats_raw
        )
    )

    footprint_stats = (
        complete_stats_dictionary(
            footprint_stats_raw
        )
    )

    # ========================================================
    # Local 60 x 60 m statistics
    # ========================================================

    local_stats_raw = (
        ee.Dictionary(
            predictors.reduceRegion(
                reducer=statistics_reducer,
                geometry=local_geometry,
                crs=ANALYSIS_CRS,
                scale=ANALYSIS_SCALE,
                maxPixels=1000,
                tileScale=4,
            )
        )
    )

    local_missing_keys = (
        get_missing_stat_keys(
            local_stats_raw
        )
    )

    local_stats = (
        complete_stats_dictionary(
            local_stats_raw
        )
    )

    return ee.Feature(
        observation
        .set(
            meteorology
        )
        .set(
            {
                "footprint_stats":
                    footprint_stats,

                "footprint_missing_keys":
                    footprint_missing_keys,

                "footprint_missing_count":
                    footprint_missing_keys.size(),

                "local_stats":
                    local_stats,

                "local_missing_keys":
                    local_missing_keys,

                "local_missing_count":
                    local_missing_keys.size(),
            }
        )
    )


# ============================================================
# Add statistics to observations
# ============================================================

def build_observations_with_stats(
    valid_observations,
    s2_collection,
    s1_collection,
    meteorology_inputs,
):
    def process_observation(
        observation,
    ):
        return calculate_observation(
            observation,
            s2_collection,
            s1_collection,
            meteorology_inputs,
        )

    return (
        ee.FeatureCollection(
            valid_observations
        )
        .map(
            process_observation
        )
    )


# ============================================================
# Build output row
# ============================================================

def build_output_row(
    observation,
    statistics_property,
    missing_count_property,
    scale_name,
    predictor_support,
):
    observation = ee.Feature(
        observation
    )

    base_properties = (
        observation.toDictionary(
            BASE_PROPERTY_NAMES
        )
    )

    statistics = (
        ee.Dictionary(
            observation.get(
                statistics_property
            )
        )
    )

    missing_count = (
        ee.Number(
            observation.get(
                missing_count_property
            )
        )
    )

    meteo_missing_count = (
        ee.Number(
            observation.get(
                "meteo_missing_count"
            )
        )
    )

    meteo_complete = (
        ee.Number(
            observation.get(
                "meteo_complete"
            )
        )
    )

    total_missing_count = (
        missing_count.add(
            meteo_missing_count
        )
    )

    stats_complete = (
        ee.Number(
            ee.Algorithms.If(
                missing_count
                .eq(0)
                .And(
                    meteo_complete.eq(1)
                ),
                1,
                0,
            )
        )
    )

    return ee.Feature(
        ee.Feature(
            None,
            base_properties,
        )
        .set(
            {
                "scale":
                    scale_name,

                "predictor_support":
                    predictor_support,

                "s1_pass":
                    S1_ORBIT_PASS,

                "s1_relative_orbit":
                    S1_RELATIVE_ORBIT,

                "missing_stats_count":
                    total_missing_count,

                "stats_complete":
                    stats_complete,
            }
        )
        .set(
            statistics
        )
    )


# ============================================================
# Footprint rows
# ============================================================

def build_footprint_rows(
    observations_with_stats,
):
    def process_observation(
        observation,
    ):
        return build_output_row(
            observation=observation,
            statistics_property=(
                "footprint_stats"
            ),
            missing_count_property=(
                "footprint_missing_count"
            ),
            scale_name="footprint",
            predictor_support=(
                "MODIS_footprint"
            ),
        )

    return (
        ee.FeatureCollection(
            observations_with_stats
        )
        .map(
            process_observation
        )
    )


# ============================================================
# Local 60 x 60 m rows
# ============================================================

def build_local_rows(
    observations_with_stats,
):
    def process_observation(
        observation,
    ):
        return build_output_row(
            observation=observation,
            statistics_property=(
                "local_stats"
            ),
            missing_count_property=(
                "local_missing_count"
            ),
            scale_name="local_60m",
            predictor_support=(
                "60m_x_60m"
            ),
        )

    return (
        ee.FeatureCollection(
            observations_with_stats
        )
        .map(
            process_observation
        )
    )


# ============================================================
# Final output table
# ============================================================

def build_output_table(
    observations_with_stats,
):
    footprint_rows = (
        build_footprint_rows(
            observations_with_stats
        )
    )

    local_rows = (
        build_local_rows(
            observations_with_stats
        )
    )

    output_table_all = (
        footprint_rows.merge(
            local_rows
        )
    )

    output_table = (
        output_table_all.filter(
            ee.Filter.eq(
                "stats_complete",
                1,
            )
        )
    )

    return {
        "all": output_table_all,

        "final": output_table,

        "footprint": (
            output_table.filter(
                ee.Filter.eq(
                    "scale",
                    "footprint",
                )
            )
        ),

        "local": (
            output_table.filter(
                ee.Filter.eq(
                    "scale",
                    "local_60m",
                )
            )
        ),
    }