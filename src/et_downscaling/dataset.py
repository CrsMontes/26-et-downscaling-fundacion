import ee

from .config import (
    ANALYSIS_CRS,
    END_DATE,
    OPTICAL_FULL_COVERAGE,
    S1_FULL_COVERAGE,
    S1_ORBIT_PASS,
    S1_RELATIVE_ORBIT,
    get_optical_output_label,
    get_optical_scale,
    normalize_optical_source,
)


from .schema import (
    BASE_PROPERTY_NAMES,
    QA_STAT_COLUMNS,
    get_satellite_extraction_bands,
    get_satellite_stat_columns,
)

from .modis import (
    get_modis_period_end,
    prepare_modis_et,
)

from .optical import (
    build_optical_predictors,
    filter_optical_period,
    get_optical_coverage,
    get_optical_date_keys,
)

from .sentinel1 import (
    build_s1_median,
    get_s1_coverage,
)


# ============================================================
# Build availability table
# ============================================================

def build_availability_table(
    modis_inputs,
    optical_collection,
    s1_collection,
    optical_source,
):
    optical_source = (
        normalize_optical_source(
            optical_source
        )
    )

    optical_label = (
        get_optical_output_label(
            optical_source
        )
    )

    optical_scale = (
        get_optical_scale(
            optical_source
        )
    )

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
                    maxPixels=100,
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
            # Optical source
            # ================================================

            optical_period = (
                filter_optical_period(
                    collection=(
                        optical_collection
                    ),
                    period_start=(
                        period_start
                    ),
                    period_end=(
                        period_end
                    ),
                    geometry=(
                        footprint_geometry
                    ),
                    source=(
                        optical_source
                    ),
                )
            )

            optical_date_keys = (
                get_optical_date_keys(
                    optical_period
                )
            )

            optical_coverage = (
                get_optical_coverage(
                    period_collection=(
                        optical_period
                    ),
                    geometry=(
                        footprint_geometry
                    ),
                    source=(
                        optical_source
                    ),
                )
            )

            optical_valid = (
                ee.Number(
                    ee.Algorithms.If(
                        optical_coverage.gte(
                            OPTICAL_FULL_COVERAGE
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
            # Legacy joint-validity flag
            #
            # This flag is retained for QA only.
            # Predictor extraction does not require optical or
            # Sentinel-1 full coverage.
            # ================================================

            valid_condition = (
                modis_good
                .eq(1)
                .And(
                    optical_valid.eq(1)
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

                    # Human-readable inclusive last date.
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

                    "optical_source":
                        optical_label,

                    "optical_scale_m":
                        optical_scale,

                    "optical_dates_total":
                        optical_date_keys.size(),

                    "optical_dates":
                        optical_date_keys.join(
                            ";"
                        ),

                    "optical_products_total":
                        optical_period.size(),

                    "optical_union_coverage_pct":
                        optical_coverage
                        .multiply(100),

                    "optical_valid":
                        optical_valid,

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

            return (
                ee.Feature(
                    feature.set(
                        et_dictionary
                    )
                )
            )

        return (
            footprint_sequence.map(
                process_station
            )
        )

    availability_list = (
        modis_sequence.map(
            process_modis_image
        )
    )

    return (
        ee.FeatureCollection(
            ee.List(
                availability_list
            ).flatten()
        )
    )


# ============================================================
# Valid observations - legacy QA view
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
    """
    Keep the neutral MODIS master.

    Optical and Sentinel-1 coverage are not hard extraction
    filters. Their completeness is retained for later local QA
    and controlled sensitivity analyses.
    """
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
    """
    Mean only.

    Within-footprint heterogeneity statistics are deliberately
    excluded because they are not scale-transferable to a
    single fine-grid prediction cell.
    """
    return (
        ee.Reducer.mean()
    )


# ============================================================
# Statistical integrity
# ============================================================

def get_empty_stats_dictionary(
    expected_keys,
):
    expected_keys = ee.List(
        expected_keys
    )

    return (
        ee.Dictionary.fromLists(
            expected_keys,
            ee.List.repeat(
                -9999,
                expected_keys.size(),
            ),
        )
    )


def complete_stats_dictionary(
    raw_dictionary,
    expected_keys,
):
    raw_dictionary = (
        ee.Dictionary(
            raw_dictionary
        )
    )

    return (
        get_empty_stats_dictionary(
            expected_keys
        )
        .combine(
            raw_dictionary,
            True,
        )
    )


def get_missing_stat_keys(
    raw_dictionary,
    expected_keys,
):
    raw_dictionary = (
        ee.Dictionary(
            raw_dictionary
        )
    )

    return (
        ee.List(
            expected_keys
        )
        .removeAll(
            raw_dictionary.keys()
        )
    )


def complete_qa_dictionary(
    raw_dictionary,
):
    raw_dictionary = (
        ee.Dictionary(
            raw_dictionary
        )
    )

    defaults = (
        ee.Dictionary.fromLists(
            ee.List(
                QA_STAT_COLUMNS
            ),
            ee.List.repeat(
                -9999,
                len(
                    QA_STAT_COLUMNS
                ),
            ),
        )
    )

    return (
        defaults.combine(
            raw_dictionary,
            True,
        )
    )


# ============================================================
# Calculate statistics for one observation
# ============================================================

def calculate_observation(
    observation,
    optical_collection,
    s1_collection,
    optical_source,
):
    optical_source = (
        normalize_optical_source(
            optical_source
        )
    )

    optical_scale = (
        get_optical_scale(
            optical_source
        )
    )

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

    # ========================================================
    # Optical predictors
    # ========================================================

    optical_period = (
        filter_optical_period(
            collection=(
                optical_collection
            ),
            period_start=(
                period_start
            ),
            period_end=(
                period_end
            ),
            geometry=(
                footprint_geometry
            ),
            source=(
                optical_source
            ),
        )
    )

    optical_predictors = (
        build_optical_predictors(
            period_collection=(
                optical_period
            ),
            geometry=(
                footprint_geometry
            ),
            source=(
                optical_source
            ),
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
    # Rich extraction predictors
    #
    # Extraction and model selection are intentionally
    # separated. Source-specific candidates are retained here,
    # while model subsets are selected later in local code.
    # ========================================================

    extraction_bands = (
        get_satellite_extraction_bands(
            optical_source
        )
    )

    stat_columns = (
        get_satellite_stat_columns(
            optical_source
        )
    )

    predictors = (
        optical_predictors
        .addBands(
            s1_predictors
        )
        .select(
            extraction_bands
        )
        .toFloat()
    )

    # Rename before reduction so exported columns remain
    # explicit *_mean variables with a mean-only reducer.
    predictor_statistics_image = (
        predictors.select(
            extraction_bands,
            stat_columns,
        )
    )

    statistics_reducer = (
        get_statistics_reducer()
    )

    footprint_stats_raw = (
        ee.Dictionary(
            predictor_statistics_image
            .reduceRegion(
                reducer=(
                    statistics_reducer
                ),
                geometry=(
                    footprint_geometry
                ),
                crs=(
                    ANALYSIS_CRS
                ),
                scale=(
                    optical_scale
                ),
                maxPixels=1e7,
                tileScale=8,
            )
        )
    )

    footprint_missing_keys = (
        get_missing_stat_keys(
            footprint_stats_raw,
            stat_columns,
        )
    )

    footprint_stats = (
        complete_stats_dictionary(
            footprint_stats_raw,
            stat_columns,
        )
    )

    # ========================================================
    # Sentinel-1 geometry QA
    #
    # Angle is exported but is not part of the extraction/model
    # predictor stack and therefore does not determine
    # predictor completeness.
    # ========================================================

    angle_image = (
        s1_predictors
        .select(
            [
                "Angle_deg",
            ],
            QA_STAT_COLUMNS,
        )
    )

    footprint_qa_stats_raw = (
        ee.Dictionary(
            angle_image.reduceRegion(
                reducer=(
                    ee.Reducer.mean()
                ),
                geometry=(
                    footprint_geometry
                ),
                crs=(
                    ANALYSIS_CRS
                ),
                scale=(
                    optical_scale
                ),
                maxPixels=1e7,
                tileScale=8,
            )
        )
    )

    footprint_qa_stats = (
        complete_qa_dictionary(
            footprint_qa_stats_raw
        )
    )

    return (
        ee.Feature(
            observation
            .set(
                {
                    "footprint_stats":
                        footprint_stats,

                    "footprint_qa_stats":
                        footprint_qa_stats,

                    "footprint_missing_keys":
                        footprint_missing_keys,

                    "footprint_missing_count":
                        footprint_missing_keys.size(),
                }
            )
        )
    )


# ============================================================
# Add statistics to observations
# ============================================================

def build_observations_with_stats(
    valid_observations,
    optical_collection,
    s1_collection,
    optical_source,
):
    def process_observation(
        observation,
    ):
        return (
            calculate_observation(
                observation=(
                    observation
                ),
                optical_collection=(
                    optical_collection
                ),
                s1_collection=(
                    s1_collection
                ),
                optical_source=(
                    optical_source
                ),
            )
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
    qa_statistics_property,
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

    qa_statistics = (
        ee.Dictionary(
            observation.get(
                qa_statistics_property
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

    total_missing_count = missing_count

    stats_complete = (
        ee.Number(
            ee.Algorithms.If(
                missing_count.eq(0),
                1,
                0,
            )
        )
    )

    return (
        ee.Feature(
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
            .set(
                qa_statistics
            )
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
        return (
            build_output_row(
                observation=(
                    observation
                ),
                statistics_property=(
                    "footprint_stats"
                ),
                qa_statistics_property=(
                    "footprint_qa_stats"
                ),
                missing_count_property=(
                    "footprint_missing_count"
                ),
                scale_name=(
                    "footprint"
                ),
                predictor_support=(
                    "MODIS_footprint"
                ),
            )
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
    output_table_all = (
        build_footprint_rows(
            observations_with_stats
        )
    )

    output_table_complete = (
        output_table_all.filter(
            ee.Filter.eq(
                "stats_complete",
                1,
            )
        )
    )

    return {
        "all":
            output_table_all,

        "complete":
            output_table_complete,
    }
