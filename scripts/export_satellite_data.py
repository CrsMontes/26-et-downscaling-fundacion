import argparse
import csv
import shutil
from datetime import date, timedelta
from pathlib import Path

import ee

from et_downscaling.config import (
    DEFAULT_OPTICAL_SOURCE,
    END_DATE,
    OUTPUT_PERIOD_LABEL,
    START_DATE,
    build_satellite_output_filename,
    get_optical_output_label,
    get_optical_scale,
    normalize_optical_source,
)
from et_downscaling.dataset import (
    build_availability_table,
    build_observations_with_stats,
    build_output_table,
    get_extraction_observations,
)
from et_downscaling.export import (
    export_feature_collection,
)
from et_downscaling.modis import (
    build_modis_inputs,
)
from et_downscaling.period import expected_observation_count
from et_downscaling.optical import (
    get_optical_collection,
)
from et_downscaling.schema import (
    get_satellite_export_selectors,
)
from et_downscaling.sentinel1 import (
    get_sentinel1_collection,
)
from et_downscaling.workspace import get_workspace_paths


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Export MODIS-footprint satellite predictors only. "
            "Meteorological processing is intentionally separate."
        )
    )

    parser.add_argument(
        "--optical-source",
        default=DEFAULT_OPTICAL_SOURCE,
        choices=[
            "S2",
            "HLS",
            "HLS_COMBINED",
        ],
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild the final source file and all processing "
            "partitions even if existing outputs are available."
        ),
    )

    return parser.parse_args()


# ============================================================
# Earth Engine initialization
# ============================================================

def initialize_earth_engine():
    while True:
        project_id = input(
            "Google Cloud Project ID: "
        ).strip()

        if not project_id:
            print(
                "Project ID cannot be empty."
            )

            continue

        try:
            ee.Initialize(
                project=project_id
            )

            ee.Number(
                1
            ).getInfo()

            print(
                "Earth Engine initialized with project:",
                project_id,
            )

            return project_id

        except Exception as error:
            print(
                "Earth Engine initialization failed:",
                error,
            )


# ============================================================
# Processing years
# ============================================================

def get_processing_years():
    start = date.fromisoformat(
        START_DATE
    )

    end = date.fromisoformat(
        END_DATE
    )

    if end <= start:
        raise ValueError(
            "END_DATE must be later than START_DATE."
        )

    last_analysis_day = (
        end
        - timedelta(
            days=1
        )
    )

    return list(
        range(
            start.year,
            last_analysis_day.year + 1,
        )
    )


# ============================================================
# Quarterly processing ranges
#
# Used for Sentinel-2 because this partition size has already
# completed successfully.
# ============================================================

def get_quarter_ranges(
    year,
):
    analysis_start = date.fromisoformat(
        START_DATE
    )

    analysis_end = date.fromisoformat(
        END_DATE
    )

    bounds = [
        (
            "Q1",
            date(
                year,
                1,
                1,
            ),
            date(
                year,
                4,
                1,
            ),
        ),
        (
            "Q2",
            date(
                year,
                4,
                1,
            ),
            date(
                year,
                7,
                1,
            ),
        ),
        (
            "Q3",
            date(
                year,
                7,
                1,
            ),
            date(
                year,
                10,
                1,
            ),
        ),
        (
            "Q4",
            date(
                year,
                10,
                1,
            ),
            date(
                year + 1,
                1,
                1,
            ),
        ),
    ]

    result = []

    for (
        name,
        start,
        end,
    ) in bounds:

        effective_start = max(
            start,
            analysis_start,
        )

        effective_end = min(
            end,
            analysis_end,
        )

        if (
            effective_start
            < effective_end
        ):
            result.append(
                (
                    name,
                    effective_start.isoformat(),
                    effective_end.isoformat(),
                )
            )

    return result


# ============================================================
# Monthly processing ranges
#
# Used for HLS to reduce the size of each Earth Engine graph.
# This changes only computational partitioning.
# ============================================================

def get_month_ranges(
    year,
):
    analysis_start = date.fromisoformat(
        START_DATE
    )

    analysis_end = date.fromisoformat(
        END_DATE
    )

    result = []

    for month in range(
        1,
        13,
    ):

        month_start = date(
            year,
            month,
            1,
        )

        if month == 12:
            month_end = date(
                year + 1,
                1,
                1,
            )

        else:
            month_end = date(
                year,
                month + 1,
                1,
            )

        effective_start = max(
            month_start,
            analysis_start,
        )

        effective_end = min(
            month_end,
            analysis_end,
        )

        if (
            effective_start
            < effective_end
        ):
            result.append(
                (
                    f"M{month:02d}",
                    effective_start.isoformat(),
                    effective_end.isoformat(),
                )
            )

    return result


# ============================================================
# Select computational partition scheme
# ============================================================

def get_partition_ranges(
    year,
    optical_source,
):
    if optical_source == "HLS":
        return get_month_ranges(
            year
        )

    return get_quarter_ranges(
        year
    )


# ============================================================
# Merge local CSV partitions
# ============================================================

def merge_csv_chunks(
    chunk_paths,
    output_path,
):
    header = None

    total_rows = 0
    footprint_rows = 0

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = None

        for chunk_path in chunk_paths:

            with chunk_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as input_file:

                reader = csv.DictReader(
                    input_file
                )

                if reader.fieldnames is None:
                    continue

                if header is None:
                    header = reader.fieldnames

                    writer = csv.DictWriter(
                        output_file,
                        fieldnames=header,
                    )

                    writer.writeheader()

                elif (
                    reader.fieldnames
                    != header
                ):
                    raise ValueError(
                        "CSV schema mismatch in "
                        f"{chunk_path}"
                    )

                for row in reader:
                    writer.writerow(
                        row
                    )

                    total_rows += 1

                    footprint_rows += int(
                        row.get(
                            "scale"
                        )
                        == "footprint"
                    )

    return (
        total_rows,
        footprint_rows,
    )


# ============================================================
# Build MODIS inputs for one computational partition
# ============================================================

def build_partition_modis_inputs(
    modis_inputs,
    station_footprints,
    station_id,
    partition_start,
    partition_end,
):
    """
    Restrict the MODIS availability graph to one station and
    one computational time partition.

    The MODIS image selected by filterDate() retains its own
    original period start. build_availability_table() continues
    to calculate the actual MODIS period end for each image.

    Therefore, this function changes the size of the Earth
    Engine graph but not the scientific temporal support of
    individual MODIS observations.
    """

    partition_station_footprints = (
        ee.FeatureCollection(
            station_footprints
        )
        .filter(
            ee.Filter.eq(
                "station_id",
                station_id,
            )
        )
    )

    partition_modis_collection = (
        ee.ImageCollection(
            modis_inputs[
                "collection"
            ]
        )
        .filterDate(
            partition_start,
            partition_end,
        )
    )

    partition_inputs = dict(
        modis_inputs
    )

    partition_inputs[
        "collection"
    ] = (
        partition_modis_collection
    )

    partition_inputs[
        "station_footprints"
    ] = (
        partition_station_footprints
    )

    return partition_inputs


# ============================================================
# Main workflow
# ============================================================

def main():
    args = parse_arguments()

    optical_source = (
        normalize_optical_source(
            args.optical_source
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

    export_selectors = (
        get_satellite_export_selectors(
            optical_source
        )
    )

    # ========================================================
    # Output paths
    # ========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    workspace = get_workspace_paths(project_root).ensure()
    source_directory = (
        workspace.raw_cache
        / "satellite"
        / optical_label
    )

    final_output = (
        source_directory
        / build_satellite_output_filename(
            optical_source
        )
    )

    if (
        final_output.exists()
        and not args.force
    ):
        print(
            "Raw satellite file already exists:",
            final_output,
        )

        print(
            "Use --force only when an intentional rebuild "
            "is required."
        )

        return

    # ========================================================
    # Earth Engine
    # ========================================================

    initialize_earth_engine()

    print(
        "Analysis period:",
        START_DATE,
        "to",
        END_DATE,
        "(exclusive)",
    )

    print(
        "Optical source:",
        optical_label,
    )

    print(
        "Optical working scale:",
        optical_scale,
        "m",
    )

    # ========================================================
    # Global source collections
    #
    # These remain lazy Earth Engine collections.
    #
    # The heavy availability and predictor graph is NOT built
    # globally. It is built later for each station-partition.
    # ========================================================

    modis_inputs = (
        build_modis_inputs()
    )

    station_footprints = (
        modis_inputs[
            "station_footprints"
        ]
    )

    optical_collection = (
        get_optical_collection(
            station_footprints,
            optical_source,
        )
    )

    s1_collection = (
        get_sentinel1_collection(
            station_footprints
        )
    )

    # ========================================================
    # Read station identifiers
    # ========================================================

    station_ids = (
        ee.FeatureCollection(
            station_footprints
        )
        .aggregate_array(
            "station_id"
        )
        .distinct()
        .sort()
        .getInfo()
    )

    number_periods = int(
        ee.ImageCollection(modis_inputs["collection"]).size().getInfo()
    )
    expected_rows = expected_observation_count(number_periods, len(station_ids))

    years = (
        get_processing_years()
    )

    print(
        "Stations:",
        len(
            station_ids
        ),
    )

    print(
        "Years:",
        years,
    )

    # ========================================================
    # Partition output directory
    # ========================================================

    chunk_directory = (
        source_directory
        / "_chunks"
        / OUTPUT_PERIOD_LABEL
    )

    chunk_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunk_paths = []

    # ========================================================
    # Process station x time partitions
    #
    # S2:
    #     station x quarter
    #
    # HLS:
    #     station x month
    #
    # Crucially, build_availability_table() is called only
    # AFTER station and temporal filtering.
    # ========================================================

    for (
        station_index,
        station_id,
    ) in enumerate(
        station_ids,
        start=1,
    ):

        for year in years:

            partition_ranges = (
                get_partition_ranges(
                    year,
                    optical_source,
                )
            )

            if not partition_ranges:
                continue

            for (
                partition_name,
                partition_start,
                partition_end,
            ) in partition_ranges:

                filename = (
                    f"station_"
                    f"{station_index:02d}_"
                    f"{year}_"
                    f"{partition_name}.csv"
                )

                relative = (
                    Path(
                        "raw"
                    )
                    / "satellite"
                    / optical_label
                    / "_chunks"
                    / OUTPUT_PERIOD_LABEL
                    / filename
                )

                path = workspace.root / relative

                # =============================================
                # Reuse completed checkpoint
                # =============================================

                if (
                    path.exists()
                    and not args.force
                ):
                    print(
                        "Using existing partition:",
                        path,
                    )

                    chunk_paths.append(
                        path
                    )

                    continue

                print(
                    f"Processing {optical_label} | "
                    f"station {station_index} | "
                    f"{year} {partition_name}"
                )

                print(
                    "Partition:",
                    partition_start,
                    "to",
                    partition_end,
                    "(exclusive)",
                )

                # =============================================
                # Restrict MODIS inputs BEFORE availability
                #
                # This is the main memory-control correction.
                # =============================================

                partition_modis_inputs = (
                    build_partition_modis_inputs(
                        modis_inputs=(
                            modis_inputs
                        ),
                        station_footprints=(
                            station_footprints
                        ),
                        station_id=(
                            station_id
                        ),
                        partition_start=(
                            partition_start
                        ),
                        partition_end=(
                            partition_end
                        ),
                    )
                )

                # =============================================
                # Build availability only for this station and
                # this computational partition.
                # =============================================

                partition_availability = (
                    build_availability_table(
                        modis_inputs=(
                            partition_modis_inputs
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

                # =============================================
                # Keep physically valid MODIS observations.
                #
                # Optical and S1 coverage thresholds remain QA
                # variables and are not applied here.
                # =============================================

                partition_observations = (
                    get_extraction_observations(
                        partition_availability
                    )
                )

                # =============================================
                # Calculate optical + Sentinel-1 statistics
                # only for the current partition.
                # =============================================

                with_stats = (
                    build_observations_with_stats(
                        valid_observations=(
                            partition_observations
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

                output = (
                    build_output_table(
                        with_stats
                    )
                )

                # =============================================
                # Request local CSV
                # =============================================

                downloaded = (
                    export_feature_collection(
                        feature_collection=(
                            output[
                                "all"
                            ]
                        ),
                        output_filename=(
                            relative.as_posix()
                        ),
                        selectors=(
                            export_selectors
                        ),
                    )
                )

                chunk_paths.append(
                    downloaded
                )

                print(
                    "Partition completed:",
                    downloaded,
                )

    # ========================================================
    # Validate partition list
    # ========================================================

    if not chunk_paths:
        raise RuntimeError(
            "No satellite CSV partitions were generated."
        )

    # ========================================================
    # Merge all completed partitions
    # ========================================================

    source_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Merging satellite partitions..."
    )

    (
        total_rows,
        footprint_rows,
    ) = merge_csv_chunks(
        chunk_paths,
        final_output,
    )

    # ========================================================
    # Validate final master
    # ========================================================

    if total_rows == 0:
        raise RuntimeError(
            "Satellite master contains zero rows."
        )

    if (
        total_rows
        != footprint_rows
    ):
        raise RuntimeError(
            "Invalid satellite master: "
            f"total={total_rows}, "
            f"footprint={footprint_rows}"
        )

    # Do not silently accept incomplete output. Derive the expectation from
    # the configured MODIS collection and station supports.
    if total_rows != expected_rows:
        raise RuntimeError(
            "Unexpected satellite master row count: "
            f"{total_rows}. Expected {expected_rows} "
            f"({number_periods} periods x {len(station_ids)} supports)."
        )

    # ========================================================
    # Remove checkpoints only after successful final merge
    # ========================================================

    shutil.rmtree(
        chunk_directory
    )

    print(
        "Satellite export completed:",
        final_output,
    )

    print(
        "Rows:",
        total_rows,
    )


if __name__ == "__main__":
    main()
