import csv
import shutil
from datetime import date, timedelta
from pathlib import Path

import ee

from et_downscaling.config import (
    END_DATE,
    OUTPUT_FILENAME,
    OUTPUT_PERIOD_LABEL,
    START_DATE,
)

from et_downscaling.dataset import (
    build_availability_table,
    build_observations_with_stats,
    build_output_table,
    get_extraction_observations,
)

from et_downscaling.export import (
    export_training_dataset,
)

from et_downscaling.meteorology import (
    build_meteorology_inputs,
)

from et_downscaling.modis import (
    build_modis_inputs,
)

from et_downscaling.sentinel1 import (
    get_sentinel1_collection,
)

from et_downscaling.sentinel2 import (
    get_sentinel2_collection,
)


# ============================================================
# Earth Engine initialization
# ============================================================

def initialize_earth_engine():
    while True:
        print()
        print(
            "Enter the Google Cloud Project ID "
            "to use with Earth Engine."
        )

        project_id = input(
            "Google Cloud Project ID: "
        ).strip()

        if not project_id:
            print()
            print(
                "Project ID cannot be empty."
            )
            continue

        try:
            ee.Initialize(
                project=project_id
            )

            # Perform a lightweight request to verify
            # that Earth Engine can use the project.
            ee.Number(1).getInfo()

        except Exception as error:
            print()
            print(
                "Earth Engine initialization failed."
            )

            print(
                "Check the Project ID, Earth Engine "
                "access, and project permissions."
            )

            print()
            print(
                "Error:",
                error,
            )

            retry = input(
                "\nTry another Project ID? [Y/n]: "
            ).strip().lower()

            if retry in {
                "n",
                "no",
            }:
                raise SystemExit(
                    "Execution cancelled."
                )

            continue

        print()
        print(
            "Earth Engine initialized successfully."
        )

        print(
            "Google Cloud Project ID:",
            project_id,
        )

        return project_id


# ============================================================
# Get processing years
# ============================================================

def get_processing_years():
    start_date = date.fromisoformat(
        START_DATE
    )

    end_date = date.fromisoformat(
        END_DATE
    )

    if end_date <= start_date:
        raise ValueError(
            "END_DATE must be later than START_DATE."
        )

    last_included_date = (
        end_date
        - timedelta(days=1)
    )

    return list(
        range(
            start_date.year,
            last_included_date.year + 1,
        )
    )


# ============================================================
# Get quarterly date ranges
# ============================================================

def get_quarter_ranges(
    year,
):
    analysis_start = (
        date.fromisoformat(
            START_DATE
        )
    )

    analysis_end = (
        date.fromisoformat(
            END_DATE
        )
    )

    quarter_bounds = [
        (
            "Q1",
            date(year, 1, 1),
            date(year, 4, 1),
        ),
        (
            "Q2",
            date(year, 4, 1),
            date(year, 7, 1),
        ),
        (
            "Q3",
            date(year, 7, 1),
            date(year, 10, 1),
        ),
        (
            "Q4",
            date(year, 10, 1),
            date(year + 1, 1, 1),
        ),
    ]

    quarter_ranges = []

    for (
        quarter_name,
        quarter_start,
        quarter_end,
    ) in quarter_bounds:

        effective_start = max(
            quarter_start,
            analysis_start,
        )

        effective_end = min(
            quarter_end,
            analysis_end,
        )

        if effective_start >= effective_end:
            continue

        quarter_ranges.append(
            (
                quarter_name,
                effective_start.isoformat(),
                effective_end.isoformat(),
            )
        )

    return quarter_ranges


# ============================================================
# Get station identifiers
# ============================================================

def get_station_ids(
    station_footprints,
):
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

    return station_ids


# ============================================================
# Merge local CSV chunks
# ============================================================

def merge_csv_chunks(
    chunk_paths,
    output_path,
):
    header_written = False

    total_rows = 0
    footprint_rows = 0
    local_rows = 0

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

                if not header_written:
                    writer = csv.DictWriter(
                        output_file,
                        fieldnames=reader.fieldnames,
                    )

                    writer.writeheader()

                    header_written = True

                for row in reader:
                    writer.writerow(
                        row
                    )

                    total_rows += 1

                    if row["scale"] == "footprint":
                        footprint_rows += 1

                    elif row["scale"] == "local_60m":
                        local_rows += 1

    return {
        "total": total_rows,
        "footprint": footprint_rows,
        "local_60m": local_rows,
    }


# ============================================================
# Main workflow
# ============================================================

def main():
    initialize_earth_engine()

    print()
    print(
        "Analysis period:"
    )

    print(
        START_DATE,
        "to",
        END_DATE,
        "(end date exclusive)",
    )

    print(
        "Output period label:",
        OUTPUT_PERIOD_LABEL,
    )

    print()
    print(
        "Building Earth Engine computation graph..."
    )

    # ========================================================
    # Build input collections
    # ========================================================

    modis_inputs = (
        build_modis_inputs()
    )

    station_footprints = (
        modis_inputs[
            "station_footprints"
        ]
    )

    s2_collection = (
        get_sentinel2_collection(
            station_footprints
        )
    )

    s1_collection = (
        get_sentinel1_collection(
            station_footprints
        )
    )

    print()
    print(
        "Resolving ERA5-Land station support..."
    )

    meteorology_inputs = (
        build_meteorology_inputs(
            station_footprints
        )
    )

    print(
        "ERA5-Land station support resolved."
    )

    # ========================================================
    # Build availability table
    # ========================================================

    availability_table = (
        build_availability_table(
            modis_inputs,
            s2_collection,
            s1_collection,
        )
    )

    # ========================================================
    # Select observations for predictor extraction
    #
    # Coverage thresholds for Sentinel-2 and Sentinel-1 are
    # deliberately not applied here. Spatial coverage is
    # preserved as continuous quality-control information.
    #
    # Extraction currently requires:
    # - observation inside the analysis period;
    # - valid MODIS ET target.
    # ========================================================

    extraction_observations = (
        get_extraction_observations(
            availability_table
        )
    )

    print(
        "Computation graph ready."
    )

    # ========================================================
    # Define processing partitions
    # ========================================================

    print()
    print(
        "Reading station identifiers..."
    )

    station_ids = (
        get_station_ids(
            station_footprints
        )
    )

    years = (
        get_processing_years()
    )

    print(
        "Stations:",
        len(station_ids),
    )

    print(
        "Years:",
        years,
    )

    # ========================================================
    # Prepare local output directories
    # ========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    output_directory = (
        project_root
        / "outputs"
    )

    chunk_directory = (
        output_directory
        / "_chunks"
        / OUTPUT_PERIOD_LABEL
    )

    chunk_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunk_paths = []

    # ========================================================
    # Process partitions
    # ========================================================

    for (
        station_index,
        station_id,
    ) in enumerate(
        station_ids,
        start=1,
    ):

        for year in years:

            annual_filename = (
                f"station_"
                f"{station_index:02d}_"
                f"{year}.csv"
            )

            annual_path = (
                chunk_directory
                / annual_filename
            )

            # =================================================
            # Reuse completed annual partition
            # =================================================

            if annual_path.exists():
                print()
                print(
                    "Using existing annual partition:"
                )

                print(
                    annual_filename
                )

                chunk_paths.append(
                    annual_path
                )

                continue

            # =================================================
            # Process missing annual partition by quarter
            # =================================================

            quarter_ranges = (
                get_quarter_ranges(
                    year
                )
            )

            if not quarter_ranges:
                continue

            print()
            print(
                "Annual partition not available."
            )

            print(
                "Station:",
                station_index,
            )

            print(
                "Year:",
                year,
            )

            print(
                "Processing by quarter..."
            )

            for (
                quarter_name,
                quarter_start,
                quarter_end,
            ) in quarter_ranges:

                quarter_filename = (
                    f"station_"
                    f"{station_index:02d}_"
                    f"{year}_"
                    f"{quarter_name}.csv"
                )

                quarter_path = (
                    chunk_directory
                    / quarter_filename
                )

                # =============================================
                # Reuse completed quarterly partition
                # =============================================

                if quarter_path.exists():
                    print()
                    print(
                        "Using existing partition:"
                    )

                    print(
                        quarter_filename
                    )

                    chunk_paths.append(
                        quarter_path
                    )

                    continue

                print()
                print(
                    "Processing partition:"
                )

                print(
                    "Station:",
                    station_index,
                )

                print(
                    "Year:",
                    year,
                )

                print(
                    "Quarter:",
                    quarter_name,
                )

                print(
                    "Period:",
                    quarter_start,
                    "to",
                    quarter_end,
                    "(end date exclusive)",
                )

                partition = (
                    extraction_observations
                    .filter(
                        ee.Filter.eq(
                            "station_id",
                            station_id,
                        )
                    )
                    .filterDate(
                        quarter_start,
                        quarter_end,
                    )
                )

                # =============================================
                # Heavy processing happens only for this subset
                # =============================================

                observations_with_stats = (
                    build_observations_with_stats(
                        partition,
                        s2_collection,
                        s1_collection,
                        meteorology_inputs,
                    )
                )

                output = (
                    build_output_table(
                        observations_with_stats
                    )
                )

                print(
                    "Requesting local CSV..."
                )

                # Export the complete master table.
                #
                # Rows with incomplete predictors are retained
                # and identified through the QC and missingness
                # fields instead of being discarded here.
                downloaded_path = (
                    export_training_dataset(
                        output["all"],
                        output_filename=(
                            f"_chunks/"
                            f"{OUTPUT_PERIOD_LABEL}/"
                            f"{quarter_filename}"
                        ),
                    )
                )

                chunk_paths.append(
                    downloaded_path
                )

                print(
                    "Partition completed."
                )

    # ========================================================
    # Check partitions
    # ========================================================

    if not chunk_paths:
        raise RuntimeError(
            "No CSV partitions were generated "
            "for the selected analysis period."
        )

    # ========================================================
    # Merge all local chunks
    # ========================================================

    print()
    print(
        "Merging local CSV files..."
    )

    final_output_path = (
        output_directory
        / OUTPUT_FILENAME
    )

    summary = (
        merge_csv_chunks(
            chunk_paths,
            final_output_path,
        )
    )

    # ========================================================
    # Validate master dataset
    # ========================================================

    print()
    print(
        "Master dataset summary:"
    )

    print(
        "Total rows:",
        summary["total"],
    )

    print(
        "Footprint rows:",
        summary["footprint"],
    )

    print(
        "Local 60 m rows:",
        summary["local_60m"],
    )

    if summary["total"] == 0:
        raise RuntimeError(
            "Master dataset is empty."
        )

    if (
        summary["total"]
        != (
            summary["footprint"]
            + summary["local_60m"]
        )
    ):
        raise RuntimeError(
            "Master dataset contains rows "
            "with unexpected scale values."
        )

    if (
        summary["footprint"]
        != summary["local_60m"]
    ):
        raise RuntimeError(
            "Footprint and local 60 m row "
            "counts do not match."
        )

    # ========================================================
    # Remove temporary chunks for this analysis period
    # ========================================================

    shutil.rmtree(
        chunk_directory
    )

    # Remove the parent _chunks directory if empty.
    chunks_parent = (
        chunk_directory.parent
    )

    if (
        chunks_parent.exists()
        and not any(
            chunks_parent.iterdir()
        )
    ):
        chunks_parent.rmdir()

    print()
    print(
        "Master dataset completed successfully."
    )

    print(
        "Output file:",
        final_output_path,
    )


if __name__ == "__main__":
    main()