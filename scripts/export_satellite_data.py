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
from et_downscaling.export import export_feature_collection
from et_downscaling.modis import build_modis_inputs
from et_downscaling.optical import get_optical_collection
from et_downscaling.schema import get_satellite_export_selectors
from et_downscaling.sentinel1 import get_sentinel1_collection


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
        choices=["S2", "HLS", "HLS_COMBINED"],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the final source file even if it already exists.",
    )
    return parser.parse_args()


def initialize_earth_engine():
    while True:
        project_id = input("Google Cloud Project ID: ").strip()
        if not project_id:
            print("Project ID cannot be empty.")
            continue
        try:
            ee.Initialize(project=project_id)
            ee.Number(1).getInfo()
            print("Earth Engine initialized with project:", project_id)
            return project_id
        except Exception as error:
            print("Earth Engine initialization failed:", error)


def get_processing_years():
    start = date.fromisoformat(START_DATE)
    end = date.fromisoformat(END_DATE)
    if end <= start:
        raise ValueError("END_DATE must be later than START_DATE.")
    return list(range(start.year, (end - timedelta(days=1)).year + 1))


def get_quarter_ranges(year):
    analysis_start = date.fromisoformat(START_DATE)
    analysis_end = date.fromisoformat(END_DATE)
    bounds = [
        ("Q1", date(year, 1, 1), date(year, 4, 1)),
        ("Q2", date(year, 4, 1), date(year, 7, 1)),
        ("Q3", date(year, 7, 1), date(year, 10, 1)),
        ("Q4", date(year, 10, 1), date(year + 1, 1, 1)),
    ]
    result = []
    for name, start, end in bounds:
        start = max(start, analysis_start)
        end = min(end, analysis_end)
        if start < end:
            result.append((name, start.isoformat(), end.isoformat()))
    return result


def merge_csv_chunks(chunk_paths, output_path):
    header = None
    total_rows = 0
    footprint_rows = 0
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = None
        for chunk_path in chunk_paths:
            with chunk_path.open("r", newline="", encoding="utf-8") as input_file:
                reader = csv.DictReader(input_file)
                if reader.fieldnames is None:
                    continue
                if header is None:
                    header = reader.fieldnames
                    writer = csv.DictWriter(output_file, fieldnames=header)
                    writer.writeheader()
                elif reader.fieldnames != header:
                    raise ValueError(f"CSV schema mismatch in {chunk_path}")
                for row in reader:
                    writer.writerow(row)
                    total_rows += 1
                    footprint_rows += int(row.get("scale") == "footprint")
    return total_rows, footprint_rows


def main():
    args = parse_arguments()
    optical_source = normalize_optical_source(args.optical_source)
    optical_label = get_optical_output_label(optical_source)
    optical_scale = get_optical_scale(optical_source)
    export_selectors = get_satellite_export_selectors(
        optical_source
    )

    project_root = Path(__file__).resolve().parents[1]
    source_directory = project_root / "outputs" / "raw" / "satellite" / optical_label
    final_output = source_directory / build_satellite_output_filename(optical_source)

    if final_output.exists() and not args.force:
        print("Raw satellite file already exists:", final_output)
        print("Use --force only when an intentional rebuild is required.")
        return

    initialize_earth_engine()
    print("Analysis period:", START_DATE, "to", END_DATE, "(exclusive)")
    print("Optical source:", optical_label)
    print("Optical working scale:", optical_scale, "m")

    modis_inputs = build_modis_inputs()
    station_footprints = modis_inputs["station_footprints"]
    optical_collection = get_optical_collection(station_footprints, optical_source)
    s1_collection = get_sentinel1_collection(station_footprints)

    availability = build_availability_table(
        modis_inputs=modis_inputs,
        optical_collection=optical_collection,
        s1_collection=s1_collection,
        optical_source=optical_source,
    )
    observations = get_extraction_observations(availability)

    station_ids = (
        ee.FeatureCollection(station_footprints)
        .aggregate_array("station_id")
        .distinct()
        .sort()
        .getInfo()
    )
    years = get_processing_years()

    chunk_directory = source_directory / "_chunks" / OUTPUT_PERIOD_LABEL
    chunk_directory.mkdir(parents=True, exist_ok=True)
    chunk_paths = []

    for station_index, station_id in enumerate(station_ids, start=1):
        for year in years:
            for quarter_name, quarter_start, quarter_end in get_quarter_ranges(year):
                filename = f"station_{station_index:02d}_{year}_{quarter_name}.csv"
                relative = Path("raw") / "satellite" / optical_label / "_chunks" / OUTPUT_PERIOD_LABEL / filename
                path = project_root / "outputs" / relative
                if path.exists() and not args.force:
                    print("Using existing partition:", path)
                    chunk_paths.append(path)
                    continue

                print(
                    f"Processing {optical_label} | station {station_index} | "
                    f"{year} {quarter_name}"
                )
                partition = (
                    observations
                    .filter(ee.Filter.eq("station_id", station_id))
                    .filterDate(quarter_start, quarter_end)
                )
                with_stats = build_observations_with_stats(
                    valid_observations=partition,
                    optical_collection=optical_collection,
                    s1_collection=s1_collection,
                    optical_source=optical_source,
                )
                output = build_output_table(with_stats)
                downloaded = export_feature_collection(
                    feature_collection=output["all"],
                    output_filename=relative.as_posix(),
                    selectors=export_selectors,
                )
                chunk_paths.append(downloaded)

    if not chunk_paths:
        raise RuntimeError("No satellite CSV partitions were generated.")

    source_directory.mkdir(parents=True, exist_ok=True)
    total_rows, footprint_rows = merge_csv_chunks(chunk_paths, final_output)
    if total_rows == 0 or total_rows != footprint_rows:
        raise RuntimeError(
            f"Invalid satellite master: total={total_rows}, footprint={footprint_rows}"
        )

    shutil.rmtree(chunk_directory)
    print("Satellite export completed:", final_output)
    print("Rows:", total_rows)


if __name__ == "__main__":
    main()
