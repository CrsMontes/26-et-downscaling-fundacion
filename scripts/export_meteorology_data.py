import argparse
import csv
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import ee

from et_downscaling.config import END_DATE, OUTPUT_PERIOD_LABEL, START_DATE
from et_downscaling.export import export_feature_collection
from et_downscaling.meteorology_export import (
    CHIRPS_EXPORT_SELECTORS,
    ERA5_EXPORT_SELECTORS,
    STATION_SUPPORT_SELECTORS,
    build_chirps_daily_table,
    build_era5_hourly_table,
    build_era5_station_supports,
    build_station_support_table,
    get_station_support,
)
from et_downscaling.modis import build_modis_inputs


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Export reusable raw ERA5-Land, CHIRPS, and static support data. "
            "All temporal derivation and reference ET are calculated locally."
        )
    )
    parser.add_argument("--force", action="store_true")
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
            return
        except Exception as error:
            print("Earth Engine initialization failed:", error)


def merge_csv_chunks(chunk_paths, output_path):
    header = None
    rows = 0
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
                    rows += 1
    return rows


def get_era5_utc_windows():
    # Period predictors reproduce the historical UTC MODIS windows.
    # Five additional UTC hours after END_DATE are exported only so
    # the final Colombia local day (UTC-5) is complete for ETo/ETr.
    raw_start = datetime.combine(
        date.fromisoformat(START_DATE),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    raw_end = datetime.combine(
        date.fromisoformat(END_DATE),
        datetime.min.time(),
        tzinfo=timezone.utc,
    ) + timedelta(hours=5)

    windows = []
    year = raw_start.year
    while datetime(year, 1, 1, tzinfo=timezone.utc) < raw_end:
        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        effective_start = max(raw_start, year_start)
        effective_end = min(raw_end, year_end)
        if effective_start < effective_end:
            windows.append(
                (
                    year,
                    effective_start.isoformat().replace("+00:00", "Z"),
                    effective_end.isoformat().replace("+00:00", "Z"),
                    int((effective_end - effective_start).total_seconds() // 3600),
                )
            )
        year += 1
    return windows


def get_chirps_year_windows():
    start = date.fromisoformat(START_DATE) - timedelta(days=30)
    end = date.fromisoformat(END_DATE)
    windows = []
    for year in range(start.year, (end - timedelta(days=1)).year + 1):
        year_start = date(year, 1, 1)
        year_end = date(year + 1, 1, 1)
        effective_start = max(start, year_start)
        effective_end = min(end, year_end)
        if effective_start < effective_end:
            windows.append((year, effective_start.isoformat(), effective_end.isoformat()))
    return windows


def main():
    args = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    output_directory = project_root / "outputs" / "raw" / "meteorology"
    support_path = output_directory / "station_support.csv"
    era5_path = output_directory / f"era5_hourly_{OUTPUT_PERIOD_LABEL}.csv"
    chirps_start = (date.fromisoformat(START_DATE) - timedelta(days=30)).strftime("%Y%m%d")
    chirps_end = (date.fromisoformat(END_DATE) - timedelta(days=1)).strftime("%Y%m%d")
    chirps_path = output_directory / f"chirps_daily_{chirps_start}_{chirps_end}.csv"

    if support_path.exists() and era5_path.exists() and chirps_path.exists() and not args.force:
        print("Reusable meteorological raw files already exist:")
        print(support_path)
        print(era5_path)
        print(chirps_path)
        print("Use --force only for an intentional rebuild.")
        return

    initialize_earth_engine()
    output_directory.mkdir(parents=True, exist_ok=True)

    modis_inputs = build_modis_inputs()
    station_footprints = modis_inputs["station_footprints"]
    era5_supports = build_era5_station_supports(station_footprints)
    support_table = build_station_support_table(station_footprints, era5_supports)

    if args.force or not support_path.exists():
        export_feature_collection(
            support_table,
            Path("raw/meteorology/station_support.csv").as_posix(),
            STATION_SUPPORT_SELECTORS,
        )

    support_info = support_table.getInfo().get("features", [])
    station_ids = [feature["properties"]["station_id"] for feature in support_info]
    if not station_ids:
        raise RuntimeError("No station support rows were resolved.")

    chunk_root = output_directory / "_chunks" / OUTPUT_PERIOD_LABEL
    era5_chunk_dir = chunk_root / "era5"
    chirps_chunk_dir = chunk_root / "chirps"
    era5_chunk_dir.mkdir(parents=True, exist_ok=True)
    chirps_chunk_dir.mkdir(parents=True, exist_ok=True)

    era5_chunks = []
    for station_index, station_id in enumerate(station_ids, start=1):
        station_support = get_station_support(support_table, station_id)
        for year, utc_start, utc_end, expected_hours in get_era5_utc_windows():
            filename = f"station_{station_index:02d}_{year}.csv"
            path = era5_chunk_dir / filename
            if not path.exists() or args.force:
                table = build_era5_hourly_table(station_support, utc_start, utc_end)
                relative = Path("raw/meteorology/_chunks") / OUTPUT_PERIOD_LABEL / "era5" / filename
                path = export_feature_collection(table, relative.as_posix(), ERA5_EXPORT_SELECTORS)
            else:
                print("Using existing ERA5 partition:", path)
            era5_chunks.append(path)

    chirps_chunks = []
    for station_index, station_id in enumerate(station_ids, start=1):
        station_support = get_station_support(support_table, station_id)
        for year, window_start, window_end in get_chirps_year_windows():
            filename = f"station_{station_index:02d}_{year}.csv"
            path = chirps_chunk_dir / filename
            if not path.exists() or args.force:
                table = build_chirps_daily_table(station_support, window_start, window_end)
                relative = Path("raw/meteorology/_chunks") / OUTPUT_PERIOD_LABEL / "chirps" / filename
                path = export_feature_collection(table, relative.as_posix(), CHIRPS_EXPORT_SELECTORS)
            else:
                print("Using existing CHIRPS partition:", path)
            chirps_chunks.append(path)

    era5_rows = merge_csv_chunks(era5_chunks, era5_path)
    chirps_rows = merge_csv_chunks(chirps_chunks, chirps_path)
    print("ERA5 rows:", era5_rows)
    print("CHIRPS rows:", chirps_rows)

    # Exact expected ERA5 row count from local calendar years.
    expected_era5 = (
        sum(window[3] for window in get_era5_utc_windows())
        * len(station_ids)
    )
    if era5_rows != expected_era5:
        raise RuntimeError(f"ERA5 row count mismatch: {era5_rows} != {expected_era5}")

    expected_chirps_days = (date.fromisoformat(END_DATE) - (date.fromisoformat(START_DATE) - timedelta(days=30))).days
    expected_chirps = expected_chirps_days * len(station_ids)
    if chirps_rows != expected_chirps:
        raise RuntimeError(f"CHIRPS row count mismatch: {chirps_rows} != {expected_chirps}")

    shutil.rmtree(chunk_root)
    print("Meteorology raw export completed:")
    print(support_path)
    print(era5_path)
    print(chirps_path)


if __name__ == "__main__":
    main()
