import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from et_downscaling.config import (
    DEFAULT_OPTICAL_SOURCE,
    END_DATE,
    OUTPUT_PERIOD_LABEL,
    START_DATE,
    build_satellite_output_filename,
    build_training_output_filename,
    get_optical_output_label,
    normalize_optical_source,
)
from et_downscaling.local_training import build_training_master


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Build the complete ET training master locally from reusable raw exports. "
            "This script does not connect to Earth Engine."
        )
    )
    parser.add_argument(
        "--optical-source",
        default=DEFAULT_OPTICAL_SOURCE,
        choices=["S2", "HLS", "HLS_COMBINED"],
    )
    return parser.parse_args()


def require_file(path: Path, upstream_command: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}\nRun first: {upstream_command}"
        )


def main():
    args = parse_arguments()
    optical_source = normalize_optical_source(args.optical_source)
    optical_label = get_optical_output_label(optical_source)
    project_root = Path(__file__).resolve().parents[1]

    satellite_path = (
        project_root
        / "outputs"
        / "raw"
        / "satellite"
        / optical_label
        / build_satellite_output_filename(optical_source)
    )
    meteorology_directory = project_root / "outputs" / "raw" / "meteorology"
    support_path = meteorology_directory / "station_support.csv"
    era5_path = meteorology_directory / f"era5_hourly_{OUTPUT_PERIOD_LABEL}.csv"
    chirps_start = (date.fromisoformat(START_DATE) - timedelta(days=30)).strftime("%Y%m%d")
    chirps_end = (date.fromisoformat(END_DATE) - timedelta(days=1)).strftime("%Y%m%d")
    chirps_path = meteorology_directory / f"chirps_daily_{chirps_start}_{chirps_end}.csv"

    require_file(
        satellite_path,
        f"python scripts/export_satellite_data.py --optical-source {optical_label}",
    )
    require_file(
        support_path,
        "python scripts/export_meteorology_data.py",
    )
    require_file(era5_path, "python scripts/export_meteorology_data.py")
    require_file(chirps_path, "python scripts/export_meteorology_data.py")

    print("Loading reusable local inputs...")
    station_dtype = {
        "station_id": "string",
    }

    satellite = pd.read_csv(
        satellite_path,
        dtype=station_dtype,
    )

    station_support = pd.read_csv(
        support_path,
        dtype=station_dtype,
    )

    era5_hourly = pd.read_csv(
        era5_path,
        dtype=station_dtype,
    )

    chirps_daily = pd.read_csv(
        chirps_path,
        dtype=station_dtype,
    )

    master, daily_reference = build_training_master(
        satellite=satellite,
        era5_hourly=era5_hourly,
        chirps_daily=chirps_daily,
        station_support=station_support,
    )

    output_directory = project_root / "outputs" / "processed" / "training" / optical_label
    output_directory.mkdir(parents=True, exist_ok=True)
    master_path = output_directory / build_training_output_filename(optical_source)
    daily_path = output_directory / f"reference_et_daily_{OUTPUT_PERIOD_LABEL}.csv"
    master.to_csv(master_path, index=False)
    daily_reference.to_csv(daily_path, index=False)

    print("Training master rows:", len(master))
    print("Reference-ET complete:", int(master["reference_et_complete"].sum()))
    print("Meteorology complete:", int(master["meteo_complete"].sum()))
    print("Target complete:", int(master["target_complete"].sum()))
    print(
        "Satellite extraction complete:",
        int(
            master[
                "satellite_extraction_complete"
            ].sum()
        ),
    )

    for threshold in (80, 90, 99):
        common_column = (
            f"training_candidate_common_ge_{threshold}"
        )

        source_column = (
            f"training_candidate_source_ge_{threshold}"
        )

        print(
            f"Common-feature candidates >= {threshold}%:",
            int(
                master[
                    common_column
                ].sum()
            ),
        )

        print(
            f"Source-feature candidates >= {threshold}%:",
            int(
                master[
                    source_column
                ].sum()
            ),
        )

    max_reconstruction_error = master["ET_reconstruction_error_mm"].abs().max()
    print("Maximum ET reconstruction error (mm/period):", max_reconstruction_error)
    print("Saved:", master_path)
    print("Saved daily reference ET:", daily_path)


if __name__ == "__main__":
    main()
