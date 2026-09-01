"""Export CHIRPS daily precipitation for the 2020-2024 experiment.

The export preserves the production sampling definition:
- CHIRPS DAILY
- native CHIRPS projection
- point sampling at each station coordinate
- 30 antecedent days before the experiment start

No ERA5 extraction or model training is performed.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import ee
import pandas as pd

from et_downscaling.export import export_feature_collection
from et_downscaling.meteorology_export import (
    CHIRPS_EXPORT_SELECTORS,
    build_chirps_daily_table,
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date-exclusive", required=True)
    parser.add_argument("--period-label", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--antecedent-days", type=int, default=30)
    parser.add_argument("--execute", action="store_true")

    return parser.parse_args()


def project_root():
    return Path(__file__).resolve().parents[1]


def build_windows(start_date, end_date):
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    windows = []

    for year in range(
        start.year,
        (end - timedelta(days=1)).year + 1,
    ):
        year_start = date(year, 1, 1)
        year_end = date(year + 1, 1, 1)

        window_start = max(start, year_start)
        window_end = min(end, year_end)

        if window_start < window_end:
            windows.append(
                (
                    window_start.isoformat(),
                    window_end.isoformat(),
                )
            )

    return windows


def load_station_support():
    path = (
        project_root()
        / "outputs"
        / "raw"
        / "meteorology"
        / "station_support.csv"
    )

    table = pd.read_csv(path)

    required = [
        "station",
        "station_id",
        "station_longitude",
        "station_latitude",
        "chirps_support_m",
    ]

    missing = [
        column
        for column in required
        if column not in table.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing station support columns: {missing}"
        )

    if len(table) != 5:
        raise RuntimeError(
            f"Expected 5 stations, found {len(table)}"
        )

    return table


def station_feature(row):
    return ee.Feature(
        None,
        {
            "station": str(row["station"]),
            "station_id": str(row["station_id"]),
            "station_longitude": float(
                row["station_longitude"]
            ),
            "station_latitude": float(
                row["station_latitude"]
            ),
            "chirps_support_m": float(
                row["chirps_support_m"]
            ),
        },
    )


def build_window_table(
    station_support,
    window_start,
    window_end,
):
    combined = ee.FeatureCollection([])

    for _, row in station_support.iterrows():
        table = build_chirps_daily_table(
            station_feature(row),
            window_start,
            window_end,
        )

        combined = combined.merge(table)

    return combined


def merge_chunks(paths, output_path):
    table = pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )

    table["date"] = pd.to_datetime(
        table["date"]
    )

    table = (
        table
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )

    table["date"] = table["date"].dt.strftime(
        "%Y-%m-%d"
    )

    table.to_csv(
        output_path,
        index=False,
    )

    return table


def main():
    args = parse_arguments()

    experiment_start = date.fromisoformat(
        args.start_date
    )

    chirps_start = (
        experiment_start
        - timedelta(days=args.antecedent_days)
    ).isoformat()

    chirps_end = args.end_date_exclusive

    windows = build_windows(
        chirps_start,
        chirps_end,
    )

    expected_days = (
        date.fromisoformat(chirps_end)
        - date.fromisoformat(chirps_start)
    ).days

    expected_rows = expected_days * 5

    print("CHIRPS export plan")
    print("==================")
    print("Experiment start:", args.start_date)
    print(
        "Experiment end exclusive:",
        args.end_date_exclusive,
    )
    print("CHIRPS start:", chirps_start)
    print("CHIRPS end exclusive:", chirps_end)
    print("Antecedent days:", args.antecedent_days)
    print("Temporal downloads:", len(windows))
    print("Expected daily rows:", expected_rows)
    print("ERA5 download = false")
    print("training_performed = false")

    for start, end in windows:
        print(" ", start, "->", end)

    if not args.execute:
        print(
            "Dry plan only: Earth Engine was not initialized."
        )
        return

    support = load_station_support()

    ee.Initialize(project=args.project)

    root = (
        project_root()
        / "outputs"
        / "diagnostics"
        / args.period_label
        / "meteorology_experiment"
    )

    chunk_dir = root / "raw" / "_chirps_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths = []

    for window_start, window_end in windows:
        path = (
            chunk_dir
            / f"chirps_{window_start}_{window_end}.csv"
        )

        if path.exists():
            print("Reusing:", path.name)
            chunk_paths.append(path)
            continue

        print(
            "Exporting:",
            window_start,
            "->",
            window_end,
        )

        collection = build_window_table(
            support,
            window_start,
            window_end,
        )

        export_feature_collection(
            collection,
            str(path),
            CHIRPS_EXPORT_SELECTORS,
        )

        chunk_paths.append(path)

    output_path = (
        root
        / "raw"
        / "chirps_daily_20191202_20241231.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = merge_chunks(
        chunk_paths,
        output_path,
    )

    duplicate_count = int(
        table.duplicated(
            ["station_id", "date"]
        ).sum()
    )

    if len(table) != expected_rows:
        raise RuntimeError(
            f"CHIRPS row mismatch: "
            f"{len(table)} != {expected_rows}"
        )

    if duplicate_count:
        raise RuntimeError(
            f"Duplicate station-date rows: "
            f"{duplicate_count}"
        )

    precipitation = pd.to_numeric(
        table["precipitation_mm"],
        errors="coerce",
    )

    manifest = {
        "experiment_start": args.start_date,
        "experiment_end_exclusive": (
            args.end_date_exclusive
        ),
        "chirps_start": chirps_start,
        "chirps_end_exclusive": chirps_end,
        "antecedent_days": args.antecedent_days,
        "rows": len(table),
        "expected_rows": expected_rows,
        "duplicate_station_date_rows": (
            duplicate_count
        ),
        "missing_precipitation_rows": int(
            precipitation.isna().sum()
        ),
        "negative_precipitation_rows": int(
            precipitation.lt(0).sum()
        ),
        "sampling_definition": (
            "station point at native CHIRPS projection"
        ),
        "era5_download_performed": False,
        "training_performed": False,
    }

    metadata_dir = root / "metadata"
    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        metadata_dir
        / "chirps_extraction_manifest.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print()
    print("CHIRPS rows:", len(table))
    print("Expected rows:", expected_rows)
    print(
        "Duplicate station-date rows:",
        duplicate_count,
    )
    print(
        "Missing precipitation:",
        manifest["missing_precipitation_rows"],
    )
    print(
        "Negative precipitation:",
        manifest["negative_precipitation_rows"],
    )
    print("Output:", output_path)
    print("ERA5 download = false")
    print("training_performed = false")


if __name__ == "__main__":
    main()
