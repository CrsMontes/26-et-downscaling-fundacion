"""Export the approved Landsat thermal availability diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_CONTEXT = ("2020-01-01", "2025-01-01", "2020_2024")


def project_root():
    return Path(__file__).resolve().parents[1]


def output_root(period_label):
    if period_label != "2020_2024":
        raise ValueError("Only the approved 2020_2024 period label is allowed")
    return project_root() / "outputs" / "diagnostics" / period_label / "thermal_availability"


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date-exclusive", required=True)
    parser.add_argument("--period-label", required=True)
    parser.add_argument("--project")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def validate_context(start, end, label):
    if (start, end, label) != EXPECTED_CONTEXT:
        raise ValueError(f"Expected {EXPECTED_CONTEXT}, received {(start, end, label)}")
    return output_root(label)


def expected_rows(number_periods=230, number_stations=5):
    return number_periods * number_stations


def merge_csv(paths, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = None
    rows = 0
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = None
        for path in sorted(paths):
            with path.open("r", newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames is None:
                    continue
                if header is None:
                    header = reader.fieldnames
                    writer = csv.DictWriter(output, fieldnames=header)
                    writer.writeheader()
                elif reader.fieldnames != header:
                    raise ValueError(f"CSV schema mismatch: {path}")
                for row in reader:
                    writer.writerow(row)
                    rows += 1
    return rows


def _name(start, end):
    return f"{start}_{end}_exclusive.csv".replace("-", "")


def adaptive_export(*, builder, modis_inputs, collection, start, end,
                    chunk_dir, selectors, exporter, records):
    from et_downscaling.thermal_availability import split_partition
    path = chunk_dir / _name(start, end)
    if path.exists():
        records.append({"start": start, "end_exclusive": end,
                        "status": "reused", "path": str(path)})
        return [path]
    try:
        table = builder(modis_inputs, collection, start, end)
        relative = path.relative_to(project_root() / "outputs")
        result = Path(exporter(table, str(relative), selectors))
        records.append({"start": start, "end_exclusive": end,
                        "status": "downloaded", "path": str(result)})
        return [result]
    except Exception as error:
        try:
            children = split_partition(start, end)
        except ValueError:
            raise RuntimeError(f"Unsplittable partition {start}..{end}") from error
        records.append({"start": start, "end_exclusive": end,
                        "status": "split_after_failure",
                        "error_type": type(error).__name__, "children": children})
        paths = []
        for child_start, child_end in children:
            paths.extend(adaptive_export(
                builder=builder, modis_inputs=modis_inputs, collection=collection,
                start=child_start, end=child_end, chunk_dir=chunk_dir,
                selectors=selectors, exporter=exporter, records=records,
            ))
        return paths


def print_plan():
    print("Approved thermal diagnostic context:")
    print("  start_date = 2020-01-01")
    print("  end_date_exclusive = 2025-01-01")
    print("  period_label = 2020_2024")
    print("  views = L8_ONLY, L8_L9_COMBINED")
    print("  expected station-periods per view = 1150")
    print("  expected initial Earth Engine requests = 5 annual downloads")
    print("  training_performed = false")


def main(argv=None):
    args = parse_arguments(argv)
    root = validate_context(args.start_date, args.end_date_exclusive, args.period_label)
    print_plan()
    if not args.execute:
        print("Dry plan only: Earth Engine was not initialized.")
        return 0
    if not args.project:
        raise ValueError("--project is required with --execute")

    import ee
    from et_downscaling.export import export_feature_collection
    from et_downscaling.thermal_availability import (
        EXPORT_SELECTORS, annual_partitions, build_thermal_availability,
        configuration_manifest, get_dynamic_modis_inputs,
        get_landsat_collection,
    )

    ee.Initialize(project=args.project)
    modis_inputs = get_dynamic_modis_inputs(args.start_date, args.end_date_exclusive)
    collection = get_landsat_collection(
        modis_inputs["station_footprints"], args.start_date, args.end_date_exclusive
    )
    raw = root / "raw"
    chunks = raw / "_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    records = []
    paths = []
    for start, end in annual_partitions(args.start_date, args.end_date_exclusive):
        paths.extend(adaptive_export(
            builder=build_thermal_availability, modis_inputs=modis_inputs,
            collection=collection, start=start, end=end, chunk_dir=chunks,
            selectors=EXPORT_SELECTORS, exporter=export_feature_collection,
            records=records,
        ))
    output = raw / "landsat_lst_station_period.csv"
    rows = merge_csv(paths, output)
    if rows != expected_rows():
        raise RuntimeError(f"Expected 1150 thermal rows, found {rows}")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root(), text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=project_root(), text=True,
        capture_output=True, check=True,
    ).stdout.splitlines()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit, "dirty": bool(status), "git_status_short": status,
        "period": {"start_date": args.start_date,
                   "end_date_exclusive": args.end_date_exclusive,
                   "period_label": args.period_label},
        "configuration": configuration_manifest(),
        "planned_requests": 5, "executed_partitions": records,
        "rows": rows, "rows_per_view": rows,
        "training_performed": False,
    }
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Thermal availability export complete. No training was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
