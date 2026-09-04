"""Export the approved 2020-2024 source-availability diagnostic tables."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_PERIOD = ("2020-01-01", "2025-01-01", "2020_2024")
SOURCES = ("modis", "s2", "hls", "s1_period")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_root(period_label: str) -> Path:
    if period_label != "2020_2024":
        raise ValueError("This approved diagnostic requires period_label=2020_2024")
    return project_root() / "outputs" / "diagnostics" / period_label / "availability"


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date-exclusive", required=True)
    parser.add_argument("--period-label", required=True)
    parser.add_argument("--project", help="Google Cloud project; required with --execute")
    parser.add_argument(
        "--execute", action="store_true",
        help="Initialize Earth Engine and execute. Without this flag, print the plan only.",
    )
    return parser.parse_args(argv)


def validate_approved_context(start_date, end_date_exclusive, period_label):
    actual = (start_date, end_date_exclusive, period_label)
    if actual != EXPECTED_PERIOD:
        raise ValueError(f"Expected approved context {EXPECTED_PERIOD}, received {actual}")
    return output_root(period_label)


def repository_state(root: Path):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=root, text=True,
        capture_output=True, check=True,
    ).stdout.splitlines()
    return commit, status


def merge_csv_files(paths, destination):
    """Merge compatible resumable chunks without pandas."""
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


def _partition_name(start, end):
    return f"{start}_{end}_exclusive".replace("-", "")


def export_with_adaptive_partition(
    *, builder, modis_inputs, collection, start, end, chunk_dir,
    selectors, export_feature_collection, executed_partitions,
):
    """Try annual export, bisect only on an EE/download failure, and resume."""
    from et_downscaling.availability_diagnostic import split_partition

    path = chunk_dir / f"{_partition_name(start, end)}.csv"
    if path.exists():
        executed_partitions.append({
            "start": start, "end_exclusive": end, "path": str(path),
            "status": "reused",
        })
        return [path]
    try:
        if collection is None:
            table = builder(modis_inputs, start, end)
        else:
            table = builder(modis_inputs, collection, start, end)
        relative = path.relative_to(project_root() / "outputs")
        exported = export_feature_collection(table, str(relative), selectors)
        executed_partitions.append({
            "start": start, "end_exclusive": end, "path": str(exported),
            "status": "downloaded",
        })
        return [Path(exported)]
    except Exception as error:
        from et_downscaling.availability_diagnostic import split_partition
        try:
            children = split_partition(start, end)
        except ValueError:
            raise RuntimeError(f"Unsplittable failed partition {start}..{end}") from error
        executed_partitions.append({
            "start": start, "end_exclusive": end, "status": "split_after_failure",
            "error_type": type(error).__name__, "children": children,
        })
        result = []
        for child_start, child_end in children:
            result.extend(export_with_adaptive_partition(
                builder=builder, modis_inputs=modis_inputs, collection=collection,
                start=child_start, end=child_end, chunk_dir=chunk_dir,
                selectors=selectors, export_feature_collection=export_feature_collection,
                executed_partitions=executed_partitions,
            ))
        return result


def print_plan():
    print("Planned Earth Engine requests:")
    print("  MODIS: 5 annual table downloads")
    print("  Sentinel-2: 5 annual table downloads")
    print("  HLS S30/L30/combined: 5 annual table downloads")
    print("  Sentinel-1 geometry inventory: 1 table download")
    print("  Sentinel-1 station-period detail: 5 annual table downloads")
    print("  Expected total: 21; adaptive bisection only after a failed request")


def main(argv=None):
    args = parse_arguments(argv)
    root = validate_approved_context(
        args.start_date, args.end_date_exclusive, args.period_label
    )
    print_plan()
    if not args.execute:
        print("Dry plan only: Earth Engine was not initialized.")
        return 0
    if not args.project:
        raise ValueError("--project is required with --execute")

    import ee
    from et_downscaling.availability_diagnostic import (
        HLS_SELECTORS, MODIS_SELECTORS, S1_INVENTORY_SELECTORS,
        S1_PERIOD_SELECTORS, S2_SELECTORS, annual_partitions,
        build_hls_availability, build_modis_availability,
        build_s1_geometry_inventory, build_s1_period_availability,
        build_s2_availability, get_dynamic_hls_collection,
        get_dynamic_modis_inputs, get_dynamic_s1_collection,
        get_dynamic_s2_collection, scientific_configuration,
    )
    from et_downscaling.export import export_feature_collection

    ee.Initialize(project=args.project)
    raw_dir = root / "raw"
    chunk_root = raw_dir / "_chunks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty_status = repository_state(project_root())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit, "dirty": bool(dirty_status),
        "git_status_short": dirty_status,
        "period": {"start_date": args.start_date,
                   "end_date_exclusive": args.end_date_exclusive,
                   "period_label": args.period_label},
        "configuration": scientific_configuration(),
        "planned_requests": 21, "executed_partitions": {},
        "training_performed": False,
    }
    modis_inputs = get_dynamic_modis_inputs(args.start_date, args.end_date_exclusive)
    footprints = modis_inputs["station_footprints"]
    collections = {
        "modis": None,
        "s2": get_dynamic_s2_collection(footprints, args.start_date, args.end_date_exclusive),
        "hls": get_dynamic_hls_collection(footprints, args.start_date, args.end_date_exclusive),
        "s1_period": get_dynamic_s1_collection(footprints, args.start_date, args.end_date_exclusive),
    }
    builders = {
        "modis": (build_modis_availability, MODIS_SELECTORS),
        "s2": (build_s2_availability, S2_SELECTORS),
        "hls": (build_hls_availability, HLS_SELECTORS),
        "s1_period": (build_s1_period_availability, S1_PERIOD_SELECTORS),
    }
    for source in SOURCES:
        builder, selectors = builders[source]
        chunk_dir = chunk_root / source
        chunk_dir.mkdir(parents=True, exist_ok=True)
        records = []
        paths = []
        for start, end in annual_partitions(args.start_date, args.end_date_exclusive):
            paths.extend(export_with_adaptive_partition(
                builder=builder, modis_inputs=modis_inputs,
                collection=collections[source], start=start, end=end,
                chunk_dir=chunk_dir, selectors=selectors,
                export_feature_collection=export_feature_collection,
                executed_partitions=records,
            ))
        merge_csv_files(paths, raw_dir / f"{source}_station_period.csv")
        manifest["executed_partitions"][source] = records

    inventory_path = raw_dir / "sentinel1_geometry_inventory.csv"
    if not inventory_path.exists():
        inventory = build_s1_geometry_inventory(collections["s1_period"], footprints)
        relative = inventory_path.relative_to(project_root() / "outputs")
        export_feature_collection(inventory, str(relative), S1_INVENTORY_SELECTORS)
        inventory_status = "downloaded"
    else:
        inventory_status = "reused"
    manifest["executed_partitions"]["s1_inventory"] = [{
        "start": args.start_date, "end_exclusive": args.end_date_exclusive,
        "path": str(inventory_path), "status": inventory_status,
    }]
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Raw availability export complete. No training was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
