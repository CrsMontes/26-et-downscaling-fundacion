"""Export paired optical predictors and indispensable ERA5 inputs for Phase 3A."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from et_downscaling.candidate_paths import get_candidate_study_paths


def project_root():
    return Path(__file__).resolve().parents[2]


def output_root(label):
    if label != "2020_2024":
        raise ValueError("Phase 3A outputs are confined to 2020_2024")
    return get_candidate_study_paths(project_root()).optical_root


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date-exclusive", required=True)
    parser.add_argument("--period-label", required=True)
    parser.add_argument("--project")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def merge_csv(paths, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = None
    rows = 0
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = None
        for path in sorted(paths):
            with Path(path).open("r", newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames is None:
                    continue
                if header is None:
                    header = reader.fieldnames
                    writer = csv.DictWriter(output, fieldnames=header)
                    writer.writeheader()
                elif header != reader.fieldnames:
                    raise ValueError(f"CSV schema mismatch: {path}")
                for row in reader:
                    writer.writerow(row)
                    rows += 1
    return rows


def adaptive_export(builder, start, end, destination, exporter, selectors, records):
    name = re.sub(r"[^0-9A-Za-z_]+", "", f"{start}_{end}_exclusive") + ".csv"
    path = destination / name
    if path.exists():
        records.append({"start": start, "end_exclusive": end, "status": "reused"})
        return [path]
    try:
        table = builder(start, end)
        relative = path.relative_to(get_candidate_study_paths(project_root()).workspace_root)
        result = exporter(table, relative.as_posix(), selectors)
        records.append({"start": start, "end_exclusive": end, "status": "downloaded"})
        return [result]
    except Exception as error:
        try:
            lower = pd.Timestamp(start)
            upper = pd.Timestamp(end)
            if upper - lower <= pd.Timedelta(hours=1):
                raise ValueError("Partition cannot be split further")
            midpoint = lower + (upper - lower) / 2
            if "T" not in start and "T" not in end:
                midpoint = midpoint.normalize()
            def format_value(value):
                if "T" in start or "T" in end:
                    return value.isoformat().replace("+00:00", "Z")
                return value.date().isoformat()
            children = (
                (format_value(lower), format_value(midpoint)),
                (format_value(midpoint), format_value(upper)),
            )
        except (ValueError, TypeError):
            raise RuntimeError(f"Unsplittable partition {start}..{end}") from error
        records.append({
            "start": start, "end_exclusive": end, "status": "split_after_failure",
            "error_type": type(error).__name__, "error": str(error),
        })
        results = []
        for child_start, child_end in children:
            results.extend(adaptive_export(
                builder, child_start, child_end, destination, exporter, selectors, records
            ))
        return results


def annual_era5_windows():
    result = []
    for year in range(2020, 2025):
        start = f"{year}-01-01T00:00:00Z"
        end = f"{year + 1}-01-01T00:00:00Z"
        if year == 2024:
            end = "2025-01-01T05:00:00Z"
        result.append((start, end))
    return result


def support_features_from_local_csv(ee):
    table = pd.read_csv(
        project_root() / "outputs" / "raw" / "meteorology" / "station_support.csv",
        dtype={"station_id": str},
    )
    if len(table) != 5 or table.station_id.nunique() != 5:
        raise RuntimeError("Expected exactly five unique station-support rows")
    features = []
    for row in table.to_dict("records"):
        properties = {
            key: value for key, value in row.items() if not pd.isna(value)
        }
        features.append(ee.Feature(None, properties))
    return features


def validate_hls_metadata(collections):
    bands = {
        "S30": ("B2", "B3", "B4", "B8A", "B11", "B12"),
        "L30": ("B2", "B3", "B4", "B5", "B6", "B7"),
    }
    result = {}
    for sensor, collection in collections.items():
        sensor_result = {}
        for band in bands[sensor]:
            key = f"{band}_scale"
            values = (
                collection.filter(__import__("ee").Filter.notNull([key]))
                .aggregate_array(key).distinct().sort().getInfo()
            )
            if values != [0.0001]:
                raise RuntimeError(f"Unexpected EE HLS metadata {sensor} {key}: {values}")
            sensor_result[key] = values
        sensor_result["additional_scaling_applied"] = False
        result[sensor] = sensor_result
    return result


def main(argv=None):
    args = parse_arguments(argv)
    from et_downscaling.optical_source_experiment import (
        END_DATE_EXCLUSIVE, EXPORT_SELECTORS, PERIOD_LABEL, START_DATE,
        expected_rows, experiment_configuration, validate_context,
    )
    validate_context(args.start_date, args.end_date_exclusive, args.period_label)
    print("Phase 3A plan: 5 paired-optical annual downloads; shared ERA5 reused")
    print("training_performed = false")
    if not args.execute:
        print("Dry plan only: Earth Engine was not initialized.")
        return 0
    if not args.project:
        raise ValueError("--project is required with --execute")

    import ee
    from et_downscaling.availability_diagnostic import (
        annual_partitions, get_dynamic_hls_collection, get_dynamic_modis_inputs,
        get_dynamic_s2_collection,
    )
    from et_downscaling.export import export_feature_collection
    from et_downscaling.optical_source_experiment import build_paired_optical_table

    ee.Initialize(project=args.project)
    root = output_root(args.period_label)
    raw = root / "raw"
    modis = get_dynamic_modis_inputs(START_DATE, END_DATE_EXCLUSIVE)
    s2 = get_dynamic_s2_collection(modis["station_footprints"], START_DATE, END_DATE_EXCLUSIVE)
    hls = get_dynamic_hls_collection(modis["station_footprints"], START_DATE, END_DATE_EXCLUSIVE)
    hls_sources = {
        "S30": hls.filter(ee.Filter.eq("sensor", "S30")),
        "L30": hls.filter(ee.Filter.eq("sensor", "L30")),
    }
    scale_metadata = validate_hls_metadata(hls_sources)
    records = {"optical": []}

    optical_chunks = raw / "_chunks" / "optical"
    optical_chunks.mkdir(parents=True, exist_ok=True)
    optical_paths = []
    for start, end in annual_partitions(START_DATE, END_DATE_EXCLUSIVE):
        optical_paths.extend(adaptive_export(
            lambda a, b: build_paired_optical_table(modis, s2, hls, a, b),
            start, end, optical_chunks, export_feature_collection,
            EXPORT_SELECTORS, records["optical"],
        ))
    optical_output = raw / "paired_optical_common.csv"
    optical_rows = merge_csv(optical_paths, optical_output)
    if optical_rows != expected_rows():
        raise RuntimeError(f"Expected {expected_rows()} optical rows, found {optical_rows}")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root(), capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=project_root(), capture_output=True,
        text=True, check=True,
    ).stdout.splitlines()
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
        "dirty": bool(status), "git_status_short": status,
        "project": args.project, "configuration": experiment_configuration(),
        "hls_earth_engine_metadata": scale_metadata,
        "requests": records, "optical_rows": optical_rows,
        "shared_meteorology_source": "raw/meteorology",
        "training_performed": False,
        "aoa_di_performed": False,
    }
    (metadata / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Optical rows: {optical_rows}")
    print("ERA5 download: false; shared raw meteorology is reused")
    print("training_performed = false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
