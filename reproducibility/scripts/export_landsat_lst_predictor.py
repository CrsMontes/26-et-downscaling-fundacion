"""Export the approved 2020-2024 Landsat L8/L9 LST predictor locally."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path



START_DATE = "2020-01-01"
END_DATE_EXCLUSIVE = "2025-01-01"
EXPECTED_ROWS = 1150

SOURCE_DIRECTORY = Path(__file__).resolve().parents[2] / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from et_downscaling.candidate_paths import get_candidate_study_paths


def project_root():
    return Path(__file__).resolve().parents[2]


def output_root():
    return get_candidate_study_paths(project_root()).landsat_lst_root


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def method_manifest():
    from et_downscaling.landsat_lst_predictor import (
        EXPORT_SELECTORS,
        configuration_manifest as predictor_manifest,
    )
    from et_downscaling.thermal_availability import configuration_manifest

    return {
        **configuration_manifest(),
        **predictor_manifest(),
        "period": {"start": START_DATE, "end_exclusive": END_DATE_EXCLUSIVE},
        "view": "L8_L9_COMBINED",
        "historical_dn_ge_293_filter_used": False,
        "status": "APPROVED_PREDICTOR_LADDER_EXTRACTION",
        "selectors": list(EXPORT_SELECTORS),
    }


def _chunk_name(start, end):
    return f"{start}_{end}_exclusive.csv".replace("-", "")


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
                elif reader.fieldnames != header:
                    raise ValueError(f"CSV schema mismatch: {path}")
                for row in reader:
                    writer.writerow(row)
                    rows += 1
    return rows


def adaptive_export(
    *, builder, modis_inputs, collection, start, end, chunk_dir,
    selectors, exporter, records,
):
    from et_downscaling.thermal_availability import split_partition

    path = chunk_dir / _chunk_name(start, end)
    if path.exists():
        records.append({
            "start": start, "end_exclusive": end,
            "status": "reused", "path": str(path),
        })
        return [path]
    try:
        table = builder(modis_inputs, collection, start, end)
        workspace_root = get_candidate_study_paths(project_root()).workspace_root
        relative = path.relative_to(workspace_root)
        result = Path(exporter(table, str(relative), selectors))
        records.append({
            "start": start, "end_exclusive": end,
            "status": "downloaded", "path": str(result),
        })
        return [result]
    except Exception as error:
        try:
            children = split_partition(start, end)
        except ValueError:
            raise RuntimeError(f"Unsplittable partition {start}..{end}") from error
        records.append({
            "start": start, "end_exclusive": end,
            "status": "split_after_failure", "error_type": type(error).__name__,
            "error": str(error), "children": children,
        })
        paths = []
        for child_start, child_end in children:
            paths.extend(adaptive_export(
                builder=builder, modis_inputs=modis_inputs,
                collection=collection, start=child_start, end=child_end,
                chunk_dir=chunk_dir, selectors=selectors, exporter=exporter,
                records=records,
            ))
        return paths


def _git(command):
    return subprocess.run(
        ["git", *command], cwd=project_root(), check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main(argv=None):
    args = parse_arguments(argv)
    print(json.dumps(method_manifest(), indent=2))
    if not args.execute:
        print("Dry plan only: Earth Engine was not initialized.")
        return 0
    if not args.project:
        raise ValueError("--project is required with --execute")

    import ee
    from et_downscaling.availability_diagnostic import get_dynamic_modis_inputs
    from et_downscaling.export import export_feature_collection
    from et_downscaling.landsat_lst_predictor import (
        EXPORT_SELECTORS,
        build_landsat_lst_predictor,
    )
    from et_downscaling.thermal_availability import (
        annual_partitions,
        get_landsat_collection,
    )

    ee.Initialize(project=args.project)
    modis_inputs = get_dynamic_modis_inputs(START_DATE, END_DATE_EXCLUSIVE)
    collection = get_landsat_collection(
        modis_inputs["station_footprints"], START_DATE, END_DATE_EXCLUSIVE
    )
    root = output_root()
    chunks = root / "_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    records = []
    paths = []
    for start, end in annual_partitions(START_DATE, END_DATE_EXCLUSIVE):
        paths.extend(adaptive_export(
            builder=build_landsat_lst_predictor,
            modis_inputs=modis_inputs,
            collection=collection,
            start=start,
            end=end,
            chunk_dir=chunks,
            selectors=EXPORT_SELECTORS,
            exporter=export_feature_collection,
            records=records,
        ))

    output = root / "landsat_lst_station_period.csv"
    rows = merge_csv(paths, output)
    if rows != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} LST rows, found {rows}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "earth_engine_project": args.project,
        "branch": _git(["branch", "--show-current"]),
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]).splitlines(),
        "configuration": method_manifest(),
        "executed_partitions": records,
        "rows": rows,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "training_performed": False,
    }
    (root / "landsat_lst_export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Landsat LST predictor export complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
