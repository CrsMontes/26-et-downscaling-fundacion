"""Materialize HLS Albedo and FVC for the approved 2020-2024 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPECTED_CONTEXT = ("2020-01-01", "2025-01-01", "2020_2024")
PREDICTORS = ("Albedo", "FVC")
EXPORT_SELECTORS = (
    "station", "station_id", "modis_pixel_id", "period_start", "period_end",
    "period_end_exclusive", "period_days", "footprint_area_m2",
    "hls_s30_products", "hls_l30_products", "hls_s30_unique_dates",
    "hls_l30_unique_dates", "hls_local_mgrs_tiles",
    "hls_selected_s30_area_m2", "hls_selected_l30_area_m2",
    "hls_selected_s30_pct", "hls_selected_l30_pct",
    "hls_reflectance_already_scaled_by_ee", "additional_hls_scaling_applied",
    "hls_Albedo_mean", "hls_FVC_mean",
)


def project_root():
    return Path(__file__).resolve().parents[2]


def output_root(label):
    if label != "2020_2024":
        raise ValueError("Only the approved 2020_2024 label is allowed")
    return project_root() / "outputs" / "diagnostics" / label / "hls_albedo_fvc"


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


def build_hls_albedo_fvc_table(
    modis_inputs, hls_collection, partition_start, partition_end
):
    """Build only HLS Albedo/FVC plus provenance for station-period rows."""
    import ee
    from et_downscaling.availability_diagnostic import (
        _base_properties, _period_context, _period_values,
        filter_dynamic_hls_to_geometry,
    )
    from et_downscaling.hls import add_hls_indices
    from et_downscaling.optical_source_experiment import (
        HLS_SOURCE_CODES, _selected_area, build_hls_medoid_with_provenance,
    )

    images, image_indexes, footprints, footprint_indexes = _period_context(
        modis_inputs, partition_start, partition_end
    )

    def process_image(image_index):
        period_start, period_end, period_days = _period_values(images.get(image_index))

        def process_footprint(footprint_index):
            footprint = ee.Feature(footprints.get(footprint_index))
            geometry = footprint.geometry()
            local = filter_dynamic_hls_to_geometry(hls_collection, geometry)
            period = local.filterDate(period_start, period_end)
            medoid = build_hls_medoid_with_provenance(period, geometry)
            predictors = add_hls_indices(medoid.select(
                ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
            )).select(list(PREDICTORS))
            source = medoid.select("hls_source_code")
            footprint_area = ee.Number(geometry.area(maxError=1))
            s30_area = _selected_area(source, HLS_SOURCE_CODES["S30"], geometry)
            l30_area = _selected_area(source, HLS_SOURCE_CODES["L30"], geometry)
            values = ee.Dictionary(predictors.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=geometry, scale=30,
                maxPixels=1_000_000, tileScale=4,
            ))
            properties = _base_properties(
                footprint, period_start, period_end, period_days
            )
            properties.update({
                "modis_pixel_id": footprint.get("modis_pixel_id"),
                "footprint_area_m2": footprint_area,
                "hls_s30_products": period.filter(ee.Filter.eq("sensor", "S30")).size(),
                "hls_l30_products": period.filter(ee.Filter.eq("sensor", "L30")).size(),
                "hls_s30_unique_dates": ee.List(period.filter(
                    ee.Filter.eq("sensor", "S30")
                ).aggregate_array("date_key")).distinct().size(),
                "hls_l30_unique_dates": ee.List(period.filter(
                    ee.Filter.eq("sensor", "L30")
                ).aggregate_array("date_key")).distinct().size(),
                "hls_local_mgrs_tiles": ee.List(local.get("local_mgrs_tiles")).join(";"),
                "hls_selected_s30_area_m2": s30_area,
                "hls_selected_l30_area_m2": l30_area,
                "hls_selected_s30_pct": s30_area.divide(footprint_area).multiply(100),
                "hls_selected_l30_pct": l30_area.divide(footprint_area).multiply(100),
                "hls_reflectance_already_scaled_by_ee": True,
                "additional_hls_scaling_applied": False,
                "hls_Albedo_mean": values.get("Albedo", -9999),
                "hls_FVC_mean": values.get("FVC", -9999),
            })
            return ee.Feature(None, properties)

        return footprint_indexes.map(process_footprint)

    return ee.FeatureCollection(image_indexes.map(process_image).flatten())


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


def adaptive_export(builder, start, end, destination, exporter, records):
    name = f"{start}_{end}_exclusive.csv".replace("-", "")
    path = destination / name
    if path.exists():
        records.append({"start": start, "end_exclusive": end, "status": "reused"})
        return [path]
    try:
        relative = path.relative_to(project_root() / "outputs")
        result = exporter(builder(start, end), relative.as_posix(), EXPORT_SELECTORS)
        records.append({"start": start, "end_exclusive": end, "status": "downloaded"})
        return [Path(result)]
    except Exception as error:
        lower, upper = pd.Timestamp(start), pd.Timestamp(end)
        if (upper - lower).days < 2:
            raise RuntimeError(f"Unsplittable partition {start}..{end}") from error
        midpoint = (lower + (upper - lower) / 2).normalize()
        children = ((lower.date().isoformat(), midpoint.date().isoformat()),
                    (midpoint.date().isoformat(), upper.date().isoformat()))
        records.append({"start": start, "end_exclusive": end,
                        "status": "split_after_failure", "error": str(error)})
        paths = []
        for child_start, child_end in children:
            paths.extend(adaptive_export(
                builder, child_start, child_end, destination, exporter, records
            ))
        return paths


def validate_local_table(table):
    keys = ["station_id", "period_start"]
    if len(table) != 1150 or table.duplicated(keys).any():
        raise RuntimeError("Expected exactly 1,150 unique station-period rows")
    for column in ("hls_Albedo_mean", "hls_FVC_mean"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
        table.loc[table[column] <= -9990, column] = pd.NA
    return {
        "rows": len(table),
        "unique_keys": len(table[keys].drop_duplicates()),
        "albedo_available": int(table["hls_Albedo_mean"].notna().sum()),
        "fvc_available": int(table["hls_FVC_mean"].notna().sum()),
    }


def main(argv=None):
    args = parse_arguments(argv)
    root = validate_context(args.start_date, args.end_date_exclusive, args.period_label)
    print("HLS Albedo/FVC plan: five annual local downloads; no training")
    if not args.execute:
        print("Dry plan only: Earth Engine was not initialized.")
        return 0
    if not args.project:
        raise ValueError("--project is required with --execute")
    import ee
    from et_downscaling.availability_diagnostic import (
        annual_partitions, get_dynamic_hls_collection, get_dynamic_modis_inputs,
    )
    from et_downscaling.export import export_feature_collection
    from et_downscaling.fvc import load_fvc_endmembers
    from et_downscaling.albedo import HLS_ALBEDO_COEFFICIENTS, HLS_ALBEDO_INTERCEPT

    ee.Initialize(project=args.project)
    modis = get_dynamic_modis_inputs(args.start_date, args.end_date_exclusive)
    hls = get_dynamic_hls_collection(
        modis["station_footprints"], args.start_date, args.end_date_exclusive
    )
    chunks = root / "raw" / "_chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    records = []
    paths = []
    for start, end in annual_partitions(args.start_date, args.end_date_exclusive):
        paths.extend(adaptive_export(
            lambda a, b: build_hls_albedo_fvc_table(modis, hls, a, b),
            start, end, chunks, export_feature_collection, records,
        ))
    output = root / "raw" / "hls_albedo_fvc.csv"
    rows = merge_csv(paths, output)
    if rows != 1150:
        raise RuntimeError(f"Expected 1,150 rows, found {rows}")
    integrity = validate_local_table(pd.read_csv(output, dtype={"station_id": str}))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root(),
                            capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=project_root(),
                            capture_output=True, text=True, check=True).stdout.splitlines()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
        "dirty": bool(status), "git_status_short": status, "project": args.project,
        "period": {"start_date": args.start_date,
                   "end_date_exclusive": args.end_date_exclusive},
        "composite": "combined_S30_L30_multiband_temporal_medoid",
        "medoid_score_bands": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "calculation_order": "indices_FVC_and_Albedo_after_combined_medoid",
        "reduction_scale_m": 30, "hls_nominal_support_m": 30,
        "albedo_coefficients": HLS_ALBEDO_COEFFICIENTS,
        "albedo_intercept": HLS_ALBEDO_INTERCEPT,
        "fvc_ndvi_endmembers": load_fvc_endmembers("HLS"),
        "requests": records, "integrity": integrity,
        "training_performed": False, "earth_engine_download_performed": True,
    }
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(integrity, indent=2))
    print("training_performed = false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
