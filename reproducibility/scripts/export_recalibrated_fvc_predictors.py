"""Freeze experimental endmembers and export recalibrated FVC predictors.

Earth Engine is initialized only with ``--execute``. Each source/year is a
validated resumable chunk, and each station-period builds its medoid once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

import analyze_fvc_candidate_stability as candidates
import export_fvc_calibration_candidates as candidate_export
import preflight_fvc_recalibration as preflight


SCHEMA_VERSION = "recalibrated-fvc-predictors-v1"
CALIBRATION_SCHEMA_VERSION = "experimental-fvc-calibration-v1"
YEARS = range(2020, 2025)
SOURCES = ("S2", "HLS")
SPATIAL = {1: ("ST05",), 2: ("ST02", "ST03"), 3: ("ST04",), 4: ("ST01",)}
HISTORICAL = {
    "S2": (0.30906052790151156, 0.9240448371180946),
    "HLS": (0.411908487478892, 0.9082510914569858),
}
EXPECTED_GLOBAL = {
    "S2": (0.30134083330631256, 0.9234938586644688, 831),
    "HLS": (0.3843640714883804, 0.9088275356582599, 635),
}
CALIBRATION_TOLERANCE = 1e-14
MAX_TRANSIENT_RETRIES = 2


def project_root():
    return Path(__file__).resolve().parents[2]


def output_root():
    return project_root() / "outputs/diagnostics/2020_2024/recalibrated_fvc_predictors"


def calibration_path():
    return output_root() / "metadata/experimental_fvc_calibration_manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _calibrate(table):
    return candidates.calculate(table, 80)[0]


def freeze_calibrations():
    manifest = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "date_start": "2020-01-01", "date_end_exclusive": "2025-01-01",
        "calibration_method": "two_stage_global_percentile",
        "coverage_threshold_pct": 80, "water_rule": "NDWI <= 0",
        "fold_assignment": {"spatial": {str(k): list(v) for k, v in SPATIAL.items()},
                            "temporal": list(YEARS)},
        "sources": {},
    }
    master = pd.read_csv(
        project_root() / "outputs/diagnostics/2020_2024/optical_source_experiment/population/paired_master.csv",
        dtype={"station_id": str},
    )[["station_id", "period_start", "spatial_block"]]
    for source in SOURCES:
        table, chunk_manifests = candidates.load_source_chunks(source, YEARS)
        table["year"] = pd.to_datetime(table.period_start).dt.year
        table = table.merge(master, on=["station_id", "period_start"], validate="one_to_one")
        global_result = _calibrate(table)
        expected_low, expected_high, expected_n = EXPECTED_GLOBAL[source]
        if global_result["n_observations"] != expected_n or abs(global_result["ndvi_low_endmember"] - expected_low) > CALIBRATION_TOLERANCE or abs(global_result["ndvi_high_endmember"] - expected_high) > CALIBRATION_TOLERANCE:
            raise RuntimeError(f"{source} global GE80 calibration does not match the approved gate")
        source_record = {
            "medoid_definition": "six_common_band_medoid" if source == "S2" else "combined_S30_L30_six_common_band_medoid",
            "scale_m": 20 if source == "S2" else 30,
            "grid": "Sentinel-2 B8A 20 m" if source == "S2" else "HLS 30 m source grid",
            "historical_fixed": {"low": HISTORICAL[source][0], "high": HISTORICAL[source][1]},
            "global_2020_2024": {"low": global_result["ndvi_low_endmember"],
                                  "high": global_result["ndvi_high_endmember"],
                                  "n": global_result["n_observations"]},
            "spatial_training_only": {}, "temporal_training_only": {},
            "candidate_chunks": chunk_manifests,
        }
        for fold, excluded in SPATIAL.items():
            training = table.loc[~table.station_id.isin(excluded)]
            if training.station_id.isin(excluded).any():
                raise RuntimeError("Spatial test observations entered calibration")
            result = _calibrate(training)
            source_record["spatial_training_only"][str(fold)] = {
                "excluded_station_ids": list(excluded), "low": result["ndvi_low_endmember"],
                "high": result["ndvi_high_endmember"], "n": result["n_observations"]}
        for year in YEARS:
            training = table.loc[table.year.ne(year)]
            if training.year.eq(year).any():
                raise RuntimeError("Temporal test observations entered calibration")
            result = _calibrate(training)
            source_record["temporal_training_only"][str(year)] = {
                "excluded_year": year, "low": result["ndvi_low_endmember"],
                "high": result["ndvi_high_endmember"], "n": result["n_observations"]}
        source_record["candidate_csv_sha256"] = {
            str(year): sha256_file(candidate_export.chunk_paths(
                source, f"{year}-01-01", f"{year + 1}-01-01"
            )[0]) for year in YEARS
        }
        manifest["sources"][source] = source_record
    manifest["manifest_sha256"] = canonical_hash(manifest)
    path = calibration_path(); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def calibration_variants(source, manifest):
    record = manifest["sources"][source]
    values = {"historical": record["historical_fixed"],
              "global_2020_2024": record["global_2020_2024"]}
    values.update({f"spatial_train_excl_fold{fold}": item
                   for fold, item in record["spatial_training_only"].items()})
    values.update({f"temporal_train_excl_{year}": item
                   for year, item in record["temporal_training_only"].items()})
    return values


def output_columns(source, manifest):
    columns = ["station_id", "period_start", "source", "valid_pixel_count",
               "optical_coverage_pct", "optical_products", "optical_unique_dates"]
    if source == "HLS":
        columns = columns[:-2] + ["hls_s30_products", "hls_l30_products",
            "hls_s30_unique_dates", "hls_l30_unique_dates", "local_mgrs_tiles"]
    for name in calibration_variants(source, manifest):
        columns.extend([f"fvc_{name}_mean", f"fvc_{name}_clipped_low_fraction",
                        f"fvc_{name}_clipped_high_fraction"])
    if source == "HLS":
        columns.append("hls_Albedo_mean")
    return columns


def _safe_ratio(numerator, denominator, name):
    return numerator.divide(denominator).updateMask(denominator.abs().gt(1e-6)).rename(name).toFloat()


def build_export_table(source, modis_inputs, collection, start, end_exclusive, manifest):
    import ee
    from et_downscaling.albedo import calculate_hls_albedo
    from et_downscaling.availability_diagnostic import _period_context, _period_values, filter_dynamic_hls_to_geometry
    from et_downscaling.hls import build_hls_medoid
    from et_downscaling.sentinel2 import build_s2_medoid

    images, image_indexes, footprints, footprint_indexes = _period_context(modis_inputs, start, end_exclusive)
    variants = calibration_variants(source, manifest)

    def process_image(image_index):
        period_start, period_end, _ = _period_values(images.get(image_index))
        def process_footprint(footprint_index):
            footprint = ee.Feature(footprints.get(footprint_index)); geometry = footprint.geometry()
            if source == "S2":
                period = ee.ImageCollection(collection).filterDate(period_start, period_end).filterBounds(geometry)
                medoid = build_s2_medoid(period, geometry)
                provenance = {"optical_products": period.size(), "optical_unique_dates": ee.List(period.aggregate_array("date_key")).distinct().size()}
                scale = 20
            else:
                local = filter_dynamic_hls_to_geometry(collection, geometry)
                period = local.filterDate(period_start, period_end)
                medoid = build_hls_medoid(period, geometry)
                s30, l30 = period.filter(ee.Filter.eq("sensor", "S30")), period.filter(ee.Filter.eq("sensor", "L30"))
                provenance = {"hls_s30_products": s30.size(), "hls_l30_products": l30.size(),
                    "hls_s30_unique_dates": ee.List(s30.aggregate_array("date_key")).distinct().size(),
                    "hls_l30_unique_dates": ee.List(l30.aggregate_array("date_key")).distinct().size(),
                    "local_mgrs_tiles": ee.List(local.get("local_mgrs_tiles")).join(";")}
                scale = 30
            nir, red = medoid.select("NIR"), medoid.select("Red")
            ndvi = _safe_ratio(nir.subtract(red), nir.add(red), "NDVI")
            valid = medoid.select(["Green", "Red", "NIR"]).mask().reduce(ee.Reducer.min()).rename("valid").uint8()
            stack = ndvi
            for name, endpoints in variants.items():
                raw = ndvi.subtract(endpoints["low"]).divide(endpoints["high"] - endpoints["low"])
                stack = stack.addBands(raw.clamp(0, 1).rename(f"fvc_{name}"))
                stack = stack.addBands(raw.lt(0).rename(f"fvc_{name}_clipped_low"))
                stack = stack.addBands(raw.gt(1).rename(f"fvc_{name}_clipped_high"))
            if source == "HLS":
                stack = stack.addBands(calculate_hls_albedo(medoid).rename("hls_Albedo"))
            raw_stats = ee.Dictionary(stack.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
                geometry=geometry, scale=scale, maxPixels=10_000_000, tileScale=4))
            coverage_raw = valid.unmask(0).reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry,
                scale=scale, maxPixels=10_000_000, tileScale=4).get("valid")
            properties = {"station_id": footprint.get("station_id"),
                "period_start": period_start.format("yyyy-MM-dd"), "source": source,
                "valid_pixel_count": raw_stats.get("NDVI_count", 0),
                "optical_coverage_pct": ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(coverage_raw, None), 0, ee.Number(coverage_raw).multiply(100)))}
            properties.update(provenance)
            for name in variants:
                properties[f"fvc_{name}_mean"] = raw_stats.get(f"fvc_{name}_mean", -9999)
                properties[f"fvc_{name}_clipped_low_fraction"] = raw_stats.get(f"fvc_{name}_clipped_low_mean", -9999)
                properties[f"fvc_{name}_clipped_high_fraction"] = raw_stats.get(f"fvc_{name}_clipped_high_mean", -9999)
            if source == "HLS": properties["hls_Albedo_mean"] = raw_stats.get("hls_Albedo_mean", -9999)
            return ee.Feature(None, properties)
        return footprint_indexes.map(process_footprint)
    return ee.FeatureCollection(image_indexes.map(process_image).flatten())


def chunk_paths(source, year):
    directory = output_root() / "chunks" / source.lower()
    return directory / f"{year}0101_{year + 1}0101_exclusive.csv", directory / f"{year}0101_{year + 1}0101_exclusive.manifest.json"


def expected_chunk_manifest(source, year, calibration, keys):
    return {"schema_version": SCHEMA_VERSION, "source": source, "start_date": f"{year}-01-01",
        "end_date_exclusive": f"{year + 1}-01-01", "expected_rows": len(keys),
        "expected_keys_sha256": preflight.key_digest(keys),
        "medoid_version": calibration["sources"][source]["medoid_definition"],
        "calibration_manifest_sha256": calibration["manifest_sha256"]}


def validate_table(table, source, start_date, end_date_exclusive, selectors, expected_rows):
    if list(table.columns) != list(selectors):
        raise ValueError("Unexpected recalibrated FVC schema")
    if len(table) != expected_rows or table.duplicated(["station_id", "period_start"]).any():
        raise ValueError("Output does not contain the expected unique keys")
    dates = pd.to_datetime(table.period_start)
    if not dates.ge(start_date).all() or not dates.lt(end_date_exclusive).all() or set(table.source) != {source}:
        raise ValueError("Output has invalid source or dates")
    numeric = [column for column in table if column.startswith("fvc_")]
    values = table[numeric].apply(pd.to_numeric, errors="coerce")
    valid = values.gt(-9990)
    means = [column for column in numeric if column.endswith("_mean")]
    fractions = [column for column in numeric if column.endswith("_fraction")]
    if not values[means].where(valid[means]).stack().dropna().between(0, 1).all(): raise ValueError("FVC outside 0..1")
    if not values[fractions].where(valid[fractions]).stack().dropna().between(0, 1).all(): raise ValueError("Clipping fraction outside 0..1")
    valid_count = pd.to_numeric(table.valid_pixel_count, errors="coerce")
    if valid_count.lt(0).any(): raise ValueError("Negative valid-pixel count")
    if values[means].isna().any(axis=1).ne(valid_count.eq(0)).any():
        raise ValueError("FVC null status disagrees with valid NDVI support")
    availability = valid[means].nunique(axis=1)
    if not availability.eq(1).all(): raise ValueError("FVC variants do not share pixel availability")
    if source == "HLS":
        albedo = pd.to_numeric(table.hls_Albedo_mean, errors="coerce").mask(lambda x: x <= -9990)
        if not albedo.dropna().between(-0.2, 1.2).all(): raise ValueError("Implausible HLS Albedo")
        if table.local_mgrs_tiles.fillna("").str.strip().eq("").any(): raise ValueError("Missing HLS MGRS provenance")
    return table


def validate_existing(csv_path, manifest_path, expected, keys, selectors):
    if not csv_path.is_file() or not manifest_path.is_file(): return False
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(saved.get(k) != v for k, v in expected.items()) or saved.get("status") != "completed": return False
    if saved.get("csv_sha256") != sha256_file(csv_path): return False
    table = pd.read_csv(csv_path, dtype={"station_id": str})
    try: validate_table(table, expected["source"], expected["start_date"],
                        expected["end_date_exclusive"], selectors, expected["expected_rows"])
    except ValueError: return False
    actual = preflight.key_digest(table[["station_id", "period_start"]])
    return actual == preflight.key_digest(keys)


def execute_chunk(source, year, project, calibration, modis, collection):
    keys = candidate_export.expected_keys(f"{year}-01-01", f"{year + 1}-01-01")
    expected = expected_chunk_manifest(source, year, calibration, keys)
    csv_path, manifest_path = chunk_paths(source, year); selectors = output_columns(source, calibration)
    if validate_existing(csv_path, manifest_path, expected, keys, selectors):
        print(f"Reusing validated chunk: {csv_path}"); return
    if csv_path.is_file() and not manifest_path.is_file():
        recovered = pd.read_csv(csv_path, dtype={"station_id": str})
        validate_table(recovered, source, expected["start_date"], expected["end_date_exclusive"], selectors, len(keys))
        if preflight.key_digest(recovered[["station_id", "period_start"]]) != preflight.key_digest(keys):
            raise ValueError("Existing manifestless export has an invalid key set")
        saved = dict(expected, actual_rows=len(recovered), csv_sha256=sha256_file(csv_path),
                     status="completed", completion_mode="recovered_after_local_validation")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        print(f"Recovered validated manifestless chunk: {csv_path}")
        return
    from et_downscaling.export import export_feature_collection
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            collection_table = build_export_table(source, modis, collection, f"{year}-01-01", f"{year + 1}-01-01", calibration)
            exported = Path(export_feature_collection(collection_table,
                csv_path.relative_to(project_root() / "outputs").as_posix(), selectors))
            table = pd.read_csv(exported, dtype={"station_id": str})
            validate_table(table, source, expected["start_date"], expected["end_date_exclusive"], selectors, len(keys))
            actual_keys = preflight.key_digest(table[["station_id", "period_start"]])
            if actual_keys != preflight.key_digest(keys): raise ValueError("Export key set differs")
            saved = dict(expected, actual_rows=len(table), csv_sha256=sha256_file(exported), status="completed")
            manifest_path.write_text(json.dumps(saved, indent=2), encoding="utf-8"); return
        except Exception as error:
            last_error = error
            if isinstance(error, (ValueError, RuntimeError)):
                raise
            if attempt < MAX_TRANSIENT_RETRIES: time.sleep(2 ** attempt)
    raise RuntimeError(f"Chunk failed after preregistered retries: {source} {year}") from last_error


def execute_subchunk(source, start_date, end_date_exclusive, calibration, modis, collection):
    """Export a deterministic child interval after an annual request stalls/fails."""
    from et_downscaling.export import export_feature_collection
    keys = candidate_export.expected_keys(start_date, end_date_exclusive)
    year = pd.Timestamp(start_date).year
    expected = expected_chunk_manifest(source, year, calibration, keys)
    expected["start_date"] = start_date
    expected["end_date_exclusive"] = end_date_exclusive
    expected["expected_rows"] = len(keys)
    stem = f"{start_date}_{end_date_exclusive}_exclusive".replace("-", "")
    directory = output_root() / "chunks" / source.lower() / "split_2024"
    csv_path, manifest_path = directory / f"{stem}.csv", directory / f"{stem}.manifest.json"
    selectors = output_columns(source, calibration)
    if validate_existing(csv_path, manifest_path, expected, keys, selectors):
        print(f"Reusing validated subchunk: {csv_path}"); return csv_path
    directory.mkdir(parents=True, exist_ok=True)
    table = build_export_table(source, modis, collection, start_date, end_date_exclusive, calibration)
    exported = Path(export_feature_collection(
        table, csv_path.relative_to(project_root() / "outputs").as_posix(), selectors
    ))
    local = pd.read_csv(exported, dtype={"station_id": str})
    validate_table(local, source, start_date, end_date_exclusive, selectors, len(keys))
    if preflight.key_digest(local[["station_id", "period_start"]]) != preflight.key_digest(keys):
        raise ValueError("Subchunk key set differs")
    saved = dict(expected, actual_rows=len(local), csv_sha256=sha256_file(exported),
                 status="completed", completion_mode="deterministic_split_after_stalled_annual_request")
    manifest_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return exported


def merge_2024_subchunks(source, calibration):
    paths = [
        output_root() / "chunks" / source.lower() / "split_2024" / "20240101_20240701_exclusive.csv",
        output_root() / "chunks" / source.lower() / "split_2024" / "20240701_20250101_exclusive.csv",
    ]
    table = pd.concat([pd.read_csv(path, dtype={"station_id": str}) for path in paths], ignore_index=True)
    selectors = output_columns(source, calibration)
    validate_table(table, source, "2024-01-01", "2025-01-01", selectors, 230)
    keys = candidate_export.expected_keys("2024-01-01", "2025-01-01")
    if preflight.key_digest(table[["station_id", "period_start"]]) != preflight.key_digest(keys):
        raise ValueError("Merged 2024 key set differs")
    csv_path, manifest_path = chunk_paths(source, 2024)
    table.sort_values(["period_start", "station_id"]).to_csv(csv_path, index=False)
    expected = expected_chunk_manifest(source, 2024, calibration, keys)
    saved = dict(expected, actual_rows=230, csv_sha256=sha256_file(csv_path), status="completed",
                 completion_mode="merged_deterministic_half_year_subchunks",
                 children=[str(path.relative_to(project_root())) for path in paths])
    manifest_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return csv_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--project")
    parser.add_argument("--source", choices=SOURCES); parser.add_argument("--year", type=int, choices=list(YEARS))
    parser.add_argument("--split-2024", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv); calibration = freeze_calibrations()
    print(json.dumps({"calibration_manifest": str(calibration_path()),
        "calibration_manifest_sha256": calibration["manifest_sha256"],
        "earth_engine_initialized": bool(args.execute)}, indent=2))
    if not args.execute: return 0
    if not args.project or not args.source or (args.year not in YEARS and not args.split_2024): raise ValueError("--execute requires --project, --source, and --year or --split-2024")
    import ee
    from et_downscaling.availability_diagnostic import get_dynamic_hls_collection, get_dynamic_modis_inputs, get_dynamic_s2_collection
    ee.Initialize(project=args.project)
    modis = get_dynamic_modis_inputs("2020-01-01", "2025-01-01")
    collection = get_dynamic_s2_collection(modis["station_footprints"], "2020-01-01", "2025-01-01") if args.source == "S2" else get_dynamic_hls_collection(modis["station_footprints"], "2020-01-01", "2025-01-01")
    if args.split_2024:
        execute_subchunk(args.source, "2024-01-01", "2024-07-01", calibration, modis, collection)
        execute_subchunk(args.source, "2024-07-01", "2025-01-01", calibration, modis, collection)
        merge_2024_subchunks(args.source, calibration)
    else:
        execute_chunk(args.source, args.year, args.project, calibration, modis, collection)
    return 0


if __name__ == "__main__": raise SystemExit(main())
