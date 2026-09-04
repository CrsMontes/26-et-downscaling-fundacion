"""Export minimal FVC calibration candidates with gated, resumable chunks.

No FVC, Albedo, target, meteorology, or model feature is exported. Earth
Engine is initialized only when ``--execute`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

import preflight_fvc_recalibration as preflight


PERIOD_LABEL = "2020_2024"
MAX_TRANSIENT_RETRIES = 2
HISTORICAL_ABSOLUTE_TOLERANCE = 1e-7
HISTORICAL_RELATIVE_TOLERANCE = 1e-7
STABILITY_THRESHOLDS = {
    "negligible_max_abs_ndvi": 0.01,
    "modest_max_abs_ndvi": 0.05,
}
COMMON_SELECTORS = list(preflight.REQUIRED_COLUMNS)
SELECTORS = {
    source: COMMON_SELECTORS + list(preflight.PROVENANCE_COLUMNS[source])
    for source in preflight.SOURCES
}
FORBIDDEN_EXPORT_TOKENS = (
    "FVC", "Albedo", "Kc", "ET_", "meteor", "VV", "VH", "LST",
    "Blue_mean", "Green_mean", "Red_mean", "NIR_mean",
)


def project_root():
    return Path(__file__).resolve().parents[2]


def output_root():
    return (
        project_root() / "outputs" / "diagnostics" / PERIOD_LABEL
        / "fvc_recalibration_candidates"
    )


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=preflight.SOURCES, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date-exclusive", required=True)
    parser.add_argument("--project")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def validate_interval(start, end_exclusive):
    lower = pd.Timestamp(start)
    upper = pd.Timestamp(end_exclusive)
    if lower >= upper:
        raise ValueError("Chunk interval must be non-empty")
    if lower < pd.Timestamp(preflight.START_DATE) or upper > pd.Timestamp(
        preflight.END_DATE_EXCLUSIVE
    ):
        raise ValueError("Chunk lies outside the approved half-open interval")
    return lower, upper


def local_universe():
    path = (
        project_root() / "outputs/diagnostics/2020_2024/optical_source_experiment"
        / "raw/paired_optical_common.csv"
    )
    table = pd.read_csv(path, dtype={"station_id": str})
    return preflight.validate_universe(table[["station_id", "period_start"]])


def expected_keys(start, end_exclusive):
    lower, upper = validate_interval(start, end_exclusive)
    universe = local_universe()
    dates = pd.to_datetime(universe["period_start"])
    result = universe.loc[dates.ge(lower) & dates.lt(upper)].copy()
    if result.empty:
        raise ValueError("Chunk interval has no expected station-period keys")
    return result


def chunk_paths(source, start, end_exclusive):
    stem = f"{start}_{end_exclusive}_exclusive".replace("-", "")
    directory = output_root() / "chunks" / source.lower()
    return directory / f"{stem}.csv", directory / f"{stem}.manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_manifest(source, start, end_exclusive, keys):
    return preflight.chunk_manifest(
        source, start, end_exclusive, preflight.key_digest(keys), len(keys)
    )


def validate_candidate_values(table, source, start, end_exclusive):
    dates = pd.to_datetime(table["period_start"], errors="raise")
    lower, upper = validate_interval(start, end_exclusive)
    if not dates.ge(lower).all() or not dates.lt(upper).all():
        raise ValueError("Chunk contains dates outside its half-open interval")
    if set(table["source"].astype(str)) != {source}:
        raise ValueError("Chunk mixes or mislabels optical sources")
    numeric = [
        "optical_coverage_pct", "nonwater_pixel_count",
        "ndvi_p05_nonwater", "ndvi_p95_nonwater",
        "valid_for_fvc_calibration",
    ]
    values = table.copy()
    for column in numeric:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    if not values["optical_coverage_pct"].between(0, 100).all():
        raise ValueError("Optical coverage is outside 0..100")
    if not values["nonwater_pixel_count"].ge(0).all():
        raise ValueError("Non-water pixel count is negative")
    valid_percentiles = values[
        ["ndvi_p05_nonwater", "ndvi_p95_nonwater"]
    ].gt(-9990).all(axis=1)
    p05 = values.loc[valid_percentiles, "ndvi_p05_nonwater"]
    p95 = values.loc[valid_percentiles, "ndvi_p95_nonwater"]
    if not p05.between(-1, 1).all() or not p95.between(-1, 1).all():
        raise ValueError("NDVI percentile is outside -1..1")
    if not p05.le(p95).all():
        raise ValueError("NDVI P05 exceeds P95")
    expected_flag = (
        values["optical_coverage_pct"].ge(preflight.THRESHOLD_PCT)
        & values["nonwater_pixel_count"].gt(0)
        & valid_percentiles
    ).astype(int)
    if not values["valid_for_fvc_calibration"].eq(expected_flag).all():
        raise ValueError("Calibration eligibility flag is inconsistent")
    if source == "HLS":
        provenance = [
            "hls_s30_products", "hls_l30_products", "hls_s30_unique_dates",
            "hls_l30_unique_dates",
        ]
        for column in provenance:
            converted = pd.to_numeric(values[column], errors="coerce")
            if converted.isna().any() or converted.lt(0).any():
                raise ValueError(f"Invalid HLS provenance: {column}")
        if values["local_mgrs_tiles"].isna().any() or values[
            "local_mgrs_tiles"
        ].astype(str).str.strip().eq("").any():
            raise ValueError("HLS local MGRS provenance is missing")
    return values


def validate_existing_chunk(csv_path, manifest_path, expected, keys):
    if not csv_path.is_file() or not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        return False
    if manifest.get("csv_sha256") != sha256_file(csv_path):
        return False
    table = pd.read_csv(csv_path, dtype={"station_id": str})
    try:
        preflight.validate_completed_chunk(table, manifest, expected, keys)
        validate_candidate_values(
            table, expected["source"], expected["start_date"],
            expected["end_date_exclusive"],
        )
    except ValueError:
        return False
    return True


def _safe_ratio(numerator, denominator, name):
    return (
        numerator.divide(denominator)
        .updateMask(denominator.abs().gt(1e-6)).rename(name).toFloat()
    )


def _candidate_statistics(medoid, geometry, scale):
    import ee
    valid_mask = (
        medoid.select(["Green", "Red", "NIR"]).mask()
        .reduce(ee.Reducer.min()).rename("valid").uint8()
    )
    coverage_raw = valid_mask.unmask(0).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry, scale=scale,
        maxPixels=10_000_000, tileScale=4,
    ).get("valid")
    coverage = ee.Number(ee.Algorithms.If(
        ee.Algorithms.IsEqual(coverage_raw, None), 0,
        ee.Number(coverage_raw).multiply(100),
    ))
    nir, red = medoid.select("NIR"), medoid.select("Red")
    green = medoid.select("Green")
    ndvi = _safe_ratio(nir.subtract(red), nir.add(red), "NDVI")
    ndwi = _safe_ratio(green.subtract(nir), green.add(nir), "NDWI")
    nonwater = ndvi.updateMask(valid_mask).updateMask(ndwi.lte(0))
    reducer = ee.Reducer.percentile(
        [5, 95], ["p05", "p95"]
    ).combine(ee.Reducer.count(), sharedInputs=True)
    raw = ee.Dictionary(nonwater.reduceRegion(
        reducer=reducer, geometry=geometry, scale=scale,
        maxPixels=10_000_000, tileScale=4,
    ))
    stats = ee.Dictionary({
        "p05": -9999, "p95": -9999, "count": 0,
    }).combine(ee.Dictionary({
        "p05": raw.get("NDVI_p05", -9999),
        "p95": raw.get("NDVI_p95", -9999),
        "count": raw.get("NDVI_count", 0),
    }), True)
    count = ee.Number(stats.get("count"))
    eligible = coverage.gte(preflight.THRESHOLD_PCT).And(count.gt(0))
    return coverage, stats, eligible.int()


def build_candidate_table(source, modis_inputs, collection, start, end_exclusive):
    import ee
    from et_downscaling.availability_diagnostic import (
        _period_context, _period_values, filter_dynamic_hls_to_geometry,
    )
    from et_downscaling.hls import build_hls_medoid
    from et_downscaling.sentinel2 import build_s2_medoid

    images, image_indexes, footprints, footprint_indexes = _period_context(
        modis_inputs, start, end_exclusive
    )

    def process_image(image_index):
        period_start, period_end, period_days = _period_values(images.get(image_index))

        def process_footprint(footprint_index):
            footprint = ee.Feature(footprints.get(footprint_index))
            geometry = footprint.geometry()
            if source == "S2":
                period = ee.ImageCollection(collection).filterDate(
                    period_start, period_end
                ).filterBounds(geometry)
                medoid = build_s2_medoid(period, geometry)
                provenance = {
                    "optical_products": period.size(),
                    "optical_unique_dates": ee.List(
                        period.aggregate_array("date_key")
                    ).distinct().size(),
                }
                scale = 20
            else:
                local = filter_dynamic_hls_to_geometry(collection, geometry)
                period = local.filterDate(period_start, period_end)
                medoid = build_hls_medoid(period, geometry)
                s30 = period.filter(ee.Filter.eq("sensor", "S30"))
                l30 = period.filter(ee.Filter.eq("sensor", "L30"))
                provenance = {
                    "hls_s30_products": s30.size(), "hls_l30_products": l30.size(),
                    "hls_s30_unique_dates": ee.List(
                        s30.aggregate_array("date_key")
                    ).distinct().size(),
                    "hls_l30_unique_dates": ee.List(
                        l30.aggregate_array("date_key")
                    ).distinct().size(),
                    "local_mgrs_tiles": ee.List(local.get("local_mgrs_tiles")).join(";"),
                }
                scale = 30
            coverage, stats, eligible = _candidate_statistics(
                medoid, geometry, scale
            )
            properties = {
                "station_id": footprint.get("station_id"),
                "period_start": period_start.format("yyyy-MM-dd"),
                "number_days": period_days,
                "source": source,
                "optical_coverage_pct": coverage,
                "nonwater_pixel_count": stats.get("count"),
                "ndvi_p05_nonwater": stats.get("p05"),
                "ndvi_p95_nonwater": stats.get("p95"),
                "valid_for_fvc_calibration": eligible,
            }
            properties.update(provenance)
            return ee.Feature(None, properties)

        return footprint_indexes.map(process_footprint)

    return ee.FeatureCollection(image_indexes.map(process_image).flatten())


def write_completed_manifest(path, expected, csv_path, actual_rows):
    manifest = dict(expected)
    manifest.update({
        "actual_key_count": int(actual_rows),
        "csv_sha256": sha256_file(csv_path),
        "status": "completed",
    })
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def execute_chunk(source, start, end_exclusive, project):
    keys = expected_keys(start, end_exclusive)
    expected = expected_manifest(source, start, end_exclusive, keys)
    csv_path, manifest_path = chunk_paths(source, start, end_exclusive)
    if validate_existing_chunk(csv_path, manifest_path, expected, keys):
        print(f"Reusing validated chunk: {csv_path}")
        return [csv_path]
    import ee
    from et_downscaling.availability_diagnostic import (
        get_dynamic_hls_collection, get_dynamic_modis_inputs,
        get_dynamic_s2_collection,
    )
    from et_downscaling.export import export_feature_collection
    modis = get_dynamic_modis_inputs(start, end_exclusive)
    collection = (
        get_dynamic_s2_collection(modis["station_footprints"], start, end_exclusive)
        if source == "S2" else
        get_dynamic_hls_collection(modis["station_footprints"], start, end_exclusive)
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    relative = csv_path.relative_to(project_root() / "outputs")
    last_error = None
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            table = build_candidate_table(source, modis, collection, start, end_exclusive)
            exported = Path(export_feature_collection(
                table, relative.as_posix(), SELECTORS[source]
            ))
            data = pd.read_csv(exported, dtype={"station_id": str})
            data = validate_candidate_values(data, source, start, end_exclusive)
            provisional = dict(expected, status="completed",
                               actual_key_count=len(data), csv_sha256=sha256_file(exported))
            preflight.validate_completed_chunk(data, provisional, expected, keys)
            write_completed_manifest(manifest_path, expected, exported, len(data))
            return [exported]
        except Exception as error:
            last_error = error
            if attempt < MAX_TRANSIENT_RETRIES:
                time.sleep(2 ** attempt)
    lower, upper = validate_interval(start, end_exclusive)
    if (upper - lower).days < 2:
        raise RuntimeError(f"Unsplittable failed chunk {start}..{end_exclusive}") from last_error
    midpoint = (lower + (upper - lower) / 2).normalize()
    children = (
        (lower.date().isoformat(), midpoint.date().isoformat()),
        (midpoint.date().isoformat(), upper.date().isoformat()),
    )
    failed = dict(expected, status="split_after_retries", error=str(last_error),
                  children=children)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
    paths = []
    for child_start, child_end in children:
        paths.extend(execute_chunk(source, child_start, child_end, project))
    return paths


def main(argv=None):
    args = parse_arguments(argv)
    keys = expected_keys(args.start_date, args.end_date_exclusive)
    print(json.dumps({
        "source": args.source, "start_date": args.start_date,
        "end_date_exclusive": args.end_date_exclusive,
        "expected_keys": len(keys), "selectors": SELECTORS[args.source],
        "earth_engine_initialized": bool(args.execute),
    }, indent=2))
    if not args.execute:
        return 0
    if not args.project:
        raise ValueError("--project is required with --execute")
    import ee
    ee.Initialize(project=args.project)
    execute_chunk(args.source, args.start_date, args.end_date_exclusive, args.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
