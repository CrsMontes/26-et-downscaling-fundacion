"""Local-only preflight utilities for experimental FVC recalibration.

This module never imports or initializes Earth Engine. It defines the exact
historical quantile calculation and validates resumable diagnostic chunks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


START_DATE = "2020-01-01"
END_DATE_EXCLUSIVE = "2025-01-01"
THRESHOLD_PCT = 80.0
EXPECTED_STATIONS = 5
EXPECTED_PERIODS = 230
EXPECTED_KEYS = 1150
SOURCES = ("S2", "HLS")
SCHEMA_VERSION = "fvc-candidates-v1"
ALGORITHM_VERSION = "historical-two-stage-global-percentile-v1"
MEDOID_DEFINITIONS = {
    "S2": "existing_s2_multiband_temporal_medoid",
    "HLS": "existing_combined_s30_l30_multiband_temporal_medoid",
}
SOURCE_SCALES_M = {"S2": 20, "HLS": 30}
REQUIRED_COLUMNS = (
    "station_id", "period_start", "number_days", "source",
    "optical_coverage_pct", "nonwater_pixel_count",
    "ndvi_p05_nonwater", "ndvi_p95_nonwater",
    "valid_for_fvc_calibration",
)
PROVENANCE_COLUMNS = {
    "S2": ("optical_products", "optical_unique_dates"),
    "HLS": (
        "hls_s30_products", "hls_l30_products", "hls_s30_unique_dates",
        "hls_l30_unique_dates", "local_mgrs_tiles",
    ),
}
HISTORICAL_CONFIG_PATH = Path("config/fvc_endmembers.json")


def calculate_endmembers(candidates):
    """Apply the historical float64, linear-interpolation quantile method."""
    required = {
        "coverage_pct", "nonwater_pixel_count", "NDVI_p05", "NDVI_p95"
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    table = candidates.copy()
    for column in required:
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("float64")
    eligible = table.loc[
        table["coverage_pct"].ge(THRESHOLD_PCT)
        & table["nonwater_pixel_count"].gt(0)
        & table["NDVI_p05"].notna()
        & table["NDVI_p95"].notna()
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible FVC candidate rows")
    low = float(eligible["NDVI_p05"].quantile(0.05, interpolation="linear"))
    high = float(eligible["NDVI_p95"].quantile(0.95, interpolation="linear"))
    if not (np.isfinite(low) and np.isfinite(high) and -1 <= low < high <= 1):
        raise ValueError(f"Invalid FVC endmembers: low={low}, high={high}")
    return {
        "ndvi_low_endmember": low,
        "ndvi_high_endmember": high,
        "n_observations": int(len(eligible)),
        "n_stations": int(eligible["station_id"].nunique())
            if "station_id" in eligible else None,
    }, eligible


def validate_universe(table):
    keys = ["station_id", "period_start"]
    missing = set(keys) - set(table.columns)
    if missing:
        raise ValueError(f"Universe is missing keys: {sorted(missing)}")
    data = table.copy()
    data["period_start"] = pd.to_datetime(data["period_start"], errors="raise")
    if data.duplicated(keys).any():
        raise ValueError("Universe contains duplicate station-period keys")
    if not data["period_start"].ge(START_DATE).all() or not data["period_start"].lt(
        END_DATE_EXCLUSIVE
    ).all():
        raise ValueError("Universe contains dates outside the half-open interval")
    if data["station_id"].astype(str).nunique() != EXPECTED_STATIONS:
        raise ValueError("Universe does not contain exactly five stations")
    if data["period_start"].nunique() != EXPECTED_PERIODS or len(data) != EXPECTED_KEYS:
        raise ValueError("Universe is not 230 periods x five stations")
    return data


def chunk_manifest(source, start, end_exclusive, expected_keys_sha256, expected_rows):
    if source not in SOURCES:
        raise ValueError(f"Unsupported source: {source}")
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "source": source,
        "start_date": start,
        "end_date_exclusive": end_exclusive,
        "expected_rows": int(expected_rows),
        "expected_keys_sha256": expected_keys_sha256,
        "medoid_definition": MEDOID_DEFINITIONS[source],
        "source_scale_m": SOURCE_SCALES_M[source],
        "coverage_bands": ["Green", "Red", "NIR"],
        "water_rule": "NDWI <= 0",
        "hls_mgrs_rule": "verified_local_mgrs_before_composite" if source == "HLS" else None,
    }


def key_digest(table):
    values = "\n".join(
        table.sort_values(["station_id", "period_start"])
        [["station_id", "period_start"]].astype(str).agg("|".join, axis=1)
    )
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def manifest_compatible(actual, expected):
    compatibility_fields = (
        "schema_version", "algorithm_version", "source", "start_date",
        "end_date_exclusive", "expected_rows", "expected_keys_sha256",
        "medoid_definition", "source_scale_m", "coverage_bands", "water_rule",
        "hls_mgrs_rule",
    )
    return all(actual.get(field) == expected.get(field) for field in compatibility_fields)


def validate_completed_chunk(table, manifest, expected_manifest, expected_keys):
    if not manifest_compatible(manifest, expected_manifest):
        raise ValueError("Chunk manifest is incompatible with the requested run")
    required = set(REQUIRED_COLUMNS) | set(PROVENANCE_COLUMNS[manifest["source"]])
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Chunk is missing columns: {sorted(missing)}")
    keys = ["station_id", "period_start"]
    if table.duplicated(keys).any():
        raise ValueError("Chunk contains duplicate keys")
    if len(table) != expected_manifest["expected_rows"]:
        raise ValueError("Chunk row count does not match its manifest")
    if key_digest(table) != key_digest(expected_keys):
        raise ValueError("Chunk keys do not match the expected universe slice")
    if manifest["source"] == "HLS" and not table["local_mgrs_tiles"].notna().all():
        raise ValueError("HLS chunk lacks verified local MGRS provenance")
    return True


def safe_experimental_output(path):
    resolved = Path(path)
    if resolved.as_posix().endswith(HISTORICAL_CONFIG_PATH.as_posix()):
        raise ValueError("Historical FVC config must never be overwritten")
    return resolved


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
