"""Provenance checks for reusable raw-cache products.

The canonical pipeline reuses expensive Earth Engine exports. Reuse is only
allowed when a sidecar manifest proves that the cache was produced with the
same scientifically relevant configuration as the current run.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import (
    END_DATE,
    OUTPUT_PERIOD_LABEL,
    S2_CLEAR_THRESHOLD,
    S2_DAILY_MOSAIC_SORT_PROPERTY,
    S2_PREPROCESSING_VERSION,
    S2_QA_BAND,
    START_DATE,
    normalize_optical_source,
)

SATELLITE_PROVENANCE_SCHEMA_VERSION = 3


def satellite_provenance_path(output_path: Path) -> Path:
    """Return the JSON sidecar path for a raw satellite CSV."""
    output_path = Path(output_path)
    return output_path.with_suffix(output_path.suffix + ".provenance.json")


def build_satellite_provenance(
    optical_source: str,
    model_only: bool = False,
) -> dict[str, object]:
    """Describe the scientifically relevant raw-satellite configuration."""
    source = normalize_optical_source(optical_source)
    payload: dict[str, object] = {
        "schema_version": SATELLITE_PROVENANCE_SCHEMA_VERSION,
        "cache_kind": "satellite_raw",
        "optical_source": source,
        "analysis_start": START_DATE,
        "analysis_end_exclusive": END_DATE,
        "period_label": OUTPUT_PERIOD_LABEL,
        "model_only": bool(model_only),
        "sentinel1_queried": not bool(model_only),
    }

    if source == "S2":
        payload.update(
            {
                "s2_qa_band": S2_QA_BAND,
                "s2_clear_threshold": float(S2_CLEAR_THRESHOLD),
                "s2_daily_mosaic_sort_property": (
                    S2_DAILY_MOSAIC_SORT_PROPERTY
                ),
                "s2_preprocessing_version": S2_PREPROCESSING_VERSION,
            }
        )

    return payload


def write_satellite_provenance(
    output_path: Path,
    optical_source: str,
    model_only: bool = False,
) -> Path:
    """Write the raw-satellite sidecar manifest after a successful export."""
    path = satellite_provenance_path(output_path)
    path.write_text(
        json.dumps(
            build_satellite_provenance(
                optical_source,
                model_only=model_only,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def validate_satellite_provenance(
    output_path: Path,
    optical_source: str,
    model_only: bool = False,
) -> dict[str, object]:
    """Reject a reusable raw cache with missing or mismatched provenance."""
    output_path = Path(output_path)
    path = satellite_provenance_path(output_path)

    if not path.is_file():
        raise RuntimeError(
            "Raw satellite cache has no provenance sidecar and cannot be "
            "safely reused. Rebuild it once with --force/--refresh-raw.\n"
            f"Cache: {output_path}\nExpected sidecar: {path}"
        )

    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read raw satellite provenance: {path}"
        ) from error

    expected = build_satellite_provenance(
        optical_source,
        model_only=model_only,
    )
    mismatches = {
        key: {
            "expected": expected_value,
            "actual": actual.get(key),
        }
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }

    if mismatches:
        raise RuntimeError(
            "Raw satellite cache provenance does not match the current "
            "scientific configuration. Rebuild with --force/--refresh-raw.\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )

    return actual
