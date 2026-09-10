"""Build the final Virtual10 RF-25 training population from extracted data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TARGET_COLUMN
from .rf25 import (
    RF25_HARMONIC_FEATURES,
    RF25_METEOROLOGICAL_FEATURES,
    RF25_MODEL_FEATURES,
    RF25_OPTICAL_FEATURES,
)

KEY_COLUMNS = ["station_id", "period_start"]
OPTICAL_COVERAGE_THRESHOLD_PCT = 90.0
SPATIAL_BLOCK_SIZE_KM = 10.0


def _add_harmonics(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    dates = pd.to_datetime(result["period_start"], errors="raise")
    doy = dates.dt.dayofyear.to_numpy(dtype=float)
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * doy / 365.25
        result[f"doy_sin{harmonic}"] = np.sin(angle)
        result[f"doy_cos{harmonic}"] = np.cos(angle)
    return result


def _resolve_coordinate_columns(data: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("station_longitude", "station_latitude"),
        ("longitude", "latitude"),
        ("footprint_centroid_longitude", "footprint_centroid_latitude"),
    ]
    for longitude_column, latitude_column in candidates:
        if longitude_column in data.columns and latitude_column in data.columns:
            return longitude_column, latitude_column
    raise ValueError("No usable longitude/latitude columns were found to derive spatial blocks.")


def _add_spatial_blocks(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    longitude_column, latitude_column = _resolve_coordinate_columns(result)
    longitude = pd.to_numeric(result[longitude_column], errors="raise")
    latitude = pd.to_numeric(result[latitude_column], errors="raise")
    km_per_degree_latitude = 111.32
    km_per_degree_longitude = 111.32 * np.cos(np.radians(latitude.mean()))
    block_x = np.floor(longitude * km_per_degree_longitude / SPATIAL_BLOCK_SIZE_KM).astype(int)
    block_y = np.floor(latitude * km_per_degree_latitude / SPATIAL_BLOCK_SIZE_KM).astype(int)
    result["spatial_block"] = block_x.astype(str) + "_" + block_y.astype(str)
    return result


def canonicalize_master(master: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw-derived master to the final RF-25 schema."""
    data = master.copy()
    base_required = {"station_id", "period_start", TARGET_COLUMN, "modis_good", "target_complete"}
    missing_base = sorted(base_required - set(data.columns))
    if missing_base:
        raise ValueError("Master dataset is missing required base columns: " + ", ".join(missing_base))

    data["station_id"] = data["station_id"].astype(str)
    data["period_start"] = pd.to_datetime(data["period_start"], errors="raise")
    if data.duplicated(KEY_COLUMNS).any():
        raise ValueError("Master dataset contains duplicate station-period keys.")

    if "s2_coverage_pct" not in data.columns:
        if "optical_union_coverage_pct" not in data.columns:
            raise ValueError("Master dataset contains neither s2_coverage_pct nor optical_union_coverage_pct.")
        data["s2_coverage_pct"] = pd.to_numeric(data["optical_union_coverage_pct"], errors="coerce")

    for production_feature in RF25_OPTICAL_FEATURES:
        if production_feature in data.columns:
            continue
        diagnostic_feature = f"s2_{production_feature}"
        if diagnostic_feature not in data.columns:
            raise ValueError(
                "Missing optical predictor in both schemas: "
                f"{production_feature} / {diagnostic_feature}"
            )
        data[production_feature] = pd.to_numeric(data[diagnostic_feature], errors="coerce")

    missing_meteorology = sorted(set(RF25_METEOROLOGICAL_FEATURES) - set(data.columns))
    if missing_meteorology:
        raise ValueError(
            "Master dataset is missing final meteorological predictors: "
            + ", ".join(missing_meteorology)
        )

    if not set(RF25_HARMONIC_FEATURES).issubset(data.columns):
        data = _add_harmonics(data)

    if "year" not in data.columns:
        data["year"] = data["period_start"].dt.year.astype(int)
    else:
        data["year"] = pd.to_numeric(data["year"], errors="raise").astype(int)

    if "spatial_block" not in data.columns:
        data = _add_spatial_blocks(data)

    numeric_columns = list(RF25_MODEL_FEATURES) + [TARGET_COLUMN, "s2_coverage_pct"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return data


def prepare_rf25_population(master: pd.DataFrame) -> pd.DataFrame:
    """Return the final GE90 population without unused-predictor eligibility gates."""
    data = canonicalize_master(master)
    eligible = (
        data["modis_good"].eq(1)
        & data["target_complete"].eq(1)
        & data[TARGET_COLUMN].notna()
        & data["s2_coverage_pct"].ge(OPTICAL_COVERAGE_THRESHOLD_PCT)
        & data[RF25_MODEL_FEATURES].notna().all(axis=1)
    )
    selected = data.loc[eligible].copy().sort_values(KEY_COLUMNS).reset_index(drop=True)
    matrix = selected[list(RF25_MODEL_FEATURES) + [TARGET_COLUMN]].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("RF-25 training matrix contains non-finite values.")
    return selected
