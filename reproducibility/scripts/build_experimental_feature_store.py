"""Build a non-destructive local feature store for the 2020-2024 experiment.

The store preserves the complete 1,150 station-period universe and missing
values. It does not define a complete-case population or access Earth Engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from et_downscaling.candidate_paths import get_candidate_study_paths


KEYS = ["station_id", "period_start"]
EXPECTED_ROWS = 1150
MISSING_SENTINEL_MAX = -9990.0
COMMON_OPTICAL = [
    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
    "NDVI", "EVI", "SAVI", "NDWI", "NDMI",
]
S2_EXTRAS = [
    "RedEdge1", "RedEdge2", "RedEdge3", "NIR_Broad",
    "NDRE", "Albedo", "FVC",
]
METEOROLOGY_PREDICTORS = [
    "Tair_mean_C", "Tair_max_C", "VPD_mean_kPa", "VPD_max_kPa",
    "SolarRad_MJ_m2_day", "Wind_mean_ms", "Precip_period_mm",
    "Precip_prev30d_mm", "ETo_mm_period", "ETr_mm_period",
    "ETo_mm_day", "ETr_mm_day",
]
HARMONICS = ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]


def project_root():
    return Path(__file__).resolve().parents[2]


def read_unique(path, label):
    table = pd.read_csv(path, dtype={"station_id": str})
    missing = set(KEYS) - set(table.columns)
    if missing:
        raise RuntimeError(f"{label} is missing keys: {sorted(missing)}")
    table["station_id"] = table["station_id"].astype(str)
    table["period_start"] = pd.to_datetime(
        table["period_start"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    if table.duplicated(KEYS).any():
        raise RuntimeError(f"{label} contains duplicate station-period keys")
    return table


def left_join(store, table, label):
    before_keys = store[KEYS].copy()
    result = store.merge(table, on=KEYS, how="left", validate="one_to_one")
    if len(result) != len(store) or not result[KEYS].equals(before_keys):
        raise RuntimeError(f"{label} changed the base universe or key order")
    return result


def namespace_columns(table, prefix, excluded=()):
    excluded = set(excluded) | set(KEYS)
    return table.rename(
        columns={column: f"{prefix}{column}" for column in table.columns if column not in excluded}
    )


def add_harmonics(store):
    dates = pd.to_datetime(store["period_start"], errors="raise")
    day_of_year = dates.dt.dayofyear.to_numpy(dtype=float)
    result = store.copy()
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * day_of_year / 365.25
        result[f"doy_sin{harmonic}"] = np.sin(angle)
        result[f"doy_cos{harmonic}"] = np.cos(angle)
    return result


def normalize_numeric_missing(store, columns):
    result = store.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result.loc[result[column] <= MISSING_SENTINEL_MAX, column] = np.nan
    return result


def build_store(root):
    paths = get_candidate_study_paths(root)
    diagnostic = paths.raw_root
    optical_root = paths.optical_root
    optical_path = optical_root / "raw" / "paired_optical_common.csv"
    rich_path = optical_root / "raw" / "s2_rich_optical.csv"
    s1_path = diagnostic / "s1_geometry_experiment" / "raw" / "s1_geometry_predictors.csv"
    thermal_path = diagnostic / "thermal_availability" / "raw" / "landsat_lst_station_period.csv"
    meteorology_path = diagnostic / "meteorology_experiment" / "processed" / "period_meteorology.csv"
    master_path = optical_root / "population" / "paired_master.csv"
    support_path = paths.station_support
    availability_root = paths.availability_root / "raw"

    optical = read_unique(optical_path, "paired optical table")
    if len(optical) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} base rows, found {len(optical)}")
    identity = [
        "station", "station_id", "period_start", "period_end",
        "period_end_exclusive", "period_days", "modis_pixel_id", "footprint_area_m2",
    ]
    optical_columns = [
        column for column in optical.columns
        if column in identity or column.startswith(("s2_", "hls_"))
    ]
    store = optical[optical_columns].copy().sort_values(KEYS).reset_index(drop=True)

    rich = read_unique(rich_path, "S2 rich optical table")
    rich_common = [f"s2_{name}_mean" for name in COMMON_OPTICAL]
    comparison = optical[KEYS + rich_common].merge(
        rich[KEYS + rich_common], on=KEYS, suffixes=("_paired", "_rich"), validate="one_to_one"
    )
    for column in rich_common:
        paired = pd.to_numeric(comparison[f"{column}_paired"], errors="coerce")
        rich_values = pd.to_numeric(comparison[f"{column}_rich"], errors="coerce")
        paired = paired.mask(paired <= MISSING_SENTINEL_MAX)
        rich_values = rich_values.mask(rich_values <= MISSING_SENTINEL_MAX)
        if not np.allclose(paired, rich_values, equal_nan=True, atol=1e-12, rtol=0):
            raise RuntimeError(f"S2 common conflict between local tables: {column}")
    rich_extras = [f"s2_{name}_mean" for name in S2_EXTRAS]
    store = left_join(store, rich[KEYS + rich_extras], "S2 rich join")

    s1 = read_unique(s1_path, "S1 geometry predictors")
    s1_columns = [column for column in s1.columns if column.startswith(("r077_", "r142_"))]
    store = left_join(store, s1[KEYS + s1_columns], "S1 geometry join")

    thermal = read_unique(thermal_path, "Landsat thermal availability")
    thermal_columns = [column for column in thermal.columns if column not in KEYS + ["station"]]
    thermal = namespace_columns(thermal[KEYS + thermal_columns], "thermal_")
    store = left_join(store, thermal, "thermal QA join")

    meteorology = read_unique(meteorology_path, "period meteorology")
    store = left_join(
        store,
        meteorology,
        "period meteorology join",
    )

    master = read_unique(master_path, "paired master")
    master_columns = [
        "ET_mm_period", "modis_good", "s2_coverage_pct", "hls_coverage_pct",
        "s2_predictors_complete", "hls_predictors_complete", "target_complete",
        "Kc_target", "spatial_block", "year", "paired_candidate_ge_80",
        "paired_candidate_ge_90", "paired_candidate_ge_99",
    ]
    target = master[KEYS + master_columns].rename(columns={"ET_mm_period": "modis_ET_mm_period"})
    store = left_join(store, target, "target and population flags join")

    for label, filename, prefix in (
        ("MODIS availability", "modis_station_period.csv", "modis_qa_"),
        ("S2 availability", "s2_station_period.csv", "s2_availability_"),
        ("HLS availability", "hls_station_period.csv", "hls_availability_"),
    ):
        table = read_unique(availability_root / filename, label)
        selected = [column for column in table.columns if column not in KEYS + ["station"]]
        table = namespace_columns(table[KEYS + selected], prefix)
        duplicate_columns = set(store.columns).intersection(set(table.columns)) - set(KEYS)
        if duplicate_columns:
            raise RuntimeError(f"Ambiguous columns in {label}: {sorted(duplicate_columns)}")
        store = left_join(store, table, f"{label} join")

    support = pd.read_csv(support_path, dtype={"station_id": str})
    if support.duplicated(["station_id"]).any():
        raise RuntimeError("Station support contains duplicate station IDs")
    support_columns = [
        "station_id", "station_longitude", "station_latitude",
        "footprint_centroid_longitude", "footprint_centroid_latitude",
        "footprint_mean_elevation_m", "elevation_sampling_method",
        "elevation_grid_scale_m", "era5_support_m", "era5_sampling_method",
        "era5_sampling_longitude", "era5_sampling_latitude",
        "era5_sampling_distance_m", "chirps_support_m",
    ]
    store = store.merge(
        support[support_columns], on="station_id", how="left", validate="many_to_one"
    )
    if len(store) != EXPECTED_ROWS:
        raise RuntimeError("Station support join changed the base universe")
    store = add_harmonics(store)

    predictor_columns = (
        [f"s2_{name}_mean" for name in COMMON_OPTICAL + S2_EXTRAS]
        + [f"hls_{name}_mean" for name in COMMON_OPTICAL]
        + [f"r{orbit}_{name}" for orbit in ("077", "142") for name in (
            "VV_dB_mean", "VH_dB_mean", "VV_minus_VH_dB_mean"
        )]
        + METEOROLOGY_PREDICTORS
        + HARMONICS
        + ["footprint_mean_elevation_m"]
    )
    store = normalize_numeric_missing(store, predictor_columns)
    if store.duplicated(KEYS).any() or len(store[KEYS].drop_duplicates()) != EXPECTED_ROWS:
        raise RuntimeError("Final feature store keys are not unique")
    return store, predictor_columns


def predictor_metadata():
    rows = []

    def add(names, family, product, resolution, support, aggregation, file, status,
            classification, wall, prediction_support, requires_ee, notes="",
            implemented=True):
        for name in names:
            rows.append({
                "feature_name": name,
                "feature_family": family,
                "source_product": product,
                "role": "predictor",
                "native_or_nominal_resolution": resolution,
                "training_support": support,
                "temporal_aggregation": aggregation,
                "current_local_file": file,
                "conceptually_available": True,
                "implemented_in_code": implemented,
                "current_experiment_status": status,
                "active_or_candidate_or_qa": classification,
                "wall_to_wall_prediction_available": wall,
                "prediction_support": prediction_support,
                "requires_new_Earth_Engine_processing": requires_ee,
                "notes": notes,
            })

    add(
        [f"s2_{name}_mean" for name in COMMON_OPTICAL], "S2 common optical",
        "COPERNICUS/S2_SR_HARMONIZED", "10-20 m native; 20 m composite grid",
        "Mean over MODIS footprint", "MODIS-period temporal medoid",
        "optical_source_experiment/raw/paired_optical_common.csv", "evaluated",
        "active", True, "20 m S2 wall-to-wall stack", False,
    )
    add(
        [f"s2_{name}_mean" for name in S2_EXTRAS], "S2 source-specific optical",
        "COPERNICUS/S2_SR_HARMONIZED", "10-20 m native; 20 m composite grid",
        "Mean over MODIS footprint", "MODIS-period temporal medoid",
        "optical_source_experiment/raw/s2_rich_optical.csv", "evaluated_ablation",
        "candidate", False, "Implemented image bands but not wired into current prediction stack",
        False, "FVC remains calibration-sensitive.",
    )
    add(
        [f"hls_{name}_mean" for name in COMMON_OPTICAL], "HLS combined common optical",
        "NASA HLS S30 v2 + L30 v2", "30 m nominal and extraction grid",
        "Mean over MODIS footprint", "Combined S30/L30 per-pixel temporal medoid",
        "optical_source_experiment/raw/paired_optical_common.csv", "evaluated",
        "candidate", False, "No current HLS wall-to-wall prediction pipeline", False,
        "A composite may mix sensors and acquisition dates across pixels.",
    )
    add(
        ["hls_Albedo_mean", "hls_FVC_mean"], "HLS source-specific optical",
        "NASA HLS S30 v2 + L30 v2", "30 m nominal", "Not materialized",
        "Would follow combined HLS medoid", "", "implemented_not_materialized",
        "candidate", False, "No current HLS wall-to-wall prediction pipeline", True,
        "Code exists, but the 2020-2024 extraction contains only common11.",
    )
    for orbit, geometry in (("077", "ascending R077"), ("142", "descending R142")):
        add(
            [f"r{orbit}_{name}" for name in (
                "VV_dB_mean", "VH_dB_mean", "VV_minus_VH_dB_mean"
            )], f"Sentinel-1 {geometry}", "COPERNICUS/S1_GRD", "10 m nominal and extraction grid",
            "Mean over MODIS footprint", "MODIS-period median for fixed pass/orbit",
            "s1_geometry_experiment/raw/s1_geometry_predictors.csv", "evaluated",
            "candidate", False, "Orbit-specific wall-to-wall stack not wired", False,
            "Spatial instability was observed, especially for ST02/ST03.",
        )
    add(
        [
            "Tair_mean_C", "Tair_max_C", "VPD_mean_kPa",
            "SolarRad_MJ_m2_day", "Wind_mean_ms", "Precip_period_mm",
            "Precip_prev30d_mm",
        ], "Meteorology and precipitation",
        "ERA5-Land + CHIRPS + local FAO-56", "ERA5 ~9 km; CHIRPS 0.05 degree",
        "Station support / nearest valid land cell", "Current MODIS period; precipitation also previous 30 days",
        "meteorology_experiment/processed/period_meteorology.csv", "partly_evaluated",
        "active_or_candidate", True,
        "Current wall-to-wall stack supports these seven meteorological variables",
        False,
    )
    add(
        ["VPD_max_kPa"], "Meteorology and precipitation", "ERA5-Land",
        "~9 km", "Nearest valid ERA5-Land support", "Maximum over current MODIS period",
        "meteorology_experiment/processed/period_meteorology.csv", "available_not_evaluated",
        "candidate", False, "Not included in the current wall-to-wall predictor stack", False,
    )
    add(
        ["ETo_mm_period", "ETr_mm_period", "ETo_mm_day", "ETr_mm_day"],
        "Reference ET", "ERA5-Land + local FAO-56", "~9 km meteorological support",
        "Nearest valid ERA5-Land support", "Current MODIS-period sum or daily mean",
        "meteorology_experiment/processed/period_meteorology.csv", "available_not_primary_predictor",
        "candidate", False, "Not included in the current wall-to-wall predictor stack", False,
        "These variables share construction information with the Kc target denominator.",
    )
    add(
        HARMONICS, "Temporal harmonics", "Derived from period_start", "Constant per date",
        "All rows", "Two annual harmonic pairs using 365.25 days", "derived_locally",
        "evaluated", "candidate", True, "Constant wall-to-wall per period", False,
    )
    add(
        ["footprint_mean_elevation_m"], "Topography", "NASADEM", "30 m source grid",
        "Mean over MODIS footprint", "Static", "outputs/raw/meteorology/station_support.csv",
        "available_not_evaluated", "candidate", False,
        "Current production stack does not expose elevation as a model band", False,
        "Currently used to calculate reference ET, not as a fitted predictor.",
    )
    add(
        ["landsat_l8_only_LST_K", "landsat_l8_l9_combined_LST_K"], "Landsat thermal",
        "LANDSAT LC08/LC09 C02 T1 L2 ST_B10", "~100 m native thermal support; 30 m distributed grid",
        "Not materialized", "Would use MODIS-period temporal medoid",
        "", "availability_only", "candidate", False,
        "No wall-to-wall thermal predictor stack", True,
        "Local files contain coverage and ST_QA summaries, not extracted LST values.",
    )
    add(
        ["terrain_slope", "terrain_aspect"], "Topography", "Unspecified DEM",
        "Not defined", "Not defined", "Static", "", "concept_only",
        "candidate", False, "No implementation", True,
        "No reproducible implementation or 2020-2024 materialization was found.",
        implemented=False,
    )
    return rows


def classify_auxiliary(column):
    if column == "Kc_target":
        return "target", "Target", "evaluated", "qa"
    if column == "modis_ET_mm_period":
        return "target", "MODIS target", "evaluated", "qa"
    if column in KEYS or column in {"station", "year", "spatial_block"}:
        return "provenance", "Identity and folds", "active", "qa"
    if any(token in column.lower() for token in (
        "coverage", "complete", "valid", "products", "dates", "area", "qa",
        "ge_", "support", "longitude", "latitude", "crs", "scale", "sensor",
        "orbit", "pass", "tile", "period_end", "period_days", "footprint", "angle",
        "candidate", "sampling", "source", "count", "min_k", "max_k", "stddev",
    )):
        return "QA", "Availability / provenance", "available", "qa"
    return "provenance", "Auxiliary", "available", "qa"


def build_inventory(store, predictor_columns):
    metadata = {row["feature_name"]: row for row in predictor_metadata()}
    inventory = []
    for column in store.columns:
        if column in metadata:
            row = metadata.pop(column)
        else:
            role, family, status, classification = classify_auxiliary(column)
            row = {
                "feature_name": column,
                "feature_family": family,
                "source_product": "See source namespace",
                "role": role,
                "native_or_nominal_resolution": "Varies / not applicable",
                "training_support": "Station-period or provenance support",
                "temporal_aggregation": "See feature name and source table",
                "current_local_file": "experimental feature store join inputs",
                "conceptually_available": True,
                "implemented_in_code": True,
                "current_experiment_status": status,
                "active_or_candidate_or_qa": classification,
                "wall_to_wall_prediction_available": False,
                "prediction_support": "Not a model predictor",
                "requires_new_Earth_Engine_processing": False,
                "notes": "Preserved for availability, QA, provenance, or explicit population selection.",
            }
        available = int(store[column].notna().sum())
        row["materialized_2020_2024"] = True
        row["available_row_count"] = available
        row["missing_row_count"] = len(store) - available
        row["usable_in_training"] = column in predictor_columns and available > 0
        inventory.append(row)
    for row in metadata.values():
        row["materialized_2020_2024"] = False
        row["available_row_count"] = 0
        row["missing_row_count"] = len(store)
        row["usable_in_training"] = False
        inventory.append(row)
    columns = [
        "feature_name", "feature_family", "source_product", "role",
        "native_or_nominal_resolution", "training_support", "temporal_aggregation",
        "current_local_file", "conceptually_available", "implemented_in_code",
        "materialized_2020_2024", "available_row_count",
        "missing_row_count", "current_experiment_status", "active_or_candidate_or_qa",
        "usable_in_training", "wall_to_wall_prediction_available", "prediction_support",
        "requires_new_Earth_Engine_processing", "notes",
    ]
    return pd.DataFrame(inventory)[columns].sort_values(
        ["role", "feature_family", "feature_name"]
    ).reset_index(drop=True)


def main():
    root = project_root()
    store, predictor_columns = build_store(root)
    inventory = build_inventory(store, predictor_columns)
    paths = get_candidate_study_paths(root)
    output = paths.intermediate_root / "feature_store"
    output.mkdir(parents=True, exist_ok=True)
    store_path = output / "feature_store.csv"
    inventory_path = output / "feature_inventory.csv"
    manifest_path = output / "feature_store_manifest.json"
    store.to_csv(store_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    real_predictors = inventory.loc[
        inventory["role"].eq("predictor")
        & inventory["materialized_2020_2024"]
        & inventory["usable_in_training"]
    ]
    manifest = {
        "experiment": "2020_2024_non_destructive_feature_store",
        "key": KEYS,
        "rows": len(store),
        "unique_keys": int(store[KEYS].drop_duplicates().shape[0]),
        "columns": len(store.columns),
        "inventory_rows": len(inventory),
        "materialized_usable_predictors": len(real_predictors),
        "predictor_families": real_predictors.groupby("feature_family").size().to_dict(),
        "global_dropna_performed": False,
        "complete_case_population_imposed": False,
        "source_files_modified": False,
        "earth_engine_access": False,
        "screening_registry": {
            "random_forest": "baseline",
            "extra_trees": "primary_nonlinear_candidate",
            "ridge_s2": "conditional_linear_candidate",
            "dummy_mean": "sanity_baseline_only",
            "hist_gradient_boosting": "not_advanced",
            "ridge_hls": "not_advanced",
            "winner_declared": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Rows: {len(store)}")
    print(f"Columns: {len(store.columns)}")
    print(f"Inventory rows: {len(inventory)}")
    print(f"Materialized usable predictors: {len(real_predictors)}")
    print(f"Feature store: {store_path}")
    print(f"Inventory: {inventory_path}")
    print("global_dropna_performed = false")
    print("earth_engine_access = false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
