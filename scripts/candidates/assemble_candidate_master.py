"""Assemble the canonical 2020-2024 candidate-predictor master.

This module performs row-preserving joins only. It contains no model fitting,
feature selection, or historical model-comparison sensitivity analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from et_downscaling.candidate_paths import get_candidate_study_paths


ROOT = Path(__file__).resolve().parents[2]
PATHS = get_candidate_study_paths(ROOT)
FEATURE_STORE = PATHS.intermediate_root / "feature_store" / "feature_store.csv"
LST_TABLE = PATHS.landsat_lst_root / "landsat_lst_station_period.csv"
KEYS = ["station_id", "modis_pixel_id", "period_start"]
EXPECTED_MASTER_ROWS = 1150

S2_COMMON = [
    "s2_Blue_mean", "s2_Green_mean", "s2_Red_mean", "s2_NIR_mean",
    "s2_SWIR1_mean", "s2_SWIR2_mean", "s2_NDVI_mean", "s2_EVI_mean",
    "s2_SAVI_mean", "s2_NDWI_mean", "s2_NDMI_mean",
]
S2_RED_EDGE = [
    "s2_RedEdge1_mean", "s2_RedEdge2_mean", "s2_RedEdge3_mean",
    "s2_NIR_Broad_mean", "s2_NDRE_mean",
]
PRODUCTION_S2 = [name.removeprefix("s2_") for name in (S2_COMMON + S2_RED_EDGE)]
S2_DERIVED = ["s2_Albedo_mean", "s2_FVC_mean"]
S1_R077 = ["r077_VV_dB_mean", "r077_VH_dB_mean", "r077_VV_minus_VH_dB_mean"]
S1_R142 = ["r142_VV_dB_mean", "r142_VH_dB_mean", "r142_VV_minus_VH_dB_mean"]
ERA5_BASE = ["Tair_mean_C", "Tair_max_C", "VPD_mean_kPa", "SolarRad_MJ_m2_day", "Wind_mean_ms"]
ERA5_ADDITIONAL = ["VPD_max_kPa"]
CHIRPS = ["Precip_period_mm", "Precip_prev30d_mm"]
SEASONALITY = ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
LST = ["LST_parent_mean_K"]
CANDIDATES = S2_COMMON + S2_RED_EDGE + S2_DERIVED + S1_R077 + S1_R142 + ERA5_BASE + ERA5_ADDITIONAL + CHIRPS + SEASONALITY + LST

def normalized_keys(table, label):
    result = table.copy()
    missing = set(KEYS) - set(result.columns)
    if missing:
        raise RuntimeError(f"{label} is missing keys: {sorted(missing)}")
    result["station_id"] = result["station_id"].astype(str)
    result["modis_pixel_id"] = result["modis_pixel_id"].astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    result["period_start"] = pd.to_datetime(
        result["period_start"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    if result.duplicated(KEYS).any():
        raise RuntimeError(f"{label} contains duplicate observation keys")
    return result


def build_master_store():
    store = normalized_keys(pd.read_csv(
        FEATURE_STORE, dtype={"station_id": str, "modis_pixel_id": str}
    ), "feature store")
    lst = normalized_keys(pd.read_csv(
        LST_TABLE, dtype={"station_id": str, "modis_pixel_id": str}
    ), "Landsat LST table")
    if len(store) != EXPECTED_MASTER_ROWS or len(lst) != EXPECTED_MASTER_ROWS:
        raise RuntimeError(
            f"Expected 1,150 rows in both inputs; found {len(store)} and {len(lst)}"
        )
    lst_columns = [
        "LST_parent_mean_K", "LST_valid_count_20m", "LST_valid_area_m2",
        "LST_valid_coverage_pct", "landsat_products", "landsat_unique_dates",
        "landsat_acquisition_dates", "landsat_dates_with_valid_lst",
        "l8_products", "l9_products", "l8_unique_dates", "l9_unique_dates",
        "sensors_present", "ST_QA_count_30m", "ST_QA_mean_K_30m",
        "ST_QA_min_K_30m", "ST_QA_max_K_30m", "ST_QA_stddev_K_30m",
        "working_grid_crs", "working_grid_m", "distributed_grid_m",
        "native_thermal_support_m_approx", "resampling_method",
        "composite_method", "footprint_aggregation_method",
    ]
    rename = {
        name: f"landsat_lst_{name}" for name in lst_columns
        if name != "LST_parent_mean_K"
    }
    lst = lst[KEYS + lst_columns].rename(columns=rename)
    master = store.merge(lst, on=KEYS, how="left", validate="one_to_one")
    if len(master) != EXPECTED_MASTER_ROWS or master[KEYS].isna().any().any():
        raise RuntimeError("The Landsat join changed or invalidated master keys")

    production_s2 = normalized_keys(
        pd.read_csv(
            PATHS.operational_s2_table,
            dtype={"station_id": str, "modis_pixel_id": str},
        ),
        "operational Sentinel-2 table",
    )

    if len(production_s2) != EXPECTED_MASTER_ROWS:
        raise RuntimeError(
            "Expected 1,150 rows in operational Sentinel-2; "
            f"found {len(production_s2)}"
        )

    missing_production_s2 = sorted(
        set(PRODUCTION_S2) - set(production_s2.columns)
    )
    if missing_production_s2:
        raise RuntimeError(
            "Operational Sentinel-2 is missing final RF-25 predictors: "
            f"{missing_production_s2}"
        )

    conflicts = sorted(set(PRODUCTION_S2).intersection(master.columns))
    if conflicts:
        raise RuntimeError(
            "Canonical master already contains production-aligned S2 columns: "
            f"{conflicts}"
        )

    master = master.merge(
        production_s2[KEYS + PRODUCTION_S2],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    if len(master) != EXPECTED_MASTER_ROWS:
        raise RuntimeError(
            "Operational Sentinel-2 join changed the master population"
        )

    for predictor in CANDIDATES + PRODUCTION_S2:
        master[predictor] = pd.to_numeric(master[predictor], errors="coerce")
        master.loc[master[predictor] <= -9990, predictor] = np.nan
    target_numeric = [
        "modis_qa_modis_value_valid", "modis_good", "modis_ET_mm_period",
        "ETo_mm_period", "Kc_target", "target_complete", "s2_coverage_pct",
        "year",
    ]
    master[target_numeric] = master[target_numeric].apply(
        pd.to_numeric, errors="coerce"
    )
    return master

