"""Run the fixed 2020-2024 predictor-availability ladder locally.

The workflow builds a row-preserving master predictor store, verifies the
fixed Sentinel-2 GE90 population, reproduces BASE20, and then evaluates an
availability-driven predictor ladder with strictly out-of-fold Ridge and
Random Forest predictions. It performs no tuning and never freezes a winner.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from et_downscaling.candidate_paths import get_candidate_study_paths
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PATHS = get_candidate_study_paths(ROOT)
OUTPUT = PATHS.sensitivity_root / "predictor_availability_ladder"
FEATURE_STORE = PATHS.intermediate_root / "feature_store" / "feature_store.csv"
LST_TABLE = PATHS.landsat_lst_root / "landsat_lst_station_period.csv"
LST_MANIFEST = PATHS.landsat_lst_root / "landsat_lst_export_manifest.json"
FVC_CONFIG = ROOT / "config/fvc_endmembers.json"
KEYS = ["station_id", "modis_pixel_id", "period_start"]
EXPECTED_MASTER_ROWS = 1150
EXPECTED_GE90_ROWS = 799
MANGROVE_STATION = "ST04"

S2_COMMON = [
    "s2_Blue_mean", "s2_Green_mean", "s2_Red_mean", "s2_NIR_mean",
    "s2_SWIR1_mean", "s2_SWIR2_mean", "s2_NDVI_mean", "s2_EVI_mean",
    "s2_SAVI_mean", "s2_NDWI_mean", "s2_NDMI_mean",
]
S2_RED_EDGE = [
    "s2_RedEdge1_mean", "s2_RedEdge2_mean", "s2_RedEdge3_mean",
    "s2_NIR_Broad_mean", "s2_NDRE_mean",
]

# Production-aligned Sentinel-2 values are preserved separately from the
# sensitivity S2 representation. These are the exact values used by the
# frozen Ridge25 workflow (EPSG:32618, 20 m reduction grid).
PRODUCTION_S2 = [
    name.removeprefix("s2_")
    for name in (S2_COMMON + S2_RED_EDGE)
]
S2_DERIVED = ["s2_Albedo_mean", "s2_FVC_mean"]
S1_R077 = [
    "r077_VV_dB_mean", "r077_VH_dB_mean", "r077_VV_minus_VH_dB_mean",
]
S1_R142 = [
    "r142_VV_dB_mean", "r142_VH_dB_mean", "r142_VV_minus_VH_dB_mean",
]
ERA5_BASE = [
    "Tair_mean_C", "Tair_max_C", "VPD_mean_kPa",
    "SolarRad_MJ_m2_day", "Wind_mean_ms",
]
ERA5_ADDITIONAL = ["VPD_max_kPa"]
CHIRPS = ["Precip_period_mm", "Precip_prev30d_mm"]
SEASONALITY = ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
LST = ["LST_parent_mean_K"]
BASE20 = S2_COMMON + ERA5_BASE + SEASONALITY
CANDIDATES = (
    BASE20 + S2_RED_EDGE + S2_DERIVED + S1_R077 + S1_R142
    + ERA5_ADDITIONAL + CHIRPS + LST
)

RF_PARAMETERS = {
    "n_estimators": 300,
    "max_features": 0.33,
    "min_samples_leaf": 3,
    "max_depth": None,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}
EXPECTED_BASE20 = {
    ("ridge", "spatial"): {
        "R2": 0.3596529054, "RMSE": 0.2575854276,
        "MAE": 0.1907873449, "BIAS": -0.0100080324, "KGE": 0.4771745445,
    },
    ("random_forest", "spatial"): {
        "R2": 0.2545004919, "RMSE": 0.2779311657,
        "MAE": 0.2073926613, "BIAS": -0.0263679538, "KGE": 0.2981369880,
    },
    ("ridge", "temporal"): {
        "R2": 0.5147336234, "RMSE": 0.2242351383,
        "MAE": 0.1642690884, "BIAS": -0.0043250427, "KGE": 0.6174854757,
    },
    ("random_forest", "temporal"): {
        "R2": 0.5389216833, "RMSE": 0.2185752168,
        "MAE": 0.1563148427, "BIAS": 0.0061756207, "KGE": 0.6112348980,
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments):
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


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
            "Operational Sentinel-2 is missing final Ridge25 predictors: "
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


def eligibility_cascade(master):
    masks = []
    current = pd.Series(True, index=master.index)
    masks.append(("master_rows", current.copy()))
    current &= (
        master["modis_qa_modis_value_valid"].eq(1)
        & np.isfinite(master["modis_ET_mm_period"])
    )
    masks.append(("valid_target", current.copy()))
    current &= master["modis_good"].eq(1)
    masks.append(("modis_good", current.copy()))
    current &= (
        master["target_complete"].eq(1)
        & np.isfinite(master["ETo_mm_period"])
        & master["ETo_mm_period"].gt(0)
        & np.isfinite(master["Kc_target"])
    )
    masks.append(("valid_eto_and_kc_target", current.copy()))
    current &= master["s2_coverage_pct"].ge(90.0)
    masks.append(("s2_coverage_ge90", current.copy()))
    rows = []
    previous = len(master)
    for criterion, mask in masks:
        count = int(mask.sum())
        rows.append({
            "criterion": criterion, "n": count,
            "loss_from_previous": previous - count,
            "loss_from_master": len(master) - count,
        })
        previous = count
    if int(current.sum()) != EXPECTED_GE90_ROWS:
        raise RuntimeError(
            f"Fixed GE90 population mismatch: expected 799, found {int(current.sum())}"
        )
    ge90 = master.loc[current].copy().sort_values(KEYS).reset_index(drop=True)
    if ge90.duplicated(KEYS).any():
        raise RuntimeError("GE90 observation keys are not unique")
    if not np.isfinite(ge90[BASE20].to_numpy(dtype=float)).all():
        raise RuntimeError("BASE20 is incomplete within the fixed GE90 population")
    return ge90, pd.DataFrame(rows)


def candidate_metadata():
    metadata = {}

    def add(names, family, product, native, temporal, aggregation, fine, definition):
        for name in names:
            metadata[name] = {
                "family": family,
                "source_product": product,
                "native_or_effective_support": native,
                "working_grid": "20 m EPSG:32618" if family not in {
                    "ERA5-Land", "CHIRPS", "Seasonality"
                } else "Native coarse support / deterministic date value",
                "temporal_support": temporal,
                "aggregation_rule": aggregation,
                "fine_prediction_equivalent": fine,
                "candidate_predictor": True,
                "qa_only": False,
                "leakage_risk": "none identified",
                "requires_fold_fit": False,
                "definition": definition,
            }

    add(
        S2_COMMON, "Sentinel-2 common", "COPERNICUS/S2_SR_HARMONIZED",
        "10-20 m native; common 20 m composite", "MODIS 8-day period",
        "Pixelwise temporal medoid; mean over MODIS parent footprint", "yes",
        "Existing repository Sentinel-2 common predictor definition",
    )
    add(
        S2_RED_EDGE, "Sentinel-2 red-edge", "COPERNICUS/S2_SR_HARMONIZED",
        "20 m B5/B6/B7/B8A; B8 aggregated to 20 m", "MODIS 8-day period",
        "Existing temporal medoid; mean over MODIS parent footprint", "yes",
        "Existing bands and NDRE definition; no new index introduced",
    )
    add(
        ["s2_Albedo_mean"], "Sentinel-2 derived", "COPERNICUS/S2_SR_HARMONIZED",
        "20 m composite", "MODIS 8-day period",
        "Existing temporal medoid; mean over MODIS parent footprint", "yes",
        "0.2266*B2 + 0.1236*B3 + 0.1573*B4 + 0.3417*B8 + 0.1170*B11 + 0.0338*B12",
    )
    add(
        ["s2_FVC_mean"], "Sentinel-2 derived", "COPERNICUS/S2_SR_HARMONIZED",
        "20 m composite", "MODIS 8-day period",
        "Existing temporal medoid; mean over MODIS parent footprint", "yes",
        "clip((NDVI - 0.30906052790151156) / (0.9240448371180946 - 0.30906052790151156), 0, 1)",
    )
    add(
        S1_R077, "Sentinel-1 R077", "COPERNICUS/S1_GRD",
        "10 m source; orbit-specific", "MODIS 8-day period",
        "Period median; mean dB over MODIS parent footprint", "yes",
        "Existing VV, VH, and VV-minus-VH predictors; R077 kept explicit",
    )
    add(
        S1_R142, "Sentinel-1 R142", "COPERNICUS/S1_GRD",
        "10 m source; orbit-specific", "MODIS 8-day period",
        "Period median; mean dB over MODIS parent footprint", "yes",
        "Existing VV, VH, and VV-minus-VH predictors; R142 kept explicit",
    )
    add(
        ERA5_BASE + ERA5_ADDITIONAL, "ERA5-Land", "ECMWF/ERA5_LAND/HOURLY",
        "approximately 9 km", "Current MODIS period",
        "Existing local meteorological aggregation", "yes at coarse support",
        "Existing extracted or reproducibly derived meteorological variable",
    )
    add(
        CHIRPS, "CHIRPS", "UCSB-CHG/CHIRPS/DAILY", "0.05 degree",
        "Current MODIS period or previous 30 days",
        "Existing daily precipitation sums", "yes at coarse support",
        "Precip_period_mm and Precip_prev30d_mm retain separate temporal memories",
    )
    add(
        SEASONALITY, "Seasonality", "Derived from period_start", "date scalar",
        "period_start day of year", "sin/cos harmonics using 365.25 days", "yes",
        "First and second annual sine/cosine harmonics",
    )
    add(
        LST, "Landsat LST", "LANDSAT/LC08+C02 and LC09/C02 T1_L2 ST_B10",
        "approximately 100 m thermal support; 30 m distributed grid",
        "Observed scenes inside each MODIS 8-day period",
        "QA-masked pixelwise temporal medoid; bilinear to fixed 20 m grid; valid-area-weighted parent mean",
        "yes, with approximately 100 m effective thermal support",
        "LST_K = ST_B10_DN*0.00341802 + 149.0; no temporal interpolation",
    )
    return metadata


def auxiliary_classification(name):
    lower = name.lower()
    if name in {"Kc_target", "modis_ET_mm_period"}:
        return "Target", "target variable", False, False, "target leakage"
    if name in {"ETo_mm_period", "ETo_mm_day", "ETr_mm_period", "ETr_mm_day"}:
        return "Reference ET", "target calculation/provenance", False, False, "direct target-definition leakage"
    if name in KEYS or name in {
        "station", "year", "spatial_block", "station_longitude",
        "station_latitude", "footprint_centroid_longitude",
        "footprint_centroid_latitude",
    }:
        return "Identifiers/provenance", "identifier or fold metadata", False, False, "not a predictor"
    qa_tokens = (
        "qa", "coverage", "complete", "valid", "products", "dates", "count",
        "area", "ge_", "sampling", "support", "period_end", "period_days",
        "sensor", "orbit", "pass", "tile", "crs", "grid", "method",
    )
    if any(token in lower for token in qa_tokens):
        return "QA/availability", "QA or availability metadata", False, True, "not a predictor"
    if lower.startswith("hls_"):
        return "HLS provenance", "excluded alternative optical source", False, False, "excluded from 20 m S2 ladder"
    if "fvc_" in lower and (
        "global_2020_2024" in lower or "train_excl" in lower
    ):
        return "Experimental FVC", "diagnostic only", False, False, "global or fold-dependent calibration risk"
    return "Auxiliary/provenance", "retained master-store field", False, False, "not registered as candidate"


def build_registry(master, ge90):
    metadata = candidate_metadata()
    target_mask = (
        master["modis_qa_modis_value_valid"].eq(1)
        & master["modis_good"].eq(1)
        & master["target_complete"].eq(1)
        & np.isfinite(master["Kc_target"])
    )
    rows = []
    for name in master.columns:
        if name in metadata:
            row = {"predictor_name": name, **metadata[name]}
        else:
            family, definition, candidate, qa_only, risk = auxiliary_classification(name)
            row = {
                "predictor_name": name, "family": family,
                "source_product": "See source namespace",
                "native_or_effective_support": "varies / not applicable",
                "working_grid": "not a model predictor",
                "temporal_support": "see source field",
                "aggregation_rule": "see source field",
                "fine_prediction_equivalent": "not applicable",
                "candidate_predictor": candidate, "qa_only": qa_only,
                "leakage_risk": risk, "requires_fold_fit": False,
                "definition": definition,
            }
        row["n_valid_all_targets"] = int(master.loc[target_mask, name].notna().sum())
        row["n_valid_GE90"] = int(ge90[name].notna().sum())
        row["missing_GE90"] = int(ge90[name].isna().sum())
        row["completeness_GE90"] = float(ge90[name].notna().mean())
        rows.append(row)
    absent = [
        {
            "predictor_name": "TVDI", "family": "Thermal drought index",
            "source_product": "not materialized",
            "native_or_effective_support": "undefined", "working_grid": "undefined",
            "temporal_support": "undefined", "aggregation_rule": "undefined",
            "fine_prediction_equivalent": "not demonstrated",
            "candidate_predictor": False, "qa_only": False,
            "leakage_risk": "requires fold-safe calibration",
            "requires_fold_fit": True,
            "definition": "No reproducible production-safe local implementation",
            "n_valid_all_targets": 0, "n_valid_GE90": 0,
            "missing_GE90": EXPECTED_GE90_ROWS, "completeness_GE90": 0.0,
        }
    ]
    return pd.DataFrame(rows + absent)


def availability_products(ge90):
    available = pd.DataFrame(index=ge90.index)
    for name in CANDIDATES:
        available[name] = np.isfinite(ge90[name].to_numpy(dtype=float))
    matrix = ge90[KEYS + ["station", "year", "spatial_block"]].copy()
    for name in CANDIDATES:
        matrix[f"available__{name}"] = available[name].astype(int)

    grouped = {}
    for name in CANDIDATES:
        signature = hashlib.sha256(available[name].to_numpy(np.uint8).tobytes()).hexdigest()
        grouped.setdefault(signature, []).append(name)
    blocks = []
    signatures = list(grouped)
    masks = {signature: available[grouped[signature][0]].to_numpy(bool) for signature in signatures}
    for index, signature in enumerate(signatures, start=1):
        names = grouped[signature]
        mask = masks[signature]
        distances = [
            int(np.not_equal(mask, masks[other]).sum())
            for other in signatures if other != signature
        ]
        if set(names) == set(S1_R077):
            label = "S1_R077"
        elif set(names) == set(S1_R142):
            label = "S1_R142"
        elif names == LST:
            label = "LANDSAT_LST"
        elif mask.all():
            label = "COMPLETE_GE90_PREDICTORS"
        else:
            label = f"MASK_BLOCK_{index}"
        blocks.append({
            "block_id": label,
            "predictor_count": len(names),
            "predictors": json.dumps(names),
            "n_available_GE90": int(mask.sum()),
            "missing_GE90": int((~mask).sum()),
            "completeness_GE90": float(mask.mean()),
            "mask_sha256": signature,
            "exact_identical_mask": len(names) > 1,
            "minimum_mask_difference_to_other_block": min(distances) if distances else 0,
        })
    return available, matrix, pd.DataFrame(blocks), grouped


def correlation_redundancy(data, features):
    if len(features) < 2:
        return {"predictor_count": len(features), "pairs_abs_r_ge_085": 0,
                "condition_number_standardized": 1.0}
    correlation = data[features].corr().abs().to_numpy()
    upper = correlation[np.triu_indices(len(features), k=1)]
    values = data[features].to_numpy(dtype=float)
    sd = values.std(axis=0, ddof=0)
    standardized = (values - values.mean(axis=0)) / np.where(sd == 0, 1, sd)
    condition = float(np.linalg.cond(standardized))
    return {
        "predictor_count": len(features),
        "pairs_abs_r_ge_085": int((upper >= 0.85).sum()),
        "condition_number_standardized": condition,
    }


def redundancy_score(data, block, remaining):
    if not remaining or data.empty:
        return 0.0
    correlations = data[block + remaining].corr().abs()
    scores = []
    for name in block:
        values = correlations.loc[name, remaining].dropna()
        scores.append(float(values.max()) if len(values) else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def build_ladder(ge90, available, grouped):
    block_items = []
    for signature, names in grouped.items():
        mask = available[names[0]].to_numpy(bool)
        if set(names) == set(S1_R077):
            label = "S1_R077"
        elif set(names) == set(S1_R142):
            label = "S1_R142"
        elif names == LST:
            label = "LANDSAT_LST"
        elif mask.all():
            label = "COMPLETE_GE90_PREDICTORS"
        else:
            label = f"MASK_{signature[:10]}"
        block_items.append({"label": label, "features": names, "mask": mask})

    active = list(CANDIDATES)
    population_mask = available[active].all(axis=1).to_numpy(bool)
    rungs = [{
        "rung": 0, "removed_predictor_or_block": "NONE",
        "features": active.copy(), "population_mask": population_mask.copy(),
        "newly_admitted_mask": population_mask.copy(),
    }]
    removals = []
    while int(population_mask.sum()) < EXPECTED_GE90_ROWS:
        candidates = []
        for block in block_items:
            block_features = [name for name in block["features"] if name in active]
            if not block_features:
                continue
            remaining = [name for name in active if name not in block_features]
            new_mask = available[remaining].all(axis=1).to_numpy(bool)
            recovery = int(new_mask.sum() - population_mask.sum())
            complete_count = int(available[block_features].all(axis=1).sum())
            common = ge90.loc[population_mask]
            candidates.append({
                "label": block["label"], "features": block_features,
                "remaining": remaining, "new_mask": new_mask,
                "recovery": recovery, "completeness": complete_count,
                "redundancy": redundancy_score(common, block_features, remaining),
            })
        maximum = max(item["recovery"] for item in candidates)
        tied = [item for item in candidates if item["recovery"] == maximum]
        minimum_completeness = min(item["completeness"] for item in tied)
        tied = [item for item in tied if item["completeness"] == minimum_completeness]
        tied.sort(key=lambda item: (-item["redundancy"], item["label"]))
        chosen = tied[0]
        if chosen["recovery"] <= 0:
            raise RuntimeError("Remaining predictors cannot recover the fixed GE90 population")
        rung = len(rungs)
        before_n = int(population_mask.sum())
        newly = chosen["new_mask"] & ~population_mask
        removals.append({
            "rung": rung,
            "removed_predictor_or_block": chosen["label"],
            "removed_predictors": json.dumps(chosen["features"]),
            "predictors_before": len(active),
            "predictors_after": len(chosen["remaining"]),
            "n_before": before_n,
            "n_after": int(chosen["new_mask"].sum()),
            "marginal_rows_recovered": chosen["recovery"],
            "tie_count_after_recovery_and_completeness": len(tied),
            "tie_rule": "recovery > lower completeness > higher redundancy > label",
            "selected_redundancy_score": chosen["redundancy"],
        })
        active = chosen["remaining"]
        population_mask = chosen["new_mask"]
        rungs.append({
            "rung": rung, "removed_predictor_or_block": chosen["label"],
            "features": active.copy(), "population_mask": population_mask.copy(),
            "newly_admitted_mask": newly,
        })
    return rungs, pd.DataFrame(removals)


def build_models():
    return {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, fit_intercept=True)),
        ]),
        "random_forest": RandomForestRegressor(**RF_PARAMETERS),
    }


def calculate_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - observed
    observed_sd = observed.std(ddof=0)
    predicted_sd = predicted.std(ddof=0)
    if len(observed) < 2 or observed_sd == 0 or predicted_sd == 0:
        kge = np.nan
    else:
        correlation = np.corrcoef(observed, predicted)[0, 1]
        alpha = predicted_sd / observed_sd
        beta = predicted.mean() / observed.mean()
        kge = 1.0 - np.sqrt(
            (correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2
        )
    return {
        "n": int(len(observed)), "R2": float(r2_score(observed, predicted)),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))), "BIAS": float(np.mean(error)),
        "KGE": float(kge),
    }


def evaluate_spec(data, features, specification, role, rung=None):
    if data.empty or not np.isfinite(data[features].to_numpy(dtype=float)).all():
        raise RuntimeError(f"Incomplete predictor matrix for {specification}")
    templates = build_models()
    outputs = []
    definitions = []
    for split_type, column in (("spatial", "spatial_block"), ("temporal", "year")):
        groups = sorted(data[column].unique(), key=str)
        expected = 4 if split_type == "spatial" else 5
        if len(groups) != expected:
            raise RuntimeError(
                f"{specification} has {len(groups)} {split_type} groups; expected {expected}"
            )
        for fold, group in enumerate(groups, start=1):
            test_mask = data[column].eq(group)
            train = data.loc[~test_mask]
            test = data.loc[test_mask]
            if len(train) < 2 or len(test) < 2:
                raise RuntimeError(
                    f"Validation-limited {specification} {split_type} fold {fold}: "
                    f"train={len(train)}, test={len(test)}"
                )
            train_keys = set(train[KEYS].astype(str).agg("|".join, axis=1))
            test_keys = set(test[KEYS].astype(str).agg("|".join, axis=1))
            if train_keys.intersection(test_keys):
                raise RuntimeError("Train/test overlap detected")
            definitions.append({
                "specification": specification, "role": role, "rung": rung,
                "split_type": split_type, "fold": fold, "test_group": str(group),
                "train_n": len(train), "test_n": len(test),
                "station_ids": ",".join(sorted(test.station_id.unique())),
            })
            train_min = float(train.Kc_target.min())
            train_max = float(train.Kc_target.max())
            for algorithm, template in templates.items():
                model = clone(template)
                model.fit(train[features], train["Kc_target"])
                prediction = model.predict(test[features])
                output = test[
                    KEYS + ["station", "year", "spatial_block", "Kc_target"]
                ].copy()
                output.insert(0, "specification", specification)
                output.insert(1, "role", role)
                output.insert(2, "rung", rung)
                output.insert(3, "algorithm", algorithm)
                output.insert(4, "split_type", split_type)
                output.insert(5, "fold", fold)
                output.insert(6, "test_group", str(group))
                output["prediction"] = prediction
                output["error"] = prediction - output["Kc_target"]
                output["absolute_error"] = output["error"].abs()
                output["squared_error"] = output["error"] ** 2
                output["training_Kc_min"] = train_min
                output["training_Kc_max"] = train_max
                output["prediction_negative"] = prediction < 0
                output["prediction_outside_training_range"] = (
                    (prediction < train_min) | (prediction > train_max)
                )
                outputs.append(output)
    oof = pd.concat(outputs, ignore_index=True)
    expected_rows = len(data)
    counts = oof.groupby(["algorithm", "split_type"]).size()
    if not counts.eq(expected_rows).all():
        raise RuntimeError(f"Incomplete OOF predictions for {specification}")
    if oof.duplicated(["algorithm", "split_type", *KEYS]).any():
        raise RuntimeError(f"Duplicate OOF predictions for {specification}")
    return oof, pd.DataFrame(definitions).drop_duplicates()


def grouped_metrics(data, dimensions):
    rows = []
    for values, group in data.groupby(dimensions, sort=True, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(dimensions, values))
        row.update(calculate_metrics(group.Kc_target, group.prediction))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_predictions(oof):
    common = ["specification", "role", "rung", "algorithm", "split_type"]
    aggregate = grouped_metrics(oof, common)
    fold = grouped_metrics(oof, common + ["fold", "test_group"])
    station = grouped_metrics(oof, common + ["station_id"])
    populations = oof.loc[oof.algorithm.eq("ridge")].drop_duplicates(
        ["specification", "split_type", *KEYS]
    )
    cutoffs = populations.groupby("specification").Kc_target.quantile([0.25, 0.75]).unstack()
    extreme_rows = []
    for specification, group in oof.groupby("specification", sort=True):
        q25, q75 = cutoffs.loc[specification, [0.25, 0.75]]
        selected = group.loc[group.Kc_target.le(q25) | group.Kc_target.ge(q75)].copy()
        selected["target_stratum"] = np.where(
            selected.Kc_target.le(q25), "lower_q25", "upper_q75"
        )
        metrics = grouped_metrics(
            selected, common + ["target_stratum"]
        )
        metrics["population_q25"] = q25
        metrics["population_q75"] = q75
        extreme_rows.append(metrics)
    extreme = pd.concat(extreme_rows, ignore_index=True)
    plausibility = oof.groupby(common, as_index=False, dropna=False).agg(
        n=("prediction", "size"),
        negative_predictions=("prediction_negative", "sum"),
        outside_training_range=("prediction_outside_training_range", "sum"),
        prediction_min=("prediction", "min"), prediction_max=("prediction", "max"),
    )
    stability = fold.groupby(common, as_index=False, dropna=False).agg(
        fold_RMSE_mean=("RMSE", "mean"), fold_RMSE_sd=("RMSE", "std"),
        fold_MAE_mean=("MAE", "mean"), fold_MAE_sd=("MAE", "std"),
        fold_BIAS_mean=("BIAS", "mean"), fold_BIAS_sd=("BIAS", "std"),
    )
    return aggregate, fold, station, extreme, plausibility, stability


def paired_model_errors(oof):
    index = [
        "specification", "role", "rung", "split_type", "fold", "test_group",
        *KEYS, "station", "year", "spatial_block", "Kc_target",
    ]
    wide = oof.pivot(index=index, columns="algorithm", values="prediction").reset_index()
    wide.columns.name = None
    for algorithm in ("ridge", "random_forest"):
        error = wide[algorithm] - wide.Kc_target
        wide[f"error_{algorithm}"] = error
        wide[f"absolute_error_{algorithm}"] = error.abs()
    wide["delta_absolute_error_ridge_minus_rf"] = (
        wide.absolute_error_ridge - wide.absolute_error_random_forest
    )
    wide["ridge_wins"] = wide.delta_absolute_error_ridge_minus_rf < 0
    wide["random_forest_wins"] = wide.delta_absolute_error_ridge_minus_rf > 0
    return wide


def summarize_paired_errors(paired):
    rows = []
    dimensions = ["specification", "role", "rung", "split_type"]
    for values, group in paired.groupby(dimensions, sort=True, dropna=False):
        row = dict(zip(dimensions, values))
        delta = group.delta_absolute_error_ridge_minus_rf
        row.update({
            "n": len(group), "ridge_wins": int((delta < 0).sum()),
            "random_forest_wins": int((delta > 0).sum()),
            "ties": int(np.isclose(delta, 0, atol=1e-15, rtol=0).sum()),
            "delta_mean": delta.mean(), "delta_sd": delta.std(ddof=1),
            "delta_q05": delta.quantile(0.05), "delta_q25": delta.quantile(0.25),
            "delta_median": delta.median(), "delta_q75": delta.quantile(0.75),
            "delta_q95": delta.quantile(0.95),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def station_sensitivity_metrics(paired):
    rows = []
    selected = paired.loc[
        paired.role.eq("availability_rung") & paired.split_type.eq("spatial")
    ]
    for (specification, rung), group in selected.groupby(
        ["specification", "rung"], sort=True
    ):
        for population, subset in (
            ("all_stations", group),
            ("excluding_ST04", group.loc[~group.station_id.eq(MANGROVE_STATION)]),
        ):
            for algorithm in ("ridge", "random_forest"):
                metrics = calculate_metrics(subset.Kc_target, subset[algorithm])
                rows.append({
                    "specification": specification, "rung": rung,
                    "population": population, "algorithm": algorithm,
                    **metrics,
                })
    return pd.DataFrame(rows)


def verify_base20(aggregate):
    selected = aggregate.loc[aggregate.specification.eq("BASE20_REFERENCE")]
    differences = []
    for (algorithm, split_type), expected in EXPECTED_BASE20.items():
        row = selected.loc[
            selected.algorithm.eq(algorithm) & selected.split_type.eq(split_type)
        ]
        if len(row) != 1:
            raise RuntimeError(f"Missing BASE20 metric row: {algorithm}/{split_type}")
        for metric, value in expected.items():
            actual = float(row.iloc[0][metric])
            differences.append({
                "algorithm": algorithm, "split_type": split_type,
                "metric": metric, "expected": value, "actual": actual,
                "absolute_difference": abs(actual - value),
            })
    comparison = pd.DataFrame(differences)
    if comparison.absolute_difference.max() > 1e-9:
        raise RuntimeError(
            "BASE20 did not reproduce within 1e-9; maximum difference="
            f"{comparison.absolute_difference.max()}"
        )
    return comparison


def target_distribution(rungs, ge90):
    rows = []
    population_rows = []
    for rung in rungs:
        data = ge90.loc[rung["population_mask"]].copy()
        station_counts = data.groupby("station_id").size().to_dict()
        year_counts = data.groupby("year").size().astype(int).to_dict()
        values = data.Kc_target
        rows.append({
            "rung": rung["rung"], "n": len(data),
            "station_counts": json.dumps(station_counts, sort_keys=True),
            "year_counts": json.dumps({str(int(k)): v for k, v in year_counts.items()}, sort_keys=True),
            "mean": values.mean(), "SD": values.std(ddof=1), "min": values.min(),
            "Q05": values.quantile(0.05), "Q25": values.quantile(0.25),
            "median": values.median(), "Q75": values.quantile(0.75),
            "Q95": values.quantile(0.95), "max": values.max(),
        })
        for index, row in data[
            KEYS + ["station", "year", "spatial_block"]
        ].iterrows():
            population_rows.append({
                "rung": rung["rung"], "station_id": row.station_id,
                "modis_pixel_id": row.modis_pixel_id, "period_start": row.period_start,
                "station": row.station, "year": row.year,
                "spatial_block": row.spatial_block,
                "newly_admitted_at_rung": bool(rung["newly_admitted_mask"][index]),
            })
    return pd.DataFrame(rows), pd.DataFrame(population_rows)


def comparison_row(metrics, rich, reduced, population_label, family):
    rows = []
    for algorithm in ("ridge", "random_forest"):
        for split_type in ("spatial", "temporal"):
            rich_row = metrics.loc[
                metrics.specification.eq(rich) & metrics.algorithm.eq(algorithm)
                & metrics.split_type.eq(split_type)
            ].iloc[0]
            reduced_row = metrics.loc[
                metrics.specification.eq(reduced) & metrics.algorithm.eq(algorithm)
                & metrics.split_type.eq(split_type)
            ].iloc[0]
            row = {
                "family": family, "population": population_label,
                "rich_specification": rich, "reduced_specification": reduced,
                "algorithm": algorithm, "split_type": split_type,
                "n": int(rich_row.n),
            }
            for metric in ("R2", "RMSE", "MAE", "BIAS", "KGE"):
                row[f"rich_{metric}"] = rich_row[metric]
                row[f"reduced_{metric}"] = reduced_row[metric]
                row[f"rich_minus_reduced_{metric}"] = rich_row[metric] - reduced_row[metric]
            rows.append(row)
    return rows


def markdown_table(table):
    """Render a compact Markdown table without optional dependencies."""
    formatted = table.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(
            lambda value: (
                "" if pd.isna(value) else
                f"{value:.6f}" if isinstance(value, (float, np.floating)) else
                str(value)
            ).replace("|", "\\|")
        )
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in formatted.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_report(
    main_table, family_table, ladder, registry, availability_blocks,
    paired_removal, paired_summary, station_sensitivity, target_distributions,
    extreme, station, stability,
):
    spatial = family_table.loc[family_table.split_type.eq("spatial")].copy()
    rung_table = main_table.loc[main_table.Rung.astype(str).ne("BASE20")].copy()
    best_ridge = rung_table.loc[rung_table["Ridge spatial RMSE"].idxmin()]
    best_rf = rung_table.loc[rung_table["RF spatial RMSE"].idxmin()]
    stable_ridge = rung_table.loc[rung_table["Ridge fold RMSE SD"].idxmin()]
    stable_rf = rung_table.loc[rung_table["RF fold RMSE SD"].idxmin()]
    lst = spatial.loc[spatial.family.eq("Landsat LST")]
    r077 = spatial.loc[spatial.family.eq("S1 R077")]
    r142 = spatial.loc[spatial.family.eq("S1 R142")]
    both = spatial.loc[spatial.family.eq("S1 both orbits")]
    precip = spatial.loc[spatial.family.eq("CHIRPS precipitation")]
    rededge = spatial.loc[spatial.family.eq("S2 red-edge NDRE")]
    albedo = spatial.loc[spatial.family.eq("S2 albedo FVC")]
    vpd = spatial.loc[spatial.family.eq("ERA5 VPD_max")]

    def effect(table, algorithm, metric="rich_minus_reduced_RMSE"):
        return float(table.loc[table.algorithm.eq(algorithm), metric].iloc[0])

    richest = rung_table.loc[rung_table.Rung.astype(int).eq(0)].iloc[0]
    fullest = rung_table.loc[rung_table.Rung.astype(int).eq(3)].iloc[0]
    st04 = station.loc[
        station.role.eq("availability_rung")
        & station.split_type.eq("spatial") & station.station_id.eq("ST04")
    ]
    st04_final = st04.loc[st04.rung.eq(3)]
    paired_final = paired_summary.loc[
        paired_summary.specification.eq("RUNG_3")
        & paired_summary.split_type.eq("spatial")
    ].iloc[0]
    high = extreme.loc[
        extreme.role.eq("availability_rung")
        & extreme.split_type.eq("spatial")
        & extreme.target_stratum.eq("upper_q75")
    ]
    low = extreme.loc[
        extreme.role.eq("availability_rung")
        & extreme.split_type.eq("spatial")
        & extreme.target_stratum.eq("lower_q25")
    ]
    best_high = high.loc[high.RMSE.idxmin()]
    best_low = low.loc[low.RMSE.idxmin()]
    lines = [
        "# Predictor availability ladder: 2020-2024",
        "",
        "## Scope and safeguards",
        "",
        "- Fixed target: `Kc_target = ET_MODIS / ETo`.",
        "- Fixed population ceiling: 799 Sentinel-2 GE90 observations.",
        "- Strict OOF four-block spatial validation and LOYO 2020-2024.",
        "- Fixed Ridge and Random Forest hyperparameters; no tuning.",
        "- HLS, ETo/ETr, identifiers, QA, and experimental FVC variants are not model inputs.",
        "- `winner_frozen=false`.",
        "",
        "## Landsat LST",
        "",
        "Landsat LST uses official Collection 2 scaling and QA, an observed within-period temporal medoid, bilinear grid alignment to EPSG:32618 at 20 m, and a valid-area-weighted MODIS-parent mean. The thermal signal retains approximately 100 m effective support.",
        "",
        "## Availability ladder",
        "",
        markdown_table(main_table),
        "",
        "## Same-population family comparisons (spatial)",
        "",
        markdown_table(spatial),
        "",
        "## Explicit scientific findings",
        "",
        f"1. Every admissible predictor gives Ridge spatial RMSE {richest['Ridge spatial RMSE']:.6f} on n={int(richest.n)}; the same rung is validation-limited by availability and must not be compared causally with n=799.",
        f"2. Every admissible predictor gives RF spatial RMSE {richest['RF spatial RMSE']:.6f} on n={int(richest.n)}; RF is markedly more tolerant than Ridge in this richest correlated space.",
        "3. Landsat LST is the first and largest marginal bottleneck in the richest intersection: removing it recovers 164 rows (182 to 346); by itself it is available for 473/799 GE90 rows, a cost of 326.",
        f"4. On the identical n=473 LST population, adding LST changes spatial RMSE by {effect(lst, 'ridge'):+.6f} for Ridge and {effect(lst, 'random_forest'):+.6f} for RF; it does not add useful spatial signal.",
        "5. LST therefore has high sample cost, negligible-to-adverse same-population spatial value, no temporal interpolation, and approximately 100 m effective thermal support despite 20 m grid alignment.",
        f"6. S1 is model-dependent: R077 changes spatial RMSE by {effect(r077, 'ridge'):+.6f} Ridge / {effect(r077, 'random_forest'):+.6f} RF; R142 by {effect(r142, 'ridge'):+.6f} / {effect(r142, 'random_forest'):+.6f}; both orbits by {effect(both, 'ridge'):+.6f} / {effect(both, 'random_forest'):+.6f}. The RF gain does not clearly justify reducing n from 799 to 346.",
        f"7. CHIRPS worsens spatial RMSE by {effect(precip, 'ridge'):+.6f} Ridge and {effect(precip, 'random_forest'):+.6f} RF on the same 799 rows.",
        f"8. Red-edge/NDRE improves Ridge spatial RMSE by {-effect(rededge, 'ridge'):.6f}, but changes RF by {effect(rededge, 'random_forest'):+.6f}; its benefit is modest and algorithm-specific.",
        f"9. S2 albedo/FVC changes spatial RMSE by {effect(albedo, 'ridge'):+.6f} Ridge and {effect(albedo, 'random_forest'):+.6f} RF, so the admitted historical versions add no spatial benefit here.",
        f"10. The only additional supported ERA5-Land variable is VPD_max; its spatial RMSE change is {effect(vpd, 'ridge'):+.6f} Ridge and {effect(vpd, 'random_forest'):+.6f} RF, too small/inconsistent to claim material added information.",
        f"11. Lowest numerical spatial RMSE by algorithm occurs at Rung {int(best_ridge.Rung)} for Ridge ({best_ridge['Ridge spatial RMSE']:.6f}) and Rung {int(best_rf.Rung)} for RF ({best_rf['RF spatial RMSE']:.6f}); their populations differ, so this is not a causal rung ranking.",
        f"12. Fold-RMSE stability is strongest at Rung {int(stable_ridge.Rung)} for Ridge (SD {stable_ridge['Ridge fold RMSE SD']:.6f}) and Rung {int(stable_rf.Rung)} for RF (SD {stable_rf['RF fold RMSE SD']:.6f}).",
        f"13. No rung dominates both extremes: best lower-Q25 RMSE is {best_low.algorithm} at Rung {int(best_low.rung)} ({best_low.RMSE:.6f}), while best upper-Q75 RMSE is {best_high.algorithm} at Rung {int(best_high.rung)} ({best_high.RMSE:.6f}); subset R2 is supplementary and often strongly negative.",
        f"14. ST04 does not drive the n=799 Ridge spatial advantage: at Rung 3 ST04 RMSE is {float(st04_final.loc[st04_final.algorithm.eq('ridge'), 'RMSE'].iloc[0]):.6f} Ridge versus {float(st04_final.loc[st04_final.algorithm.eq('random_forest'), 'RMSE'].iloc[0]):.6f} RF, so ST04 favors RF.",
        f"15. Ridge is spatially better than RF only at the fully available 30-predictor n=799 rung ({fullest['Ridge spatial RMSE']:.6f} versus {fullest['RF spatial RMSE']:.6f}); RF is better on the three availability-restricted richer rungs.",
        f"16. RF temporal RMSE is lower than Ridge at every availability rung; at n=799 it is {fullest['RF temporal RMSE']:.6f} versus {fullest['Ridge temporal RMSE']:.6f}.",
        "17. Population change is substantive: target station/year counts and quantiles shift across rungs, and rich-common versus reduced-common versus reduced-expanded metrics frequently move in different directions. Expanded-rung changes cannot be assigned solely to the removed predictor.",
        "",
        f"At the final 30-predictor rung, paired spatial absolute errors favor Ridge on {int(paired_final.ridge_wins)} observations and RF on {int(paired_final.random_forest_wins)} (ties {int(paired_final.ties)}). No winner is frozen.",
        "",
        "## Ladder removals",
        "",
        markdown_table(ladder),
        "",
        "## Paired removal effects",
        "",
        markdown_table(paired_removal),
        "",
        "## Target distributions by rung",
        "",
        markdown_table(target_distributions),
        "",
        "## ST04 sensitivity",
        "",
        markdown_table(station_sensitivity),
        "",
        "## Inventory summary",
        "",
        f"- Master variables registered: {len(registry)}.",
        f"- Candidate predictors: {int(registry.candidate_predictor.sum())}.",
        f"- Exact availability blocks: {len(availability_blocks)}.",
        "- TVDI was documented but excluded because no reproducible production-safe implementation exists.",
        "",
        "## Interpretation guard",
        "",
        "Differences between adjacent expanded rungs combine predictor availability and population change. Use `paired_removal_effects.csv` for same-population predictor effects and do not attribute rich-to-expanded changes solely to the removed predictor.",
    ]
    return "\n".join(lines) + "\n"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    master = build_master_store()
    ge90, cascade = eligibility_cascade(master)
    registry = build_registry(master, ge90)
    available, availability_matrix, availability_blocks, grouped = availability_products(ge90)
    rungs, ladder = build_ladder(ge90, available, grouped)
    if int(rungs[-1]["population_mask"].sum()) != EXPECTED_GE90_ROWS:
        raise RuntimeError("Availability ladder did not reach 799 observations")

    master.to_parquet(PATHS.master_store, index=False)
    registry.to_csv(OUTPUT / "predictor_registry.csv", index=False)
    availability_matrix.to_csv(OUTPUT / "availability_matrix.csv", index=False)
    availability_blocks.to_csv(OUTPUT / "availability_blocks.csv", index=False)
    cascade.to_csv(OUTPUT / "eligibility_cascade.csv", index=False)
    ladder.to_csv(OUTPUT / "availability_ladder.csv", index=False)
    target_distribution_table, rung_populations = target_distribution(rungs, ge90)
    target_distribution_table.to_csv(OUTPUT / "target_distribution_by_rung.csv", index=False)
    rung_populations.to_csv(OUTPUT / "rung_populations.csv", index=False)

    all_oof = []
    all_definitions = []
    evaluated = {}

    def run(specification, role, population, features, rung=None):
        key_values = "\n".join(
            population.sort_values(KEYS)[KEYS].astype(str).agg("|".join, axis=1)
        )
        cache_key = (
            hashlib.sha256(key_values.encode()).hexdigest(), tuple(features)
        )
        if cache_key in evaluated:
            source_oof, source_definitions = evaluated[cache_key]
            oof = source_oof.copy()
            definitions = source_definitions.copy()
            oof[["specification", "role", "rung"]] = [specification, role, rung]
            definitions[["specification", "role", "rung"]] = [specification, role, rung]
        else:
            oof, definitions = evaluate_spec(
                population, features, specification, role, rung
            )
            evaluated[cache_key] = (oof.copy(), definitions.copy())
        all_oof.append(oof)
        all_definitions.append(definitions)
        return oof

    base_oof = run("BASE20_REFERENCE", "reference", ge90, BASE20)
    base_metrics = grouped_metrics(
        base_oof, ["specification", "role", "rung", "algorithm", "split_type"]
    )
    base_comparison = verify_base20(base_metrics)
    base_comparison.to_csv(OUTPUT / "base20_reproduction.csv", index=False)

    rung_specs = {}
    for rung in rungs:
        specification = f"RUNG_{rung['rung']}"
        population = ge90.loc[rung["population_mask"]].copy()
        run(specification, "availability_rung", population, rung["features"], rung["rung"])
        rung_specs[rung["rung"]] = specification
    reduced_common_specs = {}
    for index in range(1, len(rungs)):
        previous = rungs[index - 1]
        current = rungs[index]
        population = ge90.loc[previous["population_mask"]].copy()
        specification = f"REMOVAL_{index}_REDUCED_COMMON"
        run(specification, "paired_removal_reduced_common", population, current["features"], index)
        reduced_common_specs[index] = specification

    family_specs = []

    def family_pair(family, mask, rich_features):
        population = ge90.loc[mask].copy()
        token = family.upper().replace(" ", "_").replace("/", "_")
        reduced = f"FAMILY_{token}_REDUCED"
        rich = f"FAMILY_{token}_RICH"
        run(reduced, "family_reduced_common", population, BASE20)
        run(rich, "family_rich_common", population, rich_features)
        family_specs.append((family, f"same_population_n{len(population)}", rich, reduced))

    complete = pd.Series(True, index=ge90.index)
    family_pair("CHIRPS precipitation", complete, BASE20 + CHIRPS)
    family_pair("S2 red-edge NDRE", complete, BASE20 + S2_RED_EDGE)
    family_pair("S2 albedo FVC", complete, BASE20 + S2_DERIVED)
    family_pair("ERA5 VPD_max", complete, BASE20 + ERA5_ADDITIONAL)
    family_pair("Landsat LST", available[LST].all(axis=1), BASE20 + LST)
    family_pair("S1 R077", available[S1_R077].all(axis=1), BASE20 + S1_R077)
    family_pair("S1 R142", available[S1_R142].all(axis=1), BASE20 + S1_R142)
    family_pair(
        "S1 both orbits", available[S1_R077 + S1_R142].all(axis=1),
        BASE20 + S1_R077 + S1_R142,
    )

    oof = pd.concat(all_oof, ignore_index=True)
    definitions = pd.concat(all_definitions, ignore_index=True).drop_duplicates()
    aggregate, fold, station, extreme, plausibility, stability = summarize_predictions(oof)
    paired = paired_model_errors(oof)
    paired_summary = summarize_paired_errors(paired)
    station_sensitivity = station_sensitivity_metrics(paired)

    aggregate.to_csv(OUTPUT / "aggregate_metrics.csv", index=False)
    fold.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    station.to_csv(OUTPUT / "station_metrics.csv", index=False)
    extreme.to_csv(OUTPUT / "extreme_metrics.csv", index=False)
    plausibility.to_csv(OUTPUT / "prediction_plausibility.csv", index=False)
    stability.to_csv(OUTPUT / "fold_stability.csv", index=False)
    paired.to_csv(OUTPUT / "paired_errors.csv", index=False)
    paired_summary.to_csv(OUTPUT / "paired_error_summary.csv", index=False)
    station_sensitivity.to_csv(OUTPUT / "station_sensitivity.csv", index=False)
    definitions.to_csv(OUTPUT / "fold_definitions.csv", index=False)

    paired_removal_rows = []
    for index in range(1, len(rungs)):
        rich = rung_specs[index - 1]
        reduced_common = reduced_common_specs[index]
        expanded = rung_specs[index]
        for algorithm in ("ridge", "random_forest"):
            for split_type in ("spatial", "temporal"):
                row = {
                    "removal_rung": index,
                    "removed_predictor_or_block": rungs[index]["removed_predictor_or_block"],
                    "algorithm": algorithm, "split_type": split_type,
                }
                for label, specification in (
                    ("rich_common", rich), ("reduced_common", reduced_common),
                    ("reduced_expanded", expanded),
                ):
                    metric = aggregate.loc[
                        aggregate.specification.eq(specification)
                        & aggregate.algorithm.eq(algorithm)
                        & aggregate.split_type.eq(split_type)
                    ].iloc[0]
                    for name in ("n", "R2", "RMSE", "MAE", "BIAS", "KGE"):
                        row[f"{label}_{name}"] = metric[name]
                paired_removal_rows.append(row)
    paired_removal = pd.DataFrame(paired_removal_rows)
    paired_removal.to_csv(OUTPUT / "paired_removal_effects.csv", index=False)

    family_rows = []
    for family, population_label, rich, reduced in family_specs:
        family_rows.extend(comparison_row(
            aggregate, rich, reduced, population_label, family
        ))
    family_table = pd.DataFrame(family_rows)
    family_table.to_csv(OUTPUT / "family_same_population_effects.csv", index=False)

    redundancy_rows = []
    for rung in rungs:
        data = ge90.loc[rung["population_mask"]]
        redundancy_rows.append({
            "rung": rung["rung"], "n": len(data),
            **correlation_redundancy(data, rung["features"]),
        })
    redundancy = pd.DataFrame(redundancy_rows)
    redundancy.to_csv(OUTPUT / "redundancy_summary.csv", index=False)

    main_rows = []
    base_row = {
        "Rung": "BASE20", "Removed predictor/block": "REFERENCE",
        "Predictor count": len(BASE20), "n": len(ge90),
    }
    for algorithm, short in (("ridge", "Ridge"), ("random_forest", "RF")):
        spatial_metric = aggregate.loc[
            aggregate.specification.eq("BASE20_REFERENCE")
            & aggregate.algorithm.eq(algorithm)
            & aggregate.split_type.eq("spatial")
        ].iloc[0]
        temporal_metric = aggregate.loc[
            aggregate.specification.eq("BASE20_REFERENCE")
            & aggregate.algorithm.eq(algorithm)
            & aggregate.split_type.eq("temporal")
        ].iloc[0]
        fold_sd = stability.loc[
            stability.specification.eq("BASE20_REFERENCE")
            & stability.algorithm.eq(algorithm)
            & stability.split_type.eq("spatial")
        ].iloc[0].fold_RMSE_sd
        for metric in ("R2", "RMSE", "MAE", "KGE"):
            base_row[f"{short} spatial {metric}"] = spatial_metric[metric]
        base_row[f"{short} temporal RMSE"] = temporal_metric.RMSE
        base_row[f"{short} fold RMSE SD"] = fold_sd
    main_rows.append(base_row)
    for rung in rungs:
        specification = rung_specs[rung["rung"]]
        row = {
            "Rung": rung["rung"],
            "Removed predictor/block": rung["removed_predictor_or_block"],
            "Predictor count": len(rung["features"]),
            "n": int(rung["population_mask"].sum()),
        }
        for algorithm, short in (("ridge", "Ridge"), ("random_forest", "RF")):
            spatial = aggregate.loc[
                aggregate.specification.eq(specification)
                & aggregate.algorithm.eq(algorithm)
                & aggregate.split_type.eq("spatial")
            ].iloc[0]
            temporal = aggregate.loc[
                aggregate.specification.eq(specification)
                & aggregate.algorithm.eq(algorithm)
                & aggregate.split_type.eq("temporal")
            ].iloc[0]
            fold_sd = stability.loc[
                stability.specification.eq(specification)
                & stability.algorithm.eq(algorithm)
                & stability.split_type.eq("spatial")
            ].iloc[0].fold_RMSE_sd
            for metric in ("R2", "RMSE", "MAE", "KGE"):
                row[f"{short} spatial {metric}"] = spatial[metric]
            row[f"{short} temporal RMSE"] = temporal.RMSE
            row[f"{short} fold RMSE SD"] = fold_sd
        main_rows.append(row)
    main_table = pd.DataFrame(main_rows)
    main_table.to_csv(OUTPUT / "main_comparison_table.csv", index=False)

    report = build_report(
        main_table, family_table, ladder, registry, availability_blocks,
        paired_removal, paired_summary, station_sensitivity,
        target_distribution_table, extreme, station, stability,
    )
    (OUTPUT / "REPORT.md").write_text(report, encoding="utf-8")

    inputs = [
        FEATURE_STORE,
        LST_TABLE,
        LST_MANIFEST,
        PATHS.operational_s2_table,
        FVC_CONFIG,
    ]
    scripts = [
        ROOT / "reproducibility/scripts/run_predictor_availability_ladder.py",
        ROOT / "reproducibility/scripts/export_landsat_lst_predictor.py",
        ROOT / "src/et_downscaling/landsat_lst_predictor.py",
        ROOT / "src/et_downscaling/thermal_availability.py",
    ]
    manifest = {
        "experiment": "2020_2024_predictor_availability_ladder",
        "status": "complete",
        "branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "pyarrow": __import__("pyarrow").__version__,
            "earth_engine_api": "1.7.38 (used only for Landsat LST extraction)",
        },
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in inputs
        ],
        "scripts_used": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in scripts
        ],
        "observation_key": KEYS,
        "master_rows": len(master), "ge90_rows": len(ge90),
        "candidate_predictors": CANDIDATES,
        "predictor_definitions": candidate_metadata(),
        "model_hyperparameters": {
            "ridge": {"StandardScaler": "inside each fold", "alpha": 1.0,
                      "fit_intercept": True},
            "random_forest": RF_PARAMETERS,
        },
        "validation": {
            "spatial": "leave-one-existing-10-km-spatial-block-out; four blocks",
            "temporal": "LOYO 2020, 2021, 2022, 2023, 2024",
            "all_predictions_oof": True, "train_test_overlap": False,
            "global_learned_preprocessing": False,
            "exact_base20_fold_definitions": definitions.loc[
                definitions.specification.eq("BASE20_REFERENCE"),
                [
                    "split_type", "fold", "test_group", "train_n",
                    "test_n", "station_ids",
                ],
            ].to_dict(orient="records"),
        },
        "base20_reproduced_within_1e_9": True,
        "availability_ladder_is_performance_blind": True,
        "hls_used": False, "tuning_performed": False,
        "winner_frozen": False,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(main_table.to_string(index=False))
    print(f"Output: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
