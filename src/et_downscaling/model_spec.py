"""Shared specification for the final Sentinel-2 Kc model.

This module is the single source of truth for predictor order, temporal
harmonics, and Random Forest hyperparameters. Training and spatial production
must import the same definitions from here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


RANDOM_STATE = 42
SPATIAL_BLOCK_SIZE_KM = 10.0
TARGET_COLUMN = "Kc_target"
CANDIDATE_COLUMN = "training_candidate_source_ge_90"
SELECTED_MODEL_NAME = "rf_common"
PRODUCTION_MODEL_FILENAME = "rf_kc_s2_production_ge90.joblib"


COMMON_SATELLITE_FEATURES = [
    "Blue_mean",
    "Green_mean",
    "Red_mean",
    "NIR_mean",
    "SWIR1_mean",
    "SWIR2_mean",
    "NDVI_mean",
    "EVI_mean",
    "SAVI_mean",
    "NDWI_mean",
    "NDMI_mean",
    "VV_dB_mean",
    "VH_dB_mean",
    "VV_minus_VH_dB_mean",
]

S2_ADDITIONAL_FEATURES = [
    "RedEdge1_mean",
    "RedEdge2_mean",
    "RedEdge3_mean",
    "NDRE_mean",
    "Albedo_mean",
    "FVC_mean",
]

METEOROLOGICAL_FEATURES = [
    "Tair_mean_C",
    "Tair_max_C",
    "VPD_mean_kPa",
    "SolarRad_MJ_m2_day",
    "Wind_mean_ms",
    "Precip_period_mm",
    "Precip_prev30d_mm",
]

HARMONIC_FEATURES = [
    "doy_sin1",
    "doy_cos1",
    "doy_sin2",
    "doy_cos2",
]

COMMON_MODEL_FEATURES = (
    COMMON_SATELLITE_FEATURES
    + METEOROLOGICAL_FEATURES
    + HARMONIC_FEATURES
)

FULL_S2_MODEL_FEATURES = (
    COMMON_SATELLITE_FEATURES
    + S2_ADDITIONAL_FEATURES
    + METEOROLOGICAL_FEATURES
    + HARMONIC_FEATURES
)

RF_PARAMETERS = {
    "n_estimators": 300,
    "max_features": 0.33,
    "min_samples_leaf": 3,
    "max_depth": None,
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def add_doy_harmonics(data: pd.DataFrame) -> pd.DataFrame:
    """Add two annual harmonic pairs using MODIS period start."""
    result = data.copy()
    dates = pd.to_datetime(result["period_start"], errors="raise")
    doy = dates.dt.dayofyear.to_numpy(dtype=float)

    harmonic_data = {}
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * doy / 365.25
        harmonic_data[f"doy_sin{harmonic}"] = np.sin(angle)
        harmonic_data[f"doy_cos{harmonic}"] = np.cos(angle)

    return pd.concat(
        [result, pd.DataFrame(harmonic_data, index=result.index)],
        axis=1,
    )


def build_random_forest() -> RandomForestRegressor:
    """Build the pre-specified Random Forest regressor."""
    return RandomForestRegressor(**RF_PARAMETERS)
