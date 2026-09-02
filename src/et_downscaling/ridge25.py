"""Final Ridge-25 Kc model specification and Earth Engine transfer utilities.

This module is intentionally separate from the legacy Random Forest model
specification so the historical workflow remains reproducible while the
accepted Ridge-25 production path is developed and audited.
"""

from __future__ import annotations

from collections.abc import Iterable

import ee
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RIDGE25_MODEL_NAME = "ridge25_s2_rededge_ge90"
RIDGE25_MODEL_FILENAME = "ridge_kc_s2_rededge25_ge90.joblib"
RIDGE25_SPEC_FILENAME = "ridge25_model_spec.json"
RIDGE25_AOA_FILENAME = "ridge25_aoa_spec.json"

RIDGE25_OPTICAL_FEATURES = [
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
    "RedEdge1_mean",
    "RedEdge2_mean",
    "RedEdge3_mean",
    "NIR_Broad_mean",
    "NDRE_mean",
]

RIDGE25_METEOROLOGICAL_FEATURES = [
    "Tair_mean_C",
    "Tair_max_C",
    "VPD_mean_kPa",
    "SolarRad_MJ_m2_day",
    "Wind_mean_ms",
]

RIDGE25_HARMONIC_FEATURES = [
    "doy_sin1",
    "doy_cos1",
    "doy_sin2",
    "doy_cos2",
]

RIDGE25_MODEL_FEATURES = (
    RIDGE25_OPTICAL_FEATURES
    + RIDGE25_METEOROLOGICAL_FEATURES
    + RIDGE25_HARMONIC_FEATURES
)


def build_ridge25_model() -> Pipeline:
    """Build the fixed final Ridge-25 sklearn pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, fit_intercept=True)),
        ]
    )


def extract_ridge25_parameters(
    model: Pipeline,
    feature_names: Iterable[str] = RIDGE25_MODEL_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Validate a fitted Ridge-25 pipeline and return transfer parameters."""
    feature_names = list(feature_names)

    if not isinstance(model, Pipeline):
        raise TypeError("Ridge-25 production model must be a sklearn Pipeline.")

    if "scaler" not in model.named_steps or "regressor" not in model.named_steps:
        raise ValueError("Ridge-25 pipeline must contain scaler and regressor steps.")

    scaler = model.named_steps["scaler"]
    regressor = model.named_steps["regressor"]

    if not isinstance(scaler, StandardScaler):
        raise TypeError("Ridge-25 scaler must be sklearn StandardScaler.")
    if not isinstance(regressor, Ridge):
        raise TypeError("Ridge-25 regressor must be sklearn Ridge.")
    if not hasattr(scaler, "mean_") or not hasattr(regressor, "coef_"):
        raise ValueError("Ridge-25 pipeline is not fitted.")

    means = np.asarray(scaler.mean_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)
    coefficients = np.asarray(regressor.coef_, dtype=float)
    intercept = float(regressor.intercept_)

    expected = len(feature_names)
    for name, values in (
        ("means", means),
        ("scales", scales),
        ("coefficients", coefficients),
    ):
        if len(values) != expected:
            raise ValueError(
                f"Ridge-25 {name} count differs from feature count: "
                f"{len(values)} != {expected}."
            )

    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("Ridge-25 StandardScaler contains invalid scales.")
    if not np.isfinite(means).all():
        raise ValueError("Ridge-25 StandardScaler contains invalid means.")
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept):
        raise ValueError("Ridge-25 coefficients contain non-finite values.")

    return means, scales, coefficients, intercept


def build_ee_ridge25_prediction(
    model_stack: ee.Image,
    model: Pipeline,
    feature_names: Iterable[str] = RIDGE25_MODEL_FEATURES,
    output_name: str = "Kc_raw",
) -> ee.Image:
    """Evaluate the exact StandardScaler + Ridge equation in Earth Engine."""
    feature_names = list(feature_names)
    means, scales, coefficients, intercept = extract_ridge25_parameters(
        model,
        feature_names,
    )

    stack = ee.Image(model_stack).select(feature_names).toDouble()
    prediction = ee.Image.constant(intercept).toDouble()

    for feature, mean, scale, coefficient in zip(
        feature_names,
        means,
        scales,
        coefficients,
        strict=True,
    ):
        term = (
            stack.select(feature)
            .subtract(float(mean))
            .divide(float(scale))
            .multiply(float(coefficient))
        )
        prediction = prediction.add(term)

    return prediction.rename(output_name).toDouble()
