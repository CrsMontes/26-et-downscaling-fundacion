"""Final Random-Forest Kc model for the Fundación ET workflow.

The accepted feature set is frozen to the same 25 predictors used in the
Virtual Station comparison. Candidate predictors may be downloaded and
archived, but they are not silently added to this model.
"""

from __future__ import annotations

from collections.abc import Iterable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


RF25_MODEL_NAME = "rf25_virtual10_ge90"
RF25_MODEL_FILENAME = "rf25_virtual10_ge90.joblib"
RF25_AOA_FILENAME = "rf25_weighted_aoa.joblib"
RF25_METADATA_FILENAME = "rf25_model_metadata.json"

RF25_OPTICAL_FEATURES = [
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

RF25_METEOROLOGICAL_FEATURES = [
    "Tair_mean_C",
    "Tair_max_C",
    "VPD_mean_kPa",
    "SolarRad_MJ_m2_day",
    "Wind_mean_ms",
]

RF25_HARMONIC_FEATURES = [
    "doy_sin1",
    "doy_cos1",
    "doy_sin2",
    "doy_cos2",
]

RF25_MODEL_FEATURES = (
    RF25_OPTICAL_FEATURES
    + RF25_METEOROLOGICAL_FEATURES
    + RF25_HARMONIC_FEATURES
)

RF25_PARAMETERS = {
    "n_estimators": 300,
    "max_features": 0.33,
    "min_samples_leaf": 3,
    "max_depth": None,
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}


def build_rf25_model() -> RandomForestRegressor:
    """Build the fixed final RF-25 estimator; no tuning is performed."""
    return RandomForestRegressor(**RF25_PARAMETERS)


def validate_rf25_model(
    model: RandomForestRegressor,
    feature_names: Iterable[str] = RF25_MODEL_FEATURES,
) -> None:
    """Validate estimator type, frozen hyperparameters and fitted schema."""
    if not isinstance(model, RandomForestRegressor):
        raise TypeError("RF-25 production model must be RandomForestRegressor.")

    params = model.get_params(deep=False)
    for name, expected in RF25_PARAMETERS.items():
        if params.get(name) != expected:
            raise ValueError(
                f"RF-25 parameter {name!r} changed: {params.get(name)!r} != {expected!r}."
            )

    if not hasattr(model, "estimators_"):
        raise ValueError("RF-25 estimator is not fitted.")

    expected_count = len(tuple(feature_names))
    if getattr(model, "n_features_in_", None) != expected_count:
        raise ValueError(
            "RF-25 fitted feature count differs from the frozen 25-predictor schema."
        )

    if hasattr(model, "feature_names_in_"):
        actual = tuple(str(value) for value in model.feature_names_in_)
        expected = tuple(feature_names)
        if actual != expected:
            raise ValueError("RF-25 fitted feature names differ from the frozen schema.")


def rf25_model_signature(model: RandomForestRegressor) -> str:
    """Return a deterministic hash of the fitted sklearn estimator state."""
    validate_rf25_model(model)
    return str(joblib.hash(model, hash_name="sha1"))
