"""Final Random-Forest Kc model for the Fundación ET workflow.

The accepted feature set is frozen to the same 25 predictors used in the
Virtual Station comparison. Candidate predictors may be downloaded and
archived, but they are not silently added to this model.
"""

from __future__ import annotations

from collections.abc import Iterable

import hashlib
import json
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


def _signature_array(value) -> np.ndarray:
    """Canonical numerical values only; never expose structured-array padding."""
    array = np.asarray(value)
    dtypes = {"b": "u1", "i": "<i8", "u": "<u8", "f": "<f8"}
    if array.dtype.kind not in dtypes:
        raise TypeError(f"Unsupported scientific-state dtype: {array.dtype}")
    return np.ascontiguousarray(array, dtype=dtypes[array.dtype.kind])


def rf25_model_signature(model: RandomForestRegressor) -> str:
    """SHA-256 of explicit fitted RF state, independent of pickle and node padding.

    Artifact-file SHA-256 is a separate provenance identifier. This function
    reads the estimator without changing its arrays, parameters or predictions.
    """
    validate_rf25_model(model)
    digest = hashlib.sha256()

    def feed(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Unsupported scientific-state parameter type: {type(value).__name__}")

    def metadata(label: str, value) -> None:
        feed(json.dumps([label, value], sort_keys=True, separators=(",", ":"),
                        allow_nan=False, default=json_default).encode("utf-8"))

    def array(label: str, value) -> None:
        canonical = _signature_array(value)
        metadata(label, {"dtype": canonical.dtype.str, "shape": canonical.shape})
        feed(canonical.tobytes(order="C"))

    metadata("signature_format", "rf25_scientific_model_state_v1")
    metadata("estimator_type", [type(model).__module__, type(model).__qualname__])
    metadata("parameters", model.get_params(deep=False))
    metadata("feature_names", list(getattr(model, "feature_names_in_", RF25_MODEL_FEATURES)))
    metadata("n_features_in", int(model.n_features_in_))
    metadata("n_outputs", int(model.n_outputs_))
    metadata("tree_count", len(model.estimators_))
    for index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        state = tree.__getstate__()
        metadata("tree", {"index": index,
                          "type": [type(estimator).__module__, type(estimator).__qualname__],
                          "parameters": estimator.get_params(deep=False),
                          "n_features_in": int(estimator.n_features_in_),
                          "n_outputs": int(estimator.n_outputs_),
                          "max_features": int(estimator.max_features_),
                          "tree_n_features": int(tree.n_features),
                          "tree_n_outputs": int(tree.n_outputs),
                          "node_count": int(state["node_count"]),
                          "max_depth": int(state["max_depth"])})
        array("n_classes", tree.n_classes)
        # Include every named field, including missing_go_to_left. A field view
        # is copied to a contiguous numerical array before hashing its bytes.
        for name in sorted(state["nodes"].dtype.names):
            array("nodes/" + name, state["nodes"][name])
        array("values", state["values"])
    return digest.hexdigest()
