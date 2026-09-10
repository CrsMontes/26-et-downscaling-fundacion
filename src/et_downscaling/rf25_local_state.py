"""Local RF-25 prediction, weighted AOA and quality-control state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aoa_rf25 import RFAOAParameters, score_rf_weighted_aoa
from .rf25 import RF25_MODEL_FEATURES


RF25_USABLE_SUPPORT_FRACTION = 0.90
RF25_RECONCILIATION_TOLERANCE_MM = 0.01


@dataclass(frozen=True)
class LocalRF25State:
    kc_raw: np.ndarray
    dissimilarity_index: np.ndarray
    local_point_density: np.ndarray
    stack_valid: np.ndarray
    aoa_inside: np.ndarray
    usable: np.ndarray


def score_local_rf25(
    predictor_cube: np.ndarray,
    model,
    aoa_parameters: RFAOAParameters,
) -> LocalRF25State:
    """Predict Kc and derive final fine-grid AOA/usable support."""
    cube = np.asarray(predictor_cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError("RF-25 predictor cube must be three-dimensional.")
    feature_count = len(RF25_MODEL_FEATURES)
    if cube.shape[2] != feature_count:
        raise ValueError("RF-25 predictor cube feature count differs from model schema.")
    if tuple(aoa_parameters.feature_names) != tuple(RF25_MODEL_FEATURES):
        raise ValueError("AOA predictor schema differs from RF-25.")

    rows, columns, _ = cube.shape
    flat = cube.reshape(-1, feature_count)
    stack_valid_flat = np.isfinite(flat).all(axis=1)

    kc_flat = np.full(flat.shape[0], np.nan, dtype=float)
    di_flat = np.full(flat.shape[0], np.nan, dtype=float)
    lpd_flat = np.zeros(flat.shape[0], dtype=np.int32)
    aoa_flat = np.zeros(flat.shape[0], dtype=bool)

    if stack_valid_flat.any():
        valid_predictors = flat[stack_valid_flat]
        frame = pd.DataFrame(valid_predictors, columns=RF25_MODEL_FEATURES)
        prediction = np.asarray(model.predict(frame), dtype=float)
        if prediction.shape != (valid_predictors.shape[0],):
            raise RuntimeError("RF-25 prediction count differs from valid pixels.")
        di, inside, lpd = score_rf_weighted_aoa(valid_predictors, aoa_parameters)
        kc_flat[stack_valid_flat] = prediction
        di_flat[stack_valid_flat] = di
        lpd_flat[stack_valid_flat] = lpd
        aoa_flat[stack_valid_flat] = inside

    kc_raw = kc_flat.reshape(rows, columns)
    dissimilarity = di_flat.reshape(rows, columns)
    lpd = lpd_flat.reshape(rows, columns)
    stack_valid = stack_valid_flat.reshape(rows, columns)
    aoa_inside = aoa_flat.reshape(rows, columns)
    usable = stack_valid & aoa_inside & np.isfinite(kc_raw) & (kc_raw >= 0)

    return LocalRF25State(
        kc_raw=kc_raw,
        dissimilarity_index=dissimilarity,
        local_point_density=lpd,
        stack_valid=stack_valid,
        aoa_inside=aoa_inside,
        usable=usable,
    )
