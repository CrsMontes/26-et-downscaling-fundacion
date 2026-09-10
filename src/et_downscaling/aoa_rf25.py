"""Model-weighted DI/AOA and Local Point Density for the final RF-25 model.

The implementation follows the predictor-space logic of Meyer & Pebesma
(2021) and current CAST ``trainDI`` behavior:

* predictors are centered and scaled using training mean/sample SD;
* standardized predictors are multiplied by model-derived importance weights;
* Euclidean distance is used in the weighted predictor space;
* DI is nearest-training distance divided by the mean of all pairwise training
  distances;
* training DI uses the same held-out spatial blocks as spatial CV;
* AOA threshold is Q3 + 1.5*IQR, capped by the maximum finite training DI;
* LPD counts training points within the AOA-distance radius for each new point.

RF importance is estimated with permutation importance on the fitted final
training population. Negative permutation scores are clipped to zero before
weighting. The raw and normalized importance values are preserved for audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from .rf25 import RF25_MODEL_FEATURES, validate_rf25_model


@dataclass(frozen=True)
class RFAOAParameters:
    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    raw_importance: np.ndarray
    weights: np.ndarray
    training_weighted: np.ndarray
    mean_training_distance: float
    threshold: float
    training_di: np.ndarray
    training_lpd: np.ndarray
    group_column: str
    method: str = "weighted_L2_spatial_CV_CAST_style"


def _validate_population(population: pd.DataFrame, group_column: str) -> np.ndarray:
    missing = sorted(set(RF25_MODEL_FEATURES + [group_column]) - set(population.columns))
    if missing:
        raise ValueError("AOA population is missing columns: " + ", ".join(missing))
    matrix = population[RF25_MODEL_FEATURES].to_numpy(dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("AOA training predictors must be a finite two-dimensional matrix.")
    if population[group_column].astype(str).nunique() < 2:
        raise ValueError("AOA requires at least two spatial CV groups.")
    return matrix


def _permutation_weights(
    population: pd.DataFrame,
    model,
    target_column: str,
    repeats: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if target_column not in population.columns:
        raise ValueError(f"AOA population is missing target column {target_column!r}.")
    validate_rf25_model(model)
    y = pd.to_numeric(population[target_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("AOA target contains non-finite values.")

    importance = permutation_importance(
        model,
        population[RF25_MODEL_FEATURES],
        y,
        scoring="neg_mean_squared_error",
        n_repeats=int(repeats),
        random_state=int(random_state),
        n_jobs=-1,
    )
    raw = np.asarray(importance.importances_mean, dtype=float)
    if raw.shape != (len(RF25_MODEL_FEATURES),) or not np.isfinite(raw).all():
        raise RuntimeError("RF permutation importance has an invalid shape or non-finite values.")

    positive = np.clip(raw, 0.0, None)
    maximum = float(np.max(positive))
    if maximum <= 0:
        raise RuntimeError("RF permutation importance contains no positive predictor weight.")

    # A common multiplicative constant cancels after DI normalization. Scaling
    # the maximum to 1 keeps distance calculations numerically stable.
    weights = positive / maximum
    return raw, weights


def build_rf_weighted_aoa(
    population: pd.DataFrame,
    model,
    *,
    target_column: str = "Kc_target",
    group_column: str = "spatial_block",
    importance_repeats: int = 20,
    random_state: int = 42,
) -> RFAOAParameters:
    """Build RF-weighted DI/AOA parameters from the final training population."""
    matrix = _validate_population(population, group_column)
    raw_importance, weights = _permutation_weights(
        population,
        model,
        target_column,
        importance_repeats,
        random_state,
    )

    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=1)
    if not np.isfinite(means).all() or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("AOA scaling parameters are invalid.")

    weighted = ((matrix - means) / scales) * weights
    distances = pairwise_distances(weighted, metric="euclidean")
    np.fill_diagonal(distances, np.nan)
    mean_training_distance = float(np.nanmean(distances))
    if not np.isfinite(mean_training_distance) or mean_training_distance <= 0:
        raise RuntimeError("Mean pairwise training distance is invalid.")

    groups = population[group_column].astype(str).to_numpy()
    training_di = np.full(len(population), np.nan, dtype=float)

    for index in range(len(population)):
        candidate = groups != groups[index]
        if not candidate.any():
            raise ValueError("Each training point must have reference points outside its CV group.")
        nearest = float(np.nanmin(distances[index, candidate]))
        training_di[index] = nearest / mean_training_distance

    finite_di = training_di[np.isfinite(training_di)]
    if finite_di.size == 0:
        raise RuntimeError("No finite training DI values were generated.")
    q1 = float(np.quantile(finite_di, 0.25))
    q3 = float(np.quantile(finite_di, 0.75))
    threshold = min(float(np.max(finite_di)), q3 + 1.5 * (q3 - q1))

    # CV-aware training LPD: count reference points available to the held-out
    # point within the same absolute distance radius used by the AOA.
    radius = threshold * mean_training_distance
    training_lpd = np.zeros(len(population), dtype=np.int32)
    for index in range(len(population)):
        candidate = groups != groups[index]
        training_lpd[index] = int(np.sum(distances[index, candidate] <= radius))

    return RFAOAParameters(
        feature_names=tuple(RF25_MODEL_FEATURES),
        means=means,
        scales=scales,
        raw_importance=raw_importance,
        weights=weights,
        training_weighted=weighted,
        mean_training_distance=mean_training_distance,
        threshold=float(threshold),
        training_di=training_di,
        training_lpd=training_lpd,
        group_column=str(group_column),
    )


def score_rf_weighted_aoa(
    predictors: np.ndarray,
    parameters: RFAOAParameters,
    *,
    lpd_chunk_size: int = 25000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return DI, inside-AOA mask and LPD for new predictor rows."""
    matrix = np.asarray(predictors, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("AOA predictor input must be two-dimensional.")
    if matrix.shape[1] != len(parameters.feature_names):
        raise ValueError("AOA predictor count differs from training schema.")

    n_rows = matrix.shape[0]
    di = np.full(n_rows, np.nan, dtype=float)
    inside = np.zeros(n_rows, dtype=bool)
    lpd = np.zeros(n_rows, dtype=np.int32)
    valid = np.isfinite(matrix).all(axis=1)
    if not valid.any():
        return di, inside, lpd

    scaled = ((matrix[valid] - parameters.means) / parameters.scales) * parameters.weights
    nearest = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean", n_jobs=-1)
    nearest.fit(parameters.training_weighted)
    distances, _ = nearest.kneighbors(scaled, return_distance=True)
    valid_di = distances[:, 0] / parameters.mean_training_distance
    di[valid] = valid_di
    inside[valid] = valid_di <= parameters.threshold

    radius = parameters.threshold * parameters.mean_training_distance
    valid_positions = np.flatnonzero(valid)
    chunk_size = max(1, int(lpd_chunk_size))
    for start in range(0, scaled.shape[0], chunk_size):
        stop = min(start + chunk_size, scaled.shape[0])
        neighbors = nearest.radius_neighbors(
            scaled[start:stop],
            radius=radius,
            return_distance=False,
        )
        counts = np.fromiter((len(item) for item in neighbors), dtype=np.int32)
        lpd[valid_positions[start:stop]] = counts

    return di, inside, lpd


def importance_table(parameters: RFAOAParameters) -> pd.DataFrame:
    """Return the final AOA weight table for provenance."""
    return pd.DataFrame(
        {
            "feature": list(parameters.feature_names),
            "permutation_importance_mse": parameters.raw_importance,
            "aoa_weight": parameters.weights,
        }
    ).sort_values("aoa_weight", ascending=False, kind="stable").reset_index(drop=True)
