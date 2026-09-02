"""Unweighted Area of Applicability for Ridge-25.

Implements the L2 Dissimilarity Index logic of CAST/Meyer & Pebesma
using the same spatial cross-validation groups as model validation.
All 25 standardized predictors receive equal weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from .ridge25 import RIDGE25_MODEL_FEATURES


@dataclass(frozen=True)
class AOAParameters:
    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    training_scaled: np.ndarray
    mean_training_distance: float
    threshold: float
    training_di: np.ndarray


def build_unweighted_aoa(
    population: pd.DataFrame,
    group_column: str = "spatial_block",
) -> AOAParameters:
    """Build the unweighted DI/AOA specification from training data."""
    features = tuple(RIDGE25_MODEL_FEATURES)

    matrix = population[list(features)].to_numpy(dtype=float)

    if not np.isfinite(matrix).all():
        raise ValueError(
            "AOA training predictors contain non-finite values."
        )

    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=1)

    if np.any(scales <= 0):
        raise ValueError(
            "AOA predictors must have non-zero variance."
        )

    scaled = (matrix - means) / scales

    distances = pairwise_distances(
        scaled,
        metric="euclidean",
    )

    np.fill_diagonal(
        distances,
        np.nan,
    )

    mean_training_distance = float(
        np.nanmean(
            np.nanmean(
                distances,
                axis=1,
            )
        )
    )

    groups = population[
        group_column
    ].astype(str).to_numpy()

    training_di = np.full(
        len(population),
        np.nan,
        dtype=float,
    )

    for index in range(len(population)):
        candidate_mask = (
            groups != groups[index]
        )

        if not candidate_mask.any():
            raise ValueError(
                "Each AOA validation point requires "
                "training observations from another group."
            )

        nearest_distance = float(
            np.nanmin(
                distances[
                    index,
                    candidate_mask,
                ]
            )
        )

        training_di[index] = (
            nearest_distance
            / mean_training_distance
        )

    q1 = float(
        np.quantile(
            training_di,
            0.25,
        )
    )
    q3 = float(
        np.quantile(
            training_di,
            0.75,
        )
    )
    iqr = q3 - q1

    threshold = min(
        float(np.max(training_di)),
        q3 + 1.5 * iqr,
    )

    return AOAParameters(
        feature_names=features,
        means=means,
        scales=scales,
        training_scaled=scaled,
        mean_training_distance=(
            mean_training_distance
        ),
        threshold=threshold,
        training_di=training_di,
    )


def score_unweighted_aoa(
    predictors: np.ndarray,
    parameters: AOAParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return DI and inside/outside AOA for new predictor rows."""
    matrix = np.asarray(
        predictors,
        dtype=float,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "AOA predictor input must be two-dimensional."
        )

    if matrix.shape[1] != len(
        parameters.feature_names
    ):
        raise ValueError(
            "AOA predictor count differs from training schema."
        )

    di = np.full(
        matrix.shape[0],
        np.nan,
        dtype=float,
    )
    inside = np.zeros(
        matrix.shape[0],
        dtype=bool,
    )

    valid = np.isfinite(
        matrix
    ).all(axis=1)

    if not valid.any():
        return di, inside

    scaled = (
        matrix[valid]
        - parameters.means
    ) / parameters.scales

    nearest = NearestNeighbors(
        n_neighbors=1,
        algorithm="brute",
        metric="euclidean",
    )
    nearest.fit(
        parameters.training_scaled
    )

    distances, _ = (
        nearest.kneighbors(
            scaled,
            return_distance=True,
        )
    )

    valid_di = (
        distances[:, 0]
        / parameters.mean_training_distance
    )

    di[valid] = valid_di
    inside[valid] = (
        valid_di
        <= parameters.threshold
    )

    return di, inside
