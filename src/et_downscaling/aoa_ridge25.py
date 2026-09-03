"""Unweighted Area of Applicability for Ridge-25.

Implements the L2 Dissimilarity Index logic of CAST/Meyer & Pebesma
using the same spatial cross-validation groups as model validation.
All 25 standardized predictors receive equal weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import ee
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


def build_ee_unweighted_aoa(
    model_stack: ee.Image,
    parameters: AOAParameters,
) -> dict[str, ee.Image]:
    """Build unweighted Ridge-25 DI and AOA images in Earth Engine.

    The calculation uses the parameters derived by
    :func:`build_unweighted_aoa` and compares every valid pixel with the
    complete standardized training population. Matrix algebra keeps the
    Earth Engine graph compact while preserving the Euclidean-distance
    definition used by :func:`score_unweighted_aoa`.
    """
    features = tuple(parameters.feature_names)

    if features != tuple(RIDGE25_MODEL_FEATURES):
        raise ValueError(
            "AOA predictors differ from the Ridge-25 model specification."
        )

    feature_count = len(features)
    means = np.asarray(parameters.means, dtype=float)
    scales = np.asarray(parameters.scales, dtype=float)
    reference = np.asarray(parameters.training_scaled, dtype=float)

    if means.shape != (feature_count,):
        raise ValueError(
            "AOA mean count differs from the Ridge-25 predictor count."
        )
    if scales.shape != (feature_count,):
        raise ValueError(
            "AOA scale count differs from the Ridge-25 predictor count."
        )
    if reference.ndim != 2 or reference.shape[1] != feature_count:
        raise ValueError(
            "AOA training reference predictor count is inconsistent."
        )
    if reference.shape[0] == 0:
        raise ValueError("AOA training reference is empty.")
    if not np.isfinite(means).all():
        raise ValueError("AOA means contain non-finite values.")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("AOA scales contain invalid values.")
    if not np.isfinite(reference).all():
        raise ValueError(
            "AOA training reference contains non-finite values."
        )

    mean_training_distance = float(parameters.mean_training_distance)
    threshold = float(parameters.threshold)
    if (
        not np.isfinite(mean_training_distance)
        or mean_training_distance <= 0
    ):
        raise ValueError("AOA mean training distance is invalid.")
    if not np.isfinite(threshold):
        raise ValueError("AOA threshold is invalid.")

    predictor_image = (
        ee.Image(model_stack)
        .select(list(features))
        .toDouble()
    )
    mean_image = (
        ee.Image.constant(means.tolist())
        .rename(list(features))
        .toDouble()
    )
    scale_image = (
        ee.Image.constant(scales.tolist())
        .rename(list(features))
        .toDouble()
    )
    standardized = (
        predictor_image
        .subtract(mean_image)
        .divide(scale_image)
    )

    predictor_column = (
        standardized
        .toArray()
        .toArray(1)
    )
    reference_matrix = ee.Image.constant(
        ee.Array(reference.tolist())
    )
    dot_product = reference_matrix.matrixMultiply(
        predictor_column
    )

    reference_norm_squared = np.sum(
        reference ** 2,
        axis=1,
    ).reshape(-1, 1)
    reference_norm_image = ee.Image.constant(
        ee.Array(reference_norm_squared.tolist())
    )
    predictor_norm_squared = (
        standardized
        .pow(2)
        .reduce(ee.Reducer.sum())
        .toArray()
        .toArray(1)
        .arrayRepeat(0, int(reference.shape[0]))
    )

    # Squared Euclidean distance for each training row:
    # ||r - x||^2 = ||r||^2 + ||x||^2 - 2(r dot x).
    distance_squared = (
        reference_norm_image
        .add(predictor_norm_squared)
        .subtract(dot_product.multiply(2))
    )
    nearest_squared = (
        distance_squared
        .arrayReduce(ee.Reducer.min(), [0])
        .arrayGet(ee.Image.constant([0, 0]))
        .max(0)
    )
    nearest_distance = (
        nearest_squared
        .sqrt()
        .rename("nearest_training_distance")
        .toDouble()
    )
    dissimilarity_index = (
        nearest_distance
        .divide(mean_training_distance)
        .rename("DI")
        .toDouble()
    )
    aoa = (
        dissimilarity_index
        .lte(threshold)
        .rename("AOA")
        .toByte()
    )

    return {
        "nearest_training_distance": nearest_distance,
        "di": dissimilarity_index,
        "aoa": aoa,
    }


def build_unweighted_aoa_images(
    model_stack: ee.Image,
    parameters: AOAParameters,
) -> dict[str, ee.Image]:
    """Compatibility name for the Ridge-25 Earth Engine AOA builder."""
    return build_ee_unweighted_aoa(model_stack, parameters)
