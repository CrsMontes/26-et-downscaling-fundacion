"""Area-of-applicability utilities for the final Kc model.

The implementation follows the dissimilarity-index framework of
Meyer and Pebesma (2021).

Training DI is calculated using observations outside each observation's
spatial validation group. Prediction DI is evaluated against the complete
final training population.

AOA describes predictor-space support and is not an independent validation
of evapotranspiration accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from .model_spec import COMMON_MODEL_FEATURES


AOA_THRESHOLD_METHOD = "observed_non_outlier_max"


def build_aoa_spec(
    training: pd.DataFrame,
    model,
    group_column: str = "spatial_block",
) -> dict:
    """Build the AOA specification for the final RF model."""

    features = list(COMMON_MODEL_FEATURES)

    required_columns = features + [group_column]
    missing_columns = [
        column
        for column in required_columns
        if column not in training.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing AOA columns: {missing_columns}"
        )

    if training[features].isna().any().any():
        raise ValueError(
            "AOA training predictors contain missing values."
        )

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "The production model does not expose feature importances."
        )

    weights = np.asarray(
        model.feature_importances_,
        dtype=float,
    )

    if len(weights) != len(features):
        raise ValueError(
            "AOA weight count differs from predictor count."
        )

    if np.any(weights < 0) or np.all(weights == 0):
        raise ValueError(
            "Invalid AOA predictor weights."
        )

    values = training[features].to_numpy(
        dtype=float
    )

    means = values.mean(axis=0)

    standard_deviations = values.std(
        axis=0,
        ddof=1,
    )

    zero_variance = standard_deviations == 0

    if zero_variance.any():
        bad_features = np.asarray(features)[
            zero_variance
        ]

        raise ValueError(
            "Zero-variance AOA predictors: "
            + ", ".join(bad_features)
        )

    standardized = (
        values - means
    ) / standard_deviations

    weighted = standardized * weights

    distance_matrix = pairwise_distances(
        weighted,
        metric="euclidean",
    )

    number_rows = len(training)

    upper_triangle = np.triu_indices(
        number_rows,
        k=1,
    )

    mean_training_distance = float(
        distance_matrix[
            upper_triangle
        ].mean()
    )

    if (
        not np.isfinite(mean_training_distance)
        or mean_training_distance <= 0
    ):
        raise ValueError(
            "Invalid mean AOA training distance."
        )

    groups = (
        training[group_column]
        .astype(str)
        .to_numpy()
    )

    train_di = np.full(
        number_rows,
        np.nan,
        dtype=float,
    )

    for row_index in range(number_rows):

        admissible = (
            groups != groups[row_index]
        )

        if not np.any(admissible):
            raise ValueError(
                "AOA requires observations outside "
                "every spatial validation group."
            )

        nearest_distance = float(
            distance_matrix[
                row_index,
                admissible,
            ].min()
        )

        train_di[row_index] = (
            nearest_distance
            / mean_training_distance
        )

    q1 = float(
        np.quantile(train_di, 0.25)
    )

    q3 = float(
        np.quantile(train_di, 0.75)
    )

    iqr = q3 - q1

    upper_fence = (
        q3 + 1.5 * iqr
    )

    non_outlier_di = train_di[
        train_di <= upper_fence
    ]

    if len(non_outlier_di) == 0:
        raise ValueError(
            "No non-outlier training DI values."
        )

    threshold = float(
        non_outlier_di.max()
    )

    cast_reference_threshold = float(
        min(
            upper_fence,
            train_di.max(),
        )
    )

    return {
        "method": (
            "Meyer and Pebesma (2021) "
            "dissimilarity index"
        ),
        "threshold_method": (
            AOA_THRESHOLD_METHOD
        ),
        "features": features,
        "training_rows": int(
            number_rows
        ),
        "spatial_groups": int(
            len(np.unique(groups))
        ),
        "means": means.tolist(),
        "standard_deviations": (
            standard_deviations.tolist()
        ),
        "weights": weights.tolist(),
        "weighted_training_reference": (
            weighted.tolist()
        ),
        "mean_training_distance": (
            mean_training_distance
        ),
        "train_di": train_di.tolist(),
        "train_di_summary": {
            "minimum": float(
                train_di.min()
            ),
            "median": float(
                np.median(train_di)
            ),
            "q1": q1,
            "q3": q3,
            "maximum": float(
                train_di.max()
            ),
            "iqr": float(iqr),
            "upper_fence": float(
                upper_fence
            ),
            "outlier_count": int(
                np.sum(
                    train_di > upper_fence
                )
            ),
        },
        "threshold": threshold,
        "cast_reference_threshold": (
            cast_reference_threshold
        ),
    }


def save_aoa_spec(
    specification: dict,
    path: Path,
) -> None:
    """Save an AOA specification as JSON."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            specification,
            file,
            indent=2,
        )


def load_aoa_spec(
    path: Path,
) -> dict:
    """Load an AOA specification from JSON."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"AOA specification not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def calculate_prediction_di(
    predictors: pd.DataFrame,
    specification: dict,
) -> pd.DataFrame:
    """Calculate DI and AOA for new predictor observations.

    New observations are compared with the complete final training
    reference population. Spatial folds are used only when deriving the
    training DI threshold, not when evaluating new predictions.
    """

    features = list(specification["features"])

    missing_columns = [
        feature
        for feature in features
        if feature not in predictors.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing AOA predictors: {missing_columns}"
        )

    if predictors[features].isna().any().any():
        raise ValueError(
            "Prediction AOA predictors contain missing values."
        )

    means = np.asarray(
        specification["means"],
        dtype=float,
    )

    standard_deviations = np.asarray(
        specification["standard_deviations"],
        dtype=float,
    )

    weights = np.asarray(
        specification["weights"],
        dtype=float,
    )

    training_reference = np.asarray(
        specification["weighted_training_reference"],
        dtype=float,
    )

    values = predictors[features].to_numpy(
        dtype=float
    )

    weighted = (
        (values - means)
        / standard_deviations
    ) * weights

    distances = pairwise_distances(
        weighted,
        training_reference,
        metric="euclidean",
    )

    nearest_distance = distances.min(axis=1)

    dissimilarity_index = (
        nearest_distance
        / float(specification["mean_training_distance"])
    )

    threshold = float(
        specification["threshold"]
    )

    result = pd.DataFrame(
        index=predictors.index,
    )

    result["nearest_training_distance"] = (
        nearest_distance
    )

    result["DI"] = dissimilarity_index

    result["AOA"] = (
        dissimilarity_index <= threshold
    ).astype("uint8")

    return result

def build_aoa_images(
    model_stack: ee.Image,
    specification: dict,
) -> dict[str, ee.Image]:
    """Calculate DI and AOA for a predictor image stack in Earth Engine.

    Prediction DI is calculated against the complete final training
    reference population. The implementation uses matrix algebra to avoid
    constructing one distance image per training observation.
    """

    features = list(specification["features"])

    if features != list(COMMON_MODEL_FEATURES):
        raise ValueError(
            "AOA predictors differ from the final model specification."
        )

    means = np.asarray(
        specification["means"],
        dtype=float,
    )

    standard_deviations = np.asarray(
        specification["standard_deviations"],
        dtype=float,
    )

    weights = np.asarray(
        specification["weights"],
        dtype=float,
    )

    reference = np.asarray(
        specification["weighted_training_reference"],
        dtype=float,
    )

    if reference.ndim != 2:
        raise ValueError(
            "AOA training reference must be a two-dimensional matrix."
        )

    if reference.shape[1] != len(features):
        raise ValueError(
            "AOA training reference predictor count is inconsistent."
        )

    reference_rows = int(reference.shape[0])

    reference_norm_squared = (
        np.sum(
            reference ** 2,
            axis=1,
        )
        .reshape(-1, 1)
    )

    predictor_image = (
        ee.Image(model_stack)
        .select(features)
        .toDouble()
    )

    mean_image = (
        ee.Image.constant(
            means.tolist()
        )
        .rename(features)
        .toDouble()
    )

    standard_deviation_image = (
        ee.Image.constant(
            standard_deviations.tolist()
        )
        .rename(features)
        .toDouble()
    )

    weight_image = (
        ee.Image.constant(
            weights.tolist()
        )
        .rename(features)
        .toDouble()
    )

    weighted = (
        predictor_image
        .subtract(mean_image)
        .divide(standard_deviation_image)
        .multiply(weight_image)
    )

    # Predictor vector per pixel: [predictors, 1].
    predictor_column = (
        weighted
        .toArray()
        .toArray(1)
    )

    # Final training reference matrix: [training rows, predictors].
    reference_matrix = ee.Image.constant(
        ee.Array(
            reference.tolist()
        )
    )

    # R * x -> [training rows, 1].
    dot_product = (
        reference_matrix
        .matrixMultiply(
            predictor_column
        )
    )

    reference_norm_image = ee.Image.constant(
        ee.Array(
            reference_norm_squared.tolist()
        )
    )

    # ||x||² repeated for all training-reference rows.
    predictor_norm_squared = (
        weighted
        .pow(2)
        .reduce(ee.Reducer.sum())
        .toArray()
        .toArray(1)
        .arrayRepeat(
            0,
            reference_rows,
        )
    )

    # Squared Euclidean distance:
    # ||r - x||² = ||r||² + ||x||² - 2(r·x).
    distance_squared = (
        reference_norm_image
        .add(predictor_norm_squared)
        .subtract(
            dot_product.multiply(2)
        )
    )

    nearest_squared = (
        distance_squared
        .arrayReduce(
            ee.Reducer.min(),
            [0],
        )
        .arrayGet(
            ee.Image.constant(
                [0, 0]
            )
        )
        .max(0)
    )

    nearest_distance = (
        nearest_squared
        .sqrt()
        .rename("nearest_training_distance")
        .toFloat()
    )

    dissimilarity_index = (
        nearest_distance
        .divide(
            float(
                specification["mean_training_distance"]
            )
        )
        .rename("DI")
        .toFloat()
    )

    aoa = (
        dissimilarity_index
        .lte(
            float(
                specification["threshold"]
            )
        )
        .rename("AOA")
        .toByte()
    )

    return {
        "nearest_training_distance": nearest_distance,
        "di": dissimilarity_index,
        "aoa": aoa,
    }

