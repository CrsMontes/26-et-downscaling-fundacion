"""Final Ridge-25 training and validation from a complete master dataset.

The accepted model specification is fixed by the completed methodological
experiments, but the fitted model is NOT a repository input. Every scientific
run rebuilds the training population, performs out-of-fold validation, and fits
a new Ridge model from the master dataset.

Reconciliation is deliberately absent from this module. It belongs only to
fine-resolution production after model validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .ridge25 import (
    RIDGE25_HARMONIC_FEATURES,
    RIDGE25_METEOROLOGICAL_FEATURES,
    RIDGE25_MODEL_FEATURES,
    RIDGE25_OPTICAL_FEATURES,
)


TARGET_COLUMN = "Kc_target"
KEY_COLUMNS = ["station_id", "period_start"]
OPTICAL_COVERAGE_THRESHOLD_PCT = 90.0
RIDGE_ALPHA = 1.0

# Reference gate used only to verify exact reproduction of the accepted
# 2020-2024 experiment. It is not a universal ecological constraint.
REFERENCE_ROWS_2020_2024 = 799
REFERENCE_SPATIAL_COUNTS_2020_2024 = {
    "-811_116": 168,
    "-814_118": 297,
    "-814_119": 169,
    "-815_118": 165,
}
REFERENCE_YEAR_COUNTS_2020_2024 = {
    2020: 171,
    2021: 151,
    2022: 142,
    2023: 185,
    2024: 150,
}

MASTER_OPTICAL_FEATURES = [f"s2_{name}" for name in RIDGE25_OPTICAL_FEATURES]
MASTER_MODEL_FEATURES = (
    MASTER_OPTICAL_FEATURES
    + list(RIDGE25_METEOROLOGICAL_FEATURES)
    + list(RIDGE25_HARMONIC_FEATURES)
)
MASTER_TO_MODEL = dict(
    zip(MASTER_MODEL_FEATURES, RIDGE25_MODEL_FEATURES, strict=True)
)


@dataclass
class Ridge25Result:
    model: Pipeline
    population: pd.DataFrame
    spatial_oof: pd.DataFrame
    temporal_oof: pd.DataFrame
    spatial_fold_metrics: pd.DataFrame
    temporal_fold_metrics: pd.DataFrame
    spatial_metrics: dict[str, float]
    temporal_metrics: dict[str, float]


def build_ridge25_model() -> Pipeline:
    """Return the fixed accepted Ridge specification."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)),
        ]
    )


def calculate_metrics(
    observed: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calculate complementary predictive-performance metrics."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - observed

    observed_sd = observed.std(ddof=0)
    predicted_sd = predicted.std(ddof=0)

    if (
        len(observed) < 2
        or observed_sd == 0
        or predicted_sd == 0
        or observed.mean() == 0
    ):
        kge = np.nan
    else:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
        alpha = float(predicted_sd / observed_sd)
        beta = float(predicted.mean() / observed.mean())
        kge = 1.0 - np.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )

    return {
        "n": int(len(observed)),
        "R2": float(r2_score(observed, predicted)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "BIAS": float(np.mean(error)),
        "KGE": float(kge),
    }


def prepare_ridge25_population(
    master: pd.DataFrame,
    *,
    verify_reference_2020_2024: bool = False,
) -> pd.DataFrame:
    """Build the GE90 Ridge-25 population from the complete master table."""
    required = set(
        KEY_COLUMNS
        + [
            TARGET_COLUMN,
            "modis_good",
            "target_complete",
            "s2_coverage_pct",
            "spatial_block",
            "year",
        ]
        + MASTER_MODEL_FEATURES
    )
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(
            "Master dataset is missing required Ridge-25 columns: "
            + ", ".join(missing)
        )

    data = master.copy()
    data["station_id"] = data["station_id"].astype(str)
    data["period_start"] = pd.to_datetime(
        data["period_start"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    if data.duplicated(KEY_COLUMNS).any():
        raise ValueError("Master dataset contains duplicate station-period keys.")

    numeric_columns = MASTER_MODEL_FEATURES + [
        TARGET_COLUMN,
        "s2_coverage_pct",
    ]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    eligible = (
        data["modis_good"].eq(1)
        & data["target_complete"].eq(1)
        & data[TARGET_COLUMN].notna()
        & data["s2_coverage_pct"].ge(OPTICAL_COVERAGE_THRESHOLD_PCT)
        & data[MASTER_MODEL_FEATURES].notna().all(axis=1)
    )

    selected = (
        data.loc[eligible]
        .copy()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    matrix = selected[MASTER_MODEL_FEATURES + [TARGET_COLUMN]].to_numpy(
        dtype=float
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Ridge-25 training matrix contains non-finite values.")

    selected = selected.rename(columns=MASTER_TO_MODEL)

    if verify_reference_2020_2024:
        verify_reference_population(selected)

    return selected


def verify_reference_population(data: pd.DataFrame) -> None:
    """Verify exact population counts for the accepted 2020-2024 gate."""
    if len(data) != REFERENCE_ROWS_2020_2024:
        raise RuntimeError(
            f"Expected {REFERENCE_ROWS_2020_2024} rows, found {len(data)}."
        )

    spatial_counts = {
        str(key): int(value)
        for key, value in data.groupby("spatial_block").size().to_dict().items()
    }
    if spatial_counts != REFERENCE_SPATIAL_COUNTS_2020_2024:
        raise RuntimeError(
            f"Unexpected spatial population: {spatial_counts}"
        )

    year_counts = {
        int(key): int(value)
        for key, value in data.groupby("year").size().to_dict().items()
    }
    if year_counts != REFERENCE_YEAR_COUNTS_2020_2024:
        raise RuntimeError(f"Unexpected year population: {year_counts}")


def _oof_by_group(
    data: pd.DataFrame,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = np.full(len(data), np.nan, dtype=float)
    fold_rows: list[dict[str, float | int | str]] = []

    groups = data[group_column].astype(str)
    for fold_number, group in enumerate(sorted(groups.unique()), start=1):
        test_mask = groups.eq(group).to_numpy()
        train_mask = ~test_mask

        model = build_ridge25_model()
        model.fit(
            data.loc[train_mask, RIDGE25_MODEL_FEATURES],
            data.loc[train_mask, TARGET_COLUMN],
        )
        prediction = model.predict(
            data.loc[test_mask, RIDGE25_MODEL_FEATURES]
        )
        predictions[test_mask] = prediction

        fold_rows.append(
            {
                "fold": fold_number,
                "group": group,
                **calculate_metrics(
                    data.loc[test_mask, TARGET_COLUMN],
                    prediction,
                ),
            }
        )

    if np.isnan(predictions).any():
        raise RuntimeError(
            f"OOF predictions for {group_column} contain missing values."
        )

    output = data[
        KEY_COLUMNS + ["spatial_block", "year", TARGET_COLUMN]
    ].copy()
    output["prediction"] = predictions
    output["error"] = predictions - data[TARGET_COLUMN].to_numpy(dtype=float)

    return output, pd.DataFrame(fold_rows)


def train_and_validate_ridge25(
    master: pd.DataFrame,
    *,
    verify_reference_2020_2024: bool = False,
) -> Ridge25Result:
    """Validate Ridge-25 out of fold, then fit it on the full eligible sample."""
    population = prepare_ridge25_population(
        master,
        verify_reference_2020_2024=verify_reference_2020_2024,
    )

    spatial_oof, spatial_fold_metrics = _oof_by_group(
        population,
        "spatial_block",
    )
    temporal_oof, temporal_fold_metrics = _oof_by_group(
        population,
        "year",
    )

    spatial_metrics = calculate_metrics(
        spatial_oof[TARGET_COLUMN],
        spatial_oof["prediction"],
    )
    temporal_metrics = calculate_metrics(
        temporal_oof[TARGET_COLUMN],
        temporal_oof["prediction"],
    )

    model = build_ridge25_model()
    model.fit(
        population[RIDGE25_MODEL_FEATURES],
        population[TARGET_COLUMN],
    )

    return Ridge25Result(
        model=model,
        population=population,
        spatial_oof=spatial_oof,
        temporal_oof=temporal_oof,
        spatial_fold_metrics=spatial_fold_metrics,
        temporal_fold_metrics=temporal_fold_metrics,
        spatial_metrics=spatial_metrics,
        temporal_metrics=temporal_metrics,
    )
