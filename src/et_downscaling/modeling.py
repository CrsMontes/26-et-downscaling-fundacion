"""Parsimonious Ridge-25 training and validation from a complete master table.

The fitted model is rebuilt in memory for every scientific run. This module
accepts both:

1. the direct local master produced from reusable raw exports; and
2. the richer diagnostic master used during the five-year predictor experiment.

It derives model-only fields locally (year, seasonal harmonics, and the
approximately 10 km spatial block) and never uses historical candidate flags
that required Sentinel-1 or CHIRPS.

Reconciliation is intentionally absent. It is a production-only operation
applied after fine-resolution prediction.
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
SPATIAL_BLOCK_SIZE_KM = 10.0

# Backward-compatible names for the diagnostic predictor-store schema.
MASTER_OPTICAL_FEATURES = [
    f"s2_{feature}"
    for feature in RIDGE25_OPTICAL_FEATURES
]
MASTER_MODEL_FEATURES = (
    MASTER_OPTICAL_FEATURES
    + list(RIDGE25_METEOROLOGICAL_FEATURES)
    + list(RIDGE25_HARMONIC_FEATURES)
)

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


def _add_harmonics(data: pd.DataFrame) -> pd.DataFrame:
    """Add the exact two annual harmonic pairs used in model selection."""
    result = data.copy()
    dates = pd.to_datetime(result["period_start"], errors="raise")
    doy = dates.dt.dayofyear.to_numpy(dtype=float)

    for harmonic in (1, 2):
        angle = (
            2.0
            * np.pi
            * harmonic
            * doy
            / 365.25
        )
        result[f"doy_sin{harmonic}"] = np.sin(angle)
        result[f"doy_cos{harmonic}"] = np.cos(angle)

    return result


def _resolve_coordinate_columns(
    data: pd.DataFrame,
) -> tuple[str, str]:
    """Resolve the same coordinate preference used by prior training."""
    candidates = [
        ("station_longitude", "station_latitude"),
        ("longitude", "latitude"),
        (
            "footprint_centroid_longitude",
            "footprint_centroid_latitude",
        ),
    ]

    for longitude_column, latitude_column in candidates:
        if (
            longitude_column in data.columns
            and latitude_column in data.columns
        ):
            return longitude_column, latitude_column

    raise ValueError(
        "No usable longitude/latitude columns were found "
        "to derive spatial blocks."
    )


def _add_spatial_blocks(data: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the approximately 10 km blocks used in prior model training."""
    result = data.copy()
    longitude_column, latitude_column = _resolve_coordinate_columns(result)

    longitude = pd.to_numeric(
        result[longitude_column],
        errors="raise",
    )
    latitude = pd.to_numeric(
        result[latitude_column],
        errors="raise",
    )

    km_per_degree_latitude = 111.32
    km_per_degree_longitude = (
        111.32
        * np.cos(
            np.radians(
                latitude.mean()
            )
        )
    )

    block_x = np.floor(
        longitude
        * km_per_degree_longitude
        / SPATIAL_BLOCK_SIZE_KM
    ).astype(int)

    block_y = np.floor(
        latitude
        * km_per_degree_latitude
        / SPATIAL_BLOCK_SIZE_KM
    ).astype(int)

    result["spatial_block"] = (
        block_x.astype(str)
        + "_"
        + block_y.astype(str)
    )

    return result


def canonicalize_master(master: pd.DataFrame) -> pd.DataFrame:
    """Convert a complete raw-derived or diagnostic master to model schema."""
    data = master.copy()

    base_required = {
        "station_id",
        "period_start",
        TARGET_COLUMN,
        "modis_good",
        "target_complete",
    }
    missing_base = sorted(base_required - set(data.columns))
    if missing_base:
        raise ValueError(
            "Master dataset is missing required base columns: "
            + ", ".join(missing_base)
        )

    data["station_id"] = data["station_id"].astype(str)
    data["period_start"] = pd.to_datetime(
        data["period_start"],
        errors="raise",
    )

    if data.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            "Master dataset contains duplicate station-period keys."
        )

    # Direct raw-derived master names this quantity optical_union_coverage_pct.
    # Diagnostic stores used s2_coverage_pct.
    if "s2_coverage_pct" not in data.columns:
        if "optical_union_coverage_pct" not in data.columns:
            raise ValueError(
                "Master dataset contains neither s2_coverage_pct nor "
                "optical_union_coverage_pct."
            )
        data["s2_coverage_pct"] = pd.to_numeric(
            data["optical_union_coverage_pct"],
            errors="coerce",
        )

    # The raw-derived S2 master uses production feature names such as
    # Blue_mean. The five-year diagnostic store prefixed them with s2_.
    for production_feature in RIDGE25_OPTICAL_FEATURES:
        if production_feature in data.columns:
            continue
        diagnostic_feature = f"s2_{production_feature}"
        if diagnostic_feature not in data.columns:
            raise ValueError(
                "Missing optical predictor in both schemas: "
                f"{production_feature} / {diagnostic_feature}"
            )
        data[production_feature] = pd.to_numeric(
            data[diagnostic_feature],
            errors="coerce",
        )

    missing_meteorology = sorted(
        set(RIDGE25_METEOROLOGICAL_FEATURES)
        - set(data.columns)
    )
    if missing_meteorology:
        raise ValueError(
            "Master dataset is missing final meteorological predictors: "
            + ", ".join(missing_meteorology)
        )

    if not set(RIDGE25_HARMONIC_FEATURES).issubset(data.columns):
        data = _add_harmonics(data)

    if "year" not in data.columns:
        data["year"] = data["period_start"].dt.year.astype(int)
    else:
        data["year"] = pd.to_numeric(
            data["year"],
            errors="raise",
        ).astype(int)

    if "spatial_block" not in data.columns:
        data = _add_spatial_blocks(data)

    numeric_columns = (
        list(RIDGE25_MODEL_FEATURES)
        + [TARGET_COLUMN, "s2_coverage_pct"]
    )
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return data


def prepare_ridge25_population(
    master: pd.DataFrame,
    *,
    verify_reference_2020_2024: bool = False,
) -> pd.DataFrame:
    """Build the final GE90 population without S1/CHIRPS eligibility gates."""
    data = canonicalize_master(master)

    eligible = (
        data["modis_good"].eq(1)
        & data["target_complete"].eq(1)
        & data[TARGET_COLUMN].notna()
        & data["s2_coverage_pct"].ge(
            OPTICAL_COVERAGE_THRESHOLD_PCT
        )
        & data[
            RIDGE25_MODEL_FEATURES
        ].notna().all(axis=1)
    )

    selected = (
        data.loc[eligible]
        .copy()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    matrix = selected[
        list(RIDGE25_MODEL_FEATURES)
        + [TARGET_COLUMN]
    ].to_numpy(dtype=float)

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Ridge-25 training matrix contains non-finite values."
        )

    if verify_reference_2020_2024:
        verify_reference_population(selected)

    return selected


def verify_reference_population(data: pd.DataFrame) -> None:
    """Verify exact counts for the accepted five-year gate."""
    if len(data) != REFERENCE_ROWS_2020_2024:
        raise RuntimeError(
            f"Expected {REFERENCE_ROWS_2020_2024} rows, "
            f"found {len(data)}."
        )

    spatial_counts = {
        str(key): int(value)
        for key, value
        in data.groupby("spatial_block").size().to_dict().items()
    }
    if spatial_counts != REFERENCE_SPATIAL_COUNTS_2020_2024:
        raise RuntimeError(
            f"Unexpected spatial population: {spatial_counts}"
        )

    year_counts = {
        int(key): int(value)
        for key, value
        in data.groupby("year").size().to_dict().items()
    }
    if year_counts != REFERENCE_YEAR_COUNTS_2020_2024:
        raise RuntimeError(
            f"Unexpected year population: {year_counts}"
        )


def _oof_by_group(
    data: pd.DataFrame,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = np.full(len(data), np.nan, dtype=float)
    fold_rows: list[dict[str, float | int | str]] = []

    groups = data[group_column].astype(str)
    for fold_number, group in enumerate(
        sorted(groups.unique()),
        start=1,
    ):
        test_mask = groups.eq(group).to_numpy()
        train_mask = ~test_mask

        model = build_ridge25_model()
        model.fit(
            data.loc[
                train_mask,
                RIDGE25_MODEL_FEATURES,
            ],
            data.loc[
                train_mask,
                TARGET_COLUMN,
            ],
        )

        prediction = model.predict(
            data.loc[
                test_mask,
                RIDGE25_MODEL_FEATURES,
            ]
        )
        predictions[test_mask] = prediction

        fold_rows.append(
            {
                "fold": fold_number,
                "group": group,
                **calculate_metrics(
                    data.loc[
                        test_mask,
                        TARGET_COLUMN,
                    ],
                    prediction,
                ),
            }
        )

    if np.isnan(predictions).any():
        raise RuntimeError(
            f"OOF predictions for {group_column} contain missing values."
        )

    output = data[
        KEY_COLUMNS
        + ["spatial_block", "year", TARGET_COLUMN]
    ].copy()
    output["prediction"] = predictions
    output["error"] = (
        predictions
        - data[TARGET_COLUMN].to_numpy(dtype=float)
    )

    return output, pd.DataFrame(fold_rows)


def train_and_validate_ridge25(
    master: pd.DataFrame,
    *,
    verify_reference_2020_2024: bool = False,
) -> Ridge25Result:
    """Run spatial and temporal OOF validation, then fit Ridge on all rows."""
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
