"""Run-level reporting for the parsimonious Ridge-25 workflow.

All outputs are derived from the current in-memory fit and are written to the
external workspace. Nothing in this module is a model input.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .modeling import (
    RIDGE25_MODEL_FEATURES,
    TARGET_COLUMN,
    Ridge25Result,
    calculate_metrics,
)


def _metrics_frame(
    label: str,
    metrics: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "validation": label,
                **metrics,
            }
        ]
    )


def _baseline_oof(
    population: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    predictions = np.full(
        len(population),
        np.nan,
        dtype=float,
    )
    groups = population[group_column].astype(str)

    for group in sorted(groups.unique()):
        test_mask = groups.eq(group).to_numpy()
        train_mask = ~test_mask
        training_mean = float(
            population.loc[
                train_mask,
                TARGET_COLUMN,
            ].mean()
        )
        predictions[test_mask] = training_mean

    output = population[
        [
            "station_id",
            "period_start",
            "spatial_block",
            "year",
            TARGET_COLUMN,
        ]
    ].copy()
    output["prediction"] = predictions
    output["error"] = (
        predictions
        - output[TARGET_COLUMN].to_numpy(dtype=float)
    )
    return output


def _model_parameter_table(
    result: Ridge25Result,
) -> pd.DataFrame:
    scaler = result.model.named_steps["scaler"]
    regressor = result.model.named_steps["regressor"]

    return pd.DataFrame(
        {
            "feature": RIDGE25_MODEL_FEATURES,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "ridge_coefficient_standardized": regressor.coef_,
        }
    )


def _by_station_metrics(
    oof: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for station_id, group in oof.groupby(
        "station_id",
        sort=True,
    ):
        rows.append(
            {
                "station_id": station_id,
                **calculate_metrics(
                    group[TARGET_COLUMN],
                    group["prediction"],
                ),
            }
        )
    return pd.DataFrame(rows)


def save_run_tables(
    result: Ridge25Result,
    run_directory: Path,
) -> dict[str, Path]:
    """Write model-derived tables for the current run."""
    run_directory = Path(run_directory)
    table_directory = run_directory / "tables"
    table_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "training_population": (
            table_directory
            / "ridge25_training_population.csv"
        ),
        "spatial_oof": (
            table_directory
            / "ridge25_spatial_oof.csv"
        ),
        "temporal_oof": (
            table_directory
            / "ridge25_loyo_oof.csv"
        ),
        "spatial_folds": (
            table_directory
            / "ridge25_spatial_fold_metrics.csv"
        ),
        "temporal_folds": (
            table_directory
            / "ridge25_loyo_fold_metrics.csv"
        ),
        "station_metrics": (
            table_directory
            / "ridge25_spatial_metrics_by_station.csv"
        ),
        "model_parameters": (
            table_directory
            / "ridge25_model_parameters.csv"
        ),
        "validation_metrics": (
            table_directory
            / "ridge25_validation_metrics.csv"
        ),
        "baseline_metrics": (
            table_directory
            / "mean_baseline_metrics.csv"
        ),
    }

    result.population.to_csv(
        paths["training_population"],
        index=False,
    )
    result.spatial_oof.to_csv(
        paths["spatial_oof"],
        index=False,
    )
    result.temporal_oof.to_csv(
        paths["temporal_oof"],
        index=False,
    )
    result.spatial_fold_metrics.to_csv(
        paths["spatial_folds"],
        index=False,
    )
    result.temporal_fold_metrics.to_csv(
        paths["temporal_folds"],
        index=False,
    )
    _by_station_metrics(
        result.spatial_oof
    ).to_csv(
        paths["station_metrics"],
        index=False,
    )
    _model_parameter_table(
        result
    ).to_csv(
        paths["model_parameters"],
        index=False,
    )

    validation = pd.concat(
        [
            _metrics_frame(
                "spatial_block_oof",
                result.spatial_metrics,
            ),
            _metrics_frame(
                "leave_one_year_out",
                result.temporal_metrics,
            ),
        ],
        ignore_index=True,
    )
    validation.to_csv(
        paths["validation_metrics"],
        index=False,
    )

    spatial_baseline = _baseline_oof(
        result.population,
        "spatial_block",
    )
    temporal_baseline = _baseline_oof(
        result.population,
        "year",
    )
    baseline = pd.concat(
        [
            _metrics_frame(
                "spatial_block_mean_baseline",
                calculate_metrics(
                    spatial_baseline[TARGET_COLUMN],
                    spatial_baseline["prediction"],
                ),
            ),
            _metrics_frame(
                "leave_one_year_mean_baseline",
                calculate_metrics(
                    temporal_baseline[TARGET_COLUMN],
                    temporal_baseline["prediction"],
                ),
            ),
        ],
        ignore_index=True,
    )
    baseline.to_csv(
        paths["baseline_metrics"],
        index=False,
    )

    return paths


def save_model_metadata(
    result: Ridge25Result,
    run_directory: Path,
    metadata: dict,
) -> Path:
    """Write a JSON provenance record without serializing a fitted model."""
    run_directory = Path(run_directory)
    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    regressor = result.model.named_steps["regressor"]
    payload = {
        **metadata,
        "model": {
            "algorithm": "StandardScaler + Ridge",
            "ridge_alpha": float(
                regressor.alpha
            ),
            "fit_intercept": bool(
                regressor.fit_intercept
            ),
            "predictor_count": len(
                RIDGE25_MODEL_FEATURES
            ),
            "predictors": list(
                RIDGE25_MODEL_FEATURES
            ),
            "fitted_model_input": False,
            "serialized_model_required": False,
        },
        "spatial_metrics": (
            result.spatial_metrics
        ),
        "temporal_metrics": (
            result.temporal_metrics
        ),
        "training_rows": int(
            len(result.population)
        ),
    }

    path = run_directory / "run_metadata.json"
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def _identity_limits(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    values = np.concatenate(
        [observed, predicted]
    )
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    margin = 0.04 * max(
        upper - lower,
        1e-6,
    )
    return lower - margin, upper + margin


def figure_oof_scatter(
    result: Ridge25Result,
    path: Path,
) -> None:
    observed = result.spatial_oof[
        TARGET_COLUMN
    ].to_numpy(dtype=float)
    predicted = result.spatial_oof[
        "prediction"
    ].to_numpy(dtype=float)

    lower, upper = _identity_limits(
        observed,
        predicted,
    )

    fig, ax = plt.subplots(
        figsize=(6.2, 6.0)
    )
    ax.scatter(
        observed,
        predicted,
        s=18,
        alpha=0.55,
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        linewidth=1.2,
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Observed Kc")
    ax.set_ylabel("Spatial OOF predicted Kc")
    ax.set_title(
        "Spatial block validation - Ridge-25"
    )
    ax.text(
        0.04,
        0.96,
        (
            f"R² = {result.spatial_metrics['R2']:.3f}\n"
            f"RMSE = {result.spatial_metrics['RMSE']:.3f}\n"
            f"MAE = {result.spatial_metrics['MAE']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def figure_coefficients(
    result: Ridge25Result,
    path: Path,
) -> None:
    table = _model_parameter_table(
        result
    ).sort_values(
        "ridge_coefficient_standardized",
        key=lambda series: series.abs(),
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 8.0)
    )
    ax.barh(
        table["feature"],
        table[
            "ridge_coefficient_standardized"
        ],
    )
    ax.axvline(
        0,
        linewidth=0.8,
    )
    ax.set_xlabel(
        "Standardized Ridge coefficient"
    )
    ax.set_title(
        "Ridge-25 coefficient structure"
    )
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def figure_station_series(
    result: Ridge25Result,
    path: Path,
) -> None:
    data = result.spatial_oof.copy()
    data["period_start"] = pd.to_datetime(
        data["period_start"]
    )

    stations = sorted(
        data["station_id"].astype(str).unique()
    )
    fig, axes = plt.subplots(
        len(stations),
        1,
        figsize=(10.0, 2.3 * len(stations)),
        sharex=True,
    )
    if len(stations) == 1:
        axes = [axes]

    for ax, station_id in zip(
        axes,
        stations,
        strict=True,
    ):
        subset = (
            data[
                data["station_id"].astype(str)
                == station_id
            ]
            .sort_values("period_start")
        )
        ax.plot(
            subset["period_start"],
            subset[TARGET_COLUMN],
            linewidth=1.0,
            label="Observed",
        )
        ax.plot(
            subset["period_start"],
            subset["prediction"],
            linewidth=1.0,
            label="Spatial OOF",
        )
        ax.set_ylabel("Kc")
        ax.set_title(str(station_id))
        ax.legend(
            loc="upper right",
            frameon=False,
        )

    axes[-1].set_xlabel("Period start")
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def figure_validation_protocol(
    result: Ridge25Result,
    path: Path,
) -> None:
    metrics = pd.DataFrame(
        [
            {
                "protocol": "Spatial blocks",
                **result.spatial_metrics,
            },
            {
                "protocol": "LOYO",
                **result.temporal_metrics,
            },
        ]
    )

    fig, ax = plt.subplots(
        figsize=(6.0, 4.5)
    )
    positions = np.arange(
        len(metrics)
    )
    ax.bar(
        positions - 0.18,
        metrics["RMSE"],
        width=0.36,
        label="RMSE",
    )
    ax.bar(
        positions + 0.18,
        metrics["MAE"],
        width=0.36,
        label="MAE",
    )
    ax.set_xticks(
        positions,
        metrics["protocol"],
    )
    ax.set_ylabel("Kc error")
    ax.set_title(
        "Effect of validation protocol"
    )
    ax.legend(
        frameon=False
    )
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def figure_station_performance(
    result: Ridge25Result,
    path: Path,
) -> None:
    metrics = _by_station_metrics(
        result.spatial_oof
    ).sort_values(
        "station_id"
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 4.5)
    )
    positions = np.arange(
        len(metrics)
    )
    ax.bar(
        positions,
        metrics["RMSE"],
    )
    ax.set_xticks(
        positions,
        metrics["station_id"],
    )
    ax.set_ylabel("Spatial OOF RMSE (Kc)")
    ax.set_xlabel("Station")
    ax.set_title(
        "Spatial transfer performance by station"
    )
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_core_figures(
    result: Ridge25Result,
    run_directory: Path,
) -> dict[str, Path]:
    """Regenerate the model-dependent core diagnostic figures."""
    figure_directory = (
        Path(run_directory)
        / "figures"
    )
    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "F02": (
            figure_directory
            / "F02_spatial_oof_observed_predicted.png"
        ),
        "F03": (
            figure_directory
            / "F03_ridge25_coefficients.png"
        ),
        "F04": (
            figure_directory
            / "F04_spatial_oof_station_series.png"
        ),
        "F17": (
            figure_directory
            / "F17_validation_protocol.png"
        ),
        "F19": (
            figure_directory
            / "F19_performance_by_station.png"
        ),
    }

    figure_oof_scatter(
        result,
        paths["F02"],
    )
    figure_coefficients(
        result,
        paths["F03"],
    )
    figure_station_series(
        result,
        paths["F04"],
    )
    figure_validation_protocol(
        result,
        paths["F17"],
    )
    figure_station_performance(
        result,
        paths["F19"],
    )

    return paths
