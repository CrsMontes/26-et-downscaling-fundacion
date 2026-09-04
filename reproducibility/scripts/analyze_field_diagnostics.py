"""Reproduce and update the ETgage field diagnostics before map export.

This script intentionally separates two purposes:

1. Diagnostic reproduction: reproduces the historical field-processing logic
   required to compare the ETgage observations with MOD16A2.
2. Current-model checkpoint: compares out-of-fold coarse-support predictions
   from the current RF against the same field-derived series.

The current-model comparison is NOT a validation of the 20 m downscaled map.
That test requires extraction from the final fine-resolution product and is
therefore deferred until after the production raster exists.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from et_downscaling.config import ANALYSIS_PERIOD, build_training_output_filename
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from et_downscaling.stations import load_station_dataframe


RANDOM_STATE = 42
FIELD_SCALE_FACTOR = 10.0
MIN_VALID_DAYS_PER_PERIOD = 5
VALID_DAILY_ET_RANGE_MM = (0.05, 12.0)

FIXED_KC = {
    "ST01": 0.85,
    "ST02": 0.95,
    "ST03": 1.10,
}

NDVI_KC_SLOPE = 1.457
NDVI_KC_INTERCEPT = -0.1725
NDVI_KC_VALID_RANGE = (0.10, 1.50)

INSTRUMENT_REPEATABILITY_MM = 0.508
FIELD_RECORDING_RESOLUTION_MM = 1.0


# ============================================================
# Paths
# ============================================================


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_paths(project_root: Path) -> dict[str, Path]:
    period_label = ANALYSIS_PERIOD.label
    return {
        "field_daily": project_root / "data" / "field" / "field_etgage.csv",
        "training_master": project_root / "outputs" / "processed" / "training" / "S2" / build_training_output_filename("S2"),
        "daily_reference": project_root / "outputs" / "processed" / "training" / "S2" / f"reference_et_daily_{period_label}.csv",
        "oof": project_root / "outputs" / "processed" / "models" / "S2" / period_label / "kc_model_oof_predictions_ge90.csv",
        "tables": project_root / "outputs" / "processed" / "field_validation" / period_label / "tables",
        "figures": project_root / "outputs" / "processed" / "field_validation" / period_label / "figures",
    }


def require_files(paths: dict[str, Path]) -> None:
    required = [
        "field_daily",
        "training_master",
        "daily_reference",
        "oof",
    ]
    missing = [str(paths[key]) for key in required if not paths[key].is_file()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


# ============================================================
# Utilities
# ============================================================


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "si", "sí"})
    )


def calculate_metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"observed": observed, "predicted": predicted}).dropna()
    if len(frame) < 2:
        return {
            "n": int(len(frame)),
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "BIAS": np.nan,
            "r": np.nan,
        }

    y = frame["observed"].to_numpy(float)
    p = frame["predicted"].to_numpy(float)
    correlation = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan

    return {
        "n": int(len(frame)),
        "R2": float(r2_score(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "BIAS": float(np.mean(p - y)),
        "r": correlation,
    }


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Load current project data
# ============================================================


def load_inputs(paths: dict[str, Path]):
    field = pd.read_csv(
        paths["field_daily"],
        dtype={"station_id": "string"},
    )
    metadata = load_station_dataframe()
    master = pd.read_csv(paths["training_master"], dtype={"station_id": "string"})
    reference = pd.read_csv(paths["daily_reference"], dtype={"station_id": "string"})
    oof = pd.read_csv(paths["oof"], dtype={"station_id": "string"})

    field["date"] = pd.to_datetime(field["date"], errors="coerce")
    metadata["installation_date"] = pd.to_datetime(metadata["installation_date"], errors="coerce")
    metadata["removal_date"] = pd.to_datetime(metadata["removal_date"], errors="coerce")
    metadata["installation_conforms_manual"] = to_bool(metadata["installation_conforms_manual"])
    metadata["inside_basin"] = to_bool(metadata["inside_basin"])

    master["period_start"] = pd.to_datetime(master["period_start"], errors="raise")
    reference["local_date"] = pd.to_datetime(reference["local_date"], errors="raise")
    oof["period_start"] = pd.to_datetime(oof["period_start"], errors="raise")

    return field, metadata, master, reference, oof


# ============================================================
# Daily field preparation
# ============================================================


def prepare_field_daily(
    field: pd.DataFrame,
    metadata: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if field["station_id"].isna().any():
        raise RuntimeError(
            "field_etgage.csv contains missing station_id values."
        )

    field = field.merge(
        reference[["station_id", "local_date", "ETo_mm_day", "ETr_mm_day"]],
        left_on=["station_id", "date"],
        right_on=["station_id", "local_date"],
        how="left",
        validate="many_to_one",
    )

    field = field.merge(
        metadata[[
            "station_id",
            "canvas",
            "reference_et",
            "installation_conforms_manual",
            "inside_basin",
        ]],
        on="station_id",
        how="left",
        validate="many_to_one",
    )

    field["within_installation_window"] = to_bool(field["within_installation_window"])
    field["etgage_daily_raw"] = pd.to_numeric(field["etgage_daily_raw"], errors="coerce")

    field["qc_within_installation"] = field["within_installation_window"]
    field["qc_nonmissing"] = field["etgage_daily_raw"].notna()
    field["qc_positive"] = field["etgage_daily_raw"] > 0
    field["etgage_scaled_mm_day"] = field["etgage_daily_raw"] * FIELD_SCALE_FACTOR
    field["qc_physical_range"] = field["etgage_scaled_mm_day"].between(
        *VALID_DAILY_ET_RANGE_MM
    )

    field["field_daily_valid"] = (
        field["qc_within_installation"]
        & field["qc_nonmissing"]
        & field["qc_positive"]
        & field["qc_physical_range"]
    )

    valid = field.loc[field["field_daily_valid"]].copy()

    valid["ETr_ETo_ratio"] = valid["ETr_mm_day"] / valid["ETo_mm_day"]
    high_reference = valid["reference_et"].astype(str).str.upper().eq("ETR")

    valid["reference_expected_mm_day"] = np.where(
        high_reference,
        valid["ETr_mm_day"],
        valid["ETo_mm_day"],
    )

    valid["etgage_eto_equivalent_mm_day"] = np.where(
        high_reference,
        valid["etgage_scaled_mm_day"] / valid["ETr_ETo_ratio"],
        valid["etgage_scaled_mm_day"],
    )

    if valid[["ETo_mm_day", "ETr_mm_day"]].isna().any().any():
        raise RuntimeError("Missing daily ETo/ETr values for one or more valid field days.")

    return field, valid


# ============================================================
# Scale-factor diagnostic
# ============================================================


def bootstrap_ratio_total(reference_values, raw_values, rng, n_boot=2000):
    reference_values = np.asarray(reference_values, dtype=float)
    raw_values = np.asarray(raw_values, dtype=float)
    n = len(reference_values)
    index = rng.integers(0, n, size=(n_boot, n))
    denominator = raw_values[index].sum(axis=1)
    numerator = reference_values[index].sum(axis=1)
    ratios = numerator / denominator
    return np.percentile(ratios, [2.5, 97.5])


def build_scale_factor_table(valid: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    rows = []
    groups = [("GLOBAL", valid)] + list(valid.groupby("station"))

    for label, group in groups:
        if label != "GLOBAL" and len(group) < 15:
            continue

        reference_values = group["reference_expected_mm_day"].to_numpy(float)
        raw_values = group["etgage_daily_raw"].to_numpy(float)

        factor_ratio_total = float(reference_values.sum() / raw_values.sum())
        lo, hi = bootstrap_ratio_total(reference_values, raw_values, rng)

        canvas = "mixed" if group["canvas"].nunique() != 1 else str(group["canvas"].iloc[0])

        rows.append({
            "scope": label,
            "n_days": int(len(group)),
            "canvas": canvas,
            "raw_median": float(np.median(raw_values)),
            "reference_median_mm_day": float(np.median(reference_values)),
            "factor_ratio_total": factor_ratio_total,
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "assumed_factor": FIELD_SCALE_FACTOR,
            "assumed_factor_inside_ci": bool(lo <= FIELD_SCALE_FACTOR <= hi),
        })

    return pd.DataFrame(rows)


# ============================================================
# Reference ETo check
# ============================================================


def build_reference_validation_table(valid: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for station, group in valid.groupby("station"):
        measured = float(group["etgage_eto_equivalent_mm_day"].mean())
        era5 = float(group["ETo_mm_day"].mean())

        rows.append({
            "station": station,
            "n_days": int(len(group)),
            "installation_conforms_manual": bool(group["installation_conforms_manual"].iloc[0]),
            "inside_basin": bool(group["inside_basin"].iloc[0]),
            "ETo_measured_mm_day": measured,
            "ETo_ERA5_mm_day": era5,
            "bias_mm_day": era5 - measured,
            "bias_pct": 100.0 * (era5 / measured - 1.0),
            "measured_to_era5_factor": measured / era5,
        })

    return pd.DataFrame(rows).sort_values("station").reset_index(drop=True)


# ============================================================
# Aggregate field observations to MODIS periods
# ============================================================


def aggregate_to_modis_periods(valid: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    rows = []

    period_columns = [
        "station",
        "station_id",
        "period_start",
        "number_days",
        "ET_mm_period",
        "ETo_mm_period",
        "NDVI_mean",
        "optical_union_coverage_pct",
        "s1_union_coverage_pct",
    ]

    for station_id, station_daily in valid.groupby("station_id"):
        station = str(
            station_daily["station"].iloc[0]
        )
        periods = master.loc[
            master["station_id"] == station_id,
            period_columns,
        ].copy()
        periods["period_end"] = periods["period_start"] + pd.to_timedelta(
            periods["number_days"], unit="D"
        )

        for period in periods.itertuples(index=False):
            group = station_daily.loc[
                (station_daily["date"] >= period.period_start)
                & (station_daily["date"] < period.period_end)
            ].copy()

            if len(group) < MIN_VALID_DAYS_PER_PERIOD:
                continue

            field_reference_mean = float(group["etgage_eto_equivalent_mm_day"].mean())
            field_reference_period = field_reference_mean * int(period.number_days)

            kc = FIXED_KC.get(station_id, np.nan)
            kc_source = "FAO-56 fixed"

            if not np.isfinite(kc):
                candidate = NDVI_KC_SLOPE * float(period.NDVI_mean) + NDVI_KC_INTERCEPT
                if NDVI_KC_VALID_RANGE[0] <= candidate <= NDVI_KC_VALID_RANGE[1]:
                    kc = candidate
                    kc_source = "NDVI-derived diagnostic reproduction"
                else:
                    kc = np.nan
                    kc_source = "NDVI-derived outside valid range"

            rows.append({
                "station": station,
                "station_id": str(period.station_id),
                "period_start": period.period_start,
                "number_days": int(period.number_days),
                "n_valid_field_days": int(len(group)),
                "field_reference_eto_mm_period": field_reference_period,
                "Kc_field_conversion": kc,
                "Kc_source": kc_source,
                "field_actual_et_diagnostic_mm_period": field_reference_period * kc if np.isfinite(kc) else np.nan,
                "ET_MODIS_mm_period": float(period.ET_mm_period),
                "ETo_mm_period": float(period.ETo_mm_period),
                "NDVI_mean": float(period.NDVI_mean),
                "optical_union_coverage_pct": float(period.optical_union_coverage_pct),
                "s1_union_coverage_pct": float(period.s1_union_coverage_pct),
            })

    return pd.DataFrame(rows).sort_values(["station", "period_start"]).reset_index(drop=True)


# ============================================================
# Current-model OOF comparison
# ============================================================


def build_oof_comparison(period_pairs: pd.DataFrame, oof: pd.DataFrame):
    columns = [
        "station_id",
        "period_start",
        "Kc_predicted_rf_common",
        "Kc_predicted_modis_persistence",
    ]

    comparison = period_pairs.merge(
        oof[columns],
        on=["station_id", "period_start"],
        how="left",
        validate="one_to_one",
    )

    comparison["ET_predicted_rf_common_mm_period"] = (
        comparison["Kc_predicted_rf_common"] * comparison["ETo_mm_period"]
    )
    comparison["ET_predicted_persistence_mm_period"] = (
        comparison["Kc_predicted_modis_persistence"] * comparison["ETo_mm_period"]
    )

    observed = "field_actual_et_diagnostic_mm_period"
    matched = comparison.dropna(
        subset=[observed, "ET_MODIS_mm_period", "ET_predicted_rf_common_mm_period"]
    ).copy()

    metric_rows = []

    subsets = {
        "all_diagnostic_reproduction": matched,
        "external_fixed_kc_only": matched.loc[
            matched["station_id"].isin(FIXED_KC)
        ],
        "clean_pasture_only_installation_conforming": matched.loc[
            matched["station_id"] == "ST01"
        ],
    }

    prediction_columns = {
        "MODIS": "ET_MODIS_mm_period",
        "rf_common_oof": "ET_predicted_rf_common_mm_period",
        "modis_persistence": "ET_predicted_persistence_mm_period",
    }

    for subset_name, subset in subsets.items():
        for model_name, column in prediction_columns.items():
            metrics = calculate_metrics(subset[observed], subset[column])
            metric_rows.append({
                "subset": subset_name,
                "model": model_name,
                **metrics,
            })

    return comparison, pd.DataFrame(metric_rows)


# ============================================================
# Instrument uncertainty
# ============================================================


def build_instrument_uncertainty(period_pairs: pd.DataFrame) -> dict[str, float]:
    observed = period_pairs["field_actual_et_diagnostic_mm_period"].dropna().to_numpy(float)
    if len(observed) < 2:
        return {}

    sigma_daily = float(
        np.hypot(
            INSTRUMENT_REPEATABILITY_MM * np.sqrt(2.0),
            FIELD_RECORDING_RESOLUTION_MM / np.sqrt(12.0),
        )
    )

    # Approximate period uncertainty using the median number of valid daily observations.
    n_days = float(period_pairs.loc[period_pairs["field_actual_et_diagnostic_mm_period"].notna(), "n_valid_field_days"].median())
    sigma_period = sigma_daily * np.sqrt(n_days)
    observed_sd = float(np.std(observed, ddof=0))
    noise_fraction = min((sigma_period**2) / (observed_sd**2), 1.0) if observed_sd > 0 else np.nan
    r_max = float(np.sqrt(max(1.0 - noise_fraction, 0.0))) if np.isfinite(noise_fraction) else np.nan

    return {
        "n_periods": int(len(observed)),
        "sigma_daily_mm": sigma_daily,
        "median_valid_days": n_days,
        "sigma_period_mm": sigma_period,
        "observed_sd_mm": observed_sd,
        "noise_variance_fraction": noise_fraction,
        "r_max": r_max,
        "R2_max": r_max**2 if np.isfinite(r_max) else np.nan,
    }


# ============================================================
# Figures
# ============================================================


def plot_daily_series(field: pd.DataFrame, output: Path) -> None:
    stations = ["Clean pasture", "Oil palm", "Banana", "Mangrove", "Dry forest"]
    fig, axes = plt.subplots(len(stations), 1, figsize=(10, 11), sharex=True)

    for axis, station in zip(axes, stations):
        group = field.loc[field["station"] == station].sort_values("date")
        axis.plot(group["date"], group["etgage_daily_raw"], marker=".", linewidth=0.8)
        invalid = group.loc[~group["field_daily_valid"] & group["etgage_daily_raw"].notna()]
        axis.scatter(invalid["date"], invalid["etgage_daily_raw"], marker="x", s=22)
        axis.set_ylabel(station)
        axis.grid(alpha=0.25)

    axes[0].set_title("ETgage raw daily records; x = excluded by installation/zero/range QC")
    axes[-1].set_xlabel("Date")
    fig.supylabel("Recorded daily change (source units)")
    save_figure(fig, output)


def plot_scale_factor(table: pd.DataFrame, output: Path) -> None:
    ordered = table.copy().sort_values("factor_ratio_total")
    y = np.arange(len(ordered))
    values = ordered["factor_ratio_total"].to_numpy(float)
    lower = values - ordered["ci95_low"].to_numpy(float)
    upper = ordered["ci95_high"].to_numpy(float) - values

    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(y, values)
    axis.errorbar(values, y, xerr=np.vstack([lower, upper]), fmt="none", capsize=4)
    axis.axvline(FIELD_SCALE_FACTOR, linestyle="--", label="Diagnostic scale factor = 10")
    axis.set_yticks(y, ordered["scope"] + "  " + ordered["canvas"].astype(str))
    axis.set_xlabel("Reference ET / raw ETgage reading")
    axis.set_title("ETgage scale-factor diagnostic")
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    save_figure(fig, output)


def plot_reference_validation(table: pd.DataFrame, output: Path) -> None:
    ordered = table.sort_values("station")
    x = np.arange(len(ordered))
    width = 0.38

    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(x - width / 2, ordered["ETo_measured_mm_day"], width, label="ETgage ETo-equivalent")
    axis.bar(x + width / 2, ordered["ETo_ERA5_mm_day"], width, label="ERA5-Land ETo")
    axis.set_xticks(x, ordered["station"], rotation=20, ha="right")
    axis.set_ylabel("Reference ET (mm day$^{-1}$)")
    axis.set_title("Reference-ET check; only Clean pasture conforms to installation guidance")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    save_figure(fig, output)


def plot_modis_scatter(period_pairs: pd.DataFrame, output: Path) -> None:
    data = period_pairs.dropna(subset=["field_actual_et_diagnostic_mm_period", "ET_MODIS_mm_period"])
    fig, axis = plt.subplots(figsize=(7, 7))

    for station, group in data.groupby("station"):
        axis.scatter(
            group["field_actual_et_diagnostic_mm_period"],
            group["ET_MODIS_mm_period"],
            label=f"{station} (n={len(group)})",
            alpha=0.8,
        )

    values = np.concatenate([
        data["field_actual_et_diagnostic_mm_period"].to_numpy(float),
        data["ET_MODIS_mm_period"].to_numpy(float),
    ])
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    axis.plot([lo, hi], [lo, hi], linestyle="--")
    metrics = calculate_metrics(
        data["field_actual_et_diagnostic_mm_period"], data["ET_MODIS_mm_period"]
    )
    axis.set_xlabel("Field-derived ET, diagnostic reproduction (mm period$^{-1}$)")
    axis.set_ylabel("MOD16A2 ET (mm period$^{-1}$)")
    axis.set_title(
        f"Diagnostic reproduction: field vs MODIS\n"
        f"n={metrics['n']}, R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.2f}, r={metrics['r']:.3f}"
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    save_figure(fig, output)


def plot_modis_series(period_pairs: pd.DataFrame, output: Path) -> None:
    stations = ["Clean pasture", "Oil palm", "Banana", "Mangrove", "Dry forest"]
    fig, axes = plt.subplots(len(stations), 1, figsize=(10, 12), sharex=True)

    for axis, station in zip(axes, stations):
        group = period_pairs.loc[period_pairs["station"] == station].sort_values("period_start")
        axis.plot(
            group["period_start"],
            group["field_actual_et_diagnostic_mm_period"],
            marker="o",
            label="Field-derived ET",
        )
        axis.plot(
            group["period_start"],
            group["ET_MODIS_mm_period"],
            marker="s",
            label="MOD16A2",
        )
        axis.set_ylabel(station)
        axis.grid(alpha=0.25)

    axes[0].legend(ncol=2)
    axes[0].set_title("Diagnostic reproduction: 8-day field-derived ET and MOD16A2")
    axes[-1].set_xlabel("MODIS period start")
    fig.supylabel("ET (mm period$^{-1}$)")
    save_figure(fig, output)


def plot_current_oof(comparison: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    observed = "field_actual_et_diagnostic_mm_period"
    data = comparison.dropna(subset=[observed, "ET_MODIS_mm_period", "ET_predicted_rf_common_mm_period"])

    fig, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(data[observed], data["ET_MODIS_mm_period"], marker="o", facecolors="none", label="MOD16A2")
    axis.scatter(data[observed], data["ET_predicted_rf_common_mm_period"], marker="s", label="Current RF OOF")

    values = np.concatenate([
        data[observed].to_numpy(float),
        data["ET_MODIS_mm_period"].to_numpy(float),
        data["ET_predicted_rf_common_mm_period"].to_numpy(float),
    ])
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    axis.plot([lo, hi], [lo, hi], linestyle="--")

    summary = metrics.loc[
        (metrics["subset"] == "all_diagnostic_reproduction")
        & (metrics["model"].isin(["MODIS", "rf_common_oof"]))
    ].set_index("model")

    text = (
        "Matched OOF periods: " + str(len(data)) + "\n"
        + f"MODIS RMSE={summary.loc['MODIS', 'RMSE']:.2f}, r={summary.loc['MODIS', 'r']:.3f}\n"
        + f"RF OOF RMSE={summary.loc['rf_common_oof', 'RMSE']:.2f}, r={summary.loc['rf_common_oof', 'r']:.3f}"
    )
    axis.text(0.03, 0.97, text, transform=axis.transAxes, va="top")

    axis.set_xlabel("Field-derived ET, diagnostic reproduction (mm period$^{-1}$)")
    axis.set_ylabel("Satellite/model ET (mm period$^{-1}$)")
    axis.set_title("Pre-map checkpoint: coarse-support OOF model vs field-derived series")
    axis.legend()
    axis.grid(alpha=0.25)
    save_figure(fig, output)


def plot_instrument_uncertainty(uncertainty: dict[str, float], metrics: pd.DataFrame, output: Path) -> None:
    if not uncertainty:
        return

    all_rf = metrics.loc[
        (metrics["subset"] == "all_diagnostic_reproduction")
        & (metrics["model"] == "rf_common_oof")
    ]
    observed_r = float(all_rf["r"].iloc[0]) if len(all_rf) else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    noise = uncertainty["noise_variance_fraction"]
    axes[0].bar(["instrument\nnoise", "remaining\nvariance"], [noise, 1.0 - noise])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Fraction of observed variance")
    axes[0].set_title(f"Instrument-noise approximation: {100 * noise:.1f}%")

    axes[1].bar(["estimated\nceiling", "RF OOF\nobserved r"], [uncertainty["r_max"], observed_r])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Correlation coefficient")
    axes[1].set_title("Field uncertainty ceiling vs current coarse-support correlation")

    for axis in axes:
        axis.grid(axis="y", alpha=0.25)

    save_figure(fig, output)


# ============================================================
# Main
# ============================================================


def main() -> None:
    project_root = get_project_root()
    paths = get_paths(project_root)
    require_files(paths)

    paths["tables"].mkdir(parents=True, exist_ok=True)
    paths["figures"].mkdir(parents=True, exist_ok=True)

    field, metadata, master, reference, oof = load_inputs(paths)
    field_all, field_valid = prepare_field_daily(field, metadata, reference)

    scale_factor = build_scale_factor_table(field_valid)
    reference_validation = build_reference_validation_table(field_valid)
    period_pairs = aggregate_to_modis_periods(field_valid, master)
    comparison, metrics = build_oof_comparison(period_pairs, oof)
    uncertainty = build_instrument_uncertainty(period_pairs)

    field_all.to_csv(paths["tables"] / "field_daily_qc.csv", index=False)
    scale_factor.to_csv(paths["tables"] / "field_scale_factor.csv", index=False)
    reference_validation.to_csv(paths["tables"] / "field_reference_eto_check.csv", index=False)
    period_pairs.to_csv(paths["tables"] / "field_modis_period_pairs_diagnostic_reproduction.csv", index=False)
    comparison.to_csv(paths["tables"] / "field_current_oof_comparison.csv", index=False)
    metrics.to_csv(paths["tables"] / "field_current_oof_metrics.csv", index=False)
    pd.DataFrame([uncertainty]).to_csv(paths["tables"] / "field_instrument_uncertainty.csv", index=False)

    plot_daily_series(field_all, paths["figures"] / "FD01_daily_raw_qc.png")
    plot_scale_factor(scale_factor, paths["figures"] / "FD02_scale_factor.png")
    plot_reference_validation(reference_validation, paths["figures"] / "FD03_reference_eto_check.png")
    plot_modis_scatter(period_pairs, paths["figures"] / "FD04_field_vs_modis_scatter.png")
    plot_modis_series(period_pairs, paths["figures"] / "FD05_field_vs_modis_series.png")
    plot_current_oof(comparison, metrics, paths["figures"] / "FD06_current_oof_vs_field.png")
    plot_instrument_uncertainty(uncertainty, metrics, paths["figures"] / "FD07_instrument_uncertainty.png")

    print("Field daily rows:", len(field_all))
    print("Valid field days:", len(field_valid))
    print("Valid days by station:")
    print(field_valid.groupby("station").size().to_string())
    print()
    print("MODIS-period field rows (>=5 valid days):", len(period_pairs))
    print("Rows with field-derived actual ET:", int(period_pairs["field_actual_et_diagnostic_mm_period"].notna().sum()))
    print("Rows matched to current RF OOF:", int(comparison["ET_predicted_rf_common_mm_period"].notna().sum()))
    print()
    print("=== CURRENT PRE-MAP FIELD CHECKPOINT ===")
    print(metrics.to_string(index=False))
    print()
    print("Important interpretation:")
    print("- 'all_diagnostic_reproduction' reproduces the diagnostic conversion logic.")
    print("- Mangrove and Dry forest use an NDVI-derived Kc and are not independent of optical information.")
    print("- 'external_fixed_kc_only' retains Clean pasture, Oil palm, and Banana only.")
    print("- Only Clean pasture is flagged as conforming to the ETgage installation guidance.")
    print("- These OOF comparisons are at MODIS-footprint model support, not a validation of a 20 m map.")
    print()
    print("Saved tables:", paths["tables"])
    print("Saved figures:", paths["figures"])


if __name__ == "__main__":
    main()
