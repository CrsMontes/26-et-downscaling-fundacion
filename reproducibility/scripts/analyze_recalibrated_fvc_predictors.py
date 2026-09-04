"""Validate and compare recalibrated FVC predictor exports entirely locally."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import export_recalibrated_fvc_predictors as exporter


def project_root():
    return Path(__file__).resolve().parents[1]


def load_source(source, calibration):
    frames = []
    for year in exporter.YEARS:
        csv_path, manifest_path = exporter.chunk_paths(source, year)
        keys = exporter.candidate_export.expected_keys(f"{year}-01-01", f"{year + 1}-01-01")
        expected = exporter.expected_chunk_manifest(source, year, calibration, keys)
        selectors = exporter.output_columns(source, calibration)
        if not exporter.validate_existing(csv_path, manifest_path, expected, keys, selectors):
            raise RuntimeError(f"Invalid final chunk: {source} {year}")
        frames.append(pd.read_csv(csv_path, dtype={"station_id": str}))
    table = pd.concat(frames, ignore_index=True)
    exporter.validate_table(table, source, "2020-01-01", "2025-01-01",
                            exporter.output_columns(source, calibration), 1150)
    if table.duplicated(["station_id", "period_start"]).any():
        raise RuntimeError("Duplicate final keys")
    table["year"] = pd.to_datetime(table.period_start).dt.year
    return table


def comparison_metrics(group):
    historical = pd.to_numeric(group.fvc_historical_mean, errors="coerce")
    global_fvc = pd.to_numeric(group.fvc_global_2020_2024_mean, errors="coerce")
    paired = pd.DataFrame({"historical": historical, "global": global_fvc}).dropna()
    delta = paired["global"] - paired["historical"]
    denominator = ((paired.historical - paired.historical.mean()) ** 2).sum()
    return {
        "n_paired": len(paired), "correlation": paired.corr().iloc[0, 1],
        "R2_concordance": 1 - ((paired["global"] - paired.historical) ** 2).sum() / denominator,
        "MAE": delta.abs().mean(), "RMSE": np.sqrt((delta ** 2).mean()),
        "bias_global_minus_historical": delta.mean(), "mean_delta": delta.mean(),
        "median_delta": delta.median(), "p05_delta": delta.quantile(.05),
        "p95_delta": delta.quantile(.95), "maximum_absolute_delta": delta.abs().max(),
        "proportion_abs_delta_gt_0_01": delta.abs().gt(.01).mean(),
        "proportion_abs_delta_gt_0_025": delta.abs().gt(.025).mean(),
        "proportion_abs_delta_gt_0_05": delta.abs().gt(.05).mean(),
        "proportion_abs_delta_gt_0_10": delta.abs().gt(.10).mean(),
    }


def analyze():
    calibration = exporter.freeze_calibrations()
    output = exporter.output_root() / "analysis"; output.mkdir(parents=True, exist_ok=True)
    metrics, clipping, integrity, fold_behavior, all_tables = [], [], [], [], []
    for source in exporter.SOURCES:
        table = load_source(source, calibration); all_tables.append(table)
        for grouping, columns in (("overall", []), ("station", ["station_id"]), ("year", ["year"])):
            groups = [("all", table)] if not columns else table.groupby(columns[0])
            for label, group in groups:
                metrics.append({"source": source, "grouping": grouping, "group": label,
                                **comparison_metrics(group)})
        for variant in ("clipped_low", "clipped_high"):
            old = pd.to_numeric(table[f"fvc_historical_{variant}_fraction"], errors="coerce")
            new = pd.to_numeric(table[f"fvc_global_2020_2024_{variant}_fraction"], errors="coerce")
            delta = new - old
            clipping.append({"source": source, "clipping": variant,
                             "n_paired": int((old.notna() & new.notna()).sum()),
                             "historical_mean_fraction": old.mean(),
                             "global_mean_fraction": new.mean(),
                             "mean_delta_fraction": delta.mean(),
                             "maximum_absolute_delta_fraction": delta.abs().max()})
        global_values = pd.to_numeric(table.fvc_global_2020_2024_mean, errors="coerce")
        for fold, test_stations in exporter.SPATIAL.items():
            column = f"fvc_spatial_train_excl_fold{fold}_mean"
            variant = pd.to_numeric(table[column], errors="coerce")
            delta = variant - global_values
            test = table.station_id.isin(test_stations)
            fold_behavior.append({"source": source, "scheme": "spatial", "fold": fold,
                "excluded_test_group": ";".join(test_stations), "column": column,
                "n_all_paired": int((variant.notna() & global_values.notna()).sum()),
                "mean_delta_vs_global_all": delta.mean(), "max_abs_delta_vs_global_all": delta.abs().max(),
                "n_test_paired": int((test & variant.notna() & global_values.notna()).sum()),
                "mean_delta_vs_global_test": delta.loc[test].mean(),
                "max_abs_delta_vs_global_test": delta.loc[test].abs().max()})
        for year in exporter.YEARS:
            column = f"fvc_temporal_train_excl_{year}_mean"
            variant = pd.to_numeric(table[column], errors="coerce")
            delta = variant - global_values; test = table.year.eq(year)
            fold_behavior.append({"source": source, "scheme": "temporal", "fold": year,
                "excluded_test_group": str(year), "column": column,
                "n_all_paired": int((variant.notna() & global_values.notna()).sum()),
                "mean_delta_vs_global_all": delta.mean(), "max_abs_delta_vs_global_all": delta.abs().max(),
                "n_test_paired": int((test & variant.notna() & global_values.notna()).sum()),
                "mean_delta_vs_global_test": delta.loc[test].mean(),
                "max_abs_delta_vs_global_test": delta.loc[test].abs().max()})
        mean_columns = [column for column in table if column.startswith("fvc_") and column.endswith("_mean")]
        albedo = (pd.to_numeric(table.hls_Albedo_mean, errors="coerce").mask(lambda value: value <= -9990)
                  if source == "HLS" else None)
        integrity.append({"source": source, "rows": len(table),
            "unique_keys": len(table[["station_id", "period_start"]].drop_duplicates()),
            "valid_pixel_count_min": pd.to_numeric(table.valid_pixel_count).min(),
            "fvc_min": table[mean_columns].min().min(), "fvc_max": table[mean_columns].max().max(),
            "rows_without_valid_pixels": int(pd.to_numeric(table.valid_pixel_count).eq(0).sum()),
            "albedo_available": int(albedo.notna().sum()) if source == "HLS" else None,
            "albedo_min": albedo.min() if source == "HLS" else None,
            "albedo_max": albedo.max() if source == "HLS" else None})
        table.to_csv(output / f"{source.lower()}_recalibrated_fvc_station_period.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "fvc_historical_vs_global_metrics.csv", index=False)
    pd.DataFrame(clipping).to_csv(output / "fvc_clipping_comparison.csv", index=False)
    pd.DataFrame(integrity).to_csv(output / "recalibrated_fvc_integrity.csv", index=False)
    pd.DataFrame(fold_behavior).to_csv(output / "fold_specific_fvc_behavior.csv", index=False)
    execution = {
        "earth_engine_download_requests": 14,
        "request_breakdown": {
            "S2_2020_initial_plus_validation_retries": 3,
            "S2_2021_2024": 4,
            "HLS_2020_2023": 4,
            "HLS_2024_stalled_annual_interrupted": 1,
            "HLS_2024_deterministic_half_year_subchunks": 2,
        },
        "additional_requests_after_completed_HLS_2024": 0,
        "training_performed": False,
    }
    (output / "execution_ledger.json").write_text(json.dumps(execution, indent=2), encoding="utf-8")
    print(pd.DataFrame(integrity).to_string(index=False))
    print(pd.DataFrame(metrics).query("grouping == 'overall'").to_string(index=False))
    print(pd.DataFrame(clipping).to_string(index=False))
    return pd.DataFrame(integrity), pd.DataFrame(metrics), pd.DataFrame(clipping)


if __name__ == "__main__":
    analyze()
