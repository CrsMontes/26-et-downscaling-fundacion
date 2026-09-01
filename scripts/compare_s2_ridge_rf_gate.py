"""Consolidate the fixed GE90 S2 Ridge-versus-RF gate from saved OOF data."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
import screen_optical_algorithms as screening


ALGORITHMS = ("ridge", "random_forest")
CONFIGURATION = "s2_base_plus_seasonality"
KEYS = ["station_id", "period_start"]
METRICS = ["R2", "RMSE", "MAE", "BIAS", "KGE"]
MANGROVE_STATION = "ST04"


def project_root():
    return Path(__file__).resolve().parents[1]


def load_oof(root):
    path = root / "outputs/diagnostics/2020_2024/feature_family_screening/oof_predictions.csv"
    data = pd.read_csv(path, dtype={"station_id": str})
    selected = data.loc[
        data["source"].eq("S2")
        & data["configuration"].eq(CONFIGURATION)
        & data["algorithm"].isin(ALGORITHMS)
    ].copy()
    expected = 550 * 2 * 2
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} saved OOF rows, found {len(selected)}")
    identity = ["algorithm", "split_type", *KEYS]
    if selected.duplicated(identity).any():
        raise RuntimeError("OOF predictions are not unique")
    counts = selected.groupby(["algorithm", "split_type"]).size()
    if not counts.eq(550).all():
        raise RuntimeError(f"OOF populations differ: {counts.to_dict()}")
    return selected


def metric_table(data, dimensions):
    rows = []
    for values, group in data.groupby(dimensions, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(dimensions, values))
        row.update(screening.calculate_metrics(group.Kc_target, group.prediction))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_errors(data):
    index = ["split_type", "fold", *KEYS, "year", "spatial_block", "Kc_target"]
    wide = data.pivot(index=index, columns="algorithm", values="prediction").reset_index()
    for algorithm in ALGORITHMS:
        error = wide[algorithm] - wide.Kc_target
        wide[f"error_{algorithm}"] = error
        wide[f"absolute_error_{algorithm}"] = error.abs()
        wide[f"squared_error_{algorithm}"] = error.pow(2)
    wide["delta_absolute_error_ridge_minus_rf"] = (
        wide.absolute_error_ridge - wide.absolute_error_random_forest
    )
    wide["delta_squared_error_ridge_minus_rf"] = (
        wide.squared_error_ridge - wide.squared_error_random_forest
    )
    return wide


def target_strata(data):
    unique = data.loc[data.algorithm.eq("ridge"), KEYS + ["Kc_target"]].drop_duplicates()
    low, high = unique.Kc_target.quantile([0.25, 0.75]).tolist()
    result = data.copy()
    result["Kc_stratum"] = np.select(
        [result.Kc_target.le(low), result.Kc_target.ge(high)],
        ["low_le_q25", "high_ge_q75"], default="middle",
    )
    metrics = metric_table(
        result.loc[result.Kc_stratum.ne("middle")],
        ["algorithm", "split_type", "Kc_stratum"],
    )
    return metrics, {"q25": float(low), "q75": float(high)}


def build_outputs(data):
    overall = metric_table(data, ["algorithm", "split_type"])
    by_fold = metric_table(data, ["algorithm", "split_type", "fold"])
    by_station = metric_table(data, ["algorithm", "split_type", "station_id"])
    by_year = metric_table(data, ["algorithm", "split_type", "year"])
    paired = paired_errors(data)
    strata, cutoffs = target_strata(data)
    mangrove = by_station.loc[by_station.station_id.eq(MANGROVE_STATION)].copy()
    fold_stability = by_fold.groupby(["algorithm", "split_type"], as_index=False).agg(
        fold_RMSE_mean=("RMSE", "mean"), fold_RMSE_sd=("RMSE", "std"),
        fold_RMSE_min=("RMSE", "min"), fold_RMSE_max=("RMSE", "max"),
        fold_MAE_mean=("MAE", "mean"), fold_MAE_sd=("MAE", "std"),
        fold_BIAS_mean=("BIAS", "mean"), fold_BIAS_sd=("BIAS", "std"),
    )
    return {
        "metrics_overall": overall, "metrics_by_fold": by_fold,
        "metrics_by_station": by_station, "metrics_by_year": by_year,
        "paired_errors": paired, "metrics_by_target_stratum": strata,
        "mangrove_metrics": mangrove, "fold_stability": fold_stability,
    }, cutoffs


def main():
    root = project_root()
    data = load_oof(root)
    tables, cutoffs = build_outputs(data)
    output = root / "outputs/diagnostics/2020_2024/s2_ridge_rf_final_gate"
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    manifest = {
        "population": "paired_GE90", "rows_per_algorithm_split": 550,
        "features": "S2 common11 + ETo drivers + seasonality",
        "feature_count": 20, "algorithms": list(ALGORITHMS),
        "configuration": CONFIGURATION, "target_stratum_cutoffs": cutoffs,
        "mangrove_station": MANGROVE_STATION,
        "source_predictions_reused": "feature_family_screening/oof_predictions.csv",
        "tuning_performed": False, "production_specification_changed": False,
        "winner_frozen": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(tables["metrics_overall"].to_string(index=False))
    print("\nMangrove (ST04):")
    print(tables["mangrove_metrics"].to_string(index=False))
    print(f"\nOutput: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
