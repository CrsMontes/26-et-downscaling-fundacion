"""Summarize Landsat LST availability and join Phase 2A locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


THRESHOLDS = (80, 90, 99)
VIEWS = {"L8_ONLY": "l8_only", "L8_L9_COMBINED": "l8_l9_combined"}
KEYS = ["station_id", "period_start"]


def project_root():
    return Path(__file__).resolve().parents[1]


def thermal_root(label):
    if label != "2020_2024":
        raise ValueError("Only thermal_availability/2020_2024 is allowed")
    return project_root() / "outputs" / "diagnostics" / label / "thermal_availability"


def phase2a_root(label):
    return project_root() / "outputs" / "diagnostics" / label / "availability" / "raw"


def expected_view_rows(data):
    return len(data)


def view_table(data, view, prefix):
    columns = {
        f"{prefix}_products": "products",
        f"{prefix}_unique_dates": "unique_dates",
        f"{prefix}_dates_with_valid_lst": "dates_with_valid_lst",
        f"{prefix}_acquisition_present": "acquisition_present",
        f"{prefix}_any_valid_lst": "any_valid_lst",
        f"{prefix}_l8_products": "l8_products",
        f"{prefix}_l9_products": "l9_products",
        f"{prefix}_sensors_present": "sensors_present",
        f"{prefix}_valid_area_m2": "valid_area_m2",
        f"{prefix}_valid_coverage_pct": "valid_coverage_pct",
        f"{prefix}_st_qa_mean_k": "st_qa_mean_k",
        f"{prefix}_historical_dn_ge_293_any_valid_lst": "historical_dn_ge_293_any_valid_lst",
        f"{prefix}_historical_dn_ge_293_coverage_pct": "historical_dn_ge_293_coverage_pct",
    }
    result = data[KEYS + list(columns)].rename(columns=columns).copy()
    result["view"] = view
    return result


def build_intersections(thermal, modis, s2, hls, s1):
    as_set = lambda frame: set(map(tuple, frame[KEYS].values))
    target = as_set(modis[modis.modis_good.eq(1)])
    rows = []
    matrix = []
    sar_sets = {}
    for orbit in (77, 142):
        subset = s1[
            s1.relative_orbit.eq(orbit) & s1.has_valid_vv_vh_coverage.eq(1)
        ]
        sar_sets[orbit] = as_set(subset)
    for view, group in thermal.groupby("view"):
        for thermal_threshold in THRESHOLDS:
            lst = as_set(group[group.valid_coverage_pct.ge(thermal_threshold)])
            rows.append([view, thermal_threshold, "target & LST", len(target & lst)])
            for optical_threshold in THRESHOLDS:
                s2_set = as_set(s2[s2.continuous_valid_coverage_pct.ge(optical_threshold)])
                hls_set = as_set(hls[
                    hls.combined_continuous_valid_coverage_pct.ge(optical_threshold)
                ])
                values = {
                    "S2 & LST": len(s2_set & lst),
                    "HLS_COMBINED & LST": len(hls_set & lst),
                    "S2 & R077 & LST": len(s2_set & sar_sets[77] & lst),
                    "S2 & R142 & LST": len(s2_set & sar_sets[142] & lst),
                }
                for name, count in values.items():
                    matrix.append([view, optical_threshold, thermal_threshold, name, count])
                    if optical_threshold == thermal_threshold:
                        rows.append([view, thermal_threshold, name, count])
    return (
        pd.DataFrame(rows, columns=["view", "threshold_pct", "intersection", "count"]),
        pd.DataFrame(matrix, columns=[
            "view", "optical_threshold_pct", "thermal_threshold_pct",
            "intersection", "count",
        ]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--period-label", default="2020_2024")
    args = parser.parse_args(argv)
    root = thermal_root(args.period_label)
    data = pd.read_csv(root / "raw" / "landsat_lst_station_period.csv",
                       dtype={"station_id": str, "period_start": str})
    if len(data) != 1150:
        raise RuntimeError(f"Expected 1150 rows per view, found {len(data)}")
    views = pd.concat([
        view_table(data, view, prefix) for view, prefix in VIEWS.items()
    ], ignore_index=True)
    phase = phase2a_root(args.period_label)
    modis = pd.read_csv(phase / "modis_station_period.csv")
    s2 = pd.read_csv(phase / "s2_station_period.csv")
    hls = pd.read_csv(phase / "hls_station_period.csv")
    s1 = pd.read_csv(phase / "s1_period_station_period.csv")
    primary, matrix = build_intersections(views, modis, s2, hls, s1)
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    views.to_csv(summary / "thermal_availability_long.csv", index=False)
    primary.to_csv(summary / "same_threshold_intersections.csv", index=False)
    matrix.to_csv(summary / "threshold_matrix_3x3.csv", index=False)
    overview = views.groupby("view").agg(
        station_periods=("period_start", "size"),
        acquisitions=("acquisition_present", "sum"),
        any_valid_lst=("any_valid_lst", "sum"),
        historical_dn_ge_293_any_valid_lst=(
            "historical_dn_ge_293_any_valid_lst", "sum"
        ),
        mean_coverage_pct=("valid_coverage_pct", "mean"),
    ).reset_index()
    for threshold in THRESHOLDS:
        counts = views.assign(ok=views.valid_coverage_pct.ge(threshold)).groupby("view").ok.sum()
        overview[f"ge_{threshold}"] = overview.view.map(counts)
        historical = views.assign(
            ok=views.historical_dn_ge_293_coverage_pct.ge(threshold)
        ).groupby("view").ok.sum()
        overview[f"historical_dn_ge_293_ge_{threshold}"] = overview.view.map(historical)
    sensitivity = overview[[
        "view", "station_periods", "any_valid_lst",
        "historical_dn_ge_293_any_valid_lst",
        *[column for threshold in THRESHOLDS for column in (
            f"ge_{threshold}", f"historical_dn_ge_293_ge_{threshold}")],
    ]]
    overview.to_csv(summary / "thermal_availability_overview.csv", index=False)
    sensitivity.to_csv(summary / "dn_293_sensitivity.csv", index=False)
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "summary_manifest.json").write_text(json.dumps({
        "period_label": args.period_label,
        "rows_per_view": 1150,
        "main_comparison": "same optical and thermal threshold",
        "secondary_comparison": "complete 3x3 threshold matrix",
        "training_performed": False,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
