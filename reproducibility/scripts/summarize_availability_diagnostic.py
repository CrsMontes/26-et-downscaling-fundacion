"""Create local summaries and availability intersections; no EE access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


THRESHOLDS = (80, 90, 99)
KEYS = ["station_id", "period_start"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def availability_root(period_label: str) -> Path:
    if period_label != "2020_2024":
        raise ValueError("Only the approved 2020_2024 namespace is allowed")
    return project_root() / "outputs" / "diagnostics" / period_label / "availability"


def read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"station_id": str, "period_start": str})


def add_year(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["year"] = pd.to_datetime(result["period_start"], errors="raise").dt.year
    return result


def optical_long(s2: pd.DataFrame, hls: pd.DataFrame) -> pd.DataFrame:
    frames = []
    s2_frame = s2[KEYS + ["continuous_valid_coverage_pct"]].copy()
    s2_frame["source"] = "S2"
    frames.append(s2_frame)
    for prefix, label in (("s30", "HLS_S30"), ("l30", "HLS_L30"),
                          ("combined", "HLS_COMBINED")):
        frame = hls[KEYS + [f"{prefix}_continuous_valid_coverage_pct"]].copy()
        frame = frame.rename(columns={
            f"{prefix}_continuous_valid_coverage_pct": "continuous_valid_coverage_pct"
        })
        frame["source"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def threshold_summary(optical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = add_year(optical)
    for threshold in THRESHOLDS:
        eligible = data["continuous_valid_coverage_pct"].ge(threshold)
        grouped = data.assign(eligible=eligible).groupby(
            ["source", "station_id", "year"], dropna=False
        )["eligible"].agg(["sum", "count"]).reset_index()
        grouped["threshold_pct"] = threshold
        grouped["availability_pct"] = grouped["sum"] / grouped["count"] * 100
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def intersection_rows(modis, optical, s1):
    target = set(map(tuple, modis.loc[modis["modis_good"].eq(1), KEYS].values))
    rows = [{"intersection": "target", "threshold_pct": pd.NA,
             "pass": pd.NA, "relative_orbit": pd.NA, "count": len(target)}]
    optical_sets = {}
    for source, group in optical.groupby("source"):
        for threshold in THRESHOLDS:
            keys = set(map(tuple, group.loc[
                group["continuous_valid_coverage_pct"].ge(threshold), KEYS
            ].values))
            optical_sets[(source, threshold)] = keys
            rows.append({"intersection": f"target ∩ {source}",
                         "threshold_pct": threshold, "pass": pd.NA,
                         "relative_orbit": pd.NA, "count": len(target & keys)})
    for threshold in THRESHOLDS:
        shared = optical_sets[("S2", threshold)] & optical_sets[("HLS_COMBINED", threshold)]
        rows.append({"intersection": "target ∩ S2 ∩ HLS_COMBINED",
                     "threshold_pct": threshold, "pass": pd.NA,
                     "relative_orbit": pd.NA, "count": len(target & shared)})
    for (orbit_pass, orbit), group in s1.groupby(["pass", "relative_orbit"], dropna=False):
        sar = set(map(tuple, group.loc[
            group["has_valid_vv_vh_coverage"].eq(1), KEYS
        ].values))
        rows.append({"intersection": "target ∩ S1", "threshold_pct": pd.NA,
                     "pass": orbit_pass, "relative_orbit": orbit,
                     "count": len(target & sar)})
        for source in ("S2", "HLS_S30", "HLS_L30", "HLS_COMBINED"):
            for threshold in THRESHOLDS:
                count = len(target & sar & optical_sets[(source, threshold)])
                rows.append({"intersection": f"target ∩ {source} ∩ S1",
                             "threshold_pct": threshold, "pass": orbit_pass,
                             "relative_orbit": orbit, "count": count})
    result = pd.DataFrame(rows)
    result["target_valid_count"] = len(target)
    result["retained_target_pct"] = result["count"].div(len(target)).mul(100) if target else pd.NA
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--period-label", default="2020_2024")
    args = parser.parse_args(argv)
    root = availability_root(args.period_label)
    raw = root / "raw"
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    modis = read_required(raw / "modis_station_period.csv")
    s2 = read_required(raw / "s2_station_period.csv")
    hls = read_required(raw / "hls_station_period.csv")
    s1 = read_required(raw / "s1_period_station_period.csv")
    optical = optical_long(s2, hls)
    threshold_summary(optical).to_csv(
        summary / "optical_availability_by_threshold.csv", index=False
    )
    intersection_rows(modis, optical, s1).to_csv(
        summary / "availability_intersections.csv", index=False
    )
    s1_summary = add_year(s1).groupby(
        ["pass", "relative_orbit", "station_id", "year"], dropna=False
    ).agg(
        station_periods=("period_start", "size"),
        scenes=("products", "sum"),
        unique_date_period_counts=("unique_dates", "sum"),
        periods_with_acquisition=("has_acquisition", "sum"),
        periods_with_valid_coverage=("has_valid_vv_vh_coverage", "sum"),
        mean_valid_coverage_pct=("continuous_valid_vv_vh_coverage_pct", "mean"),
        mean_angle_deg=("angle_mean_deg", "mean"),
    ).reset_index()
    s1_summary.to_csv(summary / "sentinel1_geometry_summary.csv", index=False)
    metadata = {
        "period_label": args.period_label,
        "local_summary_only": True,
        "thresholds_are_diagnostic": list(THRESHOLDS),
        "training_performed": False,
    }
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "summary_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
