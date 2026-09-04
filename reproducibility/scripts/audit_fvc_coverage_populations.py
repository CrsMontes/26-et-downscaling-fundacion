"""Compare local availability coverage with current FVC-candidate coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_fvc_candidate_stability as stability


KEYS = ["station_id", "period_start"]
THRESHOLDS = (80, 90, 99)


def project_root():
    return Path(__file__).resolve().parents[2]


def definitions():
    common = {
        "null": "null reducer result becomes zero",
        "rounding": "no rounding before threshold comparison; CSV floating-point serialization",
    }
    rows = []
    for source, scale in (("S2", 20), ("HLS", 30)):
        values = [
            ("valid bands", "Blue+Green+Red+NIR+SWIR1+SWIR2", "Green+Red+NIR", False,
             "Availability requires all six common bands; FVC follows historical calibration support."),
            ("temporal support", "union/max of valid daily masks before medoid", "mask of the selected period medoid", False,
             "Availability asks whether any daily observation covers a pixel; FVC asks whether the medoid candidate is valid."),
            ("water mask", "not applied", "not applied to coverage; NDWI<=0 only for NDVI candidates", True,
             "Both coverage measures precede water exclusion."),
            ("scale", f"{scale} m", f"{scale} m", True, "Source-specific scale is identical."),
            ("CRS/grid", "EPSG:32618 via get_coverage_fraction", "medoid default source grid at requested scale", False,
             "Availability uses the common analysis CRS; FVC uses the accepted source medoid grid."),
            ("reducer", "mean of uint8 union mask after unmask(0)", "mean of uint8 medoid common mask after unmask(0)", False,
             "Reducer is mean in both, but the input masks differ."),
            ("denominator", "all reducer grid cells intersecting footprint", "all reducer grid cells intersecting footprint", True,
             "Masked cells are converted to zero before the mean."),
            ("null", common["null"], common["null"], True, "Same explicit zero fallback."),
            ("rounding", common["rounding"], common["rounding"], True, "Thresholds are applied to full stored precision."),
        ]
        for component, availability, candidate, same, reason in values:
            rows.append({"source": source, "component": component,
                         "availability_definition": availability,
                         "fvc_candidate_definition": candidate, "same": same, "reason": reason})
    return pd.DataFrame(rows)


def load_tables(source):
    root = project_root()
    availability_path = root / f"outputs/diagnostics/2020_2024/availability/raw/{source.lower()}_station_period.csv"
    availability = pd.read_csv(availability_path, dtype={"station_id": str})
    coverage_column = (
        "continuous_valid_coverage_pct" if source == "S2"
        else "combined_continuous_valid_coverage_pct"
    )
    availability = availability[KEYS + [coverage_column]].rename(
        columns={coverage_column: "availability_coverage"}
    )
    candidates, _ = stability.load_source_chunks(source, range(2020, 2025))
    candidates = candidates[KEYS + ["optical_coverage_pct"]].rename(
        columns={"optical_coverage_pct": "fvc_candidate_coverage"}
    )
    joined = availability.merge(candidates, on=KEYS, validate="one_to_one")
    if len(joined) != 1150:
        raise RuntimeError(f"{source} coverage join is not the full 1150-key universe")
    joined["delta"] = joined.fvc_candidate_coverage - joined.availability_coverage
    return joined


def run():
    output = project_root() / "outputs/diagnostics/2020_2024/fvc_coverage_population_audit"
    output.mkdir(parents=True, exist_ok=True)
    definitions().to_csv(output / "coverage_method_comparison.csv", index=False)
    rows, counts = [], []
    for source in ("S2", "HLS"):
        joined = load_tables(source)
        for threshold in THRESHOLDS:
            availability_pass = joined.availability_coverage.ge(threshold)
            candidate_pass = joined.fvc_candidate_coverage.ge(threshold)
            counts.append({"source": source, "threshold": threshold,
                           "availability_n": int(availability_pass.sum()),
                           "fvc_candidate_n": int(candidate_pass.sum()),
                           "classification_differences": int(availability_pass.ne(candidate_pass).sum())})
            changed = joined.loc[availability_pass.ne(candidate_pass)].copy()
            changed["source"] = source
            changed["threshold"] = threshold
            changed["availability_pass"] = availability_pass.loc[changed.index]
            changed["candidate_pass"] = candidate_pass.loc[changed.index]
            changed["availability_distance_to_threshold"] = changed.availability_coverage - threshold
            changed["candidate_distance_to_threshold"] = changed.fvc_candidate_coverage - threshold
            rows.append(changed)
    differences = pd.concat(rows, ignore_index=True)
    differences.to_csv(output / "coverage_threshold_classification_differences.csv", index=False)
    counts = pd.DataFrame(counts)
    counts.to_csv(output / "coverage_threshold_counts.csv", index=False)
    recommendation = {
        "authoritative_for_fvc": "fvc_candidate_coverage",
        "reason": "It exactly implements historical FVC coverage: common Green+Red+NIR support on the selected medoid before water exclusion.",
        "thresholds_changed": False,
        "earth_engine_requests": 0,
    }
    (output / "coverage_audit_summary.json").write_text(
        json.dumps(recommendation, indent=2), encoding="utf-8"
    )
    print(counts.to_string(index=False))
    print(differences.to_string(index=False))
    return counts, differences


if __name__ == "__main__":
    run()
