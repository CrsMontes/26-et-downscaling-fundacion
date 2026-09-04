"""Analyze exported FVC candidates entirely locally after gated extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import export_fvc_calibration_candidates as exporter
import preflight_fvc_recalibration as preflight


HISTORICAL = {
    "S2": {"low": 0.30906052790151156, "high": 0.9240448371180946, "n": 498},
    "HLS": {"low": 0.411908487478892, "high": 0.9082510914569858, "n": 381},
}
THRESHOLDS = (80, 90, 99)
YEARS = (2020, 2021, 2022, 2023, 2024)
SPATIAL_FOLDS = {1: ("ST05",), 2: ("ST02", "ST03"), 3: ("ST04",), 4: ("ST01",)}


def project_root():
    return Path(__file__).resolve().parents[2]


def diagnostic_root():
    return exporter.output_root()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_chunks(source, years):
    frames = []
    manifests = []
    for year in years:
        csv_path, manifest_path = exporter.chunk_paths(
            source, f"{year}-01-01", f"{year + 1}-01-01"
        )
        if not csv_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing completed {source} {year} chunk")
        keys = exporter.expected_keys(f"{year}-01-01", f"{year + 1}-01-01")
        expected = exporter.expected_manifest(
            source, f"{year}-01-01", f"{year + 1}-01-01", keys
        )
        if not exporter.validate_existing_chunk(csv_path, manifest_path, expected, keys):
            raise RuntimeError(f"Invalid completed {source} {year} chunk")
        frame = pd.read_csv(csv_path, dtype={"station_id": str})
        frames.append(frame)
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    table = pd.concat(frames, ignore_index=True)
    if table.duplicated(["station_id", "period_start"]).any():
        raise RuntimeError(f"Duplicate {source} keys across chunks")
    return table, manifests


def calculate(table, threshold):
    renamed = table.rename(columns={
        "optical_coverage_pct": "coverage_pct",
        "ndvi_p05_nonwater": "NDVI_p05",
        "ndvi_p95_nonwater": "NDVI_p95",
    }).copy()
    renamed.loc[
        pd.to_numeric(renamed["coverage_pct"], errors="coerce").lt(threshold),
        "coverage_pct",
    ] = np.nan
    # The pure historical function uses >=80. Re-map selected rows to 80 so
    # threshold sensitivity changes only the explicit selection criterion.
    selected = pd.to_numeric(
        table["optical_coverage_pct"], errors="coerce"
    ).ge(threshold)
    renamed = renamed.loc[selected].copy()
    if renamed.empty:
        raise RuntimeError(f"No candidates at GE{threshold}")
    renamed["coverage_pct"] = 80.0
    result, eligible = preflight.calculate_endmembers(renamed)
    return result, table.loc[eligible.index].copy()


def compare_historical(source, result):
    expected = HISTORICAL[source]
    return {
        "n_expected": expected["n"], "n_actual": result["n_observations"],
        "low_expected": expected["low"], "low_actual": result["ndvi_low_endmember"],
        "low_absolute_difference": result["ndvi_low_endmember"] - expected["low"],
        "low_relative_difference": (
            result["ndvi_low_endmember"] - expected["low"]
        ) / expected["low"],
        "high_expected": expected["high"], "high_actual": result["ndvi_high_endmember"],
        "high_absolute_difference": result["ndvi_high_endmember"] - expected["high"],
        "high_relative_difference": (
            result["ndvi_high_endmember"] - expected["high"]
        ) / expected["high"],
    }


def historical_gate(source, table):
    subset = table.loc[pd.to_datetime(table["period_start"]).dt.year.between(2021, 2023)]
    result, eligible = calculate(subset, 80)
    comparison = compare_historical(source, result)
    passed = (
        comparison["n_actual"] == comparison["n_expected"]
        and abs(comparison["low_absolute_difference"])
        <= exporter.HISTORICAL_ABSOLUTE_TOLERANCE
        and abs(comparison["high_absolute_difference"])
        <= exporter.HISTORICAL_ABSOLUTE_TOLERANCE
        and abs(comparison["low_relative_difference"])
        <= exporter.HISTORICAL_RELATIVE_TOLERANCE
        and abs(comparison["high_relative_difference"])
        <= exporter.HISTORICAL_RELATIVE_TOLERANCE
    )
    return passed, comparison, subset, eligible


def preserve_hls_historical(table, manifests, comparison):
    output = diagnostic_root() / "historical_reconstruction"
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "hls_corrected_candidates_2021_2023.csv"
    table.sort_values(["period_start", "station_id"]).to_csv(candidate_path, index=False)
    manifest = {
        "description": (
            "historical corrected HLS candidate table independently reconstructed "
            "from current MGRS-corrected code"
        ),
        "source": "HLS", "start_date": "2021-01-01",
        "end_date_exclusive": "2024-01-01", "rows": len(table),
        "unique_keys": len(table[["station_id", "period_start"]].drop_duplicates()),
        "csv_sha256": file_sha256(candidate_path), "comparison": comparison,
        "source_chunk_manifests": manifests,
        "historical_config_overwritten": False,
    }
    (output / "hls_historical_reconstruction_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return candidate_path


def attach_blocks(table):
    master = pd.read_csv(
        project_root() / "outputs/diagnostics/2020_2024/optical_source_experiment"
        / "population/paired_master.csv", dtype={"station_id": str},
    )[["station_id", "period_start", "spatial_block"]]
    result = table.merge(
        master, on=["station_id", "period_start"], validate="one_to_one"
    )
    result["year"] = pd.to_datetime(result["period_start"]).dt.year
    return result


def calibration_record(source, threshold, scheme, split_type, fold, test_group, table):
    result, _ = calculate(table, threshold)
    historical = HISTORICAL[source]
    return {
        "source": source, "threshold": threshold, "scheme": scheme,
        "split_type": split_type, "fold": fold, "test_group": test_group,
        "n": result["n_observations"],
        "low": result["ndvi_low_endmember"], "high": result["ndvi_high_endmember"],
        "delta_low_vs_historical": result["ndvi_low_endmember"] - historical["low"],
        "delta_high_vs_historical": result["ndvi_high_endmember"] - historical["high"],
    }


def sensitivity_table(source, table):
    data = attach_blocks(table)
    rows = []
    for threshold in THRESHOLDS:
        global_record = calibration_record(
            source, threshold, "experimental_global_2020_2024", "global", 0,
            "all", data,
        )
        rows.append(global_record)
        global_low, global_high = global_record["low"], global_record["high"]
        for year in YEARS:
            training = data.loc[data["year"].ne(year)]
            if training["year"].eq(year).any():
                raise RuntimeError("Temporal test candidates entered calibration")
            rows.append(calibration_record(
                source, threshold, "fold_specific", "temporal", year, str(year), training
            ))
        for fold, blocks in SPATIAL_FOLDS.items():
            training = data.loc[~data["station_id"].isin(blocks)]
            if training["station_id"].isin(blocks).any():
                raise RuntimeError("Spatial test candidates entered calibration")
            rows.append(calibration_record(
                source, threshold, "fold_specific", "spatial", fold,
                ";".join(blocks), training,
            ))
        for row in rows:
            if row["source"] == source and row["threshold"] == threshold:
                row["delta_low_vs_global_ge_threshold"] = row["low"] - global_low
                row["delta_high_vs_global_ge_threshold"] = row["high"] - global_high
    return pd.DataFrame(rows)


def classify_stability(source_rows):
    global80 = source_rows.loc[
        source_rows["scheme"].eq("experimental_global_2020_2024")
        & source_rows["threshold"].eq(80)
    ].iloc[0]
    deviations = [
        abs(global80["delta_low_vs_historical"]),
        abs(global80["delta_high_vs_historical"]),
        source_rows["delta_low_vs_global_ge_threshold"].abs().max(),
        source_rows["delta_high_vs_global_ge_threshold"].abs().max(),
    ]
    score = float(max(deviations))
    if score <= exporter.STABILITY_THRESHOLDS["negligible_max_abs_ndvi"]:
        label = "NEGLIGIBLE"
    elif score <= exporter.STABILITY_THRESHOLDS["modest_max_abs_ndvi"]:
        label = "MODEST"
    else:
        label = "MATERIAL"
    return {
        "source": global80["source"], "stability_score_max_abs_ndvi": score,
        "classification": label,
        "low_min": source_rows["low"].min(), "low_max": source_rows["low"].max(),
        "low_sd": source_rows["low"].std(ddof=1),
        "high_min": source_rows["high"].min(), "high_max": source_rows["high"].max(),
        "high_sd": source_rows["high"].std(ddof=1),
    }


def run_gate1():
    hls, manifests = load_source_chunks("HLS", (2021, 2022, 2023))
    passed, comparison, historical_table, _ = historical_gate("HLS", hls)
    if not passed:
        raise RuntimeError(f"HLS historical Gate 1 failed: {comparison}")
    path = preserve_hls_historical(historical_table, manifests, comparison)
    print(json.dumps({"gate_1_passed": True, "comparison": comparison,
                      "preserved_table": str(path)}, indent=2))


def run_full():
    all_tables = {}
    gates = {}
    sensitivities = []
    for source in preflight.SOURCES:
        table, _ = load_source_chunks(source, YEARS)
        passed, comparison, _, _ = historical_gate(source, table)
        if not passed and source != "S2":
            raise RuntimeError(f"{source} historical gate failed: {comparison}")
        comparison["status"] = (
            "exact_historical_reproduction" if passed else
            "accepted_method_drift_current_six_band_medoid"
        )
        gates[source] = comparison
        all_tables[source] = table
        sensitivities.append(sensitivity_table(source, table))
    sensitivity = pd.concat(sensitivities, ignore_index=True)
    stability = pd.DataFrame([
        classify_stability(group) for _, group in sensitivity.groupby("source")
    ])
    output = diagnostic_root() / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(output / "endmember_sensitivity.csv", index=False)
    stability.to_csv(output / "endmember_stability_classification.csv", index=False)
    decomposition_rows = []
    for source in preflight.SOURCES:
        table = all_tables[source]
        current_2021_2023, _ = calculate(
            table.loc[pd.to_datetime(table["period_start"]).dt.year.between(2021, 2023)], 80
        )
        current_2020_2024, _ = calculate(table, 80)
        historical = HISTORICAL[source]
        decomposition_rows.extend([
            {"source": source, "effect": "method_effect",
             "comparison": "CURRENT_METHOD_2021_2023_minus_HISTORICAL_FIXED_2021_2023",
             "delta_low": current_2021_2023["ndvi_low_endmember"] - historical["low"],
             "delta_high": current_2021_2023["ndvi_high_endmember"] - historical["high"],
             "reference_n": historical["n"], "comparison_n": current_2021_2023["n_observations"]},
            {"source": source, "effect": "period_effect",
             "comparison": "CURRENT_METHOD_2020_2024_minus_CURRENT_METHOD_2021_2023",
             "delta_low": current_2020_2024["ndvi_low_endmember"] - current_2021_2023["ndvi_low_endmember"],
             "delta_high": current_2020_2024["ndvi_high_endmember"] - current_2021_2023["ndvi_high_endmember"],
             "reference_n": current_2021_2023["n_observations"], "comparison_n": current_2020_2024["n_observations"]},
        ])
    decomposition = pd.DataFrame(decomposition_rows)
    for column in ("delta_low", "delta_high"):
        decomposition[f"{column}_classification"] = decomposition[column].abs().map(
            lambda value: "NEGLIGIBLE" if value <= .01 else ("MODEST" if value <= .05 else "MATERIAL")
        )
    decomposition.to_csv(output / "method_period_effect_decomposition.csv", index=False)
    global80 = sensitivity.loc[
        sensitivity["scheme"].eq("experimental_global_2020_2024")
        & sensitivity["threshold"].eq(80)
    ]
    experimental = {
        row.source: {"ndvi_low_endmember": row.low,
                     "ndvi_high_endmember": row.high, "n_observations": int(row.n)}
        for row in global80.itertuples()
    }
    (output / "experimental_fvc_endmembers_2020_2024.json").write_text(
        json.dumps({
            "method": "two_stage_global_percentile", "coverage_threshold_pct": 80,
            "period": {"start": "2020-01-01", "end_exclusive": "2025-01-01"},
            "sources": experimental, "historical_reproduction": gates,
            "historical_config_overwritten": False,
        }, indent=2), encoding="utf-8",
    )
    print(json.dumps({"historical_gates": gates, "experimental": experimental,
                      "stability": stability.to_dict("records")}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate1", "full"), required=True)
    args = parser.parse_args(argv)
    if args.mode == "gate1":
        run_gate1()
    else:
        run_full()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
