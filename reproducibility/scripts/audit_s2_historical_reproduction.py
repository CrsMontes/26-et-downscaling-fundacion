"""Audit the failed historical S2 FVC-candidate reproduction entirely locally."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import export_fvc_calibration_candidates as exporter


HISTORICAL_COMMIT = "8722b8b"
HISTORICAL_PATH = "outputs/diagnostics/_fvc_endmember_calibration_checkpoint_v2.csv"
YEARS = (2021, 2022, 2023)
STATION_MAP = {f"{i:020d}": f"ST{i + 1:02d}" for i in range(5)}
SPATIAL_FOLD = {"ST05": 1, "ST02": 2, "ST03": 2, "ST04": 3, "ST01": 4}
TOLERANCES = (0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.005, 0.01)


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return root() / "outputs/diagnostics/2020_2024/s2_historical_reproduction_audit"


def git_bytes(spec: str) -> bytes:
    return subprocess.run(
        ["git", "show", spec], cwd=root(), check=True, capture_output=True
    ).stdout


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return exporter.sha256_file(path)


def load_historical() -> pd.DataFrame:
    raw = git_bytes(f"{HISTORICAL_COMMIT}:{HISTORICAL_PATH}")
    table = pd.read_csv(io.BytesIO(raw), dtype={"station_id": str})
    table = table.loc[table.source.eq("S2")].copy()
    table["historical_station_id"] = table.station_id
    table["station_id"] = table.station_id.map(STATION_MAP)
    if table.station_id.isna().any():
        raise RuntimeError("Historical station identifiers cannot be mapped")
    return table


def load_current() -> tuple[pd.DataFrame, list[dict]]:
    frames, records = [], []
    for year in YEARS:
        csv_path, manifest_path = exporter.chunk_paths(
            "S2", f"{year}-01-01", f"{year + 1}-01-01"
        )
        keys = exporter.expected_keys(f"{year}-01-01", f"{year + 1}-01-01")
        expected = exporter.expected_manifest(
            "S2", f"{year}-01-01", f"{year + 1}-01-01", keys
        )
        valid = exporter.validate_existing_chunk(csv_path, manifest_path, expected, keys)
        if not valid:
            raise RuntimeError(f"Invalid protected chunk: {csv_path}")
        frames.append(pd.read_csv(csv_path, dtype={"station_id": str}))
        records.append({
            "kind": "current_s2_chunk", "year": year,
            "path": str(csv_path.relative_to(root())), "sha256": sha256_file(csv_path),
            "manifest_path": str(manifest_path.relative_to(root())),
            "manifest_sha256": sha256_file(manifest_path), "validated": True,
        })
    current = pd.concat(frames, ignore_index=True)
    if len(current) != 690 or current.duplicated(["station_id", "period_start"]).any():
        raise RuntimeError("Current S2 universe is not 690 unique station-periods")
    return current, records


def validate_all_protected_chunks() -> list[dict]:
    records = []
    for source in ("S2", "HLS"):
        for year in range(2020, 2025):
            csv_path, manifest_path = exporter.chunk_paths(
                source, f"{year}-01-01", f"{year + 1}-01-01"
            )
            keys = exporter.expected_keys(f"{year}-01-01", f"{year + 1}-01-01")
            expected = exporter.expected_manifest(
                source, f"{year}-01-01", f"{year + 1}-01-01", keys
            )
            valid = exporter.validate_existing_chunk(csv_path, manifest_path, expected, keys)
            if not valid:
                raise RuntimeError(f"Invalid protected chunk: {csv_path}")
            records.append({"kind": "protected_candidate_chunk", "source": source,
                "year": year, "path": str(csv_path.relative_to(root())),
                "sha256": sha256_file(csv_path),
                "manifest_path": str(manifest_path.relative_to(root())),
                "manifest_sha256": sha256_file(manifest_path), "validated": True})
    return records


def eligible_historical(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame.coverage_pct, errors="coerce").ge(80)
        & pd.to_numeric(frame.nonwater_pixel_count, errors="coerce").gt(0)
        & pd.to_numeric(frame.NDVI_p05, errors="coerce").gt(-9990)
        & pd.to_numeric(frame.NDVI_p95, errors="coerce").gt(-9990)
    )


def build_pairwise(historical: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    h = historical.rename(columns={
        "period_end": "historical_period_end", "products": "historical_products",
        "source_scale_m": "historical_scale_m", "coverage_pct": "historical_coverage",
        "nonwater_pixel_count": "historical_nonwater_pixel_count",
        "NDVI_p05": "historical_p05", "NDVI_p95": "historical_p95",
    }).copy()
    h["historical_eligible"] = eligible_historical(historical)
    c = current.rename(columns={
        "number_days": "current_number_days", "optical_products": "current_products",
        "optical_unique_dates": "current_unique_dates",
        "optical_coverage_pct": "current_coverage",
        "nonwater_pixel_count": "current_nonwater_pixel_count",
        "ndvi_p05_nonwater": "current_p05", "ndvi_p95_nonwater": "current_p95",
        "valid_for_fvc_calibration": "current_eligible",
    }).copy()
    paired = h.merge(c, on=["station_id", "period_start"], how="outer", indicator=True,
                     validate="one_to_one", suffixes=("_historical", "_current"))
    paired["year"] = pd.to_datetime(paired.period_start).dt.year
    paired["month"] = pd.to_datetime(paired.period_start).dt.month
    paired["spatial_fold"] = paired.station_id.map(SPATIAL_FOLD)
    paired["temporal_fold"] = paired.year - 2020
    paired["date_segment"] = np.where(
        pd.to_datetime(paired.period_start).lt("2022-01-25"), "before_2022-01-25", "on_or_after_2022-01-25"
    )
    for name in ("coverage", "nonwater_pixel_count", "p05", "p95"):
        paired[f"delta_{name}"] = paired[f"current_{name}"] - paired[f"historical_{name}"]
    paired["abs_delta_p05"] = paired.delta_p05.abs()
    paired["abs_delta_p95"] = paired.delta_p95.abs()
    paired["historical_eligible"] = paired.historical_eligible.fillna(False).astype(bool)
    paired["current_eligible"] = paired.current_eligible.fillna(0).astype(int).astype(bool)
    paired["input_comparison"] = np.where(
        paired.historical_products.ne(paired.current_products), "DIFFERENT_PRODUCT_COUNT", "UNKNOWN"
    )
    paired["coverage_bin"] = pd.cut(paired.current_coverage, [-1, 80, 90, 99, 100.0001], right=False).astype(str)
    paired["nonwater_count_bin"] = pd.cut(
        paired.current_nonwater_pixel_count, [-1, 100, 300, 500, np.inf], right=False
    ).astype(str)
    return paired.sort_values(["station_id", "period_start"])


def difference_distribution(paired: pd.DataFrame) -> pd.DataFrame:
    both = paired.loc[paired.historical_eligible & paired.current_eligible]
    rows = []
    for candidate in ("p05", "p95"):
        values = both[f"abs_delta_{candidate}"].dropna()
        row = {"candidate": candidate, "n": len(values), "median": values.median(),
               "mean": values.mean(), "p90": values.quantile(.90),
               "p95": values.quantile(.95), "p99": values.quantile(.99), "maximum": values.max()}
        for threshold in TOLERANCES:
            label = "exact_equal" if threshold == 0 else f"gt_{threshold:g}"
            row[label] = int(values.eq(0).sum()) if threshold == 0 else int(values.gt(threshold).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_differences(paired: pd.DataFrame) -> pd.DataFrame:
    both = paired.loc[paired.historical_eligible & paired.current_eligible].copy()
    dimensions = ["station_id", "year", "month", "spatial_fold", "temporal_fold",
                  "date_segment", "coverage_bin", "nonwater_count_bin",
                  "current_products", "current_unique_dates", "input_comparison"]
    rows = []
    for dimension in dimensions:
        for value, group in both.groupby(dimension, dropna=False, observed=True):
            rows.append({"dimension": dimension, "value": value, "n": len(group),
                         "p05_mean_abs_delta": group.abs_delta_p05.mean(),
                         "p05_max_abs_delta": group.abs_delta_p05.max(),
                         "p95_mean_abs_delta": group.abs_delta_p95.mean(),
                         "p95_max_abs_delta": group.abs_delta_p95.max()})
    return pd.DataFrame(rows)


def quantile_controls(paired: pd.DataFrame) -> pd.DataFrame:
    both = paired.loc[paired.historical_eligible & paired.current_eligible]
    rows = []
    for version in ("historical", "current"):
        for candidate, q in (("p05", .05), ("p95", .95)):
            ordered = both.sort_values(f"{version}_{candidate}").reset_index(drop=True)
            position = (len(ordered) - 1) * q
            lower, upper = int(np.floor(position)), int(np.ceil(position))
            weight = position - lower
            for role, rank in (("lower", lower), ("upper", upper)):
                item = ordered.iloc[rank]
                rows.append({"version": version, "candidate": candidate, "quantile": q,
                             "n": len(ordered), "zero_based_rank": rank,
                             "one_based_rank": rank + 1, "role": role,
                             "upper_interpolation_weight": weight,
                             "station_id": item.station_id, "period_start": item.period_start,
                             "candidate_value": item[f"{version}_{candidate}"],
                             "other_version_value_same_key": item[f"{'current' if version == 'historical' else 'historical'}_{candidate}"],
                             "absolute_delta_same_key": item[f"abs_delta_{candidate}"]})
    return pd.DataFrame(rows)


def method_diff() -> pd.DataFrame:
    rows = [
        ("collection", "COPERNICUS/S2_SR_HARMONIZED", "COPERNICUS/S2_SR_HARMONIZED", True, True, False, "src/et_downscaling/sentinel2.py", "8722b8b and working tree"),
        ("cloud score", "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED", "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED", True, True, False, "src/et_downscaling/sentinel2.py", "8722b8b and working tree"),
        ("medoid score bands", "Blue;Green;Red;RedEdge1;RedEdge2;RedEdge3;NIR;SWIR1;SWIR2", "Blue;Green;Red;NIR;SWIR1;SWIR2", False, True, True, "src/et_downscaling/sentinel2.py", "commit 0a5cae0"),
        ("medoid construction", "daily mosaic then squared spectral distance and qualityMosaic", "daily mosaic then squared spectral distance and qualityMosaic", True, True, False, "src/et_downscaling/sentinel2.py", "8722b8b and working tree"),
        ("coverage bands", "minimum mask of Green, Red, NIR; unmask(0); mean", "minimum mask of Green, Red, NIR; unmask(0); mean", True, True, False, "historical notebook/current exporter", "8722b8b vs working tree"),
        ("non-water rule", "NDWI <= 0", "NDWI <= 0", True, True, False, "historical notebook/current exporter", "8722b8b vs working tree"),
        ("percentiles", "per-footprint NDVI P05/P95, then global P05/P95", "per-footprint NDVI P05/P95, then global P05/P95", True, True, False, "historical notebook/current exporter", "8722b8b vs working tree"),
        ("reduction CRS", "explicit medoid NIR projection plus scale=20", "scale=20; CRS omitted", False, True, True, "historical notebook/current exporter", "8722b8b vs working tree"),
        ("footprint source", "EE asset projects/ee-change/assets/ETP_samples", "local station definitions transformed to MODIS parent footprints", False, True, True, "src/et_downscaling/config.py and spatial.py", "8722b8b vs 0a5cae0"),
        ("temporal interval", "half-open MODIS periods within 2021-01-01..2024-01-01", "same half-open MODIS periods", True, True, False, "historical table/current chunks", "period_start/end and number_days"),
    ]
    return pd.DataFrame(rows, columns=["component", "historical", "current", "same",
                                      "scientifically_relevant", "could_explain_difference",
                                      "evidence_file", "evidence_line_or_commit"])


def provenance_inventory() -> pd.DataFrame:
    return pd.DataFrame([
        {"artifact": HISTORICAL_PATH, "commit": HISTORICAL_COMMIT, "source": "S2/HLS",
         "product_count": True, "unique_dates": False, "product_ids": False,
         "mgrs_tile": False, "spacecraft": False, "processing_baseline": False,
         "generation_time": False, "finding": "Historical calibration checkpoint; actual candidate table."},
        {"artifact": "notebooks/diagnostics/01_data_availability_tests.ipynb", "commit": HISTORICAL_COMMIT,
         "source": "S2/HLS", "product_count": True, "unique_dates": False,
         "product_ids": False, "mgrs_tile": False, "spacecraft": False,
         "processing_baseline": False, "generation_time": False,
         "finding": "Diagnostic output has aggregate counts, not station-period scene identity."},
        {"artifact": "current protected S2 candidate chunks", "commit": "working tree", "source": "S2",
         "product_count": True, "unique_dates": True, "product_ids": False,
         "mgrs_tile": False, "spacecraft": False, "processing_baseline": False,
         "generation_time": False, "finding": "Counts and unique-date counts only."},
    ])


def run() -> dict:
    out = output_dir(); out.mkdir(parents=True, exist_ok=True)
    protected_records = validate_all_protected_chunks()
    historical, current_records = load_historical(), load_current()
    current, current_records = current_records  # type: ignore[assignment]
    paired = build_pairwise(historical, current)
    paired.to_csv(out / "s2_candidate_pairwise_comparison.csv", index=False)
    distribution = difference_distribution(paired)
    distribution.to_csv(out / "s2_candidate_difference_distribution.csv", index=False)
    grouped_differences(paired).to_csv(out / "s2_candidate_grouped_summary.csv", index=False)
    eligible_paired = paired.loc[paired.historical_eligible & paired.current_eligible]
    top = pd.concat([
        eligible_paired.nlargest(20, "abs_delta_p05").assign(ranking="top20_p05"),
        eligible_paired.nlargest(20, "abs_delta_p95").assign(ranking="top20_p95")
    ])
    top.to_csv(out / "s2_candidate_top20_differences.csv", index=False)
    controls = quantile_controls(paired)
    controls.to_csv(out / "s2_quantile_rank_controls.csv", index=False)
    method_diff().to_csv(out / "s2_historical_current_method_diff.csv", index=False)
    provenance_inventory().to_csv(out / "historical_provenance_inventory.csv", index=False)
    geometry = pd.DataFrame({"station_id": list(SPATIAL_FOLD),
        "historical_geometry_source": "projects/ee-change/assets/ETP_samples",
        "current_geometry_source": "local station definition -> MODIS parent footprint",
        "comparison": "UNKNOWN", "reason": "No historical coordinates or serialized geometry were preserved."})
    geometry.to_csv(out / "s2_geometry_identity.csv", index=False)
    both = paired.historical_eligible & paired.current_eligible
    hv = paired.loc[both]
    historical_low = hv.historical_p05.quantile(.05)
    current_low = hv.current_p05.quantile(.05)
    historical_high = hv.historical_p95.quantile(.95)
    current_high = hv.current_p95.quantile(.95)
    eligibility = {
        "eligible_both": int(both.sum()),
        "historical_only": int((paired.historical_eligible & ~paired.current_eligible).sum()),
        "current_only": int((~paired.historical_eligible & paired.current_eligible).sum()),
        "neither": int((~paired.historical_eligible & ~paired.current_eligible).sum()),
    }
    script_paths = [root()/"reproducibility/scripts/export_fvc_calibration_candidates.py",
                    root()/"reproducibility/scripts/analyze_fvc_candidate_stability.py",
                    root()/"src/et_downscaling/sentinel2.py",
                    root()/"src/et_downscaling/spatial.py"]
    hashes = [{"kind": "historical_git_blob", "path": HISTORICAL_PATH,
               "commit": HISTORICAL_COMMIT,
               "sha256": sha256_bytes(git_bytes(f"{HISTORICAL_COMMIT}:{HISTORICAL_PATH}"))}]
    current_bytes = current.sort_values(["station_id", "period_start"]).to_csv(index=False).encode()
    hashes.append({"kind": "current_s2_2021_2023_normalized", "path": "concatenated protected chunks",
                   "sha256": sha256_bytes(current_bytes)})
    hashes += protected_records
    hashes += [{"kind": "current_script", "path": str(p.relative_to(root())),
                "sha256": sha256_file(p)} for p in script_paths]
    pd.DataFrame(hashes).to_csv(out / "protected_input_sha256.csv", index=False)
    summary = {
        "historical_rows": len(historical), "current_rows": len(current),
        "paired_rows": int(paired._merge.eq("both").sum()),
        "historical_only_keys": int(paired._merge.eq("left_only").sum()),
        "current_only_keys": int(paired._merge.eq("right_only").sum()),
        "eligibility": eligibility,
        "endmembers": {"historical_low": historical_low, "current_low": current_low,
                       "delta_low": current_low-historical_low,
                       "historical_high": historical_high, "current_high": current_high,
                       "delta_high": current_high-historical_high,
                       "historical_range": historical_high-historical_low,
                       "current_range": current_high-current_low,
                       "relative_range_difference": ((current_high-current_low)/(historical_high-historical_low)-1)},
        "root_cause": "CASE 1 - METHOD_DRIFT",
        "evidence": "Commit 0a5cae0 removed RedEdge1/2/3 from S2 medoid scoring after the historical table was produced.",
        "remaining_uncertainty": "Historical geometry is unavailable and exact remote product identities were not preserved; reduction CRS also differs.",
        "additional_gee_needed": False,
        "decision": "Do not query GEE; first decide which medoid/reduction definition is methodologically intended.",
        "gee_requests_this_audit": 0,
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
