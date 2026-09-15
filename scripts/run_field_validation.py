"""Reproduce field ET proxy scenarios separately from the frozen RF25 pipeline.

All four scientific sets are saved together. RF25 values come only from verified
final rasters. Optional production uses the unchanged canonical basin script.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from et_downscaling.field_validation import (
    VALIDATION_SCENARIOS, FIXED_KC, KEYS, RAW_TO_MM, MIN_VALID_DAYS, DAILY_RANGE,
    NDVI_SLOPE, NDVI_INTERCEPT, KC_RANGE, acquire_reference, acquire_satellite,
    aggregate_field_periods,
    load_field_inputs, modis_periods, prepare_field_daily, sample_rf25_rasters, sha256, unique,
    apply_validation_scenarios, build_validation_metrics, build_sequential_attrition,
)
from et_downscaling.field_rf25_local import audit_existing_raster
from et_downscaling.field_rf25_products import production_contract, inventory_products, produce_field_date
from et_downscaling.reference_et_local import build_daily_reference_et
from et_downscaling.field_station_identity import attach_station_identity, validate_station_table
from et_downscaling.workspace import get_workspace_paths

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", help="Earth Engine project; required only for missing field caches.")
    parser.add_argument("--scenario", choices=["all", *VALIDATION_SCENARIOS], default="all",
                        help="Console selection; outputs always retain all four scientific sets.")
    parser.add_argument("--daily-reference", type=Path, help="Existing ST01–ST05 daily modeled ETo/ETr CSV.")
    parser.add_argument("--era5-hourly", type=Path, help="Existing field hourly ERA5 CSV; requires --station-support.")
    parser.add_argument("--station-support", type=Path, help="Field support CSV for local reference-ET reconstruction.")
    parser.add_argument("--satellite-table", type=Path,
                        help="Reuse field satellite CSV with station_id, period_start, number_days, ET_MODIS_mm_period, NDVI_local_20m.")
    parser.add_argument("--raster-root", type=Path, help="Read-only existing final raster directories; execution evidence is required for attribution.")
    parser.add_argument("--produce-missing", action="store_true", help="Produce pending dates with the canonical script in an isolated evaluation workspace.")
    parser.add_argument("--production-date", action="append", help="Limit --produce-missing to these field dates; repeat as needed.")
    args = parser.parse_args()
    if args.produce_missing and not args.project:
        parser.error("--produce-missing requires --project.")
    if args.production_date and not args.produce_missing:
        parser.error("--production-date requires --produce-missing.")
    if bool(args.era5_hourly) != bool(args.station_support):
        parser.error("--era5-hourly and --station-support must be supplied together.")
    if args.daily_reference and args.era5_hourly:
        parser.error("Choose --daily-reference or raw --era5-hourly/--station-support.")
    return args


def source_hashes() -> dict:
    paths = [ROOT / "scripts/run_field_validation.py", ROOT / "data/field/field_etgage.csv",
             ROOT / "data/stations/fundacion_stations.geojson"]
    paths.extend(sorted((ROOT / "src/et_downscaling").glob("*.py")))
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def initialize_ee(project: str | None) -> None:
    if not project:
        raise ValueError("Field caches are absent. Supply --project <earth-engine-project>, or offline field --daily-reference and --satellite-table inputs.")
    import ee
    ee.Initialize(project=project)
    ee.Number(1).getInfo()


def main() -> None:
    args = parse_args()
    output = ROOT / "outputs/evaluation/field_validation"
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    # Acquisition caches are versioned by raw inputs and current acquisition code.
    # A different station geometry/date range or optical implementation cannot
    # silently reuse an earlier cache. Existing cache partitions remain untouched.
    acquisition_names = ["reference_et_local", "meteorology_export", "modis", "production",
                         "rf25_production", "sentinel2", "optical", "config", "field_validation", "field_station_identity"]
    contract = {key: value for key, value in hashes.items()
                if key.startswith("data/") or Path(key).stem in acquisition_names}
    import hashlib
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    cache = output / "cache" / signature
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    field, stations = load_field_inputs(ROOT)
    periods = modis_periods(field)
    reference_path = cache / "reference_et_daily.csv"
    source_inputs = {}
    reference_origin = "field ERA5 acquired with current repository functions"
    ee_ready = False

    def read_input(path: Path) -> pd.DataFrame:
        source_inputs[str(path.resolve())] = sha256(path)
        table = pd.read_csv(path, dtype={"station_id": str})
        validate_station_table(table, require_uid=True)
        return table

    if args.daily_reference:
        reference = read_input(args.daily_reference)
        reference_origin = str(args.daily_reference.resolve())
    elif args.era5_hourly:
        reference = attach_station_identity(build_daily_reference_et(read_input(args.era5_hourly), read_input(args.station_support)))
        reference_origin = "current build_daily_reference_et from supplied raw field ERA5/support"
    elif reference_path.is_file():
        reference = read_input(reference_path)
    else:
        reference = None
        current = get_workspace_paths(ROOT).master / "S2"
        for path in sorted(current.glob("reference_et_daily_*.csv")):
            candidate = pd.read_csv(path, dtype={"station_id": str})
            if set(stations.station_id).issubset(set(candidate.station_id)):
                try:
                    validate_station_table(candidate, require_uid=True)
                    prepare_field_daily(field, stations, candidate)
                except ValueError as error:
                    print(f"Reference cache unsuitable: {path}: {error}", flush=True)
                    continue
                reference, reference_origin = candidate, str(path.resolve())
                break
            print(f"Reference cache has no full ST01–ST05 coverage; skipping {path}", flush=True)
        if reference is None:
            initialize_ee(args.project)
            ee_ready = True
            reference = acquire_reference(field, stations, cache)
            reference.to_csv(reference_path, index=False)

    daily = prepare_field_daily(field, stations, reference)
    base = aggregate_field_periods(daily, periods, stations)
    # Only periods with a usable field series somewhere require satellite queries.
    requested = base.loc[base.field_period_valid, ["period_start", "number_days"]].drop_duplicates().sort_values("period_start")
    satellite_path = cache / "field_satellite.csv"
    if args.satellite_table:
        satellite = read_input(args.satellite_table)
    elif satellite_path.is_file():
        satellite = read_input(satellite_path)
    elif requested.empty:
        satellite = pd.DataFrame(columns=[*KEYS, "number_days", "ET_MODIS_mm_period", "NDVI_local_20m"])
    else:
        if not ee_ready:
            initialize_ee(args.project)
        satellite = acquire_satellite(requested, stations, cache)
        satellite.to_csv(satellite_path, index=False)
    satellite["period_start"] = pd.to_datetime(satellite.period_start, errors="raise")
    unique(satellite, KEYS, "field satellite")
    expected = base.loc[base.period_start.isin(requested.period_start), KEYS + ["number_days"]]
    check = expected.merge(satellite[KEYS + ["number_days"]], on=KEYS, how="left", validate="one_to_one", suffixes=("", "_satellite"))
    if not check.number_days.eq(check.number_days_satellite).all():
        raise ValueError("Satellite input must contain all five field stations and correct MODIS durations for requested periods.")
    for column in ("ET_MODIS_mm_period", "NDVI_local_20m"):
        satellite[column] = pd.to_numeric(satellite[column], errors="coerce")
        satellite.loc[satellite[column].le(-9990), column] = np.nan
    base = base.merge(satellite[KEYS + ["ET_MODIS_mm_period", "NDVI_local_20m"]], on=KEYS, how="left", validate="one_to_one")
    contract = production_contract(ROOT)
    raster_audit = audit_existing_raster(ROOT, output)
    raster_roots = [args.raster_root or get_workspace_paths(ROOT).rasters]
    raster_roots.extend(sorted((output / "production").glob("*/rasters")))
    inventory = inventory_products(requested.period_start, raster_roots, contract)
    inventory.to_csv(output / "rf25_field_product_inventory.csv", index=False)
    if args.production_date and not set(args.production_date).issubset(set(inventory.period_start)):
        raise ValueError("--production-date must be a field date with >=5 valid ETgage days somewhere.")
    produced = []
    if args.produce_missing:
        for row in inventory.itertuples(index=False):
            if row.needs_canonical_production and (not args.production_date or row.period_start in args.production_date):
                produced_root = produce_field_date(ROOT, output, row.period_start, args.project, contract)
                produced.append(row.period_start)
                if produced_root not in raster_roots:
                    raster_roots.append(produced_root)
        inventory = inventory_products(requested.period_start, raster_roots, contract)
        inventory.to_csv(output / "rf25_field_product_inventory.csv", index=False)
    tables, raster_hashes = [], {}
    for date, group in base.groupby("period_start", sort=True):
        match = inventory.loc[inventory.period_start.eq(date.strftime("%Y-%m-%d"))]
        root = Path(match.raster_root.iloc[0]) if not match.empty else raster_roots[0]
        sampled, hashes_read = sample_rf25_rasters(group, root, contract)
        tables.append(sampled)
        raster_hashes.update(hashes_read)
    base = pd.concat(tables, ignore_index=True)
    if production_contract(ROOT) != contract:
        raise RuntimeError("Frozen RF25 inputs changed during field validation.")
    pairs = apply_validation_scenarios(base)
    metrics, by_station = build_validation_metrics(pairs)
    attrition = build_sequential_attrition(pairs)
    for name, table in {
        "field_daily_qc.csv": daily, "field_period_pairs.csv": pairs,
        "field_validation_metrics.csv": metrics, "field_validation_by_station.csv": by_station,
        "field_validation_attrition.csv": attrition,
    }.items():
        temporary = output / (name + ".tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(output / name)
    for path in sorted(cache.glob("*.csv")):
        source_inputs[str(path.resolve())] = sha256(path)
    for path, digest in raster_hashes.items():
        if sha256(Path(path)) != digest:
            raise RuntimeError(f"Input raster/metadata changed during field comparison: {path}")
    metadata = {
        "workflow_version": "field_proxy_rf25_final_products_v3", "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "project": args.project, "source_hashes": hashes, "input_sha256": source_inputs,
        "sampled_raster_sha256": raster_hashes, "raster_inputs_unchanged": True,
        "RF25_input_sha256": contract, "product_inventory": inventory.to_dict("records"), "existing_raster_attribution": raster_audit["attribution"],
        "reference_origin": reference_origin, "cache_signature": signature,
        "historical_sources": {"daily_rules": ["480f50b", "857adcc", "46b7d30"],
                               "local_ndvi_support": "726b914", "frozen_main": "b592569"},
        "field_raw_rows": len(field), "field_raw_rows_by_station": field.groupby("station_id").size().to_dict(),
        "valid_daily_rows_by_station": daily.groupby("station_id").field_daily_valid.sum().to_dict(),
        "raw_unit": "cm; confirmed by project owner", "raw_to_mm": RAW_TO_MM,
        "daily_qc": {"installation_window": "versioned CSV flag", "positive": True, "range_mm": DAILY_RANGE},
        "reference_harmonization": "ETgage_mm / (modeled daily ETr / modeled daily ETo) for ETr stations; unchanged for ETo",
        "aggregation": {"minimum_valid_days": MIN_VALID_DAYS, "formula": "mean(valid daily ETo-equivalent) * actual MODIS number_days", "end_exclusive": True},
        "fixed_kc": {"fixed_kc_main": FIXED_KC["historical"], "historical_all_stations": FIXED_KC["historical"], "fao_sensitivity": FIXED_KC["fao_sensitivity"]},
        "ndvi_kc": {"slope": NDVI_SLOPE, "intercept": NDVI_INTERCEPT, "valid_range": KC_RANGE, "outside_range": "missing; never clipped"},
        "ndvi_support": "local station pixel on current RF25 20 m S2 medoid grid for all NDVI scenarios; later 726b914 convention",
        "metric_units": "mm per MODIS period; R2 and KGE dimensionless; BIAS = prediction - field proxy, or RF25 - MODIS for MODIS_vs_RF25",
        "metric_subsets": {"available_sample": "Each scenario and comparison uses all valid available pairs; no NDVI intersection is imposed on the principal result.",
                           "common_sample": "Identical keys across scenarios for each product within each declared family; MODIS sensitivity does not require an RF25 raster.",
                           "main_vs_sensitivities": "Four sets; intersection necessarily restricted to ST02, ST03, ST04.",
                           "all_station_sensitivities": "Three sensitivity sets; ST01–ST05 may contribute."},
        "attrition": attrition.to_dict("records"), "rf25_training_used_field": False,
        "rf25_training_called": False, "RF25_source": "verified_final_published_raster_pixel",
        "basin_raster_production_called": bool(produced), "produced_dates": produced, "all_scenarios_saved": list(VALIDATION_SCENARIOS),
        "limitations": [
            "Field ET is a Kc-derived proxy, not a direct actual-ET observation or independent 20 m validation.",
            "Fixed Kc citations and NDVI relation applicability are not established by repository history; fao_sensitivity is a requested assumption set.",
            "NDVI scenarios share Sentinel-2 information with RF25; no explicit water-stress correction is applied.",
            "Five valid days may be expanded to eight; complete_8of8 periods are identifiable from n_valid_field_days and number_days.",
            "ST03, ST04, ST01, ST05 are flagged nonconforming installations; ST01 is outside the published basin domain.",
            "Local station reconciliation was rejected as a replacement for globally accepted canonical publication.",
            "Absent or unattributed final products remain pending; unknown diagnostics are not scientific exclusions.",
            "Historical scenario preserves conversion rules and later local NDVI support; it does not claim to reproduce frozen legacy metrics with different spatial supports/products.",
        ],
    }
    # Round-trip through pandas to normalize numpy scalars and NaNs to JSON null.
    payload = json.loads(pd.Series(metadata).to_json(date_format="iso"))
    temporary = output / "field_validation_metadata.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "field_validation_metadata.json")
    selected = metrics if args.scenario == "all" else metrics.loc[metrics.scenario.eq(args.scenario)]
    print(selected.drop(columns=["station_id", "pair_keys_sha256"]).to_string(index=False))
    print("Observation counts:")
    counts = attrition if args.scenario == "all" else attrition.loc[attrition.scenario.eq(args.scenario)]
    print(counts.pivot(index=["scenario", "station_id"], columns="stage", values="n_remaining").to_string())
    print("Saved all four scientific sets:", output)
    if not pairs.paired_RF25.any():
        print("RF25-vs-field metrics pending: no finite published RF25 pairs for the field periods.")


if __name__ == "__main__":
    main()
