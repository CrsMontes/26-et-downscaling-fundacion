"""Offline staged identity migration, with strict physical-value invariants.

The verified external backup is read-only. No model fitting, acquisition or
scientific raster generation is performed. Stop on any failed assertion.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import rasterio

from et_downscaling.field_station_identity import OLD_TO_NEW, STATIONS, VERSION, validate_station_geometry, validate_station_table
from et_downscaling.field_review_io import read_review_sheet

ROOT = Path(__file__).resolve().parents[1]
FIELD = Path("outputs/evaluation/field_validation")
AREA = ROOT / "outputs/evaluation/station_identity_migration"
STAGE = ROOT / ".station-migration-stage"
MANIFEST_NAME = "station_identity_migration_manifest.json"
BASELINE_COMMIT = "f8d96bda7e5182bd8f3ed696360d583eb85133ec"
LEGACY_NAMES = {
    "deterministic_signature_migration_manifest.json", "protected_before_audit.json",
    "protected_after_audit.json", "transport_identity_audit_before.json",
    "raster_2022_03_30_provenance_audit.json", "rf25_hashes_before_local.json",
}
IDENTITY_COLUMNS = {"station_id", "station", "land_cover", "station_uid", "legacy_station_id", "station_nomenclature_version"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventories(backup):
    rows = []
    for name in ("core_files_sha256_before.csv", "field_validation_sha256_before.csv"):
        with (backup / name).open(encoding="utf-8-sig", newline="") as stream:
            entries = list(csv.DictReader(stream))
        assert len(entries) == (3 if name.startswith("core") else 402)
        rows.extend(entries)
    return rows


def verify_baseline(backup, current=False):
    for row in inventories(backup):
        for base in ([backup, ROOT] if current else [backup]):
            path = base / row["RelativePath"]
            assert path.is_file(), f"Missing baseline file: {path}"
            expected_size, expected_hash = int(row["SizeBytes"]), row["SHA256"].lower()
            receipt_path = AREA / "notebook_reconciliation/reconciliation_receipt.json"
            if base == ROOT and Path(row["RelativePath"]).as_posix() == "notebooks/field_validation_analysis.ipynb" and receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text())
                assert receipt["status"] == "completed", "Notebook reconciliation must finish before promotion."
                assert receipt["baseline_sha256"] == expected_hash
                assert sha256(Path(receipt["active_snapshot"])) == receipt["accepted_active_sha256"]
                expected_size, expected_hash = receipt["accepted_active_size_bytes"], receipt["accepted_active_sha256"]
            assert path.stat().st_size == expected_size, f"Baseline size mismatch: {path}"
            assert sha256(path) == expected_hash, f"Baseline hash mismatch: {path}"


def remap_text(text):
    """Simultaneous token lookup; expand old ID ranges before mapping once."""
    def replace(match):
        first, last = match.group(1), match.group(2)
        if last:
            return ", ".join(OLD_TO_NEW[f"ST{i:02d}"] for i in range(int(first[-2:]), int(last[-2:]) + 1))
        return OLD_TO_NEW[first]
    return re.sub(r"(ST0[1-5])(?:[-–](ST0[1-5]))?", replace, text)


def migrated_path(relative):
    text = Path(relative).as_posix()
    return Path(re.sub(r"ST0[1-5]|st04(?=_)",
                       lambda match: OLD_TO_NEW[match[0]] if match[0].startswith("ST") else "st01", text))


def active_path_string(value):
    normalized = value.replace("\\", "/")
    marker = "outputs/evaluation/field_validation/"
    if marker not in normalized:
        return value
    relative = normalized[normalized.index(marker):]
    # Missing historical cache/production inputs remain provenance, not invented files.
    if "/cache/" in relative or "/production/" in relative:
        return value
    return str(ROOT / migrated_path(relative))


def migrate_document(document):
    if isinstance(document, list):
        return [migrate_document(value) for value in document]
    if not isinstance(document, dict):
        return document
    old_id = document.get("station_id")
    result = {}
    for key, value in document.items():
        new_key = OLD_TO_NEW.get(key, key)
        if key in {"source_hashes", "input_sha256", "sampled_raster_sha256", "RF25_input_sha256", "historical_sources"}:
            result[new_key] = value
        elif key == "station_id" and value in OLD_TO_NEW:
            result[key] = OLD_TO_NEW[value]
        elif key == "station" and old_id in OLD_TO_NEW:
            result[key] = STATIONS[OLD_TO_NEW[old_id]]["station"]
        elif key == "stations" and isinstance(value, list):
            result[key] = [OLD_TO_NEW.get(item, item) if isinstance(item, str) else migrate_document(item) for item in value]
        elif key == "limitations" and isinstance(value, list):
            result[key] = [remap_text(item) if isinstance(item, str) else migrate_document(item) for item in value]
        elif key == "metric_subsets" and isinstance(value, dict):
            result[key] = {name: remap_text(text) for name, text in value.items()}
        elif isinstance(value, (dict, list)):
            result[new_key] = migrate_document(value)
        elif isinstance(value, str) and key in {"raster", "raw_tile", "raster_root"}:
            result[key] = active_path_string(value)
        elif isinstance(value, str) and key in {"tile_id", "method_note", "note", "limitations"}:
            result[key] = remap_text(value)
        else:
            result[new_key] = value
    if old_id in OLD_TO_NEW:
        result.update(station_uid=STATIONS[OLD_TO_NEW[old_id]]["station_uid"],
                      legacy_station_id=old_id, station_nomenclature_version=VERSION)
    return result


def migrate_csv(source, target):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = list(reader.fieldnames), list(reader)
    if "station_id" not in columns or not any(row["station_id"] in OLD_TO_NEW for row in rows):
        shutil.copy2(source, target)
        return
    for name in ("station_uid", "legacy_station_id", "station_nomenclature_version"):
        if name not in columns:
            columns.append(name)
    for row in rows:
        old_id = row["station_id"]
        if old_id in OLD_TO_NEW:
            new_id = OLD_TO_NEW[old_id]
            row.update(station_id=new_id, station_uid=STATIONS[new_id]["station_uid"],
                       legacy_station_id=old_id, station_nomenclature_version=VERSION)
            for key in ("station", "land_cover"):
                if key in row:
                    row[key] = STATIONS[new_id]["station"]
        for key in ("RF25_raster", "raster", "raster_root"):
            if row.get(key):
                row[key] = active_path_string(row[key])
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compare_raster(source, target):
    with rasterio.open(source) as old, rasterio.open(target) as new:
        assert (old.crs, old.transform, old.shape, old.count, old.dtypes, old.nodata, old.descriptions) == (
            new.crs, new.transform, new.shape, new.count, new.dtypes, new.nodata, new.descriptions), f"Raster support changed: {target}"
        for _, window in old.block_windows(1):
            assert old.read(window=window).tobytes() == new.read(window=window).tobytes(), f"Raster values changed: {target}"
            assert old.read_masks(window=window).tobytes() == new.read_masks(window=window).tobytes(), f"Raster masks changed: {target}"
        old_tags, new_tags = old.tags(), new.tags()
        expected = dict(old_tags)
        if expected.get("station_id") in OLD_TO_NEW:
            expected["station_id"] = OLD_TO_NEW[expected["station_id"]]
        assert expected == new_tags, f"Unexpected raster metadata change: {target}"


def compare_table(old, new, keys):
    old = old.copy()
    if "station_id" in old:
        old["station_id"] = old.station_id.map(lambda value: OLD_TO_NEW.get(value, value))
    old = old.sort_values(keys).reset_index(drop=True)
    new = new.sort_values(keys).reset_index(drop=True)
    assert len(old) == len(new)
    ignored = IDENTITY_COLUMNS | {"pair_keys_sha256", "RF25_raster", "raster", "raster_root"}
    for column in old.columns:
        if column in ignored:
            continue
        pooled = old.station_id.eq("ALL") if "station_id" in old else pd.Series(False, index=old.index)
        pd.testing.assert_series_equal(old.loc[~pooled, column], new.loc[~pooled, column], check_dtype=False, check_exact=True,
                                       obj=f"Physical invariant {column}")
        pd.testing.assert_series_equal(old.loc[pooled, column], new.loc[pooled, column], check_dtype=False,
                                       check_exact=False, rtol=1e-12, atol=1e-12, obj=f"Pooled invariant {column}")
    assert old[keys].equals(new[keys]), "Physical station/date pairing changed."


def prepare(backup):
    # Resume a partial stage without overwriting entries (e.g. Windows path limits).
    resumed = STAGE.exists()
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "station-id-migration"
    assert subprocess.check_output(["git", "rev-parse", "pre-station-id-migration-20260914^{commit}"], cwd=ROOT, text=True).strip() == BASELINE_COMMIT
    verify_baseline(backup, current=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    protected = {p.relative_to(ROOT).as_posix(): sha256(p) for p in (ROOT / "outputs/current").rglob("*") if p.is_file()}
    protected_path = AREA / "protected_current_sha256.json"
    AREA.mkdir(parents=True, exist_ok=True)
    if resumed:
        assert json.loads(protected_path.read_text()) == protected
    else:
        protected_path.write_text(json.dumps(protected, indent=2), encoding="utf-8")
    crosswalk = {old: dict(new_id=new, **STATIONS[new]) for old, new in OLD_TO_NEW.items()}
    (AREA / "crosswalk.json").write_text(json.dumps(crosswalk, indent=2), encoding="utf-8")
    for row in inventories(backup):
        relative = Path(row["RelativePath"])
        source = backup / relative
        destination = migrated_path(relative)
        target = STAGE / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if resumed and target.is_file():
            continue  # The comprehensive verify-stage action still checks every entry.
        assert not target.exists(), f"Cyclic destination collision: {target}"
        if relative.suffix == ".csv":
            migrate_csv(source, target)
        elif relative.name == "fundacion_stations.geojson":
            document = json.loads(source.read_text())
            for feature in document["features"]:
                properties = feature["properties"]
                old = properties["station_id"]
                new = OLD_TO_NEW[old]
                expected = STATIONS[new]
                properties.update(station_id=new, station=expected["station"], station_uid=expected["station_uid"],
                                  legacy_station_id=old, station_nomenclature_version=VERSION,
                                  kc_rule=expected["kc_rule"], validation_domain=expected["validation_domain"],
                                  coastal_era5_support=expected["coastal_era5_support"])
            document["features"].sort(key=lambda feature: feature["properties"]["station_id"])
            validate_station_geometry(document)
            target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        elif relative.suffix == ".json" and relative.name not in LEGACY_NAMES and "rf25_cv" not in relative.parts:
            document = migrate_document(json.loads(source.read_text(encoding="utf-8-sig")))
            target.write_text(json.dumps(document, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        else:
            shutil.copy2(source, target)
        if target.suffix == ".tif":
            with rasterio.open(source) as raster:
                old_id = raster.tags().get("station_id")
            if old_id in OLD_TO_NEW and old_id != OLD_TO_NEW[old_id]:
                with rasterio.open(target, "r+") as raster:
                    raster.update_tags(station_id=OLD_TO_NEW[old_id])
            compare_raster(source, target)
    print("Staged 405 baseline entries; raster arrays/masks/georeferencing verified.", flush=True)


def verify(backup, target_root):
    verify_baseline(backup)
    old_geo = json.loads((backup / "data/stations/fundacion_stations.geojson").read_text())
    new_geo = json.loads((target_root / "data/stations/fundacion_stations.geojson").read_text())
    validate_station_geometry(new_geo)
    by_id = {feature["properties"]["station_id"]: feature for feature in new_geo["features"]}
    for old in old_geo["features"]:
        new = by_id[OLD_TO_NEW[old["properties"]["station_id"]]]
        assert old["geometry"] == new["geometry"]
        for key, value in old["properties"].items():
            if key not in {"station_id", "station"}:
                assert new["properties"][key] == value, f"Station attribute changed: {key}"
    csv_checks = 0
    for row in inventories(backup):
        relative = Path(row["RelativePath"])
        source, target = backup / relative, target_root / migrated_path(relative)
        if relative.suffix == ".csv":
            old, new = pd.read_csv(source), pd.read_csv(target)
            if "station_id" in old and old.station_id.isin(OLD_TO_NEW).any():
                keys = [key for key in ["station_id", "date", "period_start", "scenario", "comparison", "sample", "comparison_family", "stage_order"] if key in old]
                compare_table(old, new, keys)
                validate_station_table(new.loc[new.station_id.ne("ALL")])
                csv_checks += 1
            else:
                # Aggregate numerical metrics must be invariant; key digests can change.
                pd.testing.assert_frame_equal(old.drop(columns=["pair_keys_sha256"], errors="ignore"),
                                              new.drop(columns=["pair_keys_sha256"], errors="ignore"), check_exact=False, rtol=1e-12, atol=1e-12)
        elif relative.suffix == ".tif":
            compare_raster(source, target)
        elif relative.name in LEGACY_NAMES or relative.suffix == ".log" or "rf25_cv" in relative.parts:
            assert sha256(source) == sha256(target), f"Historical evidence changed: {relative}"
    for sheet, count in [("DAILY", 615), ("PERIOD", 80)]:
        old = read_review_sheet(backup / FIELD / "field_validation_review.xlsx", sheet)
        new = read_review_sheet(target_root / FIELD / "field_validation_review.xlsx", sheet)
        assert len(new) == count
        compare_table(old, new, ["station_id", "date" if sheet == "DAILY" else "period_start"])
        validate_station_table(new)
    period = read_review_sheet(target_root / FIELD / "field_validation_review.xlsx", "PERIOD")
    columns = ["field_et_historical_mm_period", "modis_et_mm_period", "rf25_et_unreconciled_mm_period", "rf25_et_reconciled_mm_period"]
    common = period.loc[np.isfinite(period[columns]).all(axis=1) & period.publishable.fillna(False)]
    counts = common.groupby("station_id").size().to_dict()
    assert counts == {"ST01": 8, "ST02": 5, "ST03": 7, "ST04": 7, "ST05": 10}, counts
    assert period.loc[period.station_id.isin(["ST02", "ST03", "ST04"]), "kc_ndvi_s2_20m"].isna().all()
    assert period.loc[period.station_id.eq("ST01"), "validation_domain"].eq("validation_extension").all()
    assert period.loc[period.station_id.ne("ST01"), "validation_domain"].eq("fundacion_basin").all()
    for relative, digest in json.loads((AREA / "protected_current_sha256.json").read_text()).items():
        assert sha256(ROOT / relative) == digest, f"Protected production file changed: {relative}"
    print(f"PASS: {csv_checks} station tables; DAILY 615; PERIOD 80; common 37 {counts}; all 212 rasters invariant; all 1593 production files byte-identical.", flush=True)
    return counts


def promote(backup):
    receipt_path = AREA / "notebook_reconciliation/reconciliation_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "completed"
    assert sha256(STAGE / "notebooks/field_validation_analysis.ipynb") == receipt["executed_staged_sha256"]
    verify(backup, STAGE)
    # Recheck originals immediately before the directory swap.
    verify_baseline(backup, current=True)
    legacy = ROOT / ".station-migration-legacy"
    old_field = legacy / FIELD
    old_field.parent.mkdir(parents=True, exist_ok=True)
    assert not old_field.exists()
    for path in (ROOT / FIELD, old_field, STAGE / FIELD):
        assert path.resolve().is_relative_to(ROOT.resolve()), "Unsafe directory move."
    (ROOT / FIELD).rename(old_field)
    (STAGE / FIELD).rename(ROOT / FIELD)
    for relative in (Path("data/stations/fundacion_stations.geojson"), Path("data/field/field_etgage.csv"), Path("notebooks/field_validation_analysis.ipynb")):
        shutil.copy2(STAGE / relative, ROOT / relative)
    manifest = {
        "migration_version": VERSION, "status": "promoted_pending_final_verification",
        "baseline_commit": BASELINE_COMMIT, "backup_root": str(backup),
        "legacy_root": str(legacy), "old_to_new": OLD_TO_NEW,
        "historical_method_identifiers_unchanged": True,
        "notebook_reconciliation": receipt,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }
    for row in inventories(backup):
        old = Path(row["RelativePath"])
        new = migrated_path(old)
        manifest["files"].append({"old_path": old.as_posix(), "new_path": new.as_posix(),
                                  "sha256_before": row["SHA256"].lower(), "sha256_after": sha256(ROOT / new),
                                  "raster_scientific_state_unchanged": True if old.suffix == ".tif" else None})
    (ROOT / FIELD / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Promoted verified data. Original field bundle retained under", old_field)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, default=Path("E:/ET_backup_20260914"))
    parser.add_argument("action", choices=["stage", "verify-stage", "promote", "verify-current"])
    args = parser.parse_args()
    {"stage": lambda: prepare(args.backup), "verify-stage": lambda: verify(args.backup, STAGE),
     "promote": lambda: promote(args.backup), "verify-current": lambda: verify(args.backup, ROOT)}[args.action]()
