"""One-time, offline provenance migration of the existing 70 RF25 halo products.

Only JSON sidecars are written. The explicitly authorized legacy identities are
mapped after verifying today's frozen model/AOA; this does not reconstruct
missing historical artifact hashes or certify a halo as a basin publication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ROOT / "outputs/evaluation/field_validation/local_halo7_products"
MODEL_SIGNATURE = "8a9775111cfef5e5b5a2e02fd7ebff99078db8b3e656dff5e9e264292a98af54"
PRODUCTION_SIGNATURE = "cf967ee9be94ba3d83e5a42e719aab37bd8d36ecf0f4341f9345ac47dc0ba368"
LEGACY_CANONICAL = "c7e8e034b39c0c67452b7cf4629534825d2848bf716d61b4a4c25dfba7128e62"
LEGACY_ST04 = "2b550346a83f0b6f7ecf8b4381132223bf652bff082c45bc86a4d64384b1d530"
METHOD = "st04_validation_extension_era5_nearest_valid_land_pixel_fill_v1"
SUPPORT = (-74.30000000000001, 10.799999999999997, 7511.655789576682)
ARTIFACT_HASHES = {
    "rf25_virtual10_ge90.joblib": "f9f6ef05649cbf07ab4d92d38cb41269517cfb27db027dfba40920834ea1bed4",
    "rf25_weighted_aoa.joblib": "db3dba544514742604674428703e7d72efb37d01590c0d17005424eec78588b1",
}
MANIFEST_NAME = "deterministic_signature_migration_manifest.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extension_signature(base_signature, method, longitude, latitude, distance_m):
    # Identical formatting to produce_st04_rf25_halo.main.build_extension_signature.
    text = f"{base_signature}|{method}|{longitude:.12f}|{latitude:.12f}|{distance_m:.6f}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def migrated_metadata(document, old_signature, new_signature):
    require(document.get("scientific_signature") == old_signature, "Unexpected legacy signature")
    require("legacy_scientific_signature" not in document, "Metadata already migrated")
    return {**document, "legacy_scientific_signature": old_signature,
            "scientific_signature": new_signature}


def write_json(path, document):
    temporary = path.with_name(path.name + ".migration-tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=True)
        stream.write("\n")
    os.replace(temporary, path)


def migrate(apply=False):
    import joblib
    from et_downscaling.rf25 import rf25_model_signature
    from et_downscaling.rf25_overlap_production import (
        RF25_EXACT_OVERLAP_PRODUCTION_VERSION, build_production_scientific_signature,
    )

    manifest_path = PRODUCTS / MANIFEST_NAME
    require(not manifest_path.exists(), "Migration manifest already exists; do not overwrite its audit trail")
    model_dir = ROOT / "outputs/current/models"
    for name, expected in ARTIFACT_HASHES.items():
        require(sha256(model_dir / name) == expected, f"Artifact changed: {name}")
    model, aoa = [joblib.load(model_dir / name) for name in ARTIFACT_HASHES]
    require(rf25_model_signature(model) == MODEL_SIGNATURE, "Model identity mismatch")
    require(build_production_scientific_signature(model, aoa) == PRODUCTION_SIGNATURE,
            "Production identity mismatch")

    extension_path = PRODUCTS / "ST04_validation_extension_manifest.json"
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    require(extension["method"] == METHOD, "ST04 method mismatch")
    require(tuple(extension["era5_support_" + key] for key in
                  ("longitude", "latitude", "distance_m")) == SUPPORT, "ST04 support mismatch")
    require(extension["validation_domain"] == "extension_outside_official_basin",
            "ST04 domain mismatch")
    st04_signature = extension_signature(PRODUCTION_SIGNATURE, METHOD, *SUPPORT)
    metadata_paths = sorted(PRODUCTS.glob("????-??-??/ST??/metadata.json"))
    rasters = sorted(PRODUCTS.rglob("*.tif"))
    require(len(metadata_paths) == 70, "Expected exactly 70 final sidecars")
    require(len(rasters) == 210, "Expected 70 RF25, 70 MODIS and 70 raw TIFFs")
    planned = []
    associated = set()
    stations_dates = set()

    def plan(path, old, new, linked_rasters):
        planned.append((path, new, {
            "metadata_file": path.relative_to(ROOT).as_posix(),
            "old_signature": old.get("scientific_signature"),
            "new_signature": new["scientific_signature"],
            "associated_rasters": [p.relative_to(ROOT).as_posix() for p in linked_rasters],
            "metadata_sha256_before": sha256(path),
            "metadata_before": old,
        }))

    for path in metadata_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        station, date = path.parent.name, path.parent.parent.name
        require(station in {"ST01", "ST02", "ST03", "ST04", "ST05"}, "Unexpected station")
        require((document["station_id"], document["period_start"]) == (station, date), "Sidecar path mismatch")
        stations_dates.add((station, date))
        raster = path.parent / f"RF25_halo7_{station}_{date}_20m.tif"
        raw = path.parent / "raw" / f"{station}_halo7.tif"
        modis = path.parent / f"MODIS_halo7_{station}_{date}_native.tif"
        require(Path(document["raster"]).resolve() == raster.resolve(), "Raster association mismatch")
        require(Path(document["raw_tile"]).resolve() == raw.resolve(), "Raw association mismatch")
        require(all(p.is_file() for p in (raster, raw, modis)), "Missing associated TIFF")
        associated.update((raster, raw, modis))
        old = LEGACY_ST04 if station == "ST04" else LEGACY_CANONICAL
        new = st04_signature if station == "ST04" else PRODUCTION_SIGNATURE
        plan(path, document, migrated_metadata(document, old, new), [raster, modis])
        raw_path = raw.with_suffix(".json")
        raw_document = json.loads(raw_path.read_text(encoding="utf-8"))
        require(raw_document["production_method_version"] == RF25_EXACT_OVERLAP_PRODUCTION_VERSION,
                "Raw production version mismatch")
        require((raw_document["period_start"], raw_document["tile_id"], raw_document["tile_role"])
                == (date, f"{station}_halo7", "raw_support"), "Raw sidecar mismatch")
        plan(raw_path, raw_document, migrated_metadata(raw_document, old, new), [raw])
    dates = {date for _, date in stations_dates}
    require(len(dates) == 14 and all(sum(s == station for s, _ in stations_dates) == 14
            for station in ("ST01", "ST02", "ST03", "ST04", "ST05")), "Incomplete station/date matrix")
    require(associated == set(rasters), "Unexpected TIFF inventory")
    st04_rasters = [p for p in rasters if p.name.startswith("RF25_halo7_ST04_")]
    require("scientific_signature" not in extension, "Unexpected existing extension manifest signature")
    updated_extension = migrated_metadata(extension, None, st04_signature)
    updated_extension["canonical_production_scientific_signature"] = PRODUCTION_SIGNATURE
    plan(extension_path, extension, updated_extension, st04_rasters)
    summary_path = PRODUCTS / "production_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary["stations"] == ["ST04"] and set(summary["dates"]) == dates, "Unexpected summary scope")
    plan(summary_path, summary, migrated_metadata(summary, LEGACY_ST04, st04_signature), st04_rasters)
    require(set(PRODUCTS.rglob("*.json")) == {p for p, _, _ in planned}, "Unaccounted JSON sidecar")

    print(json.dumps({"metadata_files": len(planned), "rf25_products": 70,
                      "all_tiffs": len(rasters), "st04_signature": st04_signature, "apply": apply}))
    if not apply:
        return
    before = {p.relative_to(ROOT).as_posix(): sha256(p) for p in rasters}
    manifest = {
        "migration_version": "rf25_deterministic_signature_migration_v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared",
        "basis": "User-authorized legacy identity mapping following the structured-node padding diagnosis; no numerical recomputation or retroactive historical artifact certification.",
        "model_scientific_signature": MODEL_SIGNATURE,
        "canonical_production_scientific_signature": PRODUCTION_SIGNATURE,
        "st04_extension_signature": st04_signature,
        "artifact_file_sha256": ARTIFACT_HASHES,
        "metadata_files": [entry for _, _, entry in planned],
        "tiff_sha256_before": before,
    }
    # Persist the complete original metadata and TIFF hashes before any sidecar edit.
    write_json(manifest_path, manifest)
    for path, document, entry in planned:
        require(sha256(path) == entry["metadata_sha256_before"], f"Concurrent edit: {path}")
        write_json(path, document)
        entry["metadata_sha256_after"] = sha256(path)
    after = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(PRODUCTS.rglob("*.tif"))}
    require(before == after, "TIFF hash/inventory changed during migration")
    for name, expected in ARTIFACT_HASHES.items():
        require(sha256(model_dir / name) == expected, f"Artifact changed: {name}")
    manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    tiff_sha256_after=after, all_tiffs_byte_identical=True)
    write_json(manifest_path, manifest)
    print(f"Completed: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the verified JSON migration (default: inspect only)")
    migrate(parser.parse_args().apply)
