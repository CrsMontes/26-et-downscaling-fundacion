"""Field-only orchestration of unchanged canonical basin production.

Evidence is recorded at execution time in this separate evaluation workspace.
An old raster cannot acquire attribution merely by hashing today's model.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import numpy as np
import rasterio

from .field_validation import sha256
from .field_product_identity import DOWNLOADER, contract_identity, scientific_contract
from .config import ANALYSIS_CRS
from .ee_download import OUTPUT_NODATA
from .overlap_reconciliation import MODIS_SINUSOIDAL_LOCAL_CRS
from .rf25 import RF25_MODEL_FILENAME, RF25_AOA_FILENAME, RF25_METADATA_FILENAME
from .rf25_overlap_production import (
    RF25_EXACT_OVERLAP_PRODUCTION_VERSION as VERSION, OUTPUT_BANDS,
    CONSERVATION_SCOPE, PUBLISHED_RASTER_CONSERVATION,
)
from .workspace import get_workspace_paths, WORKSPACE_ENV_VAR


def execution_contract(root: Path) -> dict:
    workspace = get_workspace_paths(root)
    paths = {f"models/{name}": workspace.models / name for name in
             (RF25_MODEL_FILENAME, RF25_AOA_FILENAME, RF25_METADATA_FILENAME)}
    paths.update({p.relative_to(root).as_posix(): p for p in
                  (root / "src/et_downscaling").glob("*.py") if not p.name.startswith("field_")})
    for name in ("scripts/produce_rf25_rasters.py", "data/boundaries/fundacion_basin.geojson"):
        paths[name] = root / name
    return {name: sha256(path) for name, path in sorted(paths.items())}


def production_contract(root: Path) -> dict:
    return scientific_contract(execution_contract(root), (root / DOWNLOADER).read_bytes().decode("utf-8"))


def scientific_metadata(date: str) -> dict:
    """Frozen scientific settings; paths and unstable joblib signatures are separate."""
    return {
        "period_start": date, "production_method_version": VERSION,
        "analysis_crs": ANALYSIS_CRS, "prediction_scale_m": 20,
        "tile_size_m": 4000, "minimum_tile_size_m": 500,
        "support_halo_rule": "one_or_more_full_tile_rings_covering_at_least_1000_m",
        "output_bands": OUTPUT_BANDS, "usable_support_fraction": 0.90,
        "conservation_tolerance_mm": 0.01, "conservation_scope": CONSERVATION_SCOPE,
        "published_raster_conservation": PUBLISHED_RASTER_CONSERVATION,
        "applicability_rule": "complete_stack AND weighted_RF_AOA_inside AND Kc_raw >= 0",
        "aoa_method": "RF permutation-importance weighted L2 DI; spatial-CV threshold; LPD diagnostic",
        "reconciliation": "single_global_exact_overlap_after_raw_mosaic",
        "negative_et_rule": "floor_once_to_zero_then_fail_if_conservation_exceeds_tolerance",
    }


def companion_paths(raster_root: Path, date: str):
    directory = raster_root.parent / "rasters_modis" / date
    return (directory / f"MODIS_ET_{date}_native.tif",
            directory / f"production_metadata_MODIS_ET_{date}_native.json")


def output_hashes(raster_root: Path, date: str) -> dict:
    raster, metadata, _ = product_paths(raster_root, date)
    modis, modis_metadata = companion_paths(raster_root, date)
    return {name: sha256(path) for name, path in (
        ("raster_sha256", raster), ("metadata_sha256", metadata),
        ("modis_raster_sha256", modis), ("modis_metadata_sha256", modis_metadata))}


def check_product_metadata(raster_root: Path, date: str, contract: dict) -> None:
    raster, metadata_path, _ = product_paths(raster_root, date)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if any(metadata.get(k) != v for k, v in scientific_metadata(date).items()):
        raise ValueError("RF25 scientific metadata differs from the frozen contract.")
    error = metadata.get("max_abs_conservation_error_after_floor_mm", float("nan"))
    if not np.isfinite(error) or not 0 <= error <= 0.01:
        raise ValueError("RF25 global reconciliation did not pass.")
    for name in (RF25_MODEL_FILENAME, RF25_AOA_FILENAME, RF25_METADATA_FILENAME):
        if sha256(raster_root.parent / "models" / name) != contract.get(f"models/{name}"):
            raise ValueError("Isolated RF25 model/AOA bytes differ from scientific inputs.")
    with rasterio.open(raster) as src:
        t = src.transform
        if (src.crs is None or src.crs.to_string() != ANALYSIS_CRS
                or tuple(src.descriptions) != tuple(OUTPUT_BANDS)
                or any(d != "float32" for d in src.dtypes) or src.nodata != OUTPUT_NODATA
                or (t.a, t.b, t.d, t.e) != (20, 0, 0, -20) or t.c % 20 or t.f % 20):
            raise ValueError("RF25 raster grid/bands/nodata differ from the frozen contract.")
    modis_path, modis_metadata_path = companion_paths(raster_root, date)
    modis_meta = json.loads(modis_metadata_path.read_text(encoding="utf-8"))
    expected = {"period_start": date, "source_product": "MOD16A2GF v6.1",
                "band": "ET_MODIS_mm_period", "units": "mm_per_modis_period",
                "native_grid_preserved": True, "spatial_resampling": "none",
                "basin_mask_rule": "retain native MODIS cells intersecting basin (all_touched=True)"}
    if any(modis_meta.get(k) != v for k, v in expected.items()):
        raise ValueError("Native MODIS scientific metadata differs.")
    with rasterio.open(modis_path) as src:
        t = src.transform
        if (src.crs != MODIS_SINUSOIDAL_LOCAL_CRS or src.descriptions != ("ET_MODIS_mm_period",)
                or src.dtypes != ("float32",) or src.nodata != OUTPUT_NODATA
                or t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0
                or not np.isclose(t.a, 463.3127165279165, rtol=0, atol=1e-8)
                or not np.isclose(-t.e, t.a, rtol=0, atol=1e-8)
                or not np.isclose(modis_meta.get("native_pixel_size_x_m", np.nan), t.a)
                or not np.isclose(modis_meta.get("native_pixel_size_y_m", np.nan), -t.e)):
            raise ValueError("MODIS native grid/bands/nodata differ.")


def product_paths(raster_root: Path, date: str):
    directory = raster_root / date
    return (directory / f"ET_{VERSION}_{date}_20m.tif",
            directory / f"production_metadata_{VERSION}.json",
            directory / "field_execution_evidence.json")


def product_status(raster_root: Path, date: str, contract: dict | None):
    raster, metadata, evidence = product_paths(raster_root, date)
    if not raster.is_file():
        return "missing_raster"
    if not metadata.is_file():
        return "incomplete_raster"
    if not evidence.is_file() or contract is None:
        return "unverified_provenance"
    try:
        record = json.loads(evidence.read_text(encoding="utf-8"))
        supplement = None
        if record.get("protocol") == "field_canonical_execution_v1":
            supplement = json.loads((raster.parent / "legacy_scientific_verification.json").read_text(encoding="utf-8"))
        verify_execution_evidence(raster_root, date, contract, record, supplement)
    except (OSError, ValueError, KeyError, TypeError):
        return "unverified_provenance"
    return "verified_final_product"


def verify_execution_evidence(raster_root, date, contract, record, supplement=None):
    """Never discard a legacy downloader hash without its exact source snapshot."""
    if (record.get("period_start") != date or record.get("completed") is not True
            or record.get("input_sha256_before") != record.get("input_sha256_after")):
        raise ValueError("Incomplete or inconsistent execution evidence.")
    protocol = record.get("protocol")
    if protocol == "field_canonical_execution_v1":
        if (not supplement or supplement.get("protocol") != "field_legacy_scientific_audit_v1"
                or supplement.get("execution_evidence_sha256") != sha256(product_paths(raster_root, date)[2])):
            raise ValueError("Legacy execution requires a separate verified audit.")
        source = supplement["downloader_source"]
        expected_outputs = supplement["output_sha256"]
        if any(expected_outputs.get(k) != record.get(k) for k in ("raster_sha256", "metadata_sha256")):
            raise ValueError("Audit cannot replace the original execution output hashes.")
        if supplement.get("scientific_contract") != contract:
            raise ValueError("Legacy audit belongs to different scientific inputs.")
    elif protocol == "field_canonical_execution_v2":
        source = record["downloader_source"]
        expected_outputs = record["output_sha256"]
        if record.get("scientific_contract") != contract:
            raise ValueError("Execution belongs to different scientific inputs.")
    else:
        raise ValueError("Unknown field execution protocol.")
    if scientific_contract(record["input_sha256_before"], source) != contract:
        raise ValueError("Execution scientific inputs differ from current inputs.")
    if expected_outputs != output_hashes(raster_root, date):
        raise ValueError("RF25/MODIS output or metadata SHA-256 mismatch.")
    check_product_metadata(raster_root, date, contract)


def inventory_products(dates, raster_roots, contract):
    rows = []
    for date in dates:
        date = pd.Timestamp(date).strftime("%Y-%m-%d")
        choices = [(Path(root), product_status(Path(root), date, contract)) for root in raster_roots]
        chosen = next((item for item in choices if item[1] == "verified_final_product"),
                      next((item for item in choices if item[1] != "missing_raster"), choices[0]))
        root, status = chosen
        rows.append({"period_start": date, "product_status": status,
                     "raster_root": str(root.resolve()), "raster": str(product_paths(root, date)[0].resolve()),
                     "needs_canonical_production": status != "verified_final_product"})
    return pd.DataFrame(rows, columns=["period_start", "product_status", "raster_root", "raster", "needs_canonical_production"])


def produce_field_date(root: Path, output: Path, date: str, project: str, contract: dict) -> Path:
    """Run the actual entry point, with byte-identical models in an isolated workspace.

No training, local reconciliation, or retroactive certification is performed.
Only a successful subprocess and unchanged inputs can create execution evidence.
"""
    date = pd.Timestamp(date).strftime("%Y-%m-%d")
    if production_contract(root) != contract:
        raise RuntimeError("RF25 inputs changed before field production.")
    # Reuse a verified legacy product in place; no copying or contract rewriting.
    for existing in sorted((output / "production").glob("*/rasters")):
        if product_status(existing, date, contract) == "verified_final_product":
            return existing
    identity = contract_identity(contract)
    # Keep canonical tile filenames below the Windows path limit. The full
    # contract is still checked, including the unlikely truncated-hash collision.
    workspace = output / "production" / identity[:16]
    workspace.mkdir(parents=True, exist_ok=True)
    identity_path = workspace / "input_contract.json"
    if identity_path.is_file() and json.loads(identity_path.read_text(encoding="utf-8")) != contract:
        raise RuntimeError("Isolated production directory belongs to different RF25 inputs.")
    identity_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    models = workspace / "models"
    models.mkdir(parents=True, exist_ok=True)
    source = get_workspace_paths(root).models
    for name in (RF25_MODEL_FILENAME, RF25_AOA_FILENAME, RF25_METADATA_FILENAME):
        destination = models / name
        if not destination.is_file() or sha256(destination) != contract[f"models/{name}"]:
            shutil.copy2(source / name, destination)
    raster_root = workspace / "rasters"
    if product_status(raster_root, date, contract) == "verified_final_product":
        return raster_root
    command = [sys.executable, str(root / "scripts/produce_rf25_rasters.py"),
               "--project", project, "--date", date, "--tile-size-m", "4000"]
    environment = os.environ.copy()
    environment[WORKSPACE_ENV_VAR] = str(workspace.resolve())
    started = datetime.now(timezone.utc).isoformat()
    before = execution_contract(root)
    downloader_source = (root / DOWNLOADER).read_bytes().decode("utf-8")
    if scientific_contract(before, downloader_source) != contract:
        raise RuntimeError("RF25 inputs changed before subprocess execution.")
    subprocess.run(command, cwd=root, env=environment, check=True)
    after = execution_contract(root)
    if after != before or any(sha256(models / name) != contract[f"models/{name}"] for name in
                               (RF25_MODEL_FILENAME, RF25_AOA_FILENAME, RF25_METADATA_FILENAME)):
        raise RuntimeError("RF25 inputs changed during canonical field production.")
    raster, metadata, evidence = product_paths(raster_root, date)
    check_product_metadata(raster_root, date, contract)
    record = {"protocol": "field_canonical_execution_v2", "period_start": date,
              "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
              "command": command, "workspace": str(workspace.resolve()), "completed": True,
              "input_sha256_before": before, "input_sha256_after": after,
              "scientific_contract": contract, "scientific_identity": identity,
              "downloader_source": downloader_source, "output_sha256": output_hashes(raster_root, date)}
    temporary = evidence.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(evidence)
    return raster_root
