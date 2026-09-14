"""Audit helpers for the rejected local RF25 prediction proposal.

The component helpers demonstrate algebra only; they do not publish station ET.
A fixed halo is NOT a valid boundary condition for that solver. All eligible
parents connected by shared fine cells are therefore followed to closure, along
with their ineligible boundary parents. This may require many canonical tiles.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import rasterio

from .field_validation import sha256
from .overlap_reconciliation import OverlapEdges, solve_overlap_reconciliation
from .rf25 import RF25_MODEL_FILENAME, RF25_AOA_FILENAME, RF25_METADATA_FILENAME, validate_rf25_model
from .rf25_overlap_production import (
    RF25_EXACT_OVERLAP_PRODUCTION_VERSION,
    build_production_scientific_signature,
)
from .workspace import get_workspace_paths


def dependency_closure(seeds, neighbors, evaluate):
    """Return evaluated boundary + entire eligible components touching seeds.

No eligibility is assumed outside the current patch. evaluate(parent) must
load ALL its fine cells before making the canonical support decision.
"""
    pending = deque(int(x) for x in seeds)
    evaluated = {}
    while pending:
        parent = pending.popleft()
        if parent in evaluated:
            continue
        eligible = bool(evaluate(parent))
        evaluated[parent] = eligible
        if eligible:
            pending.extend(int(x) for x in neighbors(parent) if int(x) not in evaluated)
    return evaluated


def solve_closed_component(edges, parents, raw_kc, usable, modis):
    """Compact indices only; all numbers/equations are passed to the canonical solver."""
    selected = np.isin(edges.coarse_index, list(parents))
    coarse = edges.coarse_index[selected]
    fine_global, fine_local = np.unique(edges.fine_index[selected], return_inverse=True)
    represented = np.zeros(modis.shape, dtype=bool)
    represented.ravel()[list(parents)] = True
    compact_edges = OverlapEdges(coarse, fine_local.astype(np.int32), edges.overlap_area_m2[selected], represented)
    result = solve_overlap_reconciliation(
        kc_raw=raw_kc.ravel()[fine_global][None, :],
        usable=usable.ravel()[fine_global][None, :],
        modis_et=modis, edges=compact_edges,
    )
    return result, fine_global


def audit_existing_raster(root: Path, output: Path):
    """Record evidence without treating timestamps or an unstable joblib hash as proof."""
    workspace = get_workspace_paths(root)
    date = "2022-03-30"
    directory = workspace.rasters / date
    path = directory / f"ET_{RF25_EXACT_OVERLAP_PRODUCTION_VERSION}_{date}_20m.tif"
    metadata_path = directory / f"production_metadata_{RF25_EXACT_OVERLAP_PRODUCTION_VERSION}.json"
    records = []
    paths = [path, metadata_path, workspace.models / RF25_MODEL_FILENAME,
             workspace.models / RF25_AOA_FILENAME, workspace.models / RF25_METADATA_FILENAME]
    paths.extend(sorted(directory.glob("tile_manifest_*.csv")))
    for item in paths:
        if item.is_file():
            stat = item.stat()
            records.append({"path": str(item.resolve()), "sha256": sha256(item), "size_bytes": stat.st_size,
                            "created_utc": datetime.fromtimestamp(stat.st_birthtime, timezone.utc).isoformat() if hasattr(stat, "st_birthtime") else None,
                            "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    signatures, trees = [], []
    aoa = joblib.load(workspace.models / RF25_AOA_FILENAME)
    for _ in range(3):
        model = joblib.load(workspace.models / RF25_MODEL_FILENAME)
        validate_rf25_model(model)
        signatures.append(build_production_scientific_signature(model, aoa))
        digest = hashlib.sha256()
        for estimator in model.estimators_:
            state = estimator.tree_.__getstate__()
            for name in state["nodes"].dtype.names:
                digest.update(np.ascontiguousarray(state["nodes"][name]).tobytes())
            digest.update(state["values"].tobytes())
        trees.append(digest.hexdigest())
    raster_info = None
    if path.is_file():
        with rasterio.open(path) as source:
            raster_info = {"crs": source.crs.to_string(), "transform": list(source.transform),
                           "shape": source.shape, "dtypes": source.dtypes,
                           "bands": source.descriptions, "tags": source.tags()}
    report = {"files": records, "raster_metadata": metadata, "raster_info": raster_info,
              "current_model_parameters": model.get_params(deep=False),
              "current_AOA_threshold": aoa.threshold,
              "canonical_signatures_repeated_loads": signatures, "tree_numeric_sha256_repeated_loads": trees,
              "attribution": "Timestamps are not model evidence. model_source is hardcoded to fitted_in_current_run even when the entry point loads joblib files. No contemporaneous manifest binds this raster to current model/AOA bytes; attribution is unproven.",
              "signature_limitation": "Repeated loads of identical model bytes yield different joblib-based signatures with identical numeric tree fields; the canonical signature alone cannot establish scientific identity.",
              "use_in_validation": "Excluded from validation and reproduction evidence until a contemporaneous run record binds raster, model and AOA bytes."}
    (output / "raster_2022_03_30_provenance_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
