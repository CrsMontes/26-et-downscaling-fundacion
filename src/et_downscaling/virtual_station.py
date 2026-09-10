"""Virtual Station support-selection paths and validation helpers."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd


V5_NAME = "v5_basin_random_ge90_canonical"
V5_AOA_THRESHOLD = 0.800732851533
V5_DATES = ("2020-03-13", "2021-11-25", "2022-03-30")
STABLE_RUN = "20260907T162048Z_2020_2024"


def resolve_virtual_workspace(value=None, repository_root=None) -> Path:
    """Never inherit the old, mixed ET_FUNDACION_WORKSPACE implicitly."""
    override = value or os.environ.get("ET_FUNDACION_VIRTUAL_WORKSPACE", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        return path.parent if path.name.lower() == "current" else path
    root = Path(repository_root) if repository_root else Path(__file__).resolve().parents[2]
    return root / "outputs"


def resolve_reference_workspace(value) -> Path:
    if not value:
        raise ValueError("Stable5 comparison requires an explicit --reference-workspace.")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def resolve_reference_run(reference_root: Path, value=None) -> Path:
    path = Path(value).expanduser().resolve() if value else reference_root / "current" / "runs" / STABLE_RUN
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def find_v5_raster(workspace_root: Path, date_text: str) -> Path:
    paths = sorted((workspace_root / "current" / "rasters" / date_text).glob(f"ET_ridge25_*_{date_text}_20m.tif"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one frozen V5 raster for {date_text}; found {len(paths)}.")
    if paths[0].stat().st_size == 0:
        raise ValueError(f"Empty V5 raster: {paths[0]}")
    return paths[0]


def validate_frozen_v5(workspace_root: Path, *, check_rasters=True) -> dict:
    """Read frozen tables; never fit, select, recheck availability, or download."""
    selection_root = workspace_root / "training" / "selection"
    selected = pd.read_csv(selection_root / "selected_supports.csv", dtype={"virtual_id": str})
    selection = json.loads((selection_root / "selection_metadata.json").read_text(encoding="utf-8"))
    results_root = workspace_root / "evaluation" / "results"
    population = pd.read_csv(results_root / "virtual10_training_population.csv", dtype={"station_id": str})
    metadata = json.loads((results_root / "experiment_metadata.json").read_text(encoding="utf-8"))
    if selection.get("seed") != 42 or selection.get("n_supports") != 10 or selection.get("selection_complete") is not True:
        raise ValueError("Frozen V5 selection must be complete, with seed 42 and 10 supports.")
    if len(selected) != 10 or selected["virtual_id"].nunique() != 10 or selected["modis_pixel_id"].nunique() != 10:
        raise ValueError("Frozen V5 selection must have 10 unique supports and MODIS pixels.")
    if selected["spatial_block_utm10km"].nunique() != 10:
        raise ValueError("Frozen V5 selection must retain 10 distinct UTM blocks.")
    if len(population) != 1454 or population["station_id"].nunique() != 10 or population["spatial_block"].nunique() != 10:
        raise ValueError("Frozen V5 population must retain 1454 rows, 10 supports and 10 blocks.")
    if set(population["station_id"]) != set(selected["virtual_id"]):
        raise ValueError("Frozen selection and training support IDs differ.")
    if not population["station_id"].str.startswith("VF").all():
        raise ValueError("V5 population contains a real-station ID.")
    block_map = selected.set_index("virtual_id")["spatial_block_utm10km"].astype(str)
    if not population["spatial_block"].astype(str).eq(population["station_id"].map(block_map)).all():
        raise ValueError("Training blocks differ from the frozen selection.")
    threshold = float(metadata["virtual10_AOA_threshold"])
    if not math.isclose(threshold, V5_AOA_THRESHOLD, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Frozen V5 AOA threshold changed: {threshold}")
    if population.duplicated(["station_id", "period_start"]).any():
        raise ValueError("Duplicate V5 support-period rows.")
    for row in selected.itertuples(index=False):
        check = pd.read_csv(selection_root / "availability_checks" / f"{int(row.candidate_order):05d}_{int(row.modis_pixel_id)}.csv")
        frozen = set(pd.to_datetime(check.loc[pd.to_numeric(check["ge90"]).eq(1), "period_start"]).dt.strftime("%Y-%m-%d"))
        trained = set(pd.to_datetime(population.loc[population["station_id"].eq(row.virtual_id), "period_start"]).dt.strftime("%Y-%m-%d"))
        if frozen != trained or len(frozen) != int(row.ge90_total):
            raise ValueError(f"Frozen GE90 periods differ from training for {row.virtual_id}.")
    rasters = {date: str(find_v5_raster(workspace_root, date)) for date in V5_DATES} if check_rasters else {}
    return {"workspace": str(workspace_root), "seed": 42, "supports": 10, "blocks": 10, "rows": 1454, "AOA_threshold": threshold, "rasters": rasters}
