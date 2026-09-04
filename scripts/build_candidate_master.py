"""Build the canonical five-year candidate predictor master.

This script assembles the row-preserving 2020-2024 candidate master without
running predictor-selection or sensitivity analyses.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from et_downscaling.workspace import get_workspace_paths


EXPECTED_ROWS = 1150
KEYS = [
    "station_id",
    "modis_pixel_id",
    "period_start",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_master_builder(root: Path):
    script = (
        root
        / "reproducibility"
        / "scripts"
        / "run_predictor_availability_ladder.py"
    )

    spec = importlib.util.spec_from_file_location(
        "candidate_master_assembly",
        script,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load candidate-master assembly: {script}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    root = project_root()
    workspace = get_workspace_paths(root)
    module = load_master_builder(root)

    master = module.build_master_store()

    if len(master) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} master rows; found {len(master)}"
        )

    missing_keys = sorted(set(KEYS) - set(master.columns))
    if missing_keys:
        raise RuntimeError(
            f"Canonical master is missing keys: {missing_keys}"
        )

    unique_keys = len(master[KEYS].drop_duplicates())
    if unique_keys != EXPECTED_ROWS:
        raise RuntimeError(
            "Canonical master keys are not one-to-one: "
            f"{unique_keys}/{EXPECTED_ROWS}"
        )

    required_candidates = [
        "hls_Albedo_mean",
        "hls_FVC_mean",
        "LST_parent_mean_K",
    ]
    missing_candidates = sorted(
        set(required_candidates) - set(master.columns)
    )
    if missing_candidates:
        raise RuntimeError(
            "Canonical master is incomplete: "
            f"{missing_candidates}"
        )

    output = workspace.master / "master_predictor_store.parquet"
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    master.to_parquet(
        output,
        index=False,
    )

    print("=" * 80)
    print("CANONICAL FIVE-YEAR CANDIDATE MASTER")
    print("=" * 80)
    print("Rows:", len(master))
    print("Columns:", len(master.columns))
    print("Unique keys:", unique_keys)
    print(
        "HLS Albedo available:",
        int(master["hls_Albedo_mean"].notna().sum()),
    )
    print(
        "HLS FVC available:",
        int(master["hls_FVC_mean"].notna().sum()),
    )
    print(
        "Landsat LST available:",
        int(master["LST_parent_mean_K"].notna().sum()),
    )
    print("Output:", output)
    print("Sensitivity analysis performed: false")


if __name__ == "__main__":
    main()
