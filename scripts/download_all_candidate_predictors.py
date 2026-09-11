"""Materialize every implemented 2020-2024 predictor family on Virtual10.

The ten selected virtual MODIS supports are the only sampling supports used by
this archive.  All implemented candidate families are preserved for audit and
future work, while the final RF model remains frozen to the accepted 25
predictors.  Real field stations are not used for model training or candidate
materialization.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


START_DATE = "2020-01-01"
END_DATE_EXCLUSIVE = "2025-01-01"
PERIOD_LABEL = "2020_2024"
VIRTUAL_SUPPORT_COUNT = 10


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    return parser.parse_args()


def run(relative_script: str, *arguments: str) -> None:
    command = [sys.executable, str(root() / relative_script), *arguments]
    print("\n>", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=root(), check=True, env=os.environ.copy())


def configure_virtual_supports() -> Path:
    points = root() / "outputs" / "training" / "selection" / "virtual_points.geojson"
    if not points.is_file():
        raise FileNotFoundError(
            f"Virtual10 points not found: {points}\n"
            "Run `python scripts/run_pipeline.py select --project <PROJECT>` first."
        )
    os.environ["ET_STATIONS_GEOJSON"] = str(points.resolve())
    os.environ["ET_CANDIDATE_SUPPORT_COUNT"] = str(VIRTUAL_SUPPORT_COUNT)
    os.environ["ET_START_DATE"] = START_DATE
    os.environ["ET_END_DATE_EXCLUSIVE"] = END_DATE_EXCLUSIVE
    return points


def main() -> None:
    args = parse_args()
    points = configure_virtual_supports()

    print("=" * 92)
    print("COMPLETE VIRTUAL10 CANDIDATE-PREDICTOR ARCHIVE")
    print("=" * 92)
    print("Support source:", points)
    print("Supports:", VIRTUAL_SUPPORT_COUNT)
    print("Period:", START_DATE, "to", END_DATE_EXCLUSIVE)
    print("Real field stations used for candidate extraction: NO")
    print("Final RF-25 feature selection reopened: NO")

    # Canonical reusable sources on Virtual10.  Operational S2 is exported in
    # model-only mode because orbit-specific S1 is materialized separately below.
    run("scripts/export_meteorology_data.py", "--project", args.project)
    run(
        "scripts/export_satellite_data.py",
        "--project", args.project,
        "--optical-source", "S2",
        "--model-only",
    )
    run("scripts/build_training_dataset.py", "--optical-source", "S2")

    common = (
        "--start-date", START_DATE,
        "--end-date-exclusive", END_DATE_EXCLUSIVE,
        "--period-label", PERIOD_LABEL,
    )

    # Earth Engine candidate-family materialization on exactly the same ten
    # virtual supports.  These data remain candidates/diagnostics unless they
    # belong to the frozen RF-25 feature list.
    run(
        "scripts/candidates/export_availability.py",
        *common, "--project", args.project, "--execute",
    )
    run(
        "scripts/candidates/export_optical_candidates.py",
        *common, "--project", args.project, "--execute",
    )
    run(
        "scripts/candidates/export_s2_candidates.py",
        *common, "--project", args.project, "--execute",
    )
    run(
        "scripts/candidates/export_s1_candidates.py",
        *common, "--project", args.project, "--execute",
    )
    run(
        "scripts/candidates/export_hls_fvc_albedo_candidates.py",
        *common, "--project", args.project, "--execute",
    )
    run(
        "scripts/candidates/export_thermal_candidates.py",
        *common, "--project", args.project, "--execute",
    )
    run(
        "scripts/candidates/export_landsat_lst_candidates.py",
        "--project", args.project, "--execute",
    )

    # Local row-preserving assembly; no feature selection or model fitting.
    run("scripts/candidates/build_optical_candidates.py")
    run("scripts/candidates/build_meteorology_candidates.py")
    run("scripts/candidates/build_candidate_feature_store.py")
    run("scripts/candidates/build_candidate_master.py")

    output = root() / "outputs" / "current" / "master" / "master_predictor_store.parquet"
    if not output.is_file():
        raise RuntimeError(f"Candidate master was not created: {output}")

    print("\nComplete implemented Virtual10 candidate archive:", output)
    print("RF-25 remains frozen to its accepted 25 predictors.")


if __name__ == "__main__":
    main()
