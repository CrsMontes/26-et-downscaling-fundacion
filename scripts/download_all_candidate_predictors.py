"""Materialize the complete implemented 2020-2024 candidate-predictor archive.

This archive is deliberately separate from RF-25 training. It reconstructs all
candidate families for which this repository already contains audited export
code (S2, HLS, S1 R077/R142, ERA5-Land, CHIRPS, Landsat LST, albedo/FVC and
seasonality) on the canonical five field-station MODIS footprints. The final
Virtual10 RF fit remains frozen to its 25 predictors.
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


def main() -> None:
    args = parse_args()
    os.environ["ET_START_DATE"] = START_DATE
    os.environ["ET_END_DATE_EXCLUSIVE"] = END_DATE_EXCLUSIVE

    # Reusable five-station raw source tables, rebuilt from Earth Engine.
    run("scripts/export_meteorology_data.py", "--project", args.project, "--force")
    run(
        "scripts/export_satellite_data.py",
        "--project", args.project,
        "--optical-source", "S2",
        "--force",
    )
    run("scripts/build_training_dataset.py", "--optical-source", "S2")

    common = (
        "--start-date", START_DATE,
        "--end-date-exclusive", END_DATE_EXCLUSIVE,
        "--period-label", PERIOD_LABEL,
    )

    # Earth Engine candidate-family materialization.
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

    # Local assembly; no additional Earth Engine access.
    run("scripts/candidates/build_meteorology_candidates.py")
    run("scripts/candidates/build_optical_candidates.py")
    run("scripts/candidates/build_candidate_feature_store.py")
    run("scripts/candidates/build_candidate_master.py")

    print("\nComplete implemented candidate archive materialized under outputs/current/.")
    print("It is audit/future-analysis data only; RF-25 feature selection was not reopened.")


if __name__ == "__main__":
    main()
