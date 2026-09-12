"""Materialize only the sources required by the frozen RF-25 workflow."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from et_downscaling.config import build_training_output_filename
from et_downscaling.workspace import get_workspace_paths


START_DATE = "2020-01-01"
END_DATE_EXCLUSIVE = "2025-01-01"
VIRTUAL_SUPPORT_COUNT = 10


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild reusable Earth Engine exports instead of accepting valid cached files.",
    )
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
            "Run `python scripts/run_pipeline.py select --project "
            "<earth-engine-project>` first."
        )
    os.environ["ET_STATIONS_GEOJSON"] = str(points.resolve())
    os.environ["ET_CANDIDATE_SUPPORT_COUNT"] = str(VIRTUAL_SUPPORT_COUNT)
    os.environ["ET_START_DATE"] = START_DATE
    os.environ["ET_END_DATE_EXCLUSIVE"] = END_DATE_EXCLUSIVE
    return points


def main() -> None:
    args = parse_args()
    points = configure_virtual_supports()
    force_args = ("--force",) if args.force else ()

    print("=" * 92)
    print("RF-25 REQUIRED INPUT MATERIALIZATION")
    print("=" * 92)
    print("Support source:", points)
    print("Supports:", VIRTUAL_SUPPORT_COUNT)
    print("Period:", START_DATE, "to", END_DATE_EXCLUSIVE)
    print("Required sources: Sentinel-2, MODIS target support and ERA5-Land")
    print("Historical candidate families required: NO")

    run(
        "scripts/export_meteorology_data.py",
        "--project",
        args.project,
        "--model-only",
        *force_args,
    )
    run(
        "scripts/export_satellite_data.py",
        "--project",
        args.project,
        "--optical-source",
        "S2",
        "--model-only",
        *force_args,
    )
    run("scripts/build_training_dataset.py", "--optical-source", "S2", "--model-only")

    workspace = get_workspace_paths(root())
    output = workspace.master / "S2" / build_training_output_filename("S2")
    if not output.is_file():
        raise RuntimeError(f"RF-25 source master was not created: {output}")
    print("\nRF-25 source master:", output)


if __name__ == "__main__":
    main()
