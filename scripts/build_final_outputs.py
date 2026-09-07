"""Build the minimal local final outputs from the frozen production run."""

from __future__ import annotations

import argparse
from pathlib import Path

from et_downscaling.final_outputs import (
    FINAL_PERIODS,
    build_minimal_final_outputs,
)
from et_downscaling.workspace import get_workspace_paths


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Derive the three one-band ET rasters, copy the three native MODIS "
            "comparison rasters, and build the minimal local final view."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Completed run directory. When omitted, use the newest run that "
            "contains all three frozen final raster periods."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory. Defaults to ET_fundacion_workspace/final."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    workspace = get_workspace_paths(project_root)

    outputs = build_minimal_final_outputs(
        workspace_current=workspace.root,
        project_root=project_root,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        periods=FINAL_PERIODS,
    )

    print("=" * 72)
    print("MINIMAL FINAL OUTPUTS COMPLETE")
    print("=" * 72)
    print("Source run:", outputs["source_run"])
    for period in FINAL_PERIODS:
        print(f"ET {period}:", outputs[f"et_{period}"])
    for period in FINAL_PERIODS:
        print(f"MODIS {period}:", outputs[f"modis_{period}"])
    print("Raster summary:", outputs["raster_summary"])
    print("Visualization notebook:", outputs["notebook"])
    print("Scientific multiband rasters remain in current/rasters and are not copied.")


if __name__ == "__main__":
    main()
