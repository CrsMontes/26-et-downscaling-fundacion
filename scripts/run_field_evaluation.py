"""Run the final field-comparison phase from the canonical workspace.

This orchestrates the accepted field workflow:
1. in-basin spatial-OOF exact-overlap evaluation;
2. external ST04 evaluation with nearest-valid ERA5-Land support when native
   coastal ERA5-Land is masked;
3. AOA-only sensitivity with every other publication rule held fixed;
4. explicit field scenarios and metrics with local 20 m NDVI for ST04-ST05.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from et_downscaling.workspace import get_workspace_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def run(root: Path, name: str, args: list[str]) -> None:
    command = [
        sys.executable,
        str(root / "scripts" / name),
        *args,
    ]
    print()
    print(">", " ".join(command))
    subprocess.run(command, cwd=root, check=True)


def reset_external_st04_cache(root: Path) -> None:
    # Remove external ST04 fine caches before a deliberate fresh rebuild.
    workspace = get_workspace_paths(root).ensure()
    path = (
        workspace.diagnostics
        / "field_ridge25_oof_exact_overlap_external_st04_era5_nearest"
    )
    if path.exists():
        shutil.rmtree(path)
        print()
        print("Removed stale external ST04 cache:", path)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    base_args = ["--project", args.project]
    if args.restart:
        base_args.append("--restart")
        reset_external_st04_cache(root)

    run(root, "evaluate_field_ridge25.py", base_args)
    run(
        root,
        "evaluate_field_external_st04.py",
        ["--project", args.project],
    )
    run(
        root,
        "compare_field_with_without_aoa.py",
        ["--project", args.project],
    )
    run(
        root,
        "build_field_comparison_scenarios.py",
        ["--project", args.project],
    )

    print()
    print("=" * 72)
    print("FINAL FIELD COMPARISON COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
