"""Produce final 20 m RF-25 ET rasters using weighted AOA and exact overlaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import joblib

from et_downscaling.rf25 import RF25_AOA_FILENAME, RF25_MODEL_FILENAME, validate_rf25_model
from et_downscaling.rf25_overlap_production import download_rf25_basin
from et_downscaling.workspace import get_workspace_paths, require_portable_inputs


DEFAULT_DATES = ["2020-03-13", "2022-10-24", "2022-03-30"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Google Cloud project ID with Earth Engine access.")
    parser.add_argument(
        "--date",
        dest="dates",
        action="append",
        default=None,
        help="Period-start date YYYY-MM-DD. Repeat for multiple dates. Defaults to the three manuscript map dates.",
    )
    parser.add_argument("--tile-size-m", type=int, default=4000)
    parser.add_argument("--min-tile-size-m", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    require_portable_inputs(root)
    workspace = get_workspace_paths(root).ensure()
    model_path = workspace.models / RF25_MODEL_FILENAME
    aoa_path = workspace.models / RF25_AOA_FILENAME
    if not model_path.is_file() or not aoa_path.is_file():
        raise FileNotFoundError(
            "Final RF model/AOA are missing. Run scripts/train_rf25.py first.\n"
            f"Model: {model_path}\nAOA: {aoa_path}"
        )

    model = joblib.load(model_path)
    aoa = joblib.load(aoa_path)
    validate_rf25_model(model)
    dates = list(dict.fromkeys(args.dates or DEFAULT_DATES))

    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()

    for index, date_text in enumerate(dates, start=1):
        print()
        print("=" * 100)
        print(f"RF-25 PRODUCTION {index}/{len(dates)}: {date_text}")
        print("=" * 100)
        product = download_rf25_basin(
            project_root=root,
            period_start=date_text,
            model=model,
            aoa_parameters=aoa,
            tile_size_m=args.tile_size_m,
            min_tile_size_m=args.min_tile_size_m,
        )
        print("Final raster:", product["raster"])
        print("Native MODIS raster:", product["modis_raster"])


if __name__ == "__main__":
    main()
