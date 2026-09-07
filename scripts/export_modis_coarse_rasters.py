"""Export native-grid MODIS ET basin rasters for selected final periods.

This utility backfills the coarse MODIS products without retraining Ridge25 or
regenerating the fine-resolution products. Future full pipeline runs create the
same native MODIS outputs automatically during exact-overlap production.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee

from et_downscaling.ridge25_overlap_production import download_native_modis_basin


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Export native-grid MODIS ET basin rasters for one or more periods."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Google Cloud Project ID with Earth Engine access.",
    )
    parser.add_argument(
        "--raster-date",
        dest="raster_dates",
        action="append",
        required=True,
        help="MODIS period start (YYYY-MM-DD). Repeat for multiple periods.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]

    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()

    print("=" * 72)
    print("NATIVE MODIS ET EXPORT")
    print("=" * 72)

    for period in dict.fromkeys(args.raster_dates):
        product = download_native_modis_basin(
            project_root=project_root,
            period_start=period,
        )
        print()
        print("Period:", period)
        print("  Raster:", product["raster"])
        print("  Metadata:", product["metadata"])

    print()
    print("NATIVE MODIS ET EXPORT COMPLETE")


if __name__ == "__main__":
    main()
