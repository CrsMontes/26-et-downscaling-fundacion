"""Generate missing Virtual Station V5 scientific rasters from frozen training.

This command reconstructs the accepted Ridge25 model and equal-weight AOA from
the frozen V5 training population. It does not select supports or rebuild the
training population. Existing scientific rasters are never overwritten.

The old generic ET_FUNDACION_WORKSPACE variable is overridden only inside this
process so the shared production machinery writes to the Virtual Station
workspace/current directory. The previous environment value is restored before
exit.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd

from et_downscaling.virtual_station import (
    V5_AOA_THRESHOLD,
    resolve_virtual_workspace,
    validate_frozen_v5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument(
        "--date",
        dest="dates",
        action="append",
        required=True,
        help="MODIS-period start date YYYY-MM-DD. Repeat for multiple dates.",
    )
    parser.add_argument("--tile-size-m", type=int, default=4000)
    parser.add_argument("--min-tile-size-m", type=int, default=500)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def existing_scientific_raster(
    workspace_root: Path,
    date_text: str,
) -> Path | None:
    matches = sorted(
        (workspace_root / "current" / "rasters" / date_text).glob(
            f"ET_ridge25_*_{date_text}_20m.tif"
        )
    )
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected at most one scientific V5 raster for {date_text}; "
            f"found {len(matches)}."
        )
    return matches[0] if matches else None


def main() -> None:
    args = parse_args()
    root = project_root()
    workspace_root = resolve_virtual_workspace(args.workspace_root, root)

    # Validate the frozen scientific inputs without requiring rasters to exist.
    frozen = validate_frozen_v5(workspace_root, check_rasters=False)

    dates = list(dict.fromkeys(str(value).strip() for value in args.dates))
    if any(not value for value in dates):
        raise ValueError("Raster dates cannot be empty.")

    missing_dates = []
    for date_text in dates:
        existing = existing_scientific_raster(workspace_root, date_text)
        if existing is not None:
            print(f"Existing scientific raster preserved: {existing}")
        else:
            missing_dates.append(date_text)

    if not missing_dates:
        print("All requested scientific rasters already exist. Nothing to do.")
        return

    population_path = (
        workspace_root
        / "evaluation"
        / "results"
        / "virtual10_training_population.csv"
    )
    population = pd.read_csv(population_path, dtype={"station_id": str})

    from et_downscaling.aoa_ridge25 import build_unweighted_aoa
    from et_downscaling.ridge25 import (
        RIDGE25_MODEL_FEATURES,
        build_ridge25_model,
    )

    model = build_ridge25_model()
    model.fit(
        population[RIDGE25_MODEL_FEATURES],
        population["Kc_target"],
    )
    aoa = build_unweighted_aoa(
        population,
        group_column="spatial_block",
    )

    if not math.isclose(
        float(aoa.threshold),
        float(frozen["AOA_threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Reconstructed AOA threshold differs from frozen metadata: "
            f"{aoa.threshold} vs {frozen['AOA_threshold']}"
        )
    if not math.isclose(
        float(aoa.threshold),
        V5_AOA_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"Unexpected V5 AOA threshold: {aoa.threshold}"
        )

    # Shared production code still uses the historical generic workspace
    # variable. Override it only inside this process and restore on exit.
    previous_workspace = os.environ.get("ET_FUNDACION_WORKSPACE")
    os.environ["ET_FUNDACION_WORKSPACE"] = str(
        workspace_root / "current"
    )

    try:
        import ee

        from et_downscaling.ridge25_overlap_production import (
            download_ridge25_basin,
        )

        ee.Initialize(project=args.project)
        ee.Number(1).getInfo()

        for date_text in missing_dates:
            print()
            print("=" * 100)
            print(f"VIRTUAL STATION V5 PRODUCTION: {date_text}")
            print("=" * 100)

            product = download_ridge25_basin(
                project_root=root,
                period_start=date_text,
                model=model,
                aoa_parameters=aoa,
                tile_size_m=args.tile_size_m,
                min_tile_size_m=args.min_tile_size_m,
            )

            output_raster = Path(product["raster"]).resolve()
            current_root = (workspace_root / "current").resolve()
            if not output_raster.is_relative_to(current_root):
                raise RuntimeError(
                    "Production escaped the Virtual Station current workspace: "
                    f"{output_raster}"
                )

            print("Scientific raster:", output_raster)
            print("Native MODIS raster:", product["modis_raster"])

    finally:
        if previous_workspace is None:
            os.environ.pop("ET_FUNDACION_WORKSPACE", None)
        else:
            os.environ["ET_FUNDACION_WORKSPACE"] = previous_workspace


if __name__ == "__main__":
    main()
