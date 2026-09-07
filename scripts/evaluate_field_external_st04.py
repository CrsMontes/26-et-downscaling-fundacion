"""Evaluate external ST04 fine-resolution OOF ET with station-consistent ERA5-Land fallback.

This script does not modify the basin production code. It applies the accepted
external-site diagnostic rule: wherever the native fine-production ERA5-Land
stack is masked, fill only those masked meteorological pixels with the same
nearest-valid ERA5-Land support used for ST04 during footprint-scale training.

Valid native ERA5-Land pixels are never replaced.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import ee
import numpy as np
import pandas as pd

import et_downscaling.ridge25_production as ridge25_production
from et_downscaling.config import ANALYSIS_CRS
from et_downscaling.local_tiles import Tile, build_initial_tiles
from et_downscaling.ridge25 import RIDGE25_METEOROLOGICAL_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    return parser.parse_args()


def load_field_module(root: Path):
    script_path = root / "scripts" / "evaluate_field_ridge25.py"
    spec = importlib.util.spec_from_file_location("field_eval", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_external_center_tile(module, root: Path, x: float, y: float) -> Tile:
    _, grid_bounds = build_initial_tiles(
        root,
        tile_size_m=module.LOCAL_TILE_SIZE_M,
    )
    grid_xmin, _, _, grid_ymax = grid_bounds
    size = float(module.LOCAL_TILE_SIZE_M)

    col = math.floor((x - grid_xmin) / size)
    row = math.floor((grid_ymax - y) / size)

    xmin = grid_xmin + col * size
    ymax = grid_ymax - row * size

    return Tile(
        xmin=xmin,
        ymin=ymax - size,
        xmax=xmin + size,
        ymax=ymax,
        tile_id=f"external_r{row:03d}_c{col:03d}",
        level=0,
    )


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    module = load_field_module(root)

    (
        workspace,
        master,
        reference,
        field,
        metadata,
        master_path,
        reference_path,
    ) = module.load_inputs(root)

    master = master.copy()
    master["station_id"] = master["station_id"].astype(str)

    support_columns = [
        "station_id",
        "era5_sampling_longitude",
        "era5_sampling_latitude",
        "era5_sampling_distance_m",
    ]
    missing = [c for c in support_columns if c not in master.columns]
    if missing:
        raise RuntimeError(
            "Master is missing ERA5 support columns: " + ", ".join(missing)
        )

    support = (
        master.loc[
            master["station_id"].eq("ST04"),
            support_columns,
        ]
        .dropna(
            subset=[
                "era5_sampling_longitude",
                "era5_sampling_latitude",
            ]
        )
        .drop_duplicates()
    )
    if len(support) != 1:
        raise RuntimeError(
            f"Expected one ST04 ERA5 support record, found {len(support)}."
        )

    support_row = support.iloc[0]
    support_lon = float(support_row["era5_sampling_longitude"])
    support_lat = float(support_row["era5_sampling_latitude"])
    support_distance = float(support_row["era5_sampling_distance_m"])

    print()
    print("=" * 110)
    print("ST04 ERA5-LAND SUPPORT")
    print("=" * 110)
    print("Method: nearest_valid_land_pixel (from training master)")
    print(f"Longitude: {support_lon:.8f}")
    print(f"Latitude:  {support_lat:.8f}")
    print(f"Distance from station: {support_distance:.2f} m")

    print()
    print("Initializing Earth Engine...")
    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()

    # Save the unmodified production meteorology builder.
    native_builder = ridge25_production.build_ridge25_meteorological_predictors

    support_point = ee.Geometry.Point([support_lon, support_lat])
    support_geometry = support_point.buffer(1000)

    def station_consistent_meteorology(
        period_start,
        period_end,
        number_days,
        processing_geometry,
    ):
        """Fill only masked native ERA5 pixels using ST04 training support."""
        native = native_builder(
            period_start=period_start,
            period_end=period_end,
            number_days=number_days,
            processing_geometry=processing_geometry,
        )

        support_meteorology = native_builder(
            period_start=period_start,
            period_end=period_end,
            number_days=number_days,
            processing_geometry=support_geometry,
        )

        era5_projection = (
            ee.Image(
                ee.ImageCollection(
                    ridge25_production.ERA5_COLLECTION_ID
                ).first()
            )
            .select("temperature_2m")
            .projection()
        )

        values = support_meteorology.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=support_point,
            crs=era5_projection,
            scale=era5_projection.nominalScale(),
            maxPixels=100,
        )

        fallback = (
            ee.Image.constant(
                ee.List(
                    [
                        values.get(name)
                        for name in RIDGE25_METEOROLOGICAL_FEATURES
                    ]
                )
            )
            .rename(RIDGE25_METEOROLOGICAL_FEATURES)
            .toFloat()
        )

        # Native values have precedence; only masked cells are filled.
        return (
            native
            .unmask(fallback)
            .clip(processing_geometry)
            .toFloat()
        )

    # Runtime-only methodological test. No repository source is changed.
    ridge25_production.build_ridge25_meteorological_predictors = (
        station_consistent_meteorology
    )

    print()
    print("Rebuilding Ridge-25 OOF resources locally...")
    result = module.train_and_validate_ridge25(master)
    station_to_block, fold_resources = module.build_fold_resources(result)

    _, valid_daily = module.prepare_field_daily(
        field,
        reference,
        metadata,
    )
    candidates = module.aggregate_field_periods(
        valid_daily,
        master,
        metadata,
    )
    candidates["station_id"] = candidates["station_id"].astype(str)
    candidates["period_start"] = pd.to_datetime(
        candidates["period_start"]
    )

    population = result.population.copy()
    population["station_id"] = population["station_id"].astype(str)
    population["period_start"] = pd.to_datetime(
        population["period_start"]
    )

    eligible_dates = set(
        population.loc[
            population["station_id"].eq("ST04"),
            "period_start",
        ]
    )

    work = candidates.loc[
        candidates["station_id"].eq("ST04")
        & candidates["period_start"].isin(eligible_dates)
    ].copy()

    block = station_to_block["ST04"]
    fold_resource = fold_resources[block]

    x, y = module.station_xy(work.iloc[0])
    center_tile = build_external_center_tile(
        module,
        root,
        x,
        y,
    )

    output_directory = (
        workspace.diagnostics
        / "field_ridge25_oof_exact_overlap_external_st04_era5_nearest"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 110)
    print("ST04 FINE DIAGNOSTIC WITH NEAREST-VALID ERA5 FALLBACK")
    print("=" * 110)
    print("Eligible periods:", len(work))
    print("Spatial block:", block)
    print("Fold AOA threshold:", f"{fold_resource['aoa'].threshold:.6f}")
    print("Center tile:", center_tile.tile_id)
    print("Fallback rule: native ERA5 first; ST04 training support only where masked")

    rows = []

    for i, (_, row) in enumerate(work.iterrows(), start=1):
        date_text = row["period_start"].strftime("%Y-%m-%d")
        print(f"[{i}/{len(work)}] ST04 {date_text}")

        output = {
            "period_start": date_text,
            "n_valid_field_days": int(row["n_valid_field_days"]),
            "ET_MODIS_mm_period": float(row["ET_MODIS_mm_period"]),
            "era5_sampling_longitude": support_lon,
            "era5_sampling_latitude": support_lat,
            "era5_sampling_distance_m": support_distance,
        }

        try:
            data, local_metadata = module.build_local_exact_product(
                output_directory=output_directory,
                date_text=date_text,
                block=block,
                fold_resource=fold_resource,
                center_tile=center_tile,
                timeout_seconds=600,
            )
        except RuntimeError as error:
            status = module.expected_nonpublication_status(error)
            output["status"] = status or "unexpected_error"
            output["error"] = str(error)
            rows.append(output)
            continue

        sample = module.sample_product(
            data,
            center_tile,
            x,
            y,
        )
        output.update(sample)

        et = sample["ET_mm_period"]
        if np.isfinite(et):
            status = "published"
        elif (
            not np.isfinite(sample["stack_valid"])
            or sample["stack_valid"] < 0.5
        ):
            status = "stack_invalid"
        elif (
            not np.isfinite(sample["AOA_inside"])
            or sample["AOA_inside"] < 0.5
        ):
            status = "outside_AOA_fine"
        elif np.isfinite(sample["Kc_raw"]) and sample["Kc_raw"] < 0:
            status = "negative_Kc"
        elif (
            np.isfinite(sample["usable_fraction"])
            and sample["usable_fraction"] < 0.90
        ):
            status = "support_below_90"
        elif (
            np.isfinite(sample["coarse_eligible"])
            and sample["coarse_eligible"] < 0.5
        ):
            status = "modis_parent_not_eligible"
        else:
            status = "not_published_other"

        output["status"] = status
        rows.append(output)

    result_table = pd.DataFrame(rows)

    columns = [
        "period_start",
        "n_valid_field_days",
        "ET_MODIS_mm_period",
        "ET_mm_period",
        "Kc_raw",
        "dissimilarity_index",
        "stack_valid",
        "AOA_inside",
        "usable",
        "usable_fraction",
        "coarse_eligible",
        "ET_conservation_error_mm",
        "status",
    ]

    for column in columns:
        if column not in result_table:
            result_table[column] = np.nan

    print()
    print("=" * 150)
    print("ST04 MANGROVE — CORRECTED FINE 20 m OOF DIAGNOSTIC")
    print("=" * 150)
    print(
        result_table[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("STATUS COUNTS")
    print(
        result_table["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    output_path = (
        output_directory
        / "st04_external_fine_nearest_era5_diagnostic.csv"
    )
    result_table.to_csv(output_path, index=False)

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
