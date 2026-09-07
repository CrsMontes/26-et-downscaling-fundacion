"""Compare field-site fine ET with and without the AOA mask, holding all other rules fixed.

This diagnostic reuses the already-downloaded raw fine OOF tiles. It does NOT
re-query Sentinel-2 or ERA5-Land. It downloads only native MODIS ET needed to
repeat exact-overlap reconciliation.

Two scenarios are solved from the same raw Ridge fields:
  WITH_AOA:
      usable = stack_valid & AOA_inside & finite(Kc_raw) & Kc_raw >= 0
  WITHOUT_AOA:
      usable = stack_valid & finite(Kc_raw) & Kc_raw >= 0

The same support90 threshold, exact-overlap geometry, nonnegative handling,
MODIS values, and conservation tolerance are used in both scenarios.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from rasterio.transform import rowcol

from et_downscaling.config import ANALYSIS_CRS
from et_downscaling.local_tiles import Tile, build_initial_tiles
from et_downscaling.overlap_reconciliation import (
    build_overlap_edges,
    materialize_active_values,
    solve_overlap_reconciliation,
)
from et_downscaling.production import PREDICTION_SCALE_M
from et_downscaling.ridge25_overlap_production import (
    RAW_TILE_BANDS,
    _download_native_modis,
    _fine_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    return parser.parse_args()


def load_field_module(root: Path):
    path = root / "scripts" / "evaluate_field_ridge25.py"
    spec = importlib.util.spec_from_file_location("field_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def external_center_tile(module, root: Path, x: float, y: float) -> Tile:
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


def load_cached_raw_support(module, directory: Path, block: str, date_text: str, center_tile: Tile):
    tiles, bounds = module.support_tiles_around(center_tile)
    completed = []

    for tile in tiles:
        path = module.raw_tile_cache_path(
            directory,
            block,
            date_text,
            tile.tile_id,
        )
        if not path.is_file():
            return None, None, f"missing_raw_cache:{tile.tile_id}"

        payload = np.load(path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("evaluation_version") != module.FIELD_EVALUATION_VERSION:
            return None, None, f"raw_cache_version_mismatch:{tile.tile_id}"

        data = np.asarray(payload["data"], dtype=np.float64)
        completed.append((tile, data))

    raw, transform = module.mosaic_raw_tiles(completed, bounds)
    return raw, transform, None


def sample_index(transform, shape, x: float, y: float):
    row, col = rowcol(transform, x, y)
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        raise RuntimeError("Station sample falls outside the support mosaic.")
    return int(row), int(col)


def solve_scenario(
    module,
    kc_raw,
    usable,
    edges,
    modis_et,
    fine_shape,
    station_row,
    station_col,
):
    try:
        result = solve_overlap_reconciliation(
            kc_raw=kc_raw,
            usable=usable,
            modis_et=modis_et,
            edges=edges,
            usable_support_fraction=module.RIDGE25_USABLE_SUPPORT_FRACTION,
            tolerance_mm=module.RIDGE25_RECONCILIATION_TOLERANCE_MM,
        )
    except RuntimeError as exc:
        return {
            "ET": np.nan,
            "usable_fraction": np.nan,
            "coarse_eligible": np.nan,
            "conservation_error": np.nan,
            "status": module.expected_nonpublication_status(exc) or "solver_error",
            "error": str(exc),
        }

    et = materialize_active_values(
        fine_shape=fine_shape,
        active_fine=result.active_fine,
        values=result.et_final_nonnegative,
        selected_active=result.publishable_active,
    )

    support, eligible, error = _fine_diagnostics(
        edges=edges,
        result=result,
        fine_size=kc_raw.size,
    )

    support = support.reshape(fine_shape)
    eligible = eligible.reshape(fine_shape)
    error = error.reshape(fine_shape)

    et_value = float(et[station_row, station_col])
    support_value = float(support[station_row, station_col])
    eligible_value = float(eligible[station_row, station_col])
    error_value = float(error[station_row, station_col])

    if np.isfinite(et_value):
        status = "valid"
    elif np.isfinite(support_value) and support_value < module.RIDGE25_USABLE_SUPPORT_FRACTION:
        status = "support_below_90"
    elif np.isfinite(eligible_value) and eligible_value < 0.5:
        status = "modis_parent_not_eligible"
    else:
        status = "not_published_other"

    return {
        "ET": et_value if np.isfinite(et_value) else np.nan,
        "usable_fraction": support_value if np.isfinite(support_value) else np.nan,
        "coarse_eligible": eligible_value if np.isfinite(eligible_value) else np.nan,
        "conservation_error": error_value if np.isfinite(error_value) else np.nan,
        "status": status,
        "error": "",
    }


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

    pairs_path = (
        workspace.diagnostics
        / "field_ridge25_oof_exact_overlap"
        / "field_ridge25_oof_pairs.csv"
    )
    if not pairs_path.is_file():
        raise FileNotFoundError(pairs_path)

    pairs = pd.read_csv(pairs_path)
    pairs["station_id"] = pairs["station_id"].astype(str)
    pairs["period_start"] = pd.to_datetime(pairs["period_start"])

    metadata = metadata.copy()
    metadata["station_id"] = metadata["station_id"].astype(str)

    # Rebuild only station->fold mapping locally; raw OOF predictions themselves
    # are read from the existing cache.
    model_result = module.train_and_validate_ridge25(master)
    station_to_block, _ = module.build_fold_resources(model_result)

    basin_tiles, _ = build_initial_tiles(
        root,
        tile_size_m=module.LOCAL_TILE_SIZE_M,
    )

    normal_dir = (
        workspace.diagnostics
        / "field_ridge25_oof_exact_overlap"
    )
    st04_dir = (
        workspace.diagnostics
        / "field_ridge25_oof_exact_overlap_external_st04_era5_nearest"
    )

    print()
    print("Initializing Earth Engine for native MODIS ET only...")
    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()

    rows = []

    # ST01, ST02, ST03, ST05: all 48 original in-basin candidates.
    work = pairs.loc[
        pairs["station_id"].isin(["ST01", "ST02", "ST03", "ST05"])
    ].copy()

    # ST04: add the 9 periods for which the corrected nearest-valid ERA5 raw
    # fine cache was actually built.
    st04_dates = []
    st04_block = station_to_block["ST04"]
    st04_block_dir = st04_dir / "exact_overlap_raw_tile_cache" / module._safe_block(st04_block)
    if st04_block_dir.is_dir():
        st04_dates = sorted(
            pd.Timestamp(item.name)
            for item in st04_block_dir.iterdir()
            if item.is_dir()
        )

    st04_source = pairs.loc[
        pairs["station_id"].eq("ST04")
        & pairs["period_start"].isin(st04_dates)
    ].copy()

    # The main pairs file marks ST04 outside basin but still contains its field
    # candidate metadata. If for any reason it does not, rebuild candidates.
    if len(st04_source) == 0 and st04_dates:
        _, valid_daily = module.prepare_field_daily(field, reference, metadata)
        candidates = module.aggregate_field_periods(valid_daily, master, metadata)
        candidates["station_id"] = candidates["station_id"].astype(str)
        candidates["period_start"] = pd.to_datetime(candidates["period_start"])
        st04_source = candidates.loc[
            candidates["station_id"].eq("ST04")
            & candidates["period_start"].isin(st04_dates)
        ].copy()

    work = pd.concat([work, st04_source], ignore_index=True, sort=False)
    work = work.sort_values(["station_id", "period_start"]).reset_index(drop=True)

    total = len(work)
    print(f"Rows with raw fine cache to evaluate: {total}")
    print("No Sentinel-2 or ERA5-Land predictor re-download will be performed.")
    print()

    for i, row in work.iterrows():
        station_id = str(row["station_id"])
        date = pd.Timestamp(row["period_start"])
        date_text = date.strftime("%Y-%m-%d")
        block = station_to_block[station_id]

        station_meta = metadata.loc[
            metadata["station_id"].eq(station_id)
        ].iloc[0]
        x, y = module.station_xy(station_meta)

        if station_id == "ST04":
            center_tile = external_center_tile(module, root, x, y)
            cache_dir = st04_dir
            evaluation_domain = "external_outside_basin"
        else:
            center_tile = module.find_station_tile(basin_tiles, x, y)
            cache_dir = normal_dir
            evaluation_domain = "in_basin"

        print(
            f"[{i + 1}/{total}] {station_id} {date_text} "
            f"domain={evaluation_domain}"
        )

        output = {
            "station_id": station_id,
            "station": row.get("station", station_meta["station"]),
            "period_start": date_text,
            "evaluation_domain": evaluation_domain,
            "n_valid_field_days": row.get("n_valid_field_days", np.nan),
            "ET_MODIS_mm_period": row.get("ET_MODIS_mm_period", np.nan),
        }

        raw, support_transform, cache_error = load_cached_raw_support(
            module,
            cache_dir,
            block,
            date_text,
            center_tile,
        )
        if cache_error is not None:
            output.update(
                {
                    "stack_valid": np.nan,
                    "AOA_inside": np.nan,
                    "dissimilarity_index": np.nan,
                    "Kc_raw": np.nan,
                    "ET_with_AOA_mm_period": np.nan,
                    "ET_without_AOA_mm_period": np.nan,
                    "status_with_AOA": cache_error,
                    "status_without_AOA": cache_error,
                    "recovered_by_removing_AOA": False,
                }
            )
            rows.append(output)
            continue

        fine_shape = raw.shape[1:]
        kc_raw = raw[RAW_TILE_BANDS.index("Kc_raw")]
        di = raw[RAW_TILE_BANDS.index("dissimilarity_index")]
        stack_valid = raw[RAW_TILE_BANDS.index("stack_valid")] > 0.5
        aoa_inside = raw[RAW_TILE_BANDS.index("AOA_inside")] > 0.5
        usable_with_aoa = raw[RAW_TILE_BANDS.index("usable")] > 0.5
        domain = raw[RAW_TILE_BANDS.index("support_domain")] > 0.5

        # This is the ONLY methodological difference in the no-AOA scenario.
        usable_without_aoa = (
            stack_valid
            & np.isfinite(kc_raw)
            & (kc_raw >= 0)
        )

        station_row, station_col = sample_index(
            support_transform,
            fine_shape,
            x,
            y,
        )

        output.update(
            {
                "stack_valid": float(stack_valid[station_row, station_col]),
                "AOA_inside": float(aoa_inside[station_row, station_col]),
                "dissimilarity_index": float(di[station_row, station_col])
                if np.isfinite(di[station_row, station_col])
                else np.nan,
                "Kc_raw": float(kc_raw[station_row, station_col])
                if np.isfinite(kc_raw[station_row, station_col])
                else np.nan,
            }
        )

        # If the station stack itself is invalid, neither scenario can produce
        # a meaningful fine prediction. No MODIS query is needed.
        if not stack_valid[station_row, station_col]:
            output.update(
                {
                    "ET_with_AOA_mm_period": np.nan,
                    "ET_without_AOA_mm_period": np.nan,
                    "support_with_AOA": 0.0,
                    "support_without_AOA": 0.0,
                    "status_with_AOA": "stack_invalid",
                    "status_without_AOA": "stack_invalid",
                    "recovered_by_removing_AOA": False,
                }
            )
            rows.append(output)
            continue

        support_tiles, support_bounds = module.support_tiles_around(center_tile)

        modis_et, modis_grid = _download_native_modis(
            period_start=date_text,
            support_bounds=support_bounds,
            timeout_seconds=600,
        )

        edges = build_overlap_edges(
            domain=domain,
            fine_transform=support_transform,
            fine_crs=ANALYSIS_CRS,
            modis_et=modis_et,
            modis_transform=modis_grid.transform,
            modis_crs=modis_grid.local_crs,
            progress_every=0,
        )

        with_aoa = solve_scenario(
            module,
            kc_raw,
            usable_with_aoa,
            edges,
            modis_et,
            fine_shape,
            station_row,
            station_col,
        )
        without_aoa = solve_scenario(
            module,
            kc_raw,
            usable_without_aoa,
            edges,
            modis_et,
            fine_shape,
            station_row,
            station_col,
        )

        output.update(
            {
                "ET_with_AOA_mm_period": with_aoa["ET"],
                "ET_without_AOA_mm_period": without_aoa["ET"],
                "support_with_AOA": with_aoa["usable_fraction"],
                "support_without_AOA": without_aoa["usable_fraction"],
                "coarse_eligible_with_AOA": with_aoa["coarse_eligible"],
                "coarse_eligible_without_AOA": without_aoa["coarse_eligible"],
                "conservation_error_with_AOA_mm": with_aoa["conservation_error"],
                "conservation_error_without_AOA_mm": without_aoa["conservation_error"],
                "status_with_AOA": with_aoa["status"],
                "status_without_AOA": without_aoa["status"],
                "recovered_by_removing_AOA": (
                    with_aoa["status"] != "valid"
                    and without_aoa["status"] == "valid"
                ),
            }
        )
        rows.append(output)

    result = pd.DataFrame(rows)

    out_dir = (
        workspace.diagnostics
        / "field_ridge25_aoa_sensitivity"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "field_ridge25_with_vs_without_aoa.csv"
    result.to_csv(out_csv, index=False)

    print()
    print("=" * 120)
    print("AOA-ONLY SENSITIVITY COMPLETE")
    print("=" * 120)

    print()
    print("WITH AOA STATUS")
    print(result["status_with_AOA"].value_counts(dropna=False).to_string())

    print()
    print("WITHOUT AOA STATUS")
    print(result["status_without_AOA"].value_counts(dropna=False).to_string())

    print()
    print("RECOVERED ONLY BY REMOVING AOA")
    recovered = result.loc[result["recovered_by_removing_AOA"]].copy()
    print("n =", len(recovered))
    if len(recovered):
        print(
            recovered[
                [
                    "station_id",
                    "period_start",
                    "evaluation_domain",
                    "AOA_inside",
                    "dissimilarity_index",
                    "support_with_AOA",
                    "support_without_AOA",
                    "ET_without_AOA_mm_period",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    both = result.loc[
        result["status_with_AOA"].eq("valid")
        & result["status_without_AOA"].eq("valid")
    ].copy()

    print()
    print("VALID IN BOTH SCENARIOS")
    print("n =", len(both))

    print()
    print("Saved:", out_csv)
    print()
    print(
        "This table is ready for the next step: construct the final field ET "
        "proxy and make the three-ET figures with and without AOA."
    )


if __name__ == "__main__":
    main()
