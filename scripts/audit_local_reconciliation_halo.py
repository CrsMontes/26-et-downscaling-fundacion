from pathlib import Path
import os

os.environ.setdefault("PROJ_NETWORK", "OFF")

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from rasterio.warp import transform_bounds

from et_downscaling.overlap_reconciliation import (
    build_overlap_edges,
    solve_overlap_reconciliation,
)


DATE = os.environ.get("AUDIT_DATE", "2022-03-30")
HALO_RADIUS = int(os.environ.get('HALO_RADIUS', '1'))
SUPPORT_FRACTION = 0.90
TOLERANCE_MM = 0.01
NODATA = -9999.0

STATIONS = ["ST01", "ST02", "ST03", "ST05"]


def scatter_fine(values, active_fine, shape, fill_value=np.nan, dtype=float):
    values = np.asarray(values)

    if values.shape == shape:
        return values.astype(dtype, copy=False)

    if values.size == shape[0] * shape[1]:
        return values.reshape(shape).astype(dtype, copy=False)

    active_fine = np.asarray(active_fine, dtype=np.int64).ravel()

    if values.size != active_fine.size:
        raise ValueError(
            f"Fine result has {values.size} values but "
            f"active_fine has {active_fine.size} indices."
        )

    output = np.full(
        shape[0] * shape[1],
        fill_value,
        dtype=dtype,
    )
    output[active_fine] = values.ravel()

    return output.reshape(shape)


def scatter_coarse(values, eligible_coarse, shape, fill_value=np.nan):
    values = np.asarray(values)

    if values.shape == shape:
        return values.astype(float, copy=False)

    if values.size == shape[0] * shape[1]:
        return values.reshape(shape).astype(float, copy=False)

    eligible_coarse = np.asarray(
        eligible_coarse,
        dtype=np.int64,
    ).ravel()

    if values.size != eligible_coarse.size:
        raise ValueError(
            f"Coarse result has {values.size} values but "
            f"eligible_coarse has {eligible_coarse.size} indices."
        )

    output = np.full(
        shape[0] * shape[1],
        fill_value,
        dtype=float,
    )
    output[eligible_coarse] = values.ravel()

    return output.reshape(shape)


def clipped_window(window, height, width):
    row0 = max(0, int(window.row_off))
    col0 = max(0, int(window.col_off))
    row1 = min(height, int(window.row_off + window.height))
    col1 = min(width, int(window.col_off + window.width))

    return Window(
        col_off=col0,
        row_off=row0,
        width=col1 - col0,
        height=row1 - row0,
    )


def valid_float(array, nodata):
    out = array.astype(np.float64, copy=True)
    if nodata is not None:
        out[out == nodata] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def main():
    root = Path.cwd()

    production_root = (
        root
        / "outputs"
        / "evaluation"
        / "field_validation"
        / "production"
        / "578979fa05e81e66"
    )

    date_root = production_root / "rasters" / DATE
    modis_root = production_root / "rasters_modis" / DATE

    raw_path = next(
        date_root.glob(f"raw_support_*_{DATE}_20m.tif")
    )
    final_path = next(
        date_root.glob(f"ET_*_{DATE}_20m.tif")
    )
    modis_path = modis_root / f"MODIS_ET_{DATE}_native.tif"

    station_audit_path = (
        root
        / "outputs"
        / "evaluation"
        / "field_validation"
        / "station_modis_parent_audit_2022-03-30.csv"
    )

    output_dir = (
        root
        / "outputs"
        / "evaluation"
        / "field_validation"
        / "local_halo_audit"
        / DATE
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(station_audit_path)
    stations = stations[
        stations["station_id"].isin(STATIONS)
    ].copy()

    summaries = []
    pixel_records = []

    with (
        rasterio.open(raw_path) as raw_src,
        rasterio.open(final_path) as final_src,
        rasterio.open(modis_path) as modis_src,
    ):
        final_et = valid_float(
            final_src.read(1),
            final_src.nodata,
        )

        for station_id in STATIONS:
            station = stations.loc[
                stations["station_id"] == station_id
            ].iloc[0]

            parent_row = int(station["parent_row"])
            parent_col = int(station["parent_col"])

            print()
            print("=" * 78)
            print(
                f"{station_id}: parent "
                f"r{parent_row}_c{parent_col}"
            )
            print("=" * 78)

            modis_window = Window(
                parent_col - HALO_RADIUS,
                parent_row - HALO_RADIUS,
                2 * HALO_RADIUS + 1,
                2 * HALO_RADIUS + 1,
            )

            if (
                modis_window.row_off < 0
                or modis_window.col_off < 0
                or modis_window.row_off + modis_window.height
                > modis_src.height
                or modis_window.col_off + modis_window.width
                > modis_src.width
            ):
                raise RuntimeError(
                    f"{station_id}: 3x3 MODIS window falls "
                    "outside the stored MODIS raster."
                )

            modis_et = valid_float(
                modis_src.read(1, window=modis_window),
                modis_src.nodata,
            )
            modis_transform = modis_src.window_transform(
                modis_window
            )

            expected_size = 2 * HALO_RADIUS + 1
            if modis_et.shape != (expected_size, expected_size):
                raise RuntimeError(
                    f"{station_id}: expected "
                    f"{expected_size}x{expected_size} MODIS array, "
                    f"got {modis_et.shape}."
                )

            # Project the full 3x3 MODIS footprint to the 20 m grid CRS.
            left, bottom, right, top = window_bounds(
                modis_window,
                modis_src.transform,
            )

            fine_left, fine_bottom, fine_right, fine_top = (
                transform_bounds(
                    modis_src.crs,
                    raw_src.crs,
                    left,
                    bottom,
                    right,
                    top,
                    densify_pts=41,
                )
            )

            # Small spatial padding only ensures that boundary 20 m
            # pixels intersecting the transformed MODIS footprint
            # are present in the local array.
            padding_m = 40.0

            fine_window = from_bounds(
                fine_left - padding_m,
                fine_bottom - padding_m,
                fine_right + padding_m,
                fine_top + padding_m,
                transform=raw_src.transform,
            )

            fine_window = Window(
                int(np.floor(fine_window.col_off)),
                int(np.floor(fine_window.row_off)),
                int(np.ceil(fine_window.width)),
                int(np.ceil(fine_window.height)),
            )

            fine_window = clipped_window(
                fine_window,
                raw_src.height,
                raw_src.width,
            )

            raw = raw_src.read(window=fine_window)
            fine_transform = raw_src.window_transform(
                fine_window
            )

            kc_raw = valid_float(
                raw[0],
                raw_src.nodata,
            )

            usable_raw = valid_float(
                raw[5],
                raw_src.nodata,
            )

            support_raw = valid_float(
                raw[6],
                raw_src.nodata,
            )

            usable = (
                np.isfinite(usable_raw)
                & (usable_raw >= 0.5)
            )

            support_domain = (
                np.isfinite(support_raw)
                & (support_raw >= 0.5)
            )

            # The local reconciliation receives exactly the same
            # raw RF25 quantities as global production. Only its
            # reconciliation domain is restricted to the 3x3 MODIS
            # neighborhood.
            edges = build_overlap_edges(
                domain=support_domain,
                fine_transform=fine_transform,
                fine_crs=raw_src.crs,
                modis_et=modis_et,
                modis_transform=modis_transform,
                modis_crs=modis_src.crs,
                progress_every=0,
            )

            result = solve_overlap_reconciliation(
                kc_raw=kc_raw,
                usable=usable,
                modis_et=modis_et,
                edges=edges,
                usable_support_fraction=SUPPORT_FRACTION,
                tolerance_mm=TOLERANCE_MM,
            )

            local_shape = kc_raw.shape

            et_final = scatter_fine(
                result.et_final,
                result.active_fine,
                local_shape,
            )

            et_nonnegative = scatter_fine(
                result.et_final_nonnegative,
                result.active_fine,
                local_shape,
            )

            publishable = scatter_fine(
                result.publishable_active,
                result.active_fine,
                local_shape,
                fill_value=False,
                dtype=bool,
            )

            # Identify every fine-grid pixel that geometrically
            # overlaps the central MODIS parent.
            center = HALO_RADIUS
            central_modis = modis_et[
                center:center + 1,
                center:center + 1,
            ]

            central_window = Window(
                parent_col,
                parent_row,
                1,
                1,
            )

            central_transform = modis_src.window_transform(
                central_window
            )

            central_edges = build_overlap_edges(
                domain=support_domain,
                fine_transform=fine_transform,
                fine_crs=raw_src.crs,
                modis_et=central_modis,
                modis_transform=central_transform,
                modis_crs=modis_src.crs,
                progress_every=0,
            )

            central_indices = np.unique(
                central_edges.fine_index
            )

            local_rows, local_cols = np.unravel_index(
                central_indices,
                local_shape,
            )

            local_values = et_nonnegative[
                local_rows,
                local_cols,
            ]
            local_values_unclipped = et_final[
                local_rows,
                local_cols,
            ]
            local_publishable = publishable[
                local_rows,
                local_cols,
            ]

            xs, ys = rasterio.transform.xy(
                fine_transform,
                local_rows,
                local_cols,
                offset="center",
            )

            global_values = np.full(
                len(central_indices),
                np.nan,
                dtype=float,
            )

            global_inside = np.zeros(
                len(central_indices),
                dtype=bool,
            )

            global_rows = np.full(
                len(central_indices),
                -1,
                dtype=int,
            )
            global_cols = np.full(
                len(central_indices),
                -1,
                dtype=int,
            )

            for index, (x, y) in enumerate(zip(xs, ys)):
                row, col = final_src.index(x, y)

                if (
                    0 <= row < final_src.height
                    and 0 <= col < final_src.width
                ):
                    global_inside[index] = True
                    global_rows[index] = row
                    global_cols[index] = col
                    global_values[index] = final_et[row, col]

            global_publishable = np.isfinite(
                global_values
            )

            local_published_values = np.where(
                local_publishable,
                local_values,
                np.nan,
            )

            common = (
                np.isfinite(local_published_values)
                & np.isfinite(global_values)
            )

            local_only = (
                np.isfinite(local_published_values)
                & ~np.isfinite(global_values)
            )

            global_only = (
                ~np.isfinite(local_published_values)
                & np.isfinite(global_values)
            )

            if common.any():
                difference = (
                    local_published_values[common]
                    - global_values[common]
                )

                max_abs_diff = float(
                    np.max(np.abs(difference))
                )
                mean_abs_diff = float(
                    np.mean(np.abs(difference))
                )
                rmse = float(
                    np.sqrt(np.mean(difference ** 2))
                )

                n_le_1e6 = int(
                    np.sum(np.abs(difference) <= 1e-6)
                )
                n_le_1e5 = int(
                    np.sum(np.abs(difference) <= 1e-5)
                )
                n_le_1e4 = int(
                    np.sum(np.abs(difference) <= 1e-4)
                )
                n_le_1e3 = int(
                    np.sum(np.abs(difference) <= 1e-3)
                )
            else:
                max_abs_diff = np.nan
                mean_abs_diff = np.nan
                rmse = np.nan
                n_le_1e6 = 0
                n_le_1e5 = 0
                n_le_1e4 = 0
                n_le_1e3 = 0

            eligible = np.asarray(
                result.eligible_coarse_mask
            ).reshape(modis_et.shape)

            final_error = scatter_coarse(
                result.final_error_after_nonnegative,
                result.eligible_coarse,
                modis_et.shape,
            )

            summary = {
                "station_id": station_id,
                "parent_row": parent_row,
                "parent_col": parent_col,
                "halo_radius": HALO_RADIUS,
                "modis_window_size": 2 * HALO_RADIUS + 1,
                "central_modis_et_mm_period": (
                    float(modis_et[center, center])
                    if np.isfinite(modis_et[center, center])
                    else np.nan
                ),
                "central_parent_eligible_local": bool(
                    eligible[center, center]
                ),
                "central_parent_local_conservation_error_mm": (
                    float(final_error[center, center])
                    if np.isfinite(final_error[center, center])
                    else np.nan
                ),
                "central_overlap_fine_pixels": int(
                    len(central_indices)
                ),
                "local_publishable_pixels": int(
                    np.sum(
                        np.isfinite(
                            local_published_values
                        )
                    )
                ),
                "global_publishable_pixels": int(
                    np.sum(global_publishable)
                ),
                "common_publishable_pixels": int(
                    np.sum(common)
                ),
                "local_only_pixels": int(
                    np.sum(local_only)
                ),
                "global_only_pixels": int(
                    np.sum(global_only)
                ),
                "max_abs_difference_mm": max_abs_diff,
                "mean_abs_difference_mm": mean_abs_diff,
                "rmse_difference_mm": rmse,
                "n_abs_diff_le_1e-6": n_le_1e6,
                "n_abs_diff_le_1e-5": n_le_1e5,
                "n_abs_diff_le_1e-4": n_le_1e4,
                "n_abs_diff_le_1e-3": n_le_1e3,
                "solver_max_abs_error_mm": float(
                    result.max_abs_final_error_mm
                ),
                "solver_max_abs_error_after_nonnegative_mm": float(
                    result.max_abs_error_after_nonnegative_mm
                ),
            }

            summaries.append(summary)

            for i in range(len(central_indices)):
                pixel_records.append(
                    {
                        "station_id": station_id,
                        "fine_x": float(xs[i]),
                        "fine_y": float(ys[i]),
                        "local_row": int(local_rows[i]),
                        "local_col": int(local_cols[i]),
                        "global_row": int(global_rows[i]),
                        "global_col": int(global_cols[i]),
                        "local_publishable": bool(
                            local_publishable[i]
                        ),
                        "local_et_mm_period": (
                            float(local_values[i])
                            if np.isfinite(local_values[i])
                            else np.nan
                        ),
                        "local_et_unclipped_mm_period": (
                            float(local_values_unclipped[i])
                            if np.isfinite(
                                local_values_unclipped[i]
                            )
                            else np.nan
                        ),
                        "global_et_mm_period": (
                            float(global_values[i])
                            if np.isfinite(global_values[i])
                            else np.nan
                        ),
                        "difference_mm": (
                            float(
                                local_published_values[i]
                                - global_values[i]
                            )
                            if (
                                np.isfinite(
                                    local_published_values[i]
                                )
                                and np.isfinite(
                                    global_values[i]
                                )
                            )
                            else np.nan
                        ),
                    }
                )

            print(
                f"central fine pixels: "
                f"{summary['central_overlap_fine_pixels']}"
            )
            print(
                f"common published: "
                f"{summary['common_publishable_pixels']}"
            )
            print(
                f"mask mismatch local/global: "
                f"{summary['local_only_pixels']} / "
                f"{summary['global_only_pixels']}"
            )
            print(
                f"max |local-global|: "
                f"{summary['max_abs_difference_mm']:.10f} mm"
            )
            print(
                f"mean |local-global|: "
                f"{summary['mean_abs_difference_mm']:.10f} mm"
            )
            print(
                f"RMSE difference: "
                f"{summary['rmse_difference_mm']:.10f} mm"
            )
            print(
                f"central conservation error: "
                f"{summary['central_parent_local_conservation_error_mm']:.12g} mm"
            )

    summary_df = pd.DataFrame(summaries)
    pixels_df = pd.DataFrame(pixel_records)

    summary_path = (
        output_dir
        / f"halo{2 * HALO_RADIUS + 1}_vs_global_summary_{DATE}.csv"
    )
    pixels_path = (
        output_dir
        / f"halo{2 * HALO_RADIUS + 1}_vs_global_pixels_{DATE}.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    pixels_df.to_csv(pixels_path, index=False)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    columns = [
        "station_id",
        "central_overlap_fine_pixels",
        "common_publishable_pixels",
        "local_only_pixels",
        "global_only_pixels",
        "max_abs_difference_mm",
        "mean_abs_difference_mm",
        "rmse_difference_mm",
        "central_parent_local_conservation_error_mm",
    ]

    print(
        summary_df[columns].to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(summary_path)
    print(pixels_path)


if __name__ == "__main__":
    main()
