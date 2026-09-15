"""Diagnose RF25 support eligibility around ST01."""

from __future__ import annotations

import sys
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from rasterio.transform import Affine

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

import produce_field_rf25_halo as base

from et_downscaling.overlap_reconciliation import build_overlap_edges


DATE = "2022-03-30"
STATION_ID = "ST01"
REFERENCE_RADIUS = 4
RADII = [2, 3, 4]


def weighted_fraction(mask, coarse_edge, fine_edge, area, count):
    total = np.bincount(
        coarse_edge,
        weights=area,
        minlength=count,
    )

    selected = np.bincount(
        coarse_edge,
        weights=area * mask.ravel()[fine_edge].astype(float),
        minlength=count,
    )

    fraction = np.full(count, np.nan, dtype=float)
    valid = total > 0
    fraction[valid] = selected[valid] / total[valid]

    return fraction


def main():
    station = base.load_stations(ROOT)[STATION_ID]

    raw_dir = (
        ROOT
        / "outputs"
        / "evaluation"
        / "field_validation"
        / "st01_validation_extension"
        / "halo_convergence"
        / DATE
        / "raw"
    )

    raw_paths = sorted(raw_dir.glob("*.tif"))

    if len(raw_paths) != 1:
        raise RuntimeError(
            f"Expected one existing ST01 raw tile, found {len(raw_paths)}."
        )

    raw_path = raw_paths[0]

    print("Using existing raw RF25 tile:")
    print(raw_path)

    raw, fine_transform, fine_crs = base.read_raw_tile(raw_path)

    kc_raw = raw[
        base.RAW_TILE_BANDS.index("Kc_raw")
    ]

    stack_valid = (
        raw[
            base.RAW_TILE_BANDS.index("stack_valid")
        ]
        > 0.5
    )

    aoa_inside = (
        raw[
            base.RAW_TILE_BANDS.index("AOA_inside")
        ]
        > 0.5
    )

    usable = (
        raw[
            base.RAW_TILE_BANDS.index("usable")
        ]
        > 0.5
    )

    domain = (
        raw[
            base.RAW_TILE_BANDS.index("support_domain")
        ]
        > 0.5
    )

    kc_finite = np.isfinite(kc_raw)

    ee.Initialize(project="ee-sneiderquintero")
    ee.Number(1).getInfo()

    point = ee.Geometry.Point(
        [
            station["longitude"],
            station["latitude"],
        ]
    )

    context = base.build_modis_period_context(
        DATE,
        point.buffer(5000),
    )

    projection_info = (
        context["modis_projection"].getInfo()
    )

    (
        parent_row,
        parent_col,
        native_transform,
    ) = base.native_parent(
        projection_info,
        station["longitude"],
        station["latitude"],
    )

    base.HALO_RADIUS = REFERENCE_RADIUS
    base.HALO_SIZE = 2 * REFERENCE_RADIUS + 1

    transform9 = base.halo_transform(
        native_transform,
        parent_row,
        parent_col,
    )

    modis9 = base.download_modis_halo(
        context=context,
        projection_info=projection_info,
        target_transform=transform9,
        timeout_seconds=600,
    )

    rows = []

    for radius in RADII:
        size = 2 * radius + 1
        offset = REFERENCE_RADIUS - radius

        modis = modis9[
            offset:offset + size,
            offset:offset + size,
        ]

        transform = (
            transform9
            * Affine.translation(offset, offset)
        )

        edges = build_overlap_edges(
            domain=domain,
            fine_transform=fine_transform,
            fine_crs=fine_crs,
            modis_et=modis,
            modis_transform=transform,
            modis_crs=base.MODIS_SINUSOIDAL_LOCAL_CRS,
            progress_every=0,
        )

        coarse_edge = np.asarray(
            edges.coarse_index,
            dtype=np.int64,
        )
        fine_edge = np.asarray(
            edges.fine_index,
            dtype=np.int64,
        )
        area = np.asarray(
            edges.overlap_area_m2,
            dtype=float,
        )

        count = modis.size

        stack_fraction = weighted_fraction(
            stack_valid,
            coarse_edge,
            fine_edge,
            area,
            count,
        )

        aoa_fraction = weighted_fraction(
            aoa_inside,
            coarse_edge,
            fine_edge,
            area,
            count,
        )

        usable_fraction = weighted_fraction(
            usable,
            coarse_edge,
            fine_edge,
            area,
            count,
        )

        kc_fraction = weighted_fraction(
            kc_finite,
            coarse_edge,
            fine_edge,
            area,
            count,
        )

        coarse_area = np.bincount(
            coarse_edge,
            weights=area,
            minlength=count,
        )

        usable_area = np.bincount(
            coarse_edge,
            weights=(
                area
                * usable.ravel()[fine_edge].astype(float)
            ),
            minlength=count,
        )

        valid_kc_edge = (
            usable.ravel()[fine_edge]
            & np.isfinite(kc_raw.ravel()[fine_edge])
        )

        kc_sum = np.bincount(
            coarse_edge[valid_kc_edge],
            weights=(
                area[valid_kc_edge]
                * kc_raw.ravel()[
                    fine_edge[valid_kc_edge]
                ]
            ),
            minlength=count,
        )

        kc_mean = np.full(
            count,
            np.nan,
            dtype=float,
        )

        positive = usable_area > 0

        kc_mean[positive] = (
            kc_sum[positive]
            / usable_area[positive]
        )

        represented = (
            np.asarray(
                edges.represented_coarse,
                dtype=bool,
            ).ravel()
        )

        finite_modis = np.isfinite(
            modis.ravel()
        )

        eligible = (
            represented
            & finite_modis
            & np.isfinite(kc_mean)
            & (usable_fraction >= 0.90)
        )

        center = radius
        center_flat = center * size + center

        row = {
            "halo_size": size,
            "parents_total": count,
            "parents_represented": int(
                represented.sum()
            ),
            "parents_modis_finite": int(
                finite_modis.sum()
            ),
            "parents_support_ge90": int(
                np.sum(
                    np.isfinite(usable_fraction)
                    & (usable_fraction >= 0.90)
                )
            ),
            "parents_eligible": int(
                eligible.sum()
            ),
            "central_modis_et_mm_period": (
                float(modis[center, center])
                if np.isfinite(modis[center, center])
                else np.nan
            ),
            "central_represented": bool(
                represented[center_flat]
            ),
            "central_stack_valid_fraction": float(
                stack_fraction[center_flat]
            ),
            "central_aoa_inside_fraction": float(
                aoa_fraction[center_flat]
            ),
            "central_kc_finite_fraction": float(
                kc_fraction[center_flat]
            ),
            "central_usable_fraction": float(
                usable_fraction[center_flat]
            ),
            "central_kc_valid_mean": float(
                kc_mean[center_flat]
            ),
            "central_eligible": bool(
                eligible[center_flat]
            ),
        }

        rows.append(row)

    result = pd.DataFrame(rows)

    print()
    print("=" * 90)
    print("ST01 SUPPORT DIAGNOSTIC")
    print("=" * 90)
    print(result.to_string(index=False))

    output = (
        ROOT
        / "outputs"
        / "evaluation"
        / "field_validation"
        / "st01_validation_extension"
        / "halo_convergence"
        / DATE
        / "st01_support_diagnostic.csv"
    )

    result.to_csv(output, index=False)

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()
