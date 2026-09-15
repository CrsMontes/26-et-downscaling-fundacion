"""Audit ST01 local reconciliation convergence using one 9x9 download."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd
from rasterio.transform import Affine

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

import produce_field_rf25_halo as base

from et_downscaling.overlap_reconciliation import (
    build_overlap_edges,
    materialize_active_values,
    solve_overlap_reconciliation,
)


DATE = "2022-03-30"
STATION_ID = "ST01"
REFERENCE_RADIUS = 4  # 9x9
RADII = [2, 3, 4]     # 5x5, 7x7, 9x9


def central_parent_error(result, radius):
    center_flat = radius * (2 * radius + 1) + radius

    if not bool(result.eligible_coarse_mask[radius, radius]):
        return np.nan

    positions = np.flatnonzero(
        result.eligible_coarse == center_flat
    )

    if positions.size != 1:
        raise RuntimeError(
            "Central eligible parent mapping is ambiguous."
        )

    return float(
        result.final_error_after_nonnegative[
            positions[0]
        ]
    )


def solve_halo(
    radius,
    raw,
    fine_transform,
    fine_crs,
    modis_et,
    modis_transform,
):
    size = 2 * radius + 1

    base.HALO_RADIUS = radius
    base.HALO_SIZE = size

    kc_raw = raw[
        base.RAW_TILE_BANDS.index("Kc_raw")
    ]

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

    edges = build_overlap_edges(
        domain=domain,
        fine_transform=fine_transform,
        fine_crs=fine_crs,
        modis_et=modis_et,
        modis_transform=modis_transform,
        modis_crs=base.MODIS_SINUSOIDAL_LOCAL_CRS,
        progress_every=0,
    )

    finite_modis = np.isfinite(modis_et)

    missing = (
        finite_modis
        & ~edges.represented_coarse
    )

    if missing.any():
        raise RuntimeError(
            f"halo{size}: "
            f"{int(missing.sum())} finite MODIS parents "
            "are not fully represented by fine support."
        )

    result = solve_overlap_reconciliation(
        kc_raw=kc_raw,
        usable=usable,
        modis_et=modis_et,
        edges=edges,
        usable_support_fraction=(
            base.RF25_USABLE_SUPPORT_FRACTION
        ),
        tolerance_mm=(
            base.RF25_RECONCILIATION_TOLERANCE_MM
        ),
    )

    fine_shape = kc_raw.shape

    et_support = materialize_active_values(
        fine_shape=fine_shape,
        active_fine=result.active_fine,
        values=result.et_final_nonnegative,
    )

    et_published = materialize_active_values(
        fine_shape=fine_shape,
        active_fine=result.active_fine,
        values=result.et_final_nonnegative,
        selected_active=result.publishable_active,
    )

    central_mask = base.central_parent_mask(
        domain=domain,
        fine_transform=fine_transform,
        fine_crs=fine_crs,
        modis_transform=modis_transform,
    )

    center = radius

    summary = {
        "halo_size": size,
        "central_modis_et_mm_period": (
            float(modis_et[center, center])
            if np.isfinite(modis_et[center, center])
            else np.nan
        ),
        "central_parent_eligible": bool(
            result.eligible_coarse_mask[
                center,
                center,
            ]
        ),
        "central_parent_usable_fraction": float(
            result.usable_fraction[
                center,
                center,
            ]
        ),
        "central_parent_kc_valid_mean": float(
            result.kc_valid_mean[
                center,
                center,
            ]
        ),
        "central_conservation_error_mm": (
            central_parent_error(
                result,
                radius,
            )
        ),
        "central_overlap_pixels": int(
            central_mask.sum()
        ),
        "central_support_pixels": int(
            np.isfinite(
                et_support[central_mask]
            ).sum()
        ),
        "central_published_pixels": int(
            np.isfinite(
                et_published[central_mask]
            ).sum()
        ),
    }

    return {
        "summary": summary,
        "central_mask": central_mask,
        "et_support": et_support,
        "et_published": et_published,
    }


def compare(reference, candidate, label):
    mask = (
        reference["central_mask"]
        & candidate["central_mask"]
    )

    ref_values = reference[
        "et_published"
    ][mask]

    candidate_values = candidate[
        "et_published"
    ][mask]

    ref_valid = np.isfinite(ref_values)
    candidate_valid = np.isfinite(
        candidate_values
    )

    common = ref_valid & candidate_valid
    candidate_only = (
        candidate_valid & ~ref_valid
    )
    reference_only = (
        ref_valid & ~candidate_valid
    )

    row = {
        "comparison": label,
        "common_published_pixels": int(
            common.sum()
        ),
        "candidate_only_pixels": int(
            candidate_only.sum()
        ),
        "reference_only_pixels": int(
            reference_only.sum()
        ),
        "max_abs_difference_mm": np.nan,
        "mean_abs_difference_mm": np.nan,
        "rmse_difference_mm": np.nan,
    }

    if common.any():
        difference = (
            candidate_values[common]
            - ref_values[common]
        )

        row.update(
            {
                "max_abs_difference_mm": float(
                    np.max(
                        np.abs(difference)
                    )
                ),
                "mean_abs_difference_mm": float(
                    np.mean(
                        np.abs(difference)
                    )
                ),
                "rmse_difference_mm": float(
                    np.sqrt(
                        np.mean(
                            difference ** 2
                        )
                    )
                ),
            }
        )

    return row


def main():
    stations = base.load_stations(ROOT)
    station = stations[STATION_ID]

    if station["inside_basin"]:
        raise RuntimeError(
            "ST01 was expected to be outside "
            "the official basin domain."
        )

    workspace = (
        base.get_workspace_paths(ROOT)
        .ensure()
    )

    model = joblib.load(
        workspace.models
        / base.RF25_MODEL_FILENAME
    )
    aoa = joblib.load(
        workspace.models
        / base.RF25_AOA_FILENAME
    )

    base.validate_rf25_model(model)

    scientific_signature = (
        base.build_production_scientific_signature(
            model,
            aoa,
        )
    )

    ee.Initialize(
        project="ee-sneiderquintero"
    )
    ee.Number(1).getInfo()

    point = ee.Geometry.Point(
        [
            station["longitude"],
            station["latitude"],
        ]
    )

    context = (
        base.build_modis_period_context(
            DATE,
            point.buffer(5000),
        )
    )

    projection_info = (
        context["modis_projection"]
        .getInfo()
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

    print(
        "ST01 native parent:",
        f"r{parent_row}_c{parent_col}",
    )

    # Download only once using the largest halo.
    base.HALO_RADIUS = REFERENCE_RADIUS
    base.HALO_SIZE = (
        2 * REFERENCE_RADIUS + 1
    )

    transform9 = base.halo_transform(
        native_transform,
        parent_row,
        parent_col,
    )

    fine_tile = base.fine_tile_for_halo(
        "ST01_halo9_convergence",
        transform9,
    )

    output_root = (
        ROOT
        / "outputs"
        / "evaluation"
        / "field_validation"
        / "st01_validation_extension"
        / "halo_convergence"
        / DATE
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Downloading one 9x9 RF25 support:",
        f"{fine_tile.width_m:.0f} x "
        f"{fine_tile.height_m:.0f} m",
    )

    completed = base._download_raw_tile(
        period_start=DATE,
        model=model,
        aoa_parameters=aoa,
        tile=fine_tile,
        tile_directory=(
            output_root / "raw"
        ),
        timeout_seconds=600,
        scientific_signature=(
            scientific_signature
        ),
    )

    modis9 = base.download_modis_halo(
        context=context,
        projection_info=projection_info,
        target_transform=transform9,
        timeout_seconds=600,
    )

    raw, fine_transform, fine_crs = (
        base.read_raw_tile(
            completed.path
        )
    )

    results = {}
    summaries = []

    for radius in RADII:
        size = 2 * radius + 1
        offset = (
            REFERENCE_RADIUS - radius
        )

        modis = modis9[
            offset:offset + size,
            offset:offset + size,
        ]

        transform = (
            transform9
            * Affine.translation(
                offset,
                offset,
            )
        )

        print()
        print(
            "=" * 70
        )
        print(
            f"ST01 {DATE}: solving "
            f"{size}x{size}"
        )
        print(
            "=" * 70
        )

        result = solve_halo(
            radius=radius,
            raw=raw,
            fine_transform=fine_transform,
            fine_crs=fine_crs,
            modis_et=modis,
            modis_transform=transform,
        )

        results[size] = result
        summaries.append(
            result["summary"]
        )

        print(
            "Eligible:",
            result["summary"][
                "central_parent_eligible"
            ],
        )
        print(
            "Published pixels:",
            result["summary"][
                "central_published_pixels"
            ],
        )
        print(
            "Usable fraction:",
            result["summary"][
                "central_parent_usable_fraction"
            ],
        )
        print(
            "Conservation error:",
            result["summary"][
                "central_conservation_error_mm"
            ],
        )

    comparisons = [
        compare(
            results[9],
            results[5],
            "5x5_vs_9x9",
        ),
        compare(
            results[9],
            results[7],
            "7x7_vs_9x9",
        ),
    ]

    summary_df = pd.DataFrame(
        summaries
    )
    comparison_df = pd.DataFrame(
        comparisons
    )

    summary_path = (
        output_root
        / "st01_halo_summary.csv"
    )
    comparison_path = (
        output_root
        / "st01_halo_comparison.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )
    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    metadata = {
        "date": DATE,
        "station_id": STATION_ID,
        "station": station["station"],
        "inside_official_basin": (
            station["inside_basin"]
        ),
        "scientific_signature": (
            scientific_signature
        ),
        "native_parent_row": (
            parent_row
        ),
        "native_parent_col": (
            parent_col
        ),
        "largest_download_halo": "9x9",
        "tested_halos": [
            "5x5",
            "7x7",
            "9x9",
        ],
    }

    (
        output_root
        / "audit_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 70
    )
    print("HALO SUMMARY")
    print(
        "=" * 70
    )
    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 70
    )
    print("CONVERGENCE AGAINST 9x9")
    print(
        "=" * 70
    )
    print(
        comparison_df.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(summary_path)
    print(comparison_path)


if __name__ == "__main__":
    main()
