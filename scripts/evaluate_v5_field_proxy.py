"""Compare the virtual-10 Ridge25 model with the frozen field-derived ET proxy.

Scientific purpose
------------------
This is the decisive field-proxy test for the experimental training design:
10 spatially distributed virtual MODIS footprints are used for training, while
the five real field stations remain excluded from model training.

The script deliberately reuses the *frozen stable field-comparison table* to
preserve exactly the same:
- field periods;
- >=5/8 valid-day primary completeness rule;
- field-reference ET aggregation;
- fixed Kc proxies for ST01-ST03;
- local-NDVI Kc sensitivity for ST04-ST05;
- MODIS parent ET values.

Only the Ridge25 training design changes.

For each frozen field station-period, the virtual-10 full model is evaluated
using the accepted 20 m production machinery:
- same 25 predictors;
- full virtual-10 equal-weight AOA;
- exact 20 m/native-MODIS overlap reconciliation;
- same >=90% usable support rule;
- same nonnegative handling and conservation tolerance.

Both WITH_AOA and WITHOUT_AOA sensitivity are solved from the same raw virtual-10
fine fields. The no-AOA case changes only the AOA mask, holding all other rules
fixed.

Important interpretation
------------------------
The virtual-10 model has never trained on the five real station footprints.
This provides a spatially external field-site comparison relative to the
training locations. It is still NOT independent 20 m ET validation because:
1. the training target remains MODIS-derived Kc = MODIS ET / ETo; and
2. field actual ET remains a Kc-derived proxy, not a direct ET measurement.

Outputs are isolated under:
ET_fundacion_workspace_virtual_station/training/
    field_comparison/

Stable current/ and final/ are never modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil

import ee
import numpy as np
import pandas as pd

from et_downscaling.virtual_station import (
    resolve_virtual_workspace, resolve_reference_workspace, resolve_reference_run,
)

import et_downscaling.ridge25_production as ridge25_production
from et_downscaling.aoa_ridge25 import build_unweighted_aoa
from et_downscaling.config import ANALYSIS_CRS, build_training_output_filename
from et_downscaling.local_tiles import build_initial_tiles
from et_downscaling.overlap_reconciliation import build_overlap_edges
from et_downscaling.ridge25 import (
    RIDGE25_METEOROLOGICAL_FEATURES,
    RIDGE25_MODEL_FEATURES,
    build_ridge25_model,
)
from et_downscaling.ridge25_overlap_production import (
    RAW_TILE_BANDS,
    _download_native_modis,
)


EXPERIMENT_NAME = "v5_basin_random_ge90_canonical"
FIELD_EVALUATION_VERSION = "virtual10_full_model_field_exact_overlap_v1"
PRIMARY_MIN_VALID_DAYS = 5
STRICT_VALID_DAYS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate virtual-10 Ridge25 against the frozen field-derived ET proxy."
        )
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--reference-workspace", required=True)
    parser.add_argument(
        "--stable-run-dir",
        default=None,
        help=(
            "Frozen stable run used only for provenance and the existing stable "
            "comparison. Defaults to current/runs/20260907T162048Z_2020_2024."
        ),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard the virtual-10 field checkpoint and raw/product cache.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_workspace_root(value: str | None) -> Path:
    return resolve_virtual_workspace(value, project_root())



def resolve_stable_run(reference_root: Path, value: str | None) -> Path:
    return resolve_reference_run(reference_root, value)


def load_script_module(
    path: Path,
    module_name: str,
):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "si", "sÃ­"})
    )


def build_st04_meteorology_fallback(
    stable_master: pd.DataFrame,
):
    """Return a production meteorology wrapper consistent with prior ST04 audit."""
    support_columns = [
        "station_id",
        "era5_sampling_longitude",
        "era5_sampling_latitude",
        "era5_sampling_distance_m",
    ]
    missing = [
        column
        for column in support_columns
        if column not in stable_master.columns
    ]
    if missing:
        raise RuntimeError(
            "Stable master is missing ST04 ERA5 support columns: "
            + ", ".join(missing)
        )

    support = (
        stable_master.loc[
            stable_master["station_id"].astype(str).eq("ST04"),
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
            f"Expected one ST04 ERA5 support record; found {len(support)}."
        )

    support_row = support.iloc[0]
    support_lon = float(
        support_row["era5_sampling_longitude"]
    )
    support_lat = float(
        support_row["era5_sampling_latitude"]
    )
    support_distance = float(
        support_row["era5_sampling_distance_m"]
    )

    native_builder = (
        ridge25_production
        .build_ridge25_meteorological_predictors
    )

    support_point = ee.Geometry.Point(
        [support_lon, support_lat]
    )
    support_geometry = support_point.buffer(1000)

    def station_consistent_meteorology(
        period_start,
        period_end,
        number_days,
        processing_geometry,
    ):
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

        return (
            native
            .unmask(fallback)
            .clip(processing_geometry)
            .toFloat()
        )

    return (
        station_consistent_meteorology,
        {
            "era5_sampling_longitude": support_lon,
            "era5_sampling_latitude": support_lat,
            "era5_sampling_distance_m": support_distance,
        },
    )


def ensure_raw_support(
    field_module,
    output_directory: Path,
    date_text: str,
    cache_block: str,
    model_resource: dict,
    center_tile,
    timeout_seconds: int,
):
    support_tiles, support_bounds = (
        field_module.support_tiles_around(
            center_tile
        )
    )
    completed = []

    for tile in support_tiles:
        data, _ = field_module.get_raw_tile(
            output_directory=output_directory,
            date_text=date_text,
            block=cache_block,
            fold_resource=model_resource,
            tile=tile,
            timeout_seconds=timeout_seconds,
        )
        completed.append((tile, data))

    raw, support_transform = (
        field_module.mosaic_raw_tiles(
            completed,
            support_bounds,
        )
    )
    return (
        raw,
        support_transform,
        support_bounds,
    )


def evaluate_one_row(
    *,
    field_module,
    sensitivity_module,
    root: Path,
    output_directory: Path,
    model_resource: dict,
    row: pd.Series,
    basin_tiles,
    timeout_seconds: int,
    st04_fallback,
):
    station_id = str(row["station_id"])
    date_text = pd.Timestamp(
        row["period_start"]
    ).strftime("%Y-%m-%d")

    x, y = field_module.station_xy(row)

    if station_id == "ST04":
        center_tile = (
            sensitivity_module.external_center_tile(
                field_module,
                root,
                x,
                y,
            )
        )
        evaluation_domain = "external_outside_basin"

        original_builder = (
            ridge25_production
            .build_ridge25_meteorological_predictors
        )
        ridge25_production.build_ridge25_meteorological_predictors = (
            st04_fallback
        )
    else:
        center_tile = field_module.find_station_tile(
            basin_tiles,
            x,
            y,
        )
        evaluation_domain = "in_basin"
        original_builder = None

    try:
        raw, support_transform, support_bounds = (
            ensure_raw_support(
                field_module=field_module,
                output_directory=output_directory,
                date_text=date_text,
                cache_block="virtual10_full",
                model_resource=model_resource,
                center_tile=center_tile,
                timeout_seconds=timeout_seconds,
            )
        )
    finally:
        if original_builder is not None:
            (
                ridge25_production
                .build_ridge25_meteorological_predictors
            ) = original_builder

    fine_shape = raw.shape[1:]
    kc_raw = raw[
        RAW_TILE_BANDS.index("Kc_raw")
    ]
    di = raw[
        RAW_TILE_BANDS.index(
            "dissimilarity_index"
        )
    ]
    stack_valid = (
        raw[
            RAW_TILE_BANDS.index(
                "stack_valid"
            )
        ]
        > 0.5
    )
    aoa_inside = (
        raw[
            RAW_TILE_BANDS.index(
                "AOA_inside"
            )
        ]
        > 0.5
    )
    usable_with_aoa = (
        raw[
            RAW_TILE_BANDS.index(
                "usable"
            )
        ]
        > 0.5
    )
    domain = (
        raw[
            RAW_TILE_BANDS.index(
                "support_domain"
            )
        ]
        > 0.5
    )

    usable_without_aoa = (
        stack_valid
        & np.isfinite(kc_raw)
        & (kc_raw >= 0)
    )

    station_row, station_col = (
        sensitivity_module.sample_index(
            support_transform,
            fine_shape,
            x,
            y,
        )
    )

    output = {
        "station_id": station_id,
        "station": row.get(
            "station",
            station_id,
        ),
        "period_start": date_text,
        "evaluation_domain": evaluation_domain,
        "longitude": float(row["longitude"]),
        "latitude": float(row["latitude"]),
        "stack_valid": float(
            stack_valid[
                station_row,
                station_col,
            ]
        ),
        "AOA_inside": float(
            aoa_inside[
                station_row,
                station_col,
            ]
        ),
        "dissimilarity_index": (
            float(
                di[
                    station_row,
                    station_col,
                ]
            )
            if np.isfinite(
                di[
                    station_row,
                    station_col,
                ]
            )
            else np.nan
        ),
        "Kc_raw": (
            float(
                kc_raw[
                    station_row,
                    station_col,
                ]
            )
            if np.isfinite(
                kc_raw[
                    station_row,
                    station_col,
                ]
            )
            else np.nan
        ),
    }

    if not stack_valid[
        station_row,
        station_col,
    ]:
        output.update(
            {
                "ET_virtual10_with_AOA_mm_period": np.nan,
                "ET_virtual10_without_AOA_mm_period": np.nan,
                "support_with_AOA": 0.0,
                "support_without_AOA": 0.0,
                "status_with_AOA": "stack_invalid",
                "status_without_AOA": "stack_invalid",
                "recovered_by_removing_AOA": False,
            }
        )
        return output

    modis_et, modis_grid = (
        _download_native_modis(
            period_start=date_text,
            support_bounds=support_bounds,
            timeout_seconds=timeout_seconds,
        )
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

    with_aoa = (
        sensitivity_module.solve_scenario(
            field_module,
            kc_raw,
            usable_with_aoa,
            edges,
            modis_et,
            fine_shape,
            station_row,
            station_col,
        )
    )
    without_aoa = (
        sensitivity_module.solve_scenario(
            field_module,
            kc_raw,
            usable_without_aoa,
            edges,
            modis_et,
            fine_shape,
            station_row,
            station_col,
        )
    )

    output.update(
        {
            "ET_virtual10_with_AOA_mm_period": (
                with_aoa["ET"]
            ),
            "ET_virtual10_without_AOA_mm_period": (
                without_aoa["ET"]
            ),
            "support_with_AOA": (
                with_aoa["usable_fraction"]
            ),
            "support_without_AOA": (
                without_aoa[
                    "usable_fraction"
                ]
            ),
            "coarse_eligible_with_AOA": (
                with_aoa[
                    "coarse_eligible"
                ]
            ),
            "coarse_eligible_without_AOA": (
                without_aoa[
                    "coarse_eligible"
                ]
            ),
            "conservation_error_with_AOA_mm": (
                with_aoa[
                    "conservation_error"
                ]
            ),
            "conservation_error_without_AOA_mm": (
                without_aoa[
                    "conservation_error"
                ]
            ),
            "status_with_AOA": (
                with_aoa["status"]
            ),
            "status_without_AOA": (
                without_aoa["status"]
            ),
            "recovered_by_removing_AOA": (
                with_aoa["status"]
                != "valid"
                and without_aoa["status"]
                == "valid"
            ),
        }
    )
    return output


def metric_rows_for_subset(
    field_module,
    subset: pd.DataFrame,
    *,
    comparison_label: str,
    virtual_column: str,
    stable_column: str,
    include_stable: bool = True,
) -> list[dict]:
    """Calculate field metrics on the scientifically intended sample.

    Virtual-native scenarios use only the field proxy, MODIS parent, and
    Virtual10 prediction. Stable5 availability must not reduce the native
    Virtual10 sample.

    Matched scenarios include Stable5 and therefore require all three model
    predictions on the same station-period rows.
    """
    required = [
        "ET_field_proxy_mm_period",
        "ET_MODIS_parent_mm_period",
        virtual_column,
    ]

    models = [
        (
            "MODIS_parent",
            "ET_MODIS_parent_mm_period",
        ),
        (
            "Virtual10_Ridge25",
            virtual_column,
        ),
    ]

    if include_stable:
        required.append(stable_column)
        models.insert(
            1,
            (
                "Stable5_Ridge25",
                stable_column,
            ),
        )

    complete = subset.dropna(
        subset=required
    ).copy()

    rows = []

    for model_name, column in models:
        rows.append(
            {
                "comparison": comparison_label,
                "model": model_name,
                **field_module.calculate_metrics(
                    complete[
                        "ET_field_proxy_mm_period"
                    ],
                    complete[column],
                ),
            }
        )

    return rows


def build_metrics(
    field_module,
    table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    completeness_masks = {
        "primary_ge5_of_8": (
            pd.to_numeric(
                table["n_valid_field_days"],
                errors="coerce",
            )
            >= PRIMARY_MIN_VALID_DAYS
        ),
        "strict_8_of_8": (
            pd.to_numeric(
                table["n_valid_field_days"],
                errors="coerce",
            )
            >= STRICT_VALID_DAYS
        ),
    }

    fixed_kc = table["station_id"].isin(
        ["ST01", "ST02", "ST03"]
    )

    for completeness_name, completeness in completeness_masks.items():
        # Virtual-native scenarios. These describe the virtual model on the
        # frozen field-period pool but use the virtual model's own AOA status.
        virtual_with = (
            completeness
            & table[
                "virtual10_status_with_AOA"
            ].eq("valid")
        )
        virtual_without = (
            completeness
            & table[
                "virtual10_status_without_AOA"
            ].eq("valid")
        )

        for scenario_name, mask, virtual_column, stable_column in (
            (
                "virtual_native_all_with_AOA",
                virtual_with,
                "ET_virtual10_with_AOA_mm_period",
                "ET_Ridge_with_AOA_mm_period",
            ),
            (
                "virtual_native_all_without_AOA",
                virtual_without,
                "ET_virtual10_without_AOA_mm_period",
                "ET_Ridge_without_AOA_mm_period",
            ),
            (
                "virtual_native_fixed_Kc_with_AOA",
                virtual_with & fixed_kc,
                "ET_virtual10_with_AOA_mm_period",
                "ET_Ridge_with_AOA_mm_period",
            ),
            (
                "virtual_native_fixed_Kc_without_AOA",
                virtual_without & fixed_kc,
                "ET_virtual10_without_AOA_mm_period",
                "ET_Ridge_without_AOA_mm_period",
            ),
        ):
            rows.extend(
                metric_rows_for_subset(
                    field_module,
                    table.loc[mask].copy(),
                    comparison_label=(
                        f"{completeness_name}__"
                        f"{scenario_name}"
                    ),
                    virtual_column=virtual_column,
                    stable_column=stable_column,
                    include_stable=False,
                )
            )

        # Matched comparisons are the decisive A/B: Stable5 and Virtual10 must
        # both be publishable on the exact same station-period rows.
        stable_with = (
            completeness
            & table[
                "stable_included_all_with_AOA"
            ]
        )
        stable_without = (
            completeness
            & table[
                "stable_included_all_without_AOA"
            ]
        )

        matched_with = (
            stable_with
            & table[
                "virtual10_status_with_AOA"
            ].eq("valid")
        )
        matched_without = (
            stable_without
            & table[
                "virtual10_status_without_AOA"
            ].eq("valid")
        )

        for scenario_name, mask, virtual_column, stable_column in (
            (
                "matched_all_with_AOA",
                matched_with,
                "ET_virtual10_with_AOA_mm_period",
                "ET_Ridge_with_AOA_mm_period",
            ),
            (
                "matched_all_without_AOA",
                matched_without,
                "ET_virtual10_without_AOA_mm_period",
                "ET_Ridge_without_AOA_mm_period",
            ),
            (
                "matched_fixed_Kc_with_AOA",
                matched_with & fixed_kc,
                "ET_virtual10_with_AOA_mm_period",
                "ET_Ridge_with_AOA_mm_period",
            ),
            (
                "matched_fixed_Kc_without_AOA",
                matched_without & fixed_kc,
                "ET_virtual10_without_AOA_mm_period",
                "ET_Ridge_without_AOA_mm_period",
            ),
        ):
            rows.extend(
                metric_rows_for_subset(
                    field_module,
                    table.loc[mask].copy(),
                    comparison_label=(
                        f"{completeness_name}__"
                        f"{scenario_name}"
                    ),
                    virtual_column=virtual_column,
                    stable_column=stable_column,
                )
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = project_root()
    workspace_root = resolve_workspace_root(
        args.workspace_root
    )
    reference_root = resolve_reference_workspace(args.reference_workspace)
    if reference_root == workspace_root:
        raise ValueError("Stable5 reference must be separate from the V5 workspace.")
    stable_run = resolve_stable_run(
        reference_root,
        args.stable_run_dir,
    )

    experiment_root = (
        workspace_root
        / "evaluation"
    )
    results_root = (
        experiment_root
        / "results"
    )
    output_directory = (
        experiment_root
        / "field_comparison"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for forbidden in (
        (workspace_root / "current").resolve(),
        (workspace_root / "final").resolve(),
    ):
        resolved = (
            output_directory.resolve()
        )
        if (
            resolved == forbidden
            or forbidden in resolved.parents
        ):
            raise RuntimeError(
                "Experimental field output resolves inside current/ or final/."
            )

    field_module = load_script_module(
        root
        / "scripts"
        / "evaluate_field_ridge25.py",
        "virtual10_field_eval_base",
    )
    sensitivity_module = load_script_module(
        root
        / "scripts"
        / "compare_field_with_without_aoa.py",
        "virtual10_field_aoa_base",
    )

    # Give the experimental cache an explicit identity so it can never be
    # confused with the frozen stable OOF field cache.
    field_module.FIELD_EVALUATION_VERSION = (
        FIELD_EVALUATION_VERSION
    )

    stable_field_path = (
        reference_root
        / "current"
        / "diagnostics"
        / "field_ridge25_final_scenarios"
        / "field_comparison_scenarios.csv"
    )
    virtual_population_path = (
        results_root
        / "virtual10_training_population.csv"
    )
    experiment_metadata_path = (
        results_root
        / "experiment_metadata.json"
    )

    for path in (
        stable_field_path,
        virtual_population_path,
        experiment_metadata_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    stable_field = pd.read_csv(
        stable_field_path,
        dtype={"station_id": str},
    )
    stable_field[
        "period_start"
    ] = pd.to_datetime(
        stable_field["period_start"],
        errors="raise",
    )

    # The frozen final field-scenario table intentionally omits coordinates.
    # Recover them from the canonical station GeoJSON used by the accepted
    # field-evaluation pipeline. This changes no field values, memberships,
    # Kc proxies, MODIS values, or stable Ridge predictions.
    station_coordinates = (
        field_module.load_station_metadata(root)[
            ["station_id", "longitude", "latitude"]
        ]
        .copy()
    )
    station_coordinates["station_id"] = (
        station_coordinates["station_id"].astype(str)
    )

    if station_coordinates["station_id"].duplicated().any():
        raise RuntimeError(
            "Canonical station metadata contains duplicate station_id values."
        )

    stable_field["station_id"] = stable_field["station_id"].astype(str)

    coordinate_lookup = station_coordinates.set_index("station_id")
    for coordinate in ("longitude", "latitude"):
        mapped = stable_field["station_id"].map(
            coordinate_lookup[coordinate]
        )
        if coordinate in stable_field.columns:
            existing = pd.to_numeric(
                stable_field[coordinate],
                errors="coerce",
            )
            stable_field[coordinate] = existing.fillna(mapped)
        else:
            stable_field[coordinate] = mapped

    if stable_field[["longitude", "latitude"]].isna().any().any():
        unresolved = sorted(
            stable_field.loc[
                stable_field[["longitude", "latitude"]]
                .isna()
                .any(axis=1),
                "station_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise RuntimeError(
            "Could not recover canonical coordinates for field station(s): "
            + ", ".join(unresolved)
        )

    # Frozen final table columns are renamed explicitly before merging to
    # prevent accidental ambiguity about which training design they belong to.
    stable_field = stable_field.rename(
        columns={
            "included_all_with_AOA": (
                "stable_included_all_with_AOA"
            ),
            "included_all_without_AOA_pure_extrapolations": (
                "stable_included_all_without_AOA"
            ),
            "included_fixed_Kc_with_AOA": (
                "stable_included_fixed_Kc_with_AOA"
            ),
            "included_fixed_Kc_without_AOA_pure_extrapolations": (
                "stable_included_fixed_Kc_without_AOA"
            ),
            "status_with_AOA": (
                "stable_status_with_AOA"
            ),
            "status_without_AOA": (
                "stable_status_without_AOA"
            ),
            "AOA_inside": (
                "stable_AOA_inside"
            ),
            "dissimilarity_index": (
                "stable_dissimilarity_index"
            ),
        }
    )

    for column in (
        "stable_included_all_with_AOA",
        "stable_included_all_without_AOA",
        "stable_included_fixed_Kc_with_AOA",
        "stable_included_fixed_Kc_without_AOA",
    ):
        if column in stable_field.columns:
            stable_field[column] = (
                bool_series(
                    stable_field[column]
                )
            )

    required_field_columns = {
        "station_id",
        "period_start",
        "n_valid_field_days",
        "ET_field_proxy_mm_period",
        "ET_MODIS_parent_mm_period",
        "ET_Ridge_with_AOA_mm_period",
        "ET_Ridge_without_AOA_mm_period",
        "longitude",
        "latitude",
        "stable_included_all_with_AOA",
        "stable_included_all_without_AOA",
    }
    missing_field = sorted(
        required_field_columns
        - set(stable_field.columns)
    )
    if missing_field:
        raise RuntimeError(
            "Frozen stable field table is missing columns: "
            + ", ".join(missing_field)
        )

    virtual_population = pd.read_csv(
        virtual_population_path,
        dtype={"station_id": str},
    )
    if (
        virtual_population[
            "station_id"
        ]
        .nunique()
        != 10
    ):
        raise RuntimeError(
            "Virtual10 training population does not contain 10 supports."
        )
    if (
        virtual_population[
            "spatial_block"
        ]
        .nunique()
        != 10
    ):
        raise RuntimeError(
            "Virtual10 training population does not contain 10 spatial blocks."
        )
    if not (
        virtual_population[
            "station_id"
        ]
        .astype(str)
        .str.startswith("VF")
        .all()
    ):
        raise RuntimeError(
            "Virtual10 population unexpectedly contains real station IDs."
        )

    virtual_model = (
        build_ridge25_model()
    )
    virtual_model.fit(
        virtual_population[
            RIDGE25_MODEL_FEATURES
        ],
        virtual_population[
            "Kc_target"
        ],
    )
    virtual_aoa = (
        build_unweighted_aoa(
            virtual_population,
            group_column="spatial_block",
        )
    )

    experiment_metadata = json.loads(
        experiment_metadata_path.read_text(
            encoding="utf-8"
        )
    )
    expected_threshold = float(
        experiment_metadata[
            "virtual10_AOA_threshold"
        ]
    )
    if not np.isclose(
        float(virtual_aoa.threshold),
        expected_threshold,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Virtual10 AOA threshold does not reproduce the experiment: "
            f"{virtual_aoa.threshold} vs {expected_threshold}"
        )

    model_resource = {
        "model": virtual_model,
        "aoa": virtual_aoa,
        "training_rows": int(
            len(virtual_population)
        ),
        "test_rows": 0,
        "oof_max_abs_difference": np.nan,
    }

    # Stable master is used ONLY to reproduce the previously accepted ST04
    # nearest-valid ERA5 support. It is not used to fit the virtual10 model.
    stable_master_path = (
        reference_root
        / "current"
        / "master"
        / "S2"
        / build_training_output_filename(
            "S2"
        )
    )
    if not stable_master_path.is_file():
        raise FileNotFoundError(
            stable_master_path
        )
    stable_master = pd.read_csv(
        stable_master_path,
        dtype={"station_id": str},
    )

    ee.Initialize(
        project=args.project
    )
    ee.Number(1).getInfo()

    (
        st04_fallback,
        st04_support_metadata,
    ) = build_st04_meteorology_fallback(
        stable_master
    )

    if args.restart:
        checkpoint = (
            output_directory
            / "virtual10_field_checkpoint.csv"
        )
        checkpoint.unlink(
            missing_ok=True
        )
        for cache_name in (
            "exact_overlap_raw_tile_cache",
            "exact_overlap_product_cache",
        ):
            cache_dir = (
                output_directory
                / cache_name
            )
            if cache_dir.is_dir():
                shutil.rmtree(
                    cache_dir
                )

    checkpoint_path = (
        output_directory
        / "virtual10_field_checkpoint.csv"
    )

    completed = pd.DataFrame()
    if checkpoint_path.is_file():
        completed = pd.read_csv(
            checkpoint_path,
            dtype={"station_id": str},
        )
        completed[
            "period_start"
        ] = pd.to_datetime(
            completed["period_start"],
            errors="raise",
        )

    completed_keys = set()
    if not completed.empty:
        completed_keys = set(
            zip(
                completed[
                    "station_id"
                ].astype(str),
                completed[
                    "period_start"
                ].dt.strftime(
                    "%Y-%m-%d"
                ),
            )
        )

    basin_tiles, _ = (
        build_initial_tiles(
            root,
            tile_size_m=(
                field_module
                .LOCAL_TILE_SIZE_M
            ),
        )
    )

    print("=" * 100)
    print("V5 BASIN10 VS FROZEN FIELD-DERIVED ET PROXY")
    print("=" * 100)
    print("Stable run:", stable_run.name)
    print("Virtual10 training rows:", len(virtual_population))
    print("Virtual10 training supports:", virtual_population["station_id"].nunique())
    print("Virtual10 spatial blocks:", virtual_population["spatial_block"].nunique())
    print("Virtual10 AOA threshold:", f"{virtual_aoa.threshold:.12f}")
    print("Real station footprints used for Virtual10 training: NO")
    print("Frozen field rows:", len(stable_field))
    print("Primary completeness rule: >=5 valid days per 8-day period")
    print("Strict sensitivity: 8/8 valid days")
    print("Stable current/ modified: NO")
    print("Stable final/ modified: NO")
    print()

    output_rows = (
        completed.to_dict("records")
        if not completed.empty
        else []
    )

    total = len(stable_field)
    for index, (_, row) in enumerate(
        stable_field.iterrows(),
        start=1,
    ):
        station_id = str(
            row["station_id"]
        )
        date_text = (
            pd.Timestamp(
                row["period_start"]
            ).strftime("%Y-%m-%d")
        )
        key = (
            station_id,
            date_text,
        )

        if key in completed_keys:
            print(
                f"[{index:02d}/{total:02d}] "
                f"{station_id} {date_text} - checkpoint OK"
            )
            continue

        print(
            f"[{index:02d}/{total:02d}] "
            f"{station_id} {date_text}"
        )

        try:
            evaluated = evaluate_one_row(
                field_module=field_module,
                sensitivity_module=sensitivity_module,
                root=root,
                output_directory=output_directory,
                model_resource=model_resource,
                row=row,
                basin_tiles=basin_tiles,
                timeout_seconds=args.timeout_seconds,
                st04_fallback=st04_fallback,
            )
        except RuntimeError as error:
            evaluated = {
                "station_id": station_id,
                "station": row.get(
                    "station",
                    station_id,
                ),
                "period_start": date_text,
                "longitude": float(
                    row["longitude"]
                ),
                "latitude": float(
                    row["latitude"]
                ),
                "ET_virtual10_with_AOA_mm_period": np.nan,
                "ET_virtual10_without_AOA_mm_period": np.nan,
                "status_with_AOA": (
                    field_module
                    .expected_nonpublication_status(
                        error
                    )
                    or "runtime_error"
                ),
                "status_without_AOA": (
                    field_module
                    .expected_nonpublication_status(
                        error
                    )
                    or "runtime_error"
                ),
                "production_error": str(
                    error
                ),
            }

        output_rows.append(
            evaluated
        )

        checkpoint = pd.DataFrame(
            output_rows
        )
        checkpoint[
            "period_start"
        ] = pd.to_datetime(
            checkpoint[
                "period_start"
            ],
            errors="raise",
        )
        checkpoint = (
            checkpoint
            .sort_values(
                [
                    "station_id",
                    "period_start",
                ]
            )
            .drop_duplicates(
                [
                    "station_id",
                    "period_start",
                ],
                keep="last",
            )
        )
        checkpoint.to_csv(
            checkpoint_path,
            index=False,
        )
        completed_keys.add(
            key
        )

    virtual_rows = pd.read_csv(
        checkpoint_path,
        dtype={"station_id": str},
    )
    virtual_rows[
        "period_start"
    ] = pd.to_datetime(
        virtual_rows[
            "period_start"
        ],
        errors="raise",
    )

    virtual_rows = virtual_rows.rename(
        columns={
            "status_with_AOA": (
                "virtual10_status_with_AOA"
            ),
            "status_without_AOA": (
                "virtual10_status_without_AOA"
            ),
            "AOA_inside": (
                "virtual10_AOA_inside"
            ),
            "dissimilarity_index": (
                "virtual10_dissimilarity_index"
            ),
            "Kc_raw": (
                "virtual10_Kc_raw"
            ),
            "stack_valid": (
                "virtual10_stack_valid"
            ),
            "support_with_AOA": (
                "virtual10_support_with_AOA"
            ),
            "support_without_AOA": (
                "virtual10_support_without_AOA"
            ),
            "coarse_eligible_with_AOA": (
                "virtual10_coarse_eligible_with_AOA"
            ),
            "coarse_eligible_without_AOA": (
                "virtual10_coarse_eligible_without_AOA"
            ),
        }
    )

    virtual_merge_columns = [
        column
        for column in virtual_rows.columns
        if column
        not in {
            "station",
            "longitude",
            "latitude",
            "evaluation_domain",
        }
    ]

    comparison = stable_field.merge(
        virtual_rows[
            virtual_merge_columns
        ],
        on=[
            "station_id",
            "period_start",
        ],
        how="left",
        validate="one_to_one",
    )

    comparison[
        "virtual10_recovered_by_removing_AOA"
    ] = (
        comparison[
            "virtual10_status_with_AOA"
        ].ne("valid")
        & comparison[
            "virtual10_status_without_AOA"
        ].eq("valid")
    )

    metrics = build_metrics(
        field_module,
        comparison,
    )

    comparison_path = (
        output_directory
        / "virtual10_vs_stable_field_pairs.csv"
    )
    metrics_path = (
        output_directory
        / "virtual10_vs_stable_field_metrics.csv"
    )
    metadata_output_path = (
        output_directory
        / "metadata.json"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )
    metrics.to_csv(
        metrics_path,
        index=False,
    )

    metadata_output = {
        "experiment": EXPERIMENT_NAME,
        "field_evaluation_version": FIELD_EVALUATION_VERSION,
        "stable_run": stable_run.name,
        "reference_workspace": str(reference_root),
        "virtual10_training_rows": int(
            len(virtual_population)
        ),
        "virtual10_training_supports": int(
            virtual_population[
                "station_id"
            ].nunique()
        ),
        "virtual10_spatial_blocks": int(
            virtual_population[
                "spatial_block"
            ].nunique()
        ),
        "virtual10_AOA_threshold": float(
            virtual_aoa.threshold
        ),
        "real_station_footprints_used_for_virtual10_training": False,
        "field_proxy_source": str(
            stable_field_path
        ),
        "primary_min_valid_days": PRIMARY_MIN_VALID_DAYS,
        "strict_valid_days": STRICT_VALID_DAYS,
        "st04_era5_fallback": st04_support_metadata,
        "stable_current_modified": False,
        "stable_final_modified": False,
        "google_drive_used": False,
        "interpretation_guardrail": (
            "The virtual10 model is spatially external to the five real station "
            "footprints, but the training target remains MODIS-derived and the "
            "field actual ET remains a Kc-derived proxy. This is not independent "
            "20 m ET validation."
        ),
    }
    metadata_output_path.write_text(
        json.dumps(
            metadata_output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("FIELD COMPARISON COMPLETE")
    print("=" * 100)
    print()
    print("VIRTUAL10 STATUS WITH AOA")
    print(
        comparison[
            "virtual10_status_with_AOA"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )
    print()
    print("VIRTUAL10 STATUS WITHOUT AOA")
    print(
        comparison[
            "virtual10_status_without_AOA"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )
    print()
    print("FIELD METRICS")
    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
                if isinstance(
                    value,
                    (float, np.floating),
                )
                else str(value)
            ),
        )
    )
    print()
    print("Saved:")
    print(" -", comparison_path)
    print(" -", metrics_path)
    print(" -", metadata_output_path)
    print()
    print("Stable current/ modified: NO")
    print("Stable final/ modified: NO")
    print("Production model replaced: NO")


if __name__ == "__main__":
    main()
