"""Run the parsimonious Fundación ET workflow from the three canonical inputs.

Scientific execution
--------------------
1. Reuse or refresh the canonical raw Sentinel-2/MODIS and meteorological
   extractions in the external workspace.
2. Rebuild the local training master directly from those raw caches.
3. Rebuild the GE90 Ridge-25 population.
4. Perform spatial-block and leave-one-year-out OOF validation.
5. Fit Ridge-25 in memory on all eligible observations and rebuild the AOA.
6. Save current-run tables, AOA parameters, metadata and core diagnostics.
7. Run the final field-comparison phase using spatial-OOF fine ET; field
   observations never enter Ridge-25 training.
8. Optionally generate one or more locally downloaded 20 m ET rasters from the
   same fitted Ridge-25/AOA state, each followed by the single global
   exact-overlap MODIS reconciliation.
9. Finalize run provenance only after field and raster outputs exist.

A fitted model is never loaded from disk. Reconciliation is never used during
training or OOF validation. Google Drive and persistent Earth Engine assets are
not production destinations.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import et_downscaling


CANONICAL_START_DATE = "2020-01-01"
CANONICAL_END_DATE_EXCLUSIVE = "2025-01-01"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete parsimonious Fundación ET workflow."
        )
    )
    parser.add_argument(
        "--project",
        default=None,
        help=(
            "Google Cloud Project ID with Earth Engine access. "
            "Prompted interactively when omitted."
        ),
    )
    parser.add_argument(
        "--start-date",
        default=CANONICAL_START_DATE,
    )
    parser.add_argument(
        "--end-date-exclusive",
        default=CANONICAL_END_DATE_EXCLUSIVE,
    )
    parser.add_argument(
        "--refresh-raw",
        action="store_true",
        help=(
            "Force rebuilding reusable raw satellite and "
            "meteorological extractions."
        ),
    )
    parser.add_argument(
        "--no-raster",
        action="store_true",
        help=(
            "Finish after training, validation and diagnostics."
        ),
    )
    parser.add_argument(
        "--raster-date",
        dest="raster_dates",
        action="append",
        default=[],
        help=(
            "Generate a MODIS-period ET raster without an interactive prompt "
            "(YYYY-MM-DD). Repeat this option to produce multiple periods "
            "from the same fitted model/AOA state."
        ),
    )
    parser.add_argument(
        "--tile-size-m",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--min-tile-size-m",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
    )
    parser.add_argument(
        "--no-field-evaluation",
        action="store_true",
        help=(
            "Skip the final ETgage field-comparison phase. "
            "Field observations never train Ridge-25."
        ),
    )
    return parser.parse_args()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_id(value: str | None) -> str:
    project_id = (
        value.strip()
        if value
        else ""
    )
    if not project_id:
        project_id = input(
            "Google Cloud Project ID: "
        ).strip()
    if not project_id:
        raise ValueError(
            "Google Cloud Project ID cannot be empty."
        )
    return project_id


def configure_period_environment(
    start_date: str,
    end_date_exclusive: str,
) -> None:
    os.environ["ET_START_DATE"] = start_date
    os.environ[
        "ET_END_DATE_EXCLUSIVE"
    ] = end_date_exclusive


def validate_imported_package_root(
    project_root: Path,
) -> Path:

    expected = (
        project_root
        / "src"
        / "et_downscaling"
    ).resolve()
    imported = Path(
        et_downscaling.__file__
    ).resolve()

    try:
        imported.relative_to(
            expected
        )
    except ValueError:
        raise RuntimeError(
            "The imported et_downscaling package belongs "
            "to a different repository.\n"
            f"Expected: {expected}\n"
            f"Imported: {imported}\n"
            "No pipeline work was started.\n"
            "Run: python -m pip install -e ."
        ) from None

    return imported


def run_script(
    project_root: Path,
    script_name: str,
    arguments: list[str],
    project_id: str | None = None,
) -> None:
    command = [
        sys.executable,
        str(
            project_root
            / "scripts"
            / script_name
        ),
        *arguments,
    ]
    print()
    print(
        ">",
        " ".join(command),
    )

    stdin_text = (
        project_id + "\n"
        if project_id
        else None
    )
    subprocess.run(
        command,
        cwd=project_root,
        check=True,
        text=True,
        input=stdin_text,
        env=os.environ.copy(),
    )


def build_final_training_master(
    project_root: Path,
    master_path: Path,
) -> pd.DataFrame:
    """Rebuild the final Ridge-25 master from canonical reusable raw caches."""
    run_script(
        project_root,
        "build_training_dataset.py",
        [
            "--optical-source",
            "S2",
            "--ridge25-only",
        ],
    )

    if not master_path.is_file():
        raise FileNotFoundError(
            "Final Ridge-25 training master was not created:\n"
            f"{master_path}"
        )

    master = pd.read_csv(
        master_path,
        dtype={"station_id": str},
    )

    print()
    print("=== FINAL RIDGE-25 TRAINING MASTER ===")
    print("Master rows:", len(master))
    print("Master columns:", len(master.columns))
    print("Master:", master_path)
    return master



def ask_yes_no(
    prompt: str,
    default: bool = False,
) -> bool:
    suffix = (
        " [Y/n]: "
        if default
        else " [y/N]: "
    )
    answer = input(
        prompt + suffix
    ).strip().lower()

    if not answer:
        return default

    return answer in {
        "y",
        "yes",
        "s",
        "si",
        "sí",
    }


def _validate_raster_date(value: str) -> str:
    text = str(value).strip()
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid raster date {text!r}; expected YYYY-MM-DD."
        ) from exc
    return text


def resolve_raster_dates(
    args,
) -> list[str]:
    if args.no_raster:
        if args.raster_dates:
            raise ValueError(
                "--no-raster cannot be combined with --raster-date."
            )
        return []

    if args.raster_dates:
        values = [_validate_raster_date(value) for value in args.raster_dates]
        return list(dict.fromkeys(values))

    if not ask_yes_no(
        "Generate a 20 m ET raster now?",
        default=False,
    ):
        return []

    value = input(
        "MODIS period start [YYYY-MM-DD]: "
    ).strip()
    if not value:
        return []
    return [_validate_raster_date(value)]


def collect_field_output_paths(workspace) -> dict[str, Path]:
    """Return final field products that exist after field evaluation."""
    candidates = {
        "comparison_scenarios": (
            workspace.diagnostics
            / "field_ridge25_final_scenarios"
            / "field_comparison_scenarios.csv"
        ),
        "scenario_definitions": (
            workspace.diagnostics
            / "field_ridge25_final_scenarios"
            / "field_scenario_definitions.csv"
        ),
        "scenario_metrics": (
            workspace.diagnostics
            / "field_ridge25_final_scenarios"
            / "field_scenario_metrics.csv"
        ),
        "temporal_completeness_sensitivity": (
            workspace.diagnostics
            / "field_ridge25_final_scenarios"
            / "field_temporal_completeness_sensitivity.csv"
        ),
        "aoa_sensitivity": (
            workspace.diagnostics
            / "field_ridge25_aoa_sensitivity"
            / "field_ridge25_with_vs_without_aoa.csv"
        ),
        "oof_pairs": (
            workspace.diagnostics
            / "field_ridge25_oof_exact_overlap"
            / "field_ridge25_oof_pairs.csv"
        ),
        "oof_metrics": (
            workspace.diagnostics
            / "field_ridge25_oof_exact_overlap"
            / "field_ridge25_oof_metrics.csv"
        ),
        "oof_by_station": (
            workspace.diagnostics
            / "field_ridge25_oof_exact_overlap"
            / "field_ridge25_oof_by_station.csv"
        ),
        "field_reference_audit": (
            workspace.diagnostics
            / "field_ridge25_oof_exact_overlap"
            / "field_reference_et_audit.csv"
        ),
        "field_period_candidates": (
            workspace.diagnostics
            / "field_ridge25_oof_exact_overlap"
            / "field_period_candidates.csv"
        ),
        "field_metadata": (
            workspace.diagnostics
            / "field_ridge25_oof_exact_overlap"
            / "metadata.json"
        ),
        "st04_external_diagnostic": (
            workspace.diagnostics
            / "field_ridge25_oof_exact_overlap_external_st04_era5_nearest"
            / "st04_external_fine_nearest_era5_diagnostic.csv"
        ),
    }
    missing = [str(path) for path in candidates.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Final field evaluation did not create the complete output contract:\n"
            + "\n".join(missing)
        )
    return candidates


def print_metrics(
    label: str,
    metrics: dict[str, float],
) -> None:
    print(label)
    print(
        "  n    :",
        metrics["n"],
    )
    print(
        "  R2   :",
        f"{metrics['R2']:.6f}",
    )
    print(
        "  RMSE :",
        f"{metrics['RMSE']:.6f}",
    )
    print(
        "  MAE  :",
        f"{metrics['MAE']:.6f}",
    )
    print(
        "  BIAS :",
        f"{metrics['BIAS']:.6f}",
    )
    print(
        "  KGE  :",
        f"{metrics['KGE']:.6f}",
    )


def main() -> None:
    args = parse_arguments()
    requested_raster_dates = resolve_raster_dates(args)
    configure_period_environment(
        args.start_date,
        args.end_date_exclusive,
    )

    # Period-sensitive et_downscaling imports occur only after
    # the environment above has been configured.
    import ee
    import pandas as pd

    from et_downscaling.config import (
        OUTPUT_PERIOD_LABEL,
        S2_CLEAR_THRESHOLD,
        S2_DAILY_MOSAIC_SORT_PROPERTY,
        S2_PREPROCESSING_VERSION,
        build_satellite_output_filename,
        build_training_output_filename,
    )
    from et_downscaling.aoa_ridge25 import (
        build_unweighted_aoa,
    )
    from et_downscaling.ridge25_overlap_production import (
        download_ridge25_basin,
    )
    from et_downscaling.local_reconciliation import (
        RIDGE25_USABLE_SUPPORT_FRACTION,
    )
    from et_downscaling.modeling import (
        OPTICAL_COVERAGE_THRESHOLD_PCT,
        train_and_validate_ridge25,
    )
    from et_downscaling.run_reporting import (
        save_aoa_artifacts,
        save_core_figures,
        save_model_metadata,
        save_run_tables,
    )
    from et_downscaling.run_provenance import (
        build_run_provenance,
    )
    from et_downscaling.workspace import (
        get_workspace_paths,
        require_portable_inputs,
    )

    project_root = get_project_root()
    validate_imported_package_root(
        project_root
    )
    inputs = require_portable_inputs(
        project_root
    )
    workspace = get_workspace_paths(
        project_root
    ).ensure()
    project_id = resolve_project_id(
        args.project
    )

    print()
    print("=" * 72)
    print(
        "FUNDACION ET - PARSIMONIOUS RIDGE-25 PIPELINE"
    )
    print("=" * 72)
    print(
        "Project root:",
        project_root,
    )
    print(
        "External workspace:",
        workspace.root,
    )
    print(
        "Analysis period:",
        args.start_date,
        "to",
        args.end_date_exclusive,
        "(exclusive)",
    )
    print(
        "Period label:",
        OUTPUT_PERIOD_LABEL,
    )
    print(
        "Canonical local inputs:",
        len(inputs),
    )
    print(
        "Google Drive output:",
        "DISABLED",
    )
    print(
        "Pre-trained model input:",
        "DISABLED",
    )

    print()
    print("=== RAW DATA ===")
    meteorology_arguments = [
        "--ridge25-only",
    ]
    satellite_arguments = [
        "--optical-source",
        "S2",
        "--ridge25-only",
    ]
    if args.refresh_raw:
        meteorology_arguments.append(
            "--force"
        )
        satellite_arguments.append(
            "--force"
        )

    run_script(
        project_root,
        "export_meteorology_data.py",
        meteorology_arguments,
        project_id=project_id,
    )
    run_script(
        project_root,
        "export_satellite_data.py",
        satellite_arguments,
        project_id=project_id,
    )

    master_path = (
        workspace.master
        / "S2"
        / build_training_output_filename("S2")
    )
    training_sources = {
        "satellite_footprint": (
            workspace.raw_cache
            / "satellite"
            / "S2"
            / build_satellite_output_filename("S2")
        ),
        "era5_hourly": (
            workspace.raw_cache
            / "meteorology"
            / f"era5_hourly_{OUTPUT_PERIOD_LABEL}.csv"
        ),
        "station_support": (
            workspace.raw_cache
            / "meteorology"
            / "station_support.csv"
        ),
    }

    master = build_final_training_master(
        project_root=project_root,
        master_path=master_path,
    )

    print()
    print("=== RIDGE-25 TRAINING / VALIDATION ===")
    result = train_and_validate_ridge25(
        master,
    )

    print(
        "Training population:",
        len(result.population),
    )
    print(
        "Predictors:",
        25,
    )
    print(
        "Model fitted in current run:",
        "YES",
    )
    print(
        "Serialized model loaded:",
        "NO",
    )
    print()
    print_metrics(
        "Spatial block OOF:",
        result.spatial_metrics,
    )
    print()
    print_metrics(
        "Leave-one-year-out:",
        result.temporal_metrics,
    )

    aoa_parameters = build_unweighted_aoa(
        result.population
    )
    print()
    print("=== RIDGE-25 AREA OF APPLICABILITY ===")
    print(
        "AOA threshold:",
        f"{aoa_parameters.threshold:.6f}",
    )
    print(
        "AOA training rows:",
        len(aoa_parameters.training_di),
    )

    run_id = (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + OUTPUT_PERIOD_LABEL
    )
    run_directory = (
        workspace.runs
        / run_id
    )
    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    table_paths = save_run_tables(
        result,
        run_directory,
    )
    aoa_paths = save_aoa_artifacts(
        result.population,
        aoa_parameters,
        run_directory,
    )

    run_metadata = {
        "run_id": run_id,
        "analysis_start": args.start_date,
        "analysis_end_exclusive": args.end_date_exclusive,
        "period_label": OUTPUT_PERIOD_LABEL,
        "master_path": str(master_path),
        "workspace": str(workspace.root),
        "earth_engine_project": project_id,
        "raw_refreshed": bool(args.refresh_raw),
        "sentinel2_cloud_score_clear_threshold": float(S2_CLEAR_THRESHOLD),
        "sentinel2_daily_mosaic_sort_property": S2_DAILY_MOSAIC_SORT_PROPERTY,
        "sentinel2_preprocessing_version": S2_PREPROCESSING_VERSION,
        "training_optical_coverage_threshold_pct": float(
            OPTICAL_COVERAGE_THRESHOLD_PCT
        ),
        "aoa_threshold": float(aoa_parameters.threshold),
        "aoa_definition": (
            "equal-weight standardized Euclidean DI adapted from the "
            "Meyer-Pebesma AOA framework; threshold from spatial-block "
            "cross-validation training DI"
        ),
        "usable_support_fraction": float(RIDGE25_USABLE_SUPPORT_FRACTION),
        "reconciliation": "single_global_exact_overlap",
        "google_drive_used": False,
        "earth_engine_persistent_asset_created": False,
        "reconciliation_used_in_training": False,
    }

    figure_paths = {}
    if not args.no_figures:
        figure_paths = save_core_figures(
            result,
            run_directory,
        )

    # Final field comparison is part of the same scientific run.
    field_paths: dict[str, Path] = {}
    if not args.no_field_evaluation:
        print()
        print("=== FINAL FIELD COMPARISON ===")
        field_arguments = [
            "--project",
            project_id,
        ]
        if args.refresh_raw:
            field_arguments.append(
                "--restart"
            )
        run_script(
            project_root,
            "run_field_evaluation.py",
            field_arguments,
        )
        field_paths = collect_field_output_paths(workspace)
    else:
        print()
        print(
            "Field comparison: SKIPPED"
        )

    raster_products: dict[str, dict[str, object]] = {}
    if requested_raster_dates:
        print()
        print("=== 20 M ET PRODUCTION ===")
        print(
            "Requested MODIS periods:",
            ", ".join(requested_raster_dates),
        )
        print(
            "Initializing Earth Engine once for multi-period production..."
        )
        ee.Initialize(
            project=project_id
        )
        ee.Number(1).getInfo()

        for raster_date in requested_raster_dates:
            print()
            print("-" * 72)
            print("Producing MODIS period:", raster_date)
            print("-" * 72)
            raster_products[raster_date] = download_ridge25_basin(
                project_root=project_root,
                period_start=raster_date,
                model=result.model,
                aoa_parameters=aoa_parameters,
                tile_size_m=args.tile_size_m,
                min_tile_size_m=(
                    args.min_tile_size_m
                ),
            )
    else:
        print()
        print(
            "Raster generation: SKIPPED"
        )

    output_paths = {
        **{f"table:{key}": value for key, value in table_paths.items()},
        **{f"aoa:{key}": value for key, value in aoa_paths.items()},
        **{f"figure:{key}": value for key, value in figure_paths.items()},
        **{f"field:{key}": value for key, value in field_paths.items()},
    }
    production_output_metadata: dict[str, dict[str, str]] = {}
    for period, product in raster_products.items():
        production_output_metadata[period] = {
            "raster": str(product["raster"]),
            "tile_manifest": str(product["manifest"]),
            "production_metadata": str(product["metadata"]),
            "modis_raster": str(product["modis_raster"]),
            "modis_metadata": str(product["modis_metadata"]),
        }
        output_paths[f"raster:{period}:scientific"] = Path(product["raster"])
        output_paths[f"raster:{period}:manifest"] = Path(product["manifest"])
        output_paths[f"raster:{period}:metadata"] = Path(product["metadata"])
        output_paths[f"raster:{period}:modis_native"] = Path(product["modis_raster"])
        output_paths[f"raster:{period}:modis_metadata"] = Path(product["modis_metadata"])

    run_metadata["field_evaluation"] = {
        "executed": not args.no_field_evaluation,
        "primary_valid_day_rule": "at_least_5_of_8_days",
        "complete_case_sensitivity": "8_of_8_days",
        "scenario_counts_expected_from_frozen_workflow": [21, 17, 11, 9],
        "outputs": {name: str(path) for name, path in field_paths.items()},
    }
    run_metadata["raster_periods"] = list(requested_raster_dates)
    run_metadata["production_outputs"] = production_output_metadata
    run_metadata["provenance"] = build_run_provenance(
        project_root=project_root,
        canonical_inputs=inputs,
        training_sources=training_sources,
        master_path=master_path,
        output_paths=output_paths,
    )
    metadata_path = save_model_metadata(
        result,
        run_directory,
        run_metadata,
    )

    print()
    print("=== CURRENT-RUN OUTPUTS ===")
    print(
        "Run directory:",
        run_directory,
    )
    print(
        "Metadata:",
        metadata_path,
    )
    print(
        "Tables:",
        len(table_paths),
    )
    print(
        "AOA artifacts:",
        len(aoa_paths),
    )
    print(
        "Core figures:",
        len(figure_paths),
    )
    print(
        "Field outputs hashed:",
        len(field_paths),
    )
    print(
        "Raster periods:",
        len(raster_products),
    )

    print()
    print("=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print(
        "Model source:",
        "fitted in current run",
    )
    for period, product in raster_products.items():
        print()
        print("Period:", period)
        print("  Raster:", product["raster"])
        print("  Tile manifest:", product["manifest"])
        print("  Production metadata:", product["metadata"])
        print("  Native MODIS ET:", product["modis_raster"])
        print("  Native MODIS metadata:", product["modis_metadata"])
    print(
        "Google Drive used:",
        "NO",
    )
    print(
        "Persistent Earth Engine asset created:",
        "NO",
    )


if __name__ == "__main__":
    main()
