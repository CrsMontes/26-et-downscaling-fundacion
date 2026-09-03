"""Run the parsimonious Fundación ET workflow from the three canonical inputs.

Scientific execution
--------------------
1. Reuse or refresh complete raw Earth Engine extractions in the external
   workspace.
2. Rebuild the local complete master database.
3. Rebuild the GE90 Ridge-25 population.
4. Perform spatial-block and leave-one-year-out OOF validation.
5. Fit Ridge-25 in memory on all eligible observations.
6. Save current-run tables, metadata and core diagnostic figures.
7. Optionally generate one locally downloaded, adaptively tiled 20 m ET raster.

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
        default=None,
        help=(
            "Generate one MODIS-period ET raster without an "
            "interactive prompt (YYYY-MM-DD)."
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
        "--skip-reference-check",
        action="store_true",
        help=(
            "Do not enforce the 799-row canonical 2020-2024 "
            "reference gate."
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
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


def resolve_raster_date(
    args,
) -> str | None:
    if args.no_raster:
        return None

    if args.raster_date:
        return args.raster_date

    if not ask_yes_no(
        "Generate a 20 m ET raster now?",
        default=False,
    ):
        return None

    value = input(
        "MODIS period start [YYYY-MM-DD]: "
    ).strip()
    if not value:
        return None
    return value


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
        build_training_output_filename,
        get_optical_output_label,
    )
    from et_downscaling.aoa_ridge25 import (
        build_unweighted_aoa,
    )
    from et_downscaling.ridge25_local_production import (
        download_ridge25_basin,
    )
    from et_downscaling.modeling import (
        train_and_validate_ridge25,
    )
    from et_downscaling.run_reporting import (
        save_core_figures,
        save_model_metadata,
        save_run_tables,
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
    meteorology_arguments = []
    satellite_arguments = [
        "--optical-source",
        "S2",
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

    print()
    print("=== COMPLETE MASTER ===")
    run_script(
        project_root,
        "build_training_dataset.py",
        [
            "--optical-source",
            "S2",
        ],
    )

    optical_label = (
        get_optical_output_label(
            "S2"
        )
    )
    master_path = (
        workspace.master
        / optical_label
        / build_training_output_filename(
            "S2"
        )
    )
    if not master_path.is_file():
        raise FileNotFoundError(
            f"Master was not created: {master_path}"
        )

    master = pd.read_csv(
        master_path,
        dtype={
            "station_id": "string",
        },
    )

    verify_reference = (
        args.start_date
        == CANONICAL_START_DATE
        and args.end_date_exclusive
        == CANONICAL_END_DATE_EXCLUSIVE
        and not args.skip_reference_check
    )

    print()
    print("=== RIDGE-25 TRAINING / VALIDATION ===")
    result = train_and_validate_ridge25(
        master,
        verify_reference_2020_2024=(
            verify_reference
        ),
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

    metadata_path = save_model_metadata(
        result,
        run_directory,
        {
            "run_id": run_id,
            "analysis_start": args.start_date,
            "analysis_end_exclusive": (
                args.end_date_exclusive
            ),
            "period_label": (
                OUTPUT_PERIOD_LABEL
            ),
            "master_path": str(
                master_path
            ),
            "workspace": str(
                workspace.root
            ),
            "earth_engine_project": (
                project_id
            ),
            "raw_refreshed": bool(
                args.refresh_raw
            ),
            "canonical_reference_check": bool(
                verify_reference
            ),
            "google_drive_used": False,
            "earth_engine_persistent_asset_created": False,
            "reconciliation_used_in_training": False,
        },
    )

    figure_paths = {}
    if not args.no_figures:
        figure_paths = save_core_figures(
            result,
            run_directory,
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
        "Core figures:",
        len(figure_paths),
    )

    raster_date = resolve_raster_date(
        args
    )
    if raster_date is None:
        print()
        print(
            "Raster generation: SKIPPED"
        )
        return

    print()
    print("=== 20 M ET PRODUCTION ===")
    print(
        "Requested MODIS period:",
        raster_date,
    )
    print(
        "Initializing Earth Engine for tiled production..."
    )
    ee.Initialize(
        project=project_id
    )
    ee.Number(1).getInfo()

    aoa_parameters = build_unweighted_aoa(
        result.population
    )

    product = download_ridge25_basin(
        project_root=project_root,
        period_start=raster_date,
        model=result.model,
        aoa_parameters=aoa_parameters,
        tile_size_m=args.tile_size_m,
        min_tile_size_m=(
            args.min_tile_size_m
        ),
    )

    print()
    print("=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print(
        "Model source:",
        "fitted in current run",
    )
    print(
        "Raster:",
        product["raster"],
    )
    print(
        "Tile manifest:",
        product["manifest"],
    )
    print(
        "Production metadata:",
        product["metadata"],
    )
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
