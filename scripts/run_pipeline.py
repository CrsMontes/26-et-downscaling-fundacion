"""Command dispatcher for the ET Fundacion Virtual Station workflow.

The default command is intentionally non-destructive: running this file without
a subcommand only prints help. Expensive or state-changing operations require
an explicit reproduction/production subcommand.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import et_downscaling


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_imported_package_root(
    repository_root: Path,
) -> Path:
    """Fail before execution if et_downscaling comes from another checkout."""
    expected = (
        Path(repository_root)
        / "src"
        / "et_downscaling"
    ).resolve()

    imported = Path(
        et_downscaling.__file__
    ).resolve()

    try:
        imported.relative_to(expected)
    except ValueError:
        raise RuntimeError(
            "The imported et_downscaling package belongs "
            "to a different repository.\n"
            f"Expected: {expected}\n"
            f"Imported: {imported}\n"
            "No pipeline work was started.\n"
            "Set PYTHONPATH to this checkout's src directory "
            "or install this repository in editable mode."
        ) from None

    return imported


def run_script(script_name: str, args: list[str]) -> None:
    command = [
        sys.executable,
        str(project_root() / "scripts" / script_name),
        *args,
    ]
    print(">", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=project_root(), check=True)


def add_workspace_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace-root",
        default=None,
        help=(
            "Virtual Station workspace root. Defaults to sibling "
            "ET_fundacion_workspace_virtual_station."
        ),
    )


def append_workspace(args: list[str], value: str | None) -> None:
    if value:
        args.extend(["--workspace-root", value])


def main() -> None:
    validate_imported_package_root(project_root())

    parser = argparse.ArgumentParser(
        description="ET Fundacion Virtual Station V5 workflow."
    )
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser(
        "validate",
        help="Validate the frozen V5 selection, population, AOA and rasters.",
    )
    add_workspace_argument(validate)

    audit = subparsers.add_parser(
        "audit",
        help="Audit frozen selection versus extracted training periods.",
    )
    add_workspace_argument(audit)
    audit.add_argument("--write-report", action="store_true")

    reproduce_selection = subparsers.add_parser(
        "reproduce-selection",
        help=(
            "Explicitly replay the frozen candidate-order/GE90 selection. "
            "This queries Earth Engine and is not run by default."
        ),
    )
    add_workspace_argument(reproduce_selection)
    reproduce_selection.add_argument("--project", required=True)
    reproduce_selection.add_argument("--seed", type=int, default=42)
    reproduce_selection.add_argument("--min-ge90-per-year", type=int, default=1)

    reproduce_training = subparsers.add_parser(
        "reproduce-training",
        help=(
            "Explicitly rebuild the V5 extraction/training/CV state. "
            "This may query Earth Engine and rewrite training/evaluation outputs."
        ),
    )
    add_workspace_argument(reproduce_training)
    reproduce_training.add_argument("--project", required=True)
    reproduce_training.add_argument("--reference-workspace", default=None)
    reproduce_training.add_argument("--stable-run-dir", default=None)

    evaluate_field = subparsers.add_parser(
        "evaluate-field",
        help="Reproduce the field-derived ET proxy comparison.",
    )
    add_workspace_argument(evaluate_field)
    evaluate_field.add_argument("--project", required=True)
    evaluate_field.add_argument("--reference-workspace", required=True)
    evaluate_field.add_argument("--stable-run-dir", default=None)

    produce = subparsers.add_parser(
        "produce",
        help=(
            "Generate missing V5 20 m rasters for explicitly requested dates. "
            "Existing scientific rasters are never overwritten."
        ),
    )
    add_workspace_argument(produce)
    produce.add_argument("--project", required=True)
    produce.add_argument(
        "--date",
        dest="dates",
        action="append",
        required=True,
        help="MODIS-period start date YYYY-MM-DD. Repeat for multiple dates.",
    )
    produce.add_argument("--tile-size-m", type=int, default=4000)
    produce.add_argument("--min-tile-size-m", type=int, default=500)

    compare = subparsers.add_parser(
        "compare-coverage",
        help="Compare existing V5 and Stable5 rasters without regeneration.",
    )
    add_workspace_argument(compare)
    compare.add_argument("--reference-workspace", required=True)
    compare.add_argument(
        "--date",
        dest="dates",
        action="append",
        default=None,
    )

    summarize = subparsers.add_parser(
        "summarize",
        help="Summarize preserved V5 scientific results locally.",
    )
    add_workspace_argument(summarize)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "validate":
        command_args: list[str] = []
        append_workspace(command_args, args.workspace_root)
        run_script("validate_frozen_state.py", command_args)
        return

    if args.command == "audit":
        command_args = []
        append_workspace(command_args, args.workspace_root)
        if args.write_report:
            command_args.append("--write-report")
        run_script("audit_v5_selection_training_consistency.py", command_args)
        return

    if args.command == "reproduce-selection":
        command_args = [
            "--project",
            args.project,
            "--seed",
            str(args.seed),
            "--n-supports",
            "10",
            "--min-ge90-per-year",
            str(args.min_ge90_per_year),
            "--max-candidates",
            "9123",
        ]
        append_workspace(command_args, args.workspace_root)
        run_script("select_v5_basin_random_sequential_ge90.py", command_args)
        return

    if args.command == "reproduce-training":
        command_args = ["--project", args.project]
        append_workspace(command_args, args.workspace_root)
        if args.reference_workspace:
            command_args.extend(
                ["--reference-workspace", args.reference_workspace]
            )
        if args.stable_run_dir:
            command_args.extend(["--stable-run-dir", args.stable_run_dir])
        run_script("run_v5_basin_experiment.py", command_args)
        return

    if args.command == "evaluate-field":
        command_args = [
            "--project",
            args.project,
            "--reference-workspace",
            args.reference_workspace,
        ]
        append_workspace(command_args, args.workspace_root)
        if args.stable_run_dir:
            command_args.extend(["--stable-run-dir", args.stable_run_dir])
        run_script("evaluate_v5_field_proxy.py", command_args)
        return

    if args.command == "produce":
        command_args = [
            "--project",
            args.project,
            "--tile-size-m",
            str(args.tile_size_m),
            "--min-tile-size-m",
            str(args.min_tile_size_m),
        ]
        append_workspace(command_args, args.workspace_root)
        for date_text in args.dates:
            command_args.extend(["--date", date_text])
        run_script("produce_virtual_rasters.py", command_args)
        return

    if args.command == "compare-coverage":
        command_args = [
            "--reference-workspace",
            args.reference_workspace,
        ]
        append_workspace(command_args, args.workspace_root)
        if args.dates:
            for date_text in args.dates:
                command_args.extend(["--date", date_text])
        run_script("compare_v5_basin_coverage.py", command_args)
        return

    if args.command == "summarize":
        command_args = []
        append_workspace(command_args, args.workspace_root)
        run_script("summarize_results.py", command_args)
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
