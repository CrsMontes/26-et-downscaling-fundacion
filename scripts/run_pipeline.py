"""Run or resume the accepted Fundación ET workflow.

Raw exports are reused when present. Training, validation, and QA outputs are
reused unless --rebuild-model or --rebuild-all is requested. After the
pipeline is ready, the user may generate one or more 8-day ET products.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from et_downscaling.aoa import load_aoa_spec
from et_downscaling.model_spec import RF_PARAMETERS
from et_downscaling.production import (
    MODIS_CONSERVATION_TOLERANCE_MM,
    MODIS_RECONCILIATION_PASSES,
    PREDICTION_SCALE_M,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run or resume the complete Fundación ET workflow."
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Google Cloud Project ID with Earth Engine access.",
    )
    parser.add_argument(
        "--rebuild-model",
        action="store_true",
        help="Reuse raw exports but rebuild training, validation, and QA.",
    )
    parser.add_argument(
        "--rebuild-all",
        action="store_true",
        help="Force raw exports and rebuild the complete workflow.",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Do not prompt for an ET map after pipeline QA.",
    )
    parser.add_argument(
        "--drive-folder",
        default="ET_Fundacion",
    )
    return parser.parse_args()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_id(project_id: str | None) -> str:
    value = project_id.strip() if project_id else ""
    if not value:
        value = input("Google Cloud Project ID: ").strip()
    if not value:
        raise ValueError("Google Cloud Project ID cannot be empty.")
    return value


def get_paths(project_root: Path) -> dict[str, Path]:
    model_directory = (
        project_root
        / "outputs"
        / "processed"
        / "models"
        / "S2"
    )
    field_tables = (
        project_root
        / "outputs"
        / "processed"
        / "field_validation"
        / "tables"
    )
    return {
        "model": model_directory / "rf_kc_s2_production_ge90.joblib",
        "aoa": model_directory / "aoa_spec.json",
        "model_metrics": model_directory / "kc_model_comparison_ge90.csv",
        "field_metrics": field_tables / "field_oof_downscaled_20m_metrics.csv",
    }


def require_canonical_inputs(project_root: Path) -> None:
    required = [
        project_root / "data" / "boundaries" / "fundacion_basin.geojson",
        project_root / "data" / "stations" / "fundacion_stations.geojson",
        project_root / "data" / "field" / "field_etgage.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing canonical project inputs:\n" + "\n".join(missing)
        )


def run_script(
    project_root: Path,
    script_name: str,
    arguments: list[str] | None = None,
    stdin_text: str | None = None,
) -> None:
    arguments = arguments or []
    command = [
        sys.executable,
        str(project_root / "scripts" / script_name),
        *arguments,
    ]
    print()
    print(">", " ".join(command))
    subprocess.run(
        command,
        cwd=project_root,
        check=True,
        text=True,
        input=stdin_text,
    )


def final_outputs_ready(paths: dict[str, Path]) -> bool:
    return all(path.is_file() for path in paths.values())


def build_or_reuse(
    project_root: Path,
    project_id: str,
    args,
    paths: dict[str, Path],
) -> None:
    project_input = project_id + "\n"

    print()
    print("=== RAW INPUTS ===")
    meteorology_args = ["--force"] if args.rebuild_all else []
    satellite_args = ["--optical-source", "S2"]
    if args.rebuild_all:
        satellite_args.append("--force")

    # These scripts already reuse their outputs when --force is absent.
    run_script(
        project_root,
        "export_meteorology_data.py",
        meteorology_args,
        stdin_text=project_input,
    )
    run_script(
        project_root,
        "export_satellite_data.py",
        satellite_args,
        stdin_text=project_input,
    )

    rebuild_analysis = (
        args.rebuild_all
        or args.rebuild_model
        or not final_outputs_ready(paths)
    )

    if not rebuild_analysis:
        print()
        print("=== TRAINING / VALIDATION ===")
        print("Existing accepted outputs: REUSED.")
        print(
            "Use --rebuild-model after an intentional model-method change, "
            "or --rebuild-all after an intentional raw-extraction change."
        )
        return

    print()
    print("=== TRAINING / VALIDATION ===")
    run_script(
        project_root,
        "build_training_dataset.py",
        ["--optical-source", "S2"],
    )
    run_script(project_root, "train_s2_kc_models.py")
    run_script(project_root, "analyze_field_diagnostics.py")

    validation_args = ["--project", project_id]
    if args.rebuild_all:
        validation_args.append("--restart")

    run_script(
        project_root,
        "validate_field_downscaling.py",
        validation_args,
    )
    run_script(
        project_root,
        "smoke_test_spatial_prediction.py",
        ["--project", project_id],
    )


def print_summary(paths: dict[str, Path]) -> None:
    if not final_outputs_ready(paths):
        raise RuntimeError(
            "The pipeline did not produce all required final outputs."
        )

    metrics = pd.read_csv(paths["model_metrics"])
    aoa = load_aoa_spec(paths["aoa"])
    field = pd.read_csv(paths["field_metrics"])

    print()
    print("=" * 72)
    print("FINAL PIPELINE SUMMARY")
    print("=" * 72)

    print()
    print("=== MODEL PERFORMANCE: SPATIAL VALIDATION ===")
    print(metrics.to_string(index=False))

    print()
    print("=== AREA OF APPLICABILITY ===")
    print("Training rows:", aoa["training_rows"])
    print("Predictors:", len(aoa["features"]))
    print("Spatial groups:", aoa["spatial_groups"])
    print("DI threshold:", aoa["threshold"])
    print(
        "Training DI outliers:",
        aoa["train_di_summary"]["outlier_count"],
    )

    print()
    print("=== FIELD COMPARISON ===")
    print(field.to_string(index=False))
    print(
        "\nField comparison evaluates local redistribution and is not "
        "independent validation of true ET at 20 m."
    )

    print()
    print("=== PRODUCTION QA ===")
    print("RF trees:", RF_PARAMETERS["n_estimators"])
    print("Production grid (m):", PREDICTION_SCALE_M)
    print("MODIS reconciliation passes:", MODIS_RECONCILIATION_PASSES)
    print(
        "MODIS conservation tolerance (mm/period):",
        MODIS_CONSERVATION_TOLERANCE_MM,
    )
    print("Smoke test: PASS")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(prompt + suffix).strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes", "s", "si", "sí"}


def map_loop(
    project_root: Path,
    project_id: str,
    drive_folder: str,
) -> None:
    """Interactively evaluate or export one or more 8-day ET products."""

    if not ask_yes_no(
        "Generate or evaluate an 8-day ET product?",
        default=False,
    ):
        return

    while True:
        period_start = input(
            "MODIS period start [YYYY-MM-DD] "
            "(press Enter to finish): "
        ).strip()

        if not period_start:
            break

        export = ask_yes_no(
            "Export the 4-band GeoTIFF to Google Drive?",
            default=True,
        )

        arguments = [
            "--project",
            project_id,
            "--period-start",
            period_start,
            "--drive-folder",
            drive_folder,
        ]

        if export:
            arguments.append("--export")

        try:
            run_script(
                project_root,
                "run_et_prediction.py",
                arguments,
            )
        except subprocess.CalledProcessError:
            print()
            print(
                "The requested period failed QA or lacked required inputs. "
                "No silent fallback was applied."
            )

        print()
        print(
            "Enter another MODIS period start, "
            "or press Enter to finish."
        )


def main():
    args = parse_arguments()
    project_root = get_project_root()
    require_canonical_inputs(project_root)
    project_id = resolve_project_id(args.project)
    paths = get_paths(project_root)

    print()
    print("=" * 72)
    print("FUNDACION ET DOWNSCALING PIPELINE")
    print("=" * 72)
    print("Project root:", project_root)
    print("Earth Engine project:", project_id)

    build_or_reuse(
        project_root,
        project_id,
        args,
        paths,
    )
    print_summary(paths)

    if not args.no_map:
        map_loop(
            project_root,
            project_id,
            args.drive_folder,
        )

    print()
    print("Pipeline finished.")


if __name__ == "__main__":
    main()
