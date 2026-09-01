"""Run or resume the accepted Fundación ET workflow.

Raw exports are reused when present. Training, validation, and QA outputs are
reused unless --rebuild-model or --rebuild-all is requested. After the
pipeline is ready, the user may generate one or more 8-day ET products.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import et_downscaling
import pandas as pd

from et_downscaling.aoa import load_aoa_spec
from et_downscaling.model_spec import RF_PARAMETERS
from et_downscaling.production import (
    MODIS_CONSERVATION_TOLERANCE_MM,
    MODIS_RECONCILIATION_PASSES,
    PREDICTION_SCALE_M,
)
from et_downscaling.period import AnalysisPeriod, require_matching_period_metadata


OPTIONAL_OUTPUT_KEYS = {
    "field_checkpoint",
    "field_figure_01",
    "field_figure_02",
    "field_figure_03",
    "field_figure_04",
    "field_figure_05",
    "field_figure_06",
    "field_figure_07",
    "field_figure_08",
    "field_figure_09",
}

QA_OUTPUT_KEYS = {
    "smoke_qa",
    "conservative_qa",
}


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
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date-exclusive", default="2024-01-01")
    return parser.parse_args()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_imported_package_root(project_root: Path) -> Path:
    """Fail before pipeline work if the editable package belongs elsewhere."""
    expected_package_root = (
        Path(project_root).resolve()
        / "src"
        / "et_downscaling"
    ).resolve()
    imported_file_value = getattr(et_downscaling, "__file__", None)

    if not imported_file_value:
        raise RuntimeError(
            "Cannot determine the imported et_downscaling package path.\n"
            f"Expected package root: {expected_package_root}\n"
            "Reinstall this working copy with: python -m pip install -e ."
        )

    imported_file = Path(imported_file_value).resolve()
    try:
        imported_file.relative_to(expected_package_root)
    except ValueError:
        raise RuntimeError(
            "Imported et_downscaling belongs to a different repository.\n"
            f"Expected package root: {expected_package_root}\n"
            f"Actually imported from: {imported_file}\n"
            "No pipeline work was started. Reinstall this working copy with: "
            "python -m pip install -e ."
        ) from None

    return imported_file


def resolve_project_id(project_id: str | None) -> str:
    value = project_id.strip() if project_id else ""
    if not value:
        value = input("Google Cloud Project ID: ").strip()
    if not value:
        raise ValueError("Google Cloud Project ID cannot be empty.")
    return value


def get_paths(project_root: Path, period: AnalysisPeriod) -> dict[str, Path]:
    model_directory = (
        project_root
        / "outputs"
        / "processed"
        / "models"
        / "S2"
        / period.label
    )
    field_tables = (
        project_root
        / "outputs"
        / "processed"
        / "field_validation"
        / period.label
        / "tables"
    )
    field_figures = field_tables.parent / "figures"
    return {
        "model": model_directory / "rf_kc_s2_production_ge90.joblib",
        "common_model": model_directory / "rf_kc_s2_common_ge90.joblib",
        "full_model": model_directory / "rf_kc_s2_full_ge90.joblib",
        "aoa": model_directory / "aoa_spec.json",
        "training_population": (
            model_directory / "kc_model_training_population_ge90.csv"
        ),
        "model_metrics": model_directory / "kc_model_comparison_ge90.csv",
        "fold_metrics": model_directory / "kc_model_spatial_folds_ge90.csv",
        "oof_predictions": (
            model_directory / "kc_model_oof_predictions_ge90.csv"
        ),
        "model_metadata": model_directory / "kc_model_comparison_ge90.json",
        "field_daily_qc": field_tables / "field_daily_qc.csv",
        "field_scale_factor": field_tables / "field_scale_factor.csv",
        "field_reference_eto": field_tables / "field_reference_eto_check.csv",
        "field_pairs_diagnostic": (
            field_tables / "field_modis_period_pairs_diagnostic_reproduction.csv"
        ),
        "field_current_oof": field_tables / "field_current_oof_comparison.csv",
        "field_current_metrics": field_tables / "field_current_oof_metrics.csv",
        "field_uncertainty": field_tables / "field_instrument_uncertainty.csv",
        "field_checkpoint": (
            field_tables / "field_oof_downscaling_checkpoint.csv"
        ),
        "field_pairs_20m": field_tables / "field_oof_downscaled_20m_pairs.csv",
        "field_metrics": field_tables / "field_oof_downscaled_20m_metrics.csv",
        "field_by_station": (
            field_tables / "field_oof_downscaled_20m_by_station.csv"
        ),
        "field_figure_01": field_figures / "FD01_daily_raw_qc.png",
        "field_figure_02": field_figures / "FD02_scale_factor.png",
        "field_figure_03": field_figures / "FD03_reference_eto_check.png",
        "field_figure_04": field_figures / "FD04_field_vs_modis_scatter.png",
        "field_figure_05": field_figures / "FD05_field_vs_modis_series.png",
        "field_figure_06": field_figures / "FD06_current_oof_vs_field.png",
        "field_figure_07": field_figures / "FD07_instrument_uncertainty.png",
        "field_figure_08": field_figures / "FD08_oof_downscaling_vs_field.png",
        "field_figure_09": field_figures / "FD09_oof_downscaled_series.png",
        "smoke_qa": (
            project_root
            / "outputs"
            / "processed"
            / "qa"
            / period.label
            / "spatial_smoke_test.json"
        ),
        "conservative_qa": (
            project_root
            / "outputs"
            / "processed"
            / "qa"
            / period.label
            / "conservative_reconciliation"
            / "conservative_reconciliation.json"
        ),
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
    period: AnalysisPeriod | None = None,
) -> None:
    arguments = arguments or []
    command = [
        sys.executable,
        str(project_root / "scripts" / script_name),
        *arguments,
    ]
    print()
    print(">", " ".join(command))
    environment = os.environ.copy()
    if period is not None:
        environment["ET_START_DATE"] = period.start_date.isoformat()
        environment["ET_END_DATE_EXCLUSIVE"] = period.end_date_exclusive.isoformat()
    subprocess.run(
        command,
        cwd=project_root,
        check=True,
        text=True,
        input=stdin_text,
        env=environment,
    )


def final_outputs_ready(paths: dict[str, Path]) -> bool:
    return all(
        path.is_file()
        for key, path in paths.items()
        if key not in OPTIONAL_OUTPUT_KEYS
    )


def analysis_outputs_ready(paths: dict[str, Path]) -> bool:
    return all(
        path.is_file()
        for key, path in paths.items()
        if key not in OPTIONAL_OUTPUT_KEYS | QA_OUTPUT_KEYS
    )


def build_or_reuse(
    project_root: Path,
    project_id: str,
    args,
    paths: dict[str, Path],
    period: AnalysisPeriod,
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
        period=period,
    )
    run_script(
        project_root,
        "export_satellite_data.py",
        satellite_args,
        stdin_text=project_input,
        period=period,
    )

    rebuild_analysis = (
        args.rebuild_all
        or args.rebuild_model
        or not analysis_outputs_ready(paths)
    )

    if not rebuild_analysis:
        require_matching_period_metadata(paths["model_metadata"], period)
        print()
        print("=== TRAINING / VALIDATION ===")
        print("Existing accepted outputs: REUSED.")
        print(
            "Use --rebuild-model after an intentional model-method change, "
            "or --rebuild-all after an intentional raw-extraction change."
        )
    else:
        print()
        print("=== TRAINING / VALIDATION ===")
        run_script(
            project_root,
            "build_training_dataset.py",
            ["--optical-source", "S2"],
            period=period,
        )
        run_script(project_root, "train_s2_kc_models.py", period=period)
        run_script(project_root, "analyze_field_diagnostics.py", period=period)

        validation_args = ["--project", project_id]
        if args.rebuild_all:
            validation_args.append("--restart")

        run_script(
            project_root,
            "validate_field_downscaling.py",
            validation_args,
            period=period,
        )

    rebuild_smoke = (
        args.rebuild_all
        or args.rebuild_model
        or not paths["smoke_qa"].is_file()
    )
    if rebuild_smoke:
        print()
        print("=== SPATIAL PRODUCTION QA ===")
        run_script(
            project_root,
            "smoke_test_spatial_prediction.py",
            ["--project", project_id],
            period=period,
        )
    else:
        require_matching_period_metadata(paths["smoke_qa"], period)
        print("Existing spatial smoke-test QA: REUSED.")

    rebuild_conservative_qa = (
        args.rebuild_all
        or args.rebuild_model
        or not paths["conservative_qa"].is_file()
    )
    if rebuild_conservative_qa:
        print()
        print("=== CONSERVATIVE RECONCILIATION QA ===")
        run_script(
            project_root,
            "qa_conservative_reconciliation.py",
            ["--project", project_id],
            period=period,
        )
    else:
        require_matching_period_metadata(paths["conservative_qa"], period)
        print("Existing conservative-reconciliation QA: REUSED.")


def print_summary(paths: dict[str, Path]) -> None:
    if not final_outputs_ready(paths):
        raise RuntimeError(
            "The pipeline did not produce all required final outputs."
        )

    metrics = pd.read_csv(paths["model_metrics"])
    aoa = load_aoa_spec(paths["aoa"])
    field = pd.read_csv(paths["field_metrics"])
    with paths["smoke_qa"].open("r", encoding="utf-8") as file:
        smoke_qa = json.load(file)
    if smoke_qa.get("status") != "PASS":
        raise RuntimeError("Spatial smoke-test QA record is not PASS.")
    with paths["conservative_qa"].open("r", encoding="utf-8") as file:
        conservative_qa = json.load(file)
    if conservative_qa.get("status") != "PASS":
        raise RuntimeError("Conservative-reconciliation QA record is not PASS.")

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
    print("Smoke test:", smoke_qa["status"])
    print("Smoke period:", smoke_qa["period_start"])
    print("Smoke conservation error (mm/period):", smoke_qa["conservation_error_mm"])
    print(
        "Multi-parent conservation parents:",
        conservative_qa["multi_parent_conservation"]["eligible_parent_count"],
    )
    print(
        "Multi-parent maximum error (mm/period):",
        conservative_qa["multi_parent_conservation"]["maximum_abs_error_mm"],
    )
    print(
        "Fine-fill QA:",
        conservative_qa["fine_fill"]["status"],
    )


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
    period: AnalysisPeriod,
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
                period=period,
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
    period = AnalysisPeriod.from_strings(args.start_date, args.end_date_exclusive)
    project_root = get_project_root()
    validate_imported_package_root(project_root)
    require_canonical_inputs(project_root)
    project_id = resolve_project_id(args.project)
    paths = get_paths(project_root, period)

    print()
    print("=" * 72)
    print("FUNDACION ET DOWNSCALING PIPELINE")
    print("=" * 72)
    print("Project root:", project_root)
    print("Earth Engine project:", project_id)
    print("Analysis period:", period.start_date, "to", period.end_date_exclusive, "(exclusive)")
    print("Period label:", period.label)

    build_or_reuse(
        project_root,
        project_id,
        args,
        paths,
        period,
    )
    print_summary(paths)

    if not args.no_map:
        map_loop(
            project_root,
            project_id,
            args.drive_folder,
            period,
        )

    print()
    print("Pipeline finished.")


if __name__ == "__main__":
    main()
