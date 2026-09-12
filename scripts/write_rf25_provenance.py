"""Write a compact provenance manifest for one completed RF-25 run."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from et_downscaling.config import build_training_output_filename
from et_downscaling.rf25 import (
    RF25_AOA_FILENAME,
    RF25_METADATA_FILENAME,
    RF25_MODEL_FEATURES,
    RF25_MODEL_FILENAME,
    RF25_PARAMETERS,
)
from et_downscaling.run_provenance import build_run_provenance
from et_downscaling.workspace import get_workspace_paths, require_rf25_inputs


DEFAULT_DATES = ["2020-03-13", "2024-07-11", "2022-03-30"]
EXPECTED_POPULATION_ROWS = 1526


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        dest="dates",
        action="append",
        default=None,
        help="Illustrative production/QC date. Repeat for multiple dates.",
    )
    return parser.parse_args()


def resolve_product_path(value: str, metadata_directory: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    fallback = metadata_directory / path.name
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(path)


def relative_record_paths(value: object, root: Path) -> None:
    """Replace absolute paths in file records when they are below the repository."""
    if isinstance(value, dict):
        if set(("path", "size_bytes", "sha256")).issubset(value):
            path = Path(str(value["path"]))
            try:
                value["path"] = path.relative_to(root).as_posix()
            except ValueError:
                pass
        for child in value.values():
            relative_record_paths(child, root)
    elif isinstance(value, list):
        for child in value:
            relative_record_paths(child, root)


def main() -> None:
    args = parse_args()
    root = project_root()
    workspace = get_workspace_paths(root).ensure()
    results = root / "outputs" / "evaluation" / "results"
    selection = root / "outputs" / "training" / "selection"
    dates = list(dict.fromkeys(args.dates or DEFAULT_DATES))

    source_master = workspace.master / "S2" / build_training_output_filename("S2")
    population = results / "virtual10_training_population.csv"
    model_metadata_path = workspace.models / RF25_METADATA_FILENAME
    model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))

    selected_supports_path = selection / "selected_supports.csv"
    training_sources: dict[str, Path] = {
        "rf25_source_master": source_master,
        "selected_supports": selected_supports_path,
        "virtual_points": selection / "virtual_points.geojson",
    }
    with selected_supports_path.open("r", encoding="utf-8", newline="") as handle:
        selected_rows = list(csv.DictReader(handle))
    if len(selected_rows) != 10:
        raise RuntimeError(
            "Expected exactly 10 selected Virtual10 supports; "
            f"found {len(selected_rows)}."
        )
    availability_checks = []
    for row in selected_rows:
        path = selection / "availability_checks" / (
            f"{int(row['candidate_order']):05d}_{int(row['modis_pixel_id'])}.csv"
        )
        availability_checks.append(path)
    for index, path in enumerate(availability_checks, start=1):
        training_sources[f"availability_check_{index:02d}"] = path

    output_paths: dict[str, Path] = {
        "training_population": population,
        "training_population_metadata": results / "training_population_metadata.json",
        "rf25_metrics": results / "rf25_metrics.csv",
        "spatial_oof": results / "rf25_spatial_oof.csv",
        "temporal_oof": results / "rf25_temporal_oof.csv",
        "aoa_weights": results / "rf25_aoa_weights.csv",
        "training_aoa": results / "rf25_training_aoa.csv",
        "rf25_model": workspace.models / RF25_MODEL_FILENAME,
        "rf25_aoa": workspace.models / RF25_AOA_FILENAME,
        "rf25_model_metadata": model_metadata_path,
    }
    production_configuration: dict[str, dict[str, object]] = {}
    for date_text in dates:
        directory = workspace.rasters / date_text
        metadata_files = sorted(directory.glob("production_metadata_rf25_*.json"))
        if len(metadata_files) != 1:
            raise RuntimeError(
                f"{date_text}: expected exactly one RF-25 production metadata file; "
                f"found {len(metadata_files)}."
            )
        metadata_path = metadata_files[0]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        output_paths[f"{date_text}_production_metadata"] = metadata_path
        output_paths[f"{date_text}_raster"] = resolve_product_path(
            metadata["raster"], directory
        )
        modis_directory = workspace.root / "rasters_modis" / date_text
        output_paths[f"{date_text}_modis_raster"] = resolve_product_path(
            metadata["modis_raster"], modis_directory
        )
        output_paths[f"{date_text}_modis_metadata"] = resolve_product_path(
            metadata["modis_metadata"], modis_directory
        )
        output_paths[f"{date_text}_tile_manifest"] = resolve_product_path(
            metadata["tile_manifest"], directory
        )
        production_configuration[date_text] = {
            "role": "illustrative_cartographic_qc_example",
            "production_method_version": metadata["production_method_version"],
            "scientific_signature": metadata["scientific_signature"],
            "prediction_scale_m": metadata["prediction_scale_m"],
            "conservation_tolerance_mm": metadata["conservation_tolerance_mm"],
            "eligible_modis_parents": metadata["eligible_modis_parents"],
        }

    provenance = build_run_provenance(
        project_root=root,
        canonical_inputs=require_rf25_inputs(root),
        training_sources=training_sources,
        master_path=population,
        output_paths=output_paths,
    )
    relative_record_paths(provenance, root)
    manifest = {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "rf25_virtual10_ge90",
        "scientific_configuration": {
            "target": "Kc_target = MODIS_ET / ETo",
            "expected_population_rows": EXPECTED_POPULATION_ROWS,
            "features": RF25_MODEL_FEATURES,
            "hyperparameters": RF25_PARAMETERS,
            "aoa": model_metadata["aoa"],
            "production_examples": production_configuration,
            "dates_are_climatological_sample": False,
        },
        **provenance,
    }
    manifest_path = workspace.logs / "rf25_run_provenance.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("RF-25 provenance manifest:", manifest_path)


if __name__ == "__main__":
    main()
