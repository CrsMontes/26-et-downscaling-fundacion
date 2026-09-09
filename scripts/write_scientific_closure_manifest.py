"""Write the final scientific closure manifest for Virtual Station V5."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from et_downscaling.run_provenance import (
    repository_state,
    sha256_file,
)


SCIENTIFIC_TAG = "virtual-station-v5-stable-v1"

FINAL_PERIODS = (
    "2020-03-13",
    "2021-11-25",
    "2022-03-30",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write the Virtual Station V5 scientific closure manifest. "
            "The repository must be clean."
        )
    )
    parser.add_argument(
        "--workspace-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def git_tag_commit(project_root: Path, tag: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{tag}^{{}}"],
        cwd=project_root,
        text=True,
    ).strip()


def find_single(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one file matching {pattern!r} "
            f"in {root}; found {len(matches)}."
        )
    return matches[0]


def relative_file_record(
    path: Path,
    *,
    base_root: Path,
) -> dict[str, object]:
    """Return portable size/SHA-256 provenance for one file."""
    path = Path(path).resolve()
    base_root = Path(base_root).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    return {
        "path": path.relative_to(base_root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def relative_manifest(
    paths: dict[str, Path],
    *,
    base_root: Path,
) -> dict[str, dict[str, object]]:
    """Hash named files while storing paths relative to their root."""
    return {
        name: relative_file_record(
            paths[name],
            base_root=base_root,
        )
        for name in sorted(paths)
    }


def build_closure_manifest(
    *,
    project_root: Path,
    workspace_root: Path,
) -> dict[str, object]:
    project_root = project_root.resolve()
    workspace_root = workspace_root.resolve()

    repo = repository_state(project_root)

    if not repo["git_available"]:
        raise RuntimeError("Git repository state could not be determined.")

    if repo["dirty"]:
        raise RuntimeError(
            "Scientific closure requires a clean Git working tree."
        )

    environment_lock = project_root / "environment-lock.yml"

    tabular_paths = {
        "selected_supports": (
            workspace_root
            / "training"
            / "selection"
            / "selected_supports.csv"
        ),
        "selection_metadata": (
            workspace_root
            / "training"
            / "selection"
            / "selection_metadata.json"
        ),
        "training_population": (
            workspace_root
            / "evaluation"
            / "results"
            / "virtual10_training_population.csv"
        ),
        "spatial_oof": (
            workspace_root
            / "evaluation"
            / "results"
            / "virtual10_spatial_oof.csv"
        ),
        "temporal_oof": (
            workspace_root
            / "evaluation"
            / "results"
            / "virtual10_temporal_oof.csv"
        ),
        "experiment_metadata_historical": (
            workspace_root
            / "evaluation"
            / "results"
            / "experiment_metadata.json"
        ),
        "model_comparison_metrics": (
            workspace_root
            / "evaluation"
            / "results"
            / "virtual10_vs_stable_metrics.csv"
        ),
        "persistence_baselines_historical": (
            workspace_root
            / "evaluation"
            / "results"
            / "persistence_baselines.csv"
        ),
        "field_metrics": (
            workspace_root
            / "evaluation"
            / "field_comparison"
            / "virtual10_vs_stable_field_metrics.csv"
        ),
    }

    historical_metadata = json.loads(
        tabular_paths["experiment_metadata_historical"].read_text(
            encoding="utf-8"
        )
    )
    frozen_aoa_threshold = float(
        historical_metadata["virtual10_AOA_threshold"]
    )

    raster_paths: dict[str, Path] = {}
    raster_contract: dict[str, object] = {}

    for period in FINAL_PERIODS:
        fine_root = workspace_root / "current" / "rasters" / period
        modis_root = (
            workspace_root
            / "current"
            / "rasters_modis"
            / period
        )

        fine_raster = find_single(
            fine_root,
            f"ET_*_{period}_20m.tif",
        )
        fine_metadata = find_single(
            fine_root,
            "production_metadata_*.json",
        )
        modis_raster = (
            modis_root
            / f"MODIS_ET_{period}_native.tif"
        )
        modis_metadata = find_single(
            modis_root,
            "production_metadata_*.json",
        )

        raster_paths[f"{period}_fine_et"] = fine_raster
        raster_paths[f"{period}_fine_metadata"] = fine_metadata
        raster_paths[f"{period}_native_modis"] = modis_raster
        raster_paths[f"{period}_native_modis_metadata"] = modis_metadata

        metadata = json.loads(
            fine_metadata.read_text(encoding="utf-8")
        )

        raster_contract[period] = {
            "analysis_crs": metadata["analysis_crs"],
            "prediction_scale_m": metadata["prediction_scale_m"],
            "published_basin_pixels": metadata[
                "published_basin_pixels"
            ],
            "conservation_tolerance_mm": metadata[
                "conservation_tolerance_mm"
            ],
            "max_abs_conservation_error_after_floor_mm": metadata[
                "max_abs_conservation_error_after_floor_mm"
            ],
            "conservation_scope": metadata[
                "conservation_scope"
            ],
            "published_raster_conservation": metadata[
                "published_raster_conservation"
            ],
        }

    return {
        "manifest_type": "scientific_closure",
        "scientific_status": "operational_virtual_station",
        "scientific_design": (
            "V5 Basin10 canonical virtual-support design"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository_at_closure": repo,
        "scientific_tag": SCIENTIFIC_TAG,
        "scientific_tag_commit": git_tag_commit(
            project_root,
            SCIENTIFIC_TAG,
        ),
        "historical_provenance_policy": {
            "historical_artifacts_rewritten": False,
            "note": (
                "Historical experimental and migration provenance is "
                "preserved as generated. This closure manifest records "
                "the later operational scientific state."
            ),
        },
        "path_semantics": {
            "environment_lock": "relative_to_repository_root",
            "tabular_artifacts": "relative_to_workspace_root",
            "raster_artifacts": "relative_to_workspace_root",
        },
        "environment_lock": relative_file_record(
            environment_lock,
            base_root=project_root,
        ),
        "model_reconstruction": {
            "serialized_model_required": False,
            "serialized_aoa_required": False,
            "training_population": (
                "evaluation/results/virtual10_training_population.csv"
            ),
            "target_definition": historical_metadata["target_definition"],
            "ridge": {
                "source": "src/et_downscaling/ridge25.py",
                "predictor_count": 25,
                "pipeline": "StandardScaler -> Ridge",
                "alpha": 1.0,
                "fit_intercept": True,
            },
            "aoa": {
                "source": "src/et_downscaling/aoa_ridge25.py",
                "predictor_count": 25,
                "feature_weighting": "equal",
                "standardization": (
                    "training mean and sample standard deviation (ddof=1)"
                ),
                "distance_metric": "euclidean",
                "training_di_reference": (
                    "nearest training observation outside the same "
                    "spatial_block"
                ),
                "normalization": (
                    "mean pairwise Euclidean distance among standardized "
                    "training observations"
                ),
                "threshold_rule": (
                    "min(max(training_di), q3 + 1.5 * IQR)"
                ),
                "prediction_reference": (
                    "complete standardized final training population"
                ),
                "frozen_threshold": frozen_aoa_threshold,
            },
            "note": (
                "The operational Ridge model and equal-weight AOA are "
                "reconstructed deterministically from the frozen training "
                "population by the repository code."
            ),
        },
        "tabular_artifacts": relative_manifest(
            tabular_paths,
            base_root=workspace_root,
        ),
        "raster_artifacts": relative_manifest(
            raster_paths,
            base_root=workspace_root,
        ),
        "raster_contract": raster_contract,
        "interpretation_guardrail": (
            "The 20 m product is a model-based spatial downscaling. "
            "Conservation is enforced on the full reconciled MODIS "
            "support before the publication mask and is not guaranteed "
            "exactly on the final masked published raster. This manifest "
            "does not constitute independent validation of ET at 20 m."
        ),
    }


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    workspace_root = args.workspace_root.resolve()

    output = (
        args.output.resolve()
        if args.output is not None
        else workspace_root / "SCIENTIFIC_CLOSURE_MANIFEST.json"
    )

    manifest = build_closure_manifest(
        project_root=project_root,
        workspace_root=workspace_root,
    )

    output.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Scientific closure manifest written: {output}")


if __name__ == "__main__":
    main()
