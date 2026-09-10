from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "write_scientific_closure_manifest.py"

spec = importlib.util.spec_from_file_location(
    "write_scientific_closure_manifest",
    SCRIPT_PATH,
)
closure = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(closure)


def write_text(path: Path, text: str = "test\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_relative_file_record_is_portable_and_hashed(tmp_path):
    root = tmp_path / "workspace"
    path = write_text(root / "nested" / "artifact.csv", "a,b\n1,2\n")

    record = closure.relative_file_record(
        path,
        base_root=root,
    )

    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    assert record["path"] == "nested/artifact.csv"
    assert record["size_bytes"] == path.stat().st_size
    assert record["sha256"] == expected_hash
    assert str(tmp_path) not in record["path"]


def test_relative_file_record_rejects_path_outside_root(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()

    outside = write_text(
        tmp_path / "outside" / "artifact.csv",
        "x\n",
    )

    with pytest.raises(ValueError):
        closure.relative_file_record(
            outside,
            base_root=root,
        )


def test_closure_rejects_dirty_repository(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    workspace_root = tmp_path / "workspace"
    project_root.mkdir()
    workspace_root.mkdir()

    monkeypatch.setattr(
        closure,
        "repository_state",
        lambda _: {
            "git_available": True,
            "dirty": True,
            "commit": "dirty-commit",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="clean Git working tree",
    ):
        closure.build_closure_manifest(
            project_root=project_root,
            workspace_root=workspace_root,
        )


def test_build_closure_manifest_contract(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    workspace_root = tmp_path / "workspace"

    write_text(
        project_root / "environment-lock.yml",
        "name: et-fundacion\n",
    )

    required_tabular = {
        "training/selection/selected_supports.csv": "id\n1\n",
        "training/selection/selection_metadata.json": "{}\n",
        (
            "evaluation/results/"
            "virtual10_training_population.csv"
        ): "Kc_target\n1.0\n",
        "evaluation/results/virtual10_spatial_oof.csv": "x\n1\n",
        "evaluation/results/virtual10_temporal_oof.csv": "x\n1\n",
        (
            "evaluation/results/"
            "virtual10_vs_stable_metrics.csv"
        ): "x\n1\n",
        "evaluation/results/persistence_baselines.csv": "x\n1\n",
        (
            "evaluation/field_comparison/"
            "virtual10_vs_stable_field_metrics.csv"
        ): "x\n1\n",
        (
            "evaluation/field_comparison/"
            "virtual10_vs_stable_field_pairs.csv"
        ): "station_id,period_start\nST01,2022-01-01\n",
        (
            "evaluation/field_comparison/"
            "metadata.json"
        ): "{}\n",
    }

    for relative_path, contents in required_tabular.items():
        write_text(
            workspace_root / relative_path,
            contents,
        )

    historical_metadata = {
        "virtual10_AOA_threshold": 0.8007328515330622,
        "target_definition": (
            "Kc_target = ET_mm_period / ETo_mm_period"
        ),
    }

    write_text(
        workspace_root
        / "evaluation"
        / "results"
        / "experiment_metadata.json",
        json.dumps(historical_metadata),
    )

    for period in closure.FINAL_PERIODS:
        fine_root = (
            workspace_root
            / "current"
            / "rasters"
            / period
        )
        modis_root = (
            workspace_root
            / "current"
            / "rasters_modis"
            / period
        )

        write_text(
            fine_root / f"ET_test_{period}_20m.tif",
            "fine raster",
        )

        fine_metadata = {
            "analysis_crs": "EPSG:32618",
            "prediction_scale_m": 20,
            "published_basin_pixels": 100,
            "conservation_tolerance_mm": 0.01,
            "max_abs_conservation_error_after_floor_mm": 0.001,
            "conservation_scope": (
                "full_reconciled_modis_support_before_publication_mask"
            ),
            "published_raster_conservation": (
                "not_guaranteed_after_publication_mask"
            ),
        }

        write_text(
            fine_root / "production_metadata_test.json",
            json.dumps(fine_metadata),
        )

        write_text(
            modis_root / f"MODIS_ET_{period}_native.tif",
            "native modis",
        )

        write_text(
            modis_root / "production_metadata_test.json",
            "{}\n",
        )

    monkeypatch.setattr(
        closure,
        "repository_state",
        lambda _: {
            "git_available": True,
            "dirty": False,
            "commit": "closure-commit",
        },
    )

    monkeypatch.setattr(
        closure,
        "git_tag_commit",
        lambda *_: "scientific-tag-commit",
    )

    manifest = closure.build_closure_manifest(
        project_root=project_root,
        workspace_root=workspace_root,
    )

    assert manifest["manifest_type"] == "scientific_closure"
    assert manifest["scientific_status"] == (
        "operational_virtual_station"
    )

    assert manifest["scientific_tag"] == (
        "virtual-station-v5-stable-v1"
    )
    assert manifest["scientific_tag_commit"] == (
        "scientific-tag-commit"
    )

    assert manifest["environment_lock"]["path"] == (
        "environment-lock.yml"
    )

    training = manifest["tabular_artifacts"]["training_population"]
    assert training["path"] == (
        "evaluation/results/virtual10_training_population.csv"
    )
    assert len(training["sha256"]) == 64
    assert str(tmp_path) not in training["path"]

    field_pairs = manifest["tabular_artifacts"]["field_pairs"]
    assert field_pairs["path"] == (
        "evaluation/field_comparison/"
        "virtual10_vs_stable_field_pairs.csv"
    )
    assert len(field_pairs["sha256"]) == 64

    field_metrics = manifest["tabular_artifacts"]["field_metrics"]
    assert field_metrics["path"] == (
        "evaluation/field_comparison/"
        "virtual10_vs_stable_field_metrics.csv"
    )
    assert len(field_metrics["sha256"]) == 64

    field_metadata = manifest[
        "tabular_artifacts"
    ]["field_comparison_metadata"]
    assert field_metadata["path"] == (
        "evaluation/field_comparison/metadata.json"
    )
    assert len(field_metadata["sha256"]) == 64

    ridge = manifest["model_reconstruction"]["ridge"]
    assert ridge["predictor_count"] == 25
    assert ridge["alpha"] == 1.0
    assert ridge["fit_intercept"] is True

    aoa = manifest["model_reconstruction"]["aoa"]
    assert aoa["predictor_count"] == 25
    assert aoa["feature_weighting"] == "equal"
    assert aoa["distance_metric"] == "euclidean"
    assert aoa["frozen_threshold"] == pytest.approx(
        0.8007328515330622
    )

    assert (
        manifest["historical_provenance_policy"][
            "historical_artifacts_rewritten"
        ]
        is False
    )

    for record in manifest["raster_artifacts"].values():
        assert not Path(record["path"]).is_absolute()
        assert str(tmp_path) not in record["path"]
        assert len(record["sha256"]) == 64
