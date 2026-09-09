"""Scientific contracts inherited by the Virtual Station workflow."""

from pathlib import Path


def test_final_pipeline_has_no_legacy_rf_or_drive_path():
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "scripts" / "run_pipeline.py"
    source = path.read_text(encoding="utf-8")

    assert "train_s2_kc_models.py" not in source
    assert "run_et_prediction.py" not in source
    assert "--drive-folder" not in source
    assert "joblib.load" not in source
    assert "rf_kc" not in source.lower()


def test_final_pipeline_does_not_freeze_query_dependent_reference_results():
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "scripts" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "--skip-reference-check" not in source
    assert "verify_reference_2020_2024" not in source
    assert "verify_reference_aoa" not in source
    assert "canonical_reference_check" not in source


def test_ridge25_spatial_production_does_not_compute_rejected_fvc_albedo():
    project_root = Path(__file__).resolve().parents[1]
    source = (
        project_root
        / "src"
        / "et_downscaling"
        / "ridge25_production.py"
    ).read_text(encoding="utf-8")

    assert "add_s2_spectral_indices" in source
    assert "build_optical_predictors" not in source
    assert "add_s2_indices" not in source
    assert "add_fvc_band" not in source
    assert "add_s2_albedo" not in source


def test_legacy_random_forest_model_spec_is_not_labeled_final_source_of_truth():
    project_root = Path(__file__).resolve().parents[1]
    source = (
        project_root / "src" / "et_downscaling" / "model_spec.py"
    ).read_text(encoding="utf-8")

    assert "Legacy Random-Forest specification" in source
    assert "source of truth for the accepted Ridge-25 model" in source
