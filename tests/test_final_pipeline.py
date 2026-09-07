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


def test_final_pipeline_default_period_is_five_year_gate():
    project_root = Path(__file__).resolve().parents[1]
    source = (
        project_root
        / "scripts"
        / "run_pipeline.py"
    ).read_text(encoding="utf-8")

    assert 'CANONICAL_START_DATE = "2020-01-01"' in source
    assert 'CANONICAL_END_DATE_EXCLUSIVE = "2025-01-01"' in source


def test_final_pipeline_trains_before_optional_raster():
    project_root = Path(__file__).resolve().parents[1]
    source = (
        project_root
        / "scripts"
        / "run_pipeline.py"
    ).read_text(encoding="utf-8")

    training_index = source.index(
        "result = train_and_validate_ridge25("
    )
    raster_index = source.index(
        "download_ridge25_basin("
    )

    assert training_index < raster_index


def test_final_pipeline_uses_direct_training_master_not_candidate_experiments():
    project_root = Path(__file__).resolve().parents[1]
    source = (
        project_root
        / "scripts"
        / "run_pipeline.py"
    ).read_text(encoding="utf-8")

    assert "build_training_dataset.py" in source
    assert "build_candidate_master.py" not in source
    assert "run_repro_script" not in source


def test_final_pipeline_builds_aoa_before_optional_raster():
    project_root = Path(__file__).resolve().parents[1]
    source = (
        project_root
        / "scripts"
        / "run_pipeline.py"
    ).read_text(encoding="utf-8")

    aoa_index = source.index("aoa_parameters = build_unweighted_aoa(")
    raster_index = source.index("download_ridge25_basin(")
    assert aoa_index < raster_index


def test_final_pipeline_does_not_freeze_query_dependent_reference_results():
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "scripts" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "--skip-reference-check" not in source
    assert "verify_reference_2020_2024" not in source
    assert "verify_reference_aoa" not in source
    assert "canonical_reference_check" not in source


def test_final_pipeline_requests_only_accepted_raw_dependencies():
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "scripts" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert source.count('"--ridge25-only"') >= 3


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


def test_final_pipeline_supports_repeated_raster_dates():
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "scripts" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert 'dest="raster_dates"' in source
    assert 'action="append"' in source
    assert "for raster_date in requested_raster_dates:" in source


def test_final_pipeline_finalizes_provenance_after_raster_production():
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "scripts" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )

    raster_index = source.index(
        "raster_products[raster_date] = download_ridge25_basin("
    )
    provenance_index = source.index(
        'run_metadata["provenance"] = build_run_provenance('
    )
    metadata_index = source.index("metadata_path = save_model_metadata(")

    assert raster_index < provenance_index < metadata_index


def test_final_pipeline_hashes_field_and_final_raster_outputs():
    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "scripts" / "run_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert 'f"field:{key}"' in source
    assert 'f"raster:{period}:scientific"' in source
    assert 'f"raster:{period}:manifest"' in source
    assert 'f"raster:{period}:metadata"' in source


def test_legacy_random_forest_model_spec_is_not_labeled_final_source_of_truth():
    project_root = Path(__file__).resolve().parents[1]
    source = (
        project_root / "src" / "et_downscaling" / "model_spec.py"
    ).read_text(encoding="utf-8")

    assert "Legacy Random-Forest specification" in source
    assert "source of truth for the accepted Ridge-25 model" in source
