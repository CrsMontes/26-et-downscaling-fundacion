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
        "product = download_ridge25_basin("
    )

    assert training_index < raster_index
