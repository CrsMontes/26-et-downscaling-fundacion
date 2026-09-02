from pathlib import Path

from et_downscaling.export import get_output_directory
from et_downscaling.workspace import WORKSPACE_ENV_VAR


def test_earth_engine_download_root_uses_external_workspace(
    monkeypatch,
    tmp_path,
):
    external = tmp_path / "external"
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(external))

    output = get_output_directory()

    assert output == external.resolve()
    assert output.is_dir()


def test_core_scripts_do_not_reference_repository_outputs():
    project_root = Path(__file__).resolve().parents[1]
    paths = [
        project_root / "scripts" / "export_satellite_data.py",
        project_root / "scripts" / "export_meteorology_data.py",
        project_root / "scripts" / "build_training_dataset.py",
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert 'project_root / "outputs"' not in text
