from pathlib import Path

from et_downscaling.virtual_station import (
    V5_AOA_THRESHOLD,
    V5_DATES,
    resolve_virtual_workspace,
)


def test_virtual_station_frozen_constants():
    assert V5_AOA_THRESHOLD == 0.800732851533
    assert V5_DATES == ("2020-03-13", "2021-11-25", "2022-03-30")


def test_virtual_workspace_default_is_separate(tmp_path):
    repo = tmp_path / "26-et-downscaling-fundacion-virtual-station"
    repo.mkdir()
    resolved = resolve_virtual_workspace(repository_root=repo)
    assert resolved == tmp_path / "ET_fundacion_workspace_virtual_station"


def test_virtual_resolver_does_not_use_old_generic_environment(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("ET_FUNDACION_WORKSPACE", str(tmp_path / "old_mixed"))
    monkeypatch.delenv("ET_FUNDACION_VIRTUAL_WORKSPACE", raising=False)
    resolved = resolve_virtual_workspace(repository_root=repo)
    assert resolved == tmp_path / "ET_fundacion_workspace_virtual_station"


def test_run_pipeline_is_non_destructive_by_default():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert "add_subparsers" in source
    assert "if args.command is None:" in source
    assert "parser.print_help()" in source
    assert "download_ridge25_basin" not in source
    assert "ee.Initialize" not in source
