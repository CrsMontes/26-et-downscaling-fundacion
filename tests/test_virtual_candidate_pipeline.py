from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from et_downscaling.candidate_context import candidate_expected_rows
from et_downscaling.stations import get_station_geojson_path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_virtual_candidate_row_count(monkeypatch):
    monkeypatch.setenv("ET_CANDIDATE_SUPPORT_COUNT", "10")
    assert candidate_expected_rows() == 2300


def test_station_geojson_can_be_overridden_for_virtual_candidate_extraction(tmp_path, monkeypatch):
    support_path = tmp_path / "virtual_points.geojson"
    support_path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    monkeypatch.setenv("ET_STATIONS_GEOJSON", str(support_path))
    assert get_station_geojson_path() == support_path.resolve()


def test_fresh_core_uses_only_rf25_sources_before_training(monkeypatch):
    run_pipeline = load_script("run_pipeline.py")
    calls = []

    def fake_run_script(name, arguments):
        calls.append((name, list(arguments)))

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)
    run_pipeline.run_core("ee-test", ["2020-03-13"])

    names = [name for name, _ in calls]
    assert names == [
        "select_virtual_stations.py",
        "download_rf25_inputs.py",
        "build_rf25_training_population.py",
        "train_rf25.py",
        "produce_rf25_rasters.py",
        "write_rf25_provenance.py",
    ]


def test_extract_force_is_propagated_to_rf25_downloader(monkeypatch):
    run_pipeline = load_script("run_pipeline.py")
    calls = []

    monkeypatch.setattr(run_pipeline, "validate_imported_package_root", lambda root: root)
    monkeypatch.setattr(
        run_pipeline,
        "run_script",
        lambda name, arguments: calls.append((name, list(arguments))),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_pipeline.py", "extract", "--project", "ee-test", "--force"],
    )

    run_pipeline.main()

    assert calls == [
        ("download_rf25_inputs.py", ["--project", "ee-test", "--force"]),
        ("build_rf25_training_population.py", []),
    ]
