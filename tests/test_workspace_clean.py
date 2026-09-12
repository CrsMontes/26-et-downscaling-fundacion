from et_downscaling.workspace import (
    WORKSPACE_ENV_VAR,
    field_validation_input,
    get_workspace_paths,
    require_rf25_inputs,
)


def test_workspace_override_is_external(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "external_workspace"

    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(external))
    paths = get_workspace_paths(repo).ensure()

    assert paths.root == external.resolve()
    assert paths.raw_cache.is_dir()
    assert paths.master.is_dir()
    assert paths.runs.is_dir()
    assert paths.diagnostics.is_dir()
    assert paths.rasters.is_dir()
    assert paths.archive.is_dir()


def test_default_workspace_is_repository_local(monkeypatch, tmp_path):
    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    repo = tmp_path / "26-et-downscaling-fundacion"
    repo.mkdir()

    paths = get_workspace_paths(repo)

    assert paths.root == (repo / "outputs" / "current").resolve()


def test_rf25_inputs_do_not_require_field_data(tmp_path):
    repo = tmp_path / "repo"
    required = {
        "basin": repo / "data" / "boundaries" / "fundacion_basin.geojson",
        "stations": repo / "data" / "stations" / "fundacion_stations.geojson",
    }
    for path in required.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test", encoding="utf-8")

    found = require_rf25_inputs(repo)

    assert found == required
    assert not field_validation_input(repo).exists()


def test_field_validation_path_remains_available(tmp_path):
    repo = tmp_path / "repo"
    expected = repo / "data" / "field" / "field_etgage.csv"

    assert field_validation_input(repo) == expected.resolve()
