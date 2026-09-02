from et_downscaling.workspace import (
    WORKSPACE_ENV_VAR,
    get_workspace_paths,
    require_portable_inputs,
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


def test_default_workspace_is_sibling(monkeypatch, tmp_path):
    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    repo = tmp_path / "26-et-downscaling-fundacion"
    repo.mkdir()

    paths = get_workspace_paths(repo)

    assert paths.root == (
        tmp_path / "ET_fundacion_workspace" / "current"
    ).resolve()


def test_portable_inputs_are_exactly_three(tmp_path):
    repo = tmp_path / "repo"
    required = {
        "basin": repo / "data" / "boundaries" / "fundacion_basin.geojson",
        "stations": repo / "data" / "stations" / "fundacion_stations.geojson",
        "field": repo / "data" / "field" / "field_etgage.csv",
    }
    for path in required.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test", encoding="utf-8")

    found = require_portable_inputs(repo)

    assert found == required
