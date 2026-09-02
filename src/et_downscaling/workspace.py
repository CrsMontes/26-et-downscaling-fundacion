"""External workspace paths for ET Fundacion.

Generated data are stored outside the Git repository. The repository contains
only source code, documentation, tests, and the three canonical portable inputs.

Set ET_FUNDACION_WORKSPACE to override the default sibling workspace directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


WORKSPACE_ENV_VAR = "ET_FUNDACION_WORKSPACE"


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    raw_cache: Path
    master: Path
    runs: Path
    diagnostics: Path
    rasters: Path
    archive: Path

    def ensure(self) -> "WorkspacePaths":
        for path in (
            self.root,
            self.raw_cache,
            self.master,
            self.runs,
            self.diagnostics,
            self.rasters,
            self.archive,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def get_workspace_paths(project_root: Path) -> WorkspacePaths:
    """Return the external workspace used by the current repository."""
    project_root = Path(project_root).resolve()
    override = os.environ.get(WORKSPACE_ENV_VAR, "").strip()

    if override:
        root = Path(override).expanduser().resolve()
    else:
        root = project_root.parent / "ET_fundacion_workspace" / "current"

    return WorkspacePaths(
        root=root,
        raw_cache=root / "raw",
        master=root / "master",
        runs=root / "runs",
        diagnostics=root / "diagnostics",
        rasters=root / "rasters",
        archive=root / "archive",
    )


def require_portable_inputs(project_root: Path) -> dict[str, Path]:
    """Validate the three canonical local inputs kept inside the repository."""
    project_root = Path(project_root).resolve()
    inputs = {
        "basin": project_root / "data" / "boundaries" / "fundacion_basin.geojson",
        "stations": project_root / "data" / "stations" / "fundacion_stations.geojson",
        "field": project_root / "data" / "field" / "field_etgage.csv",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing canonical portable input(s):\n" + "\n".join(missing)
        )
    return inputs
