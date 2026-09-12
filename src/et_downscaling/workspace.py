"""Repository-local generated-data workspace for ET Fundación.

All generated data are written below ``outputs/`` in this repository and are
ignored by Git. Portable scientific inputs remain tracked under ``data/``.
``ET_FUNDACION_WORKSPACE`` can still override the operational
``outputs/current`` path for advanced use, but the default is intentionally
self-contained.
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
    models: Path
    figures: Path
    logs: Path

    def ensure(self) -> "WorkspacePaths":
        for path in (
            self.root,
            self.raw_cache,
            self.master,
            self.runs,
            self.diagnostics,
            self.rasters,
            self.archive,
            self.models,
            self.figures,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def get_workspace_paths(project_root: Path) -> WorkspacePaths:
    """Return the operational workspace, repository-local by default."""
    project_root = Path(project_root).resolve()
    override = os.environ.get(WORKSPACE_ENV_VAR, "").strip()
    root = (
        Path(override).expanduser().resolve()
        if override
        else project_root / "outputs" / "current"
    )
    return WorkspacePaths(
        root=root,
        raw_cache=root / "raw",
        master=root / "master",
        runs=root / "runs",
        diagnostics=root / "diagnostics",
        rasters=root / "rasters",
        archive=root / "archive",
        models=root / "models",
        figures=root / "figures",
        logs=root / "logs",
    )


def require_rf25_inputs(project_root: Path) -> dict[str, Path]:
    """Validate the local inputs required by the canonical RF-25 workflow."""
    project_root = Path(project_root).resolve()
    inputs = {
        "basin": project_root / "data" / "boundaries" / "fundacion_basin.geojson",
        "stations": project_root / "data" / "stations" / "fundacion_stations.geojson",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing canonical portable input(s):\n" + "\n".join(missing)
        )
    return inputs


def field_validation_input(project_root: Path) -> Path:
    """Return the separate field-comparison input path without requiring it."""
    return Path(project_root).resolve() / "data" / "field" / "field_etgage.csv"
