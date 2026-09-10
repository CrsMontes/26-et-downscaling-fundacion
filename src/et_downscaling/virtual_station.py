"""Path helpers for the final Virtual10 training-support workflow."""

from __future__ import annotations

import os
from pathlib import Path


VIRTUAL_WORKSPACE_ENV_VAR = "ET_FUNDACION_VIRTUAL_WORKSPACE"


def resolve_virtual_workspace(value=None, repository_root=None) -> Path:
    """Return the generated-data root used by support selection and extraction."""
    override = value or os.environ.get(VIRTUAL_WORKSPACE_ENV_VAR, "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        return path.parent if path.name.lower() == "current" else path
    root = Path(repository_root) if repository_root else Path(__file__).resolve().parents[2]
    return root / "outputs"
