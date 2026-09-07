"""Deterministic file and repository provenance for scientific runs."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Mapping


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a local file."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    """Return a compact path/size/SHA-256 record for one file."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def file_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    """Hash a named collection of files in deterministic key order."""
    return {
        str(name): file_record(Path(paths[name]))
        for name in sorted(paths)
    }


def repository_state(project_root: Path) -> dict[str, object]:
    """Return Git commit, branch and complete porcelain status."""
    project_root = Path(project_root).resolve()

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = git("rev-parse", "HEAD")
        branch = git("branch", "--show-current") or "DETACHED"
        status_text = git("status", "--porcelain=v1", "--untracked-files=all")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "git_available": False,
            "commit": "unknown",
            "branch": "unknown",
            "dirty": None,
            "status_porcelain": [],
        }

    status = [line for line in status_text.splitlines() if line.strip()]
    return {
        "git_available": True,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain": status,
    }


def build_run_provenance(
    *,
    project_root: Path,
    canonical_inputs: Mapping[str, Path],
    training_sources: Mapping[str, Path],
    master_path: Path,
    output_paths: Mapping[str, Path],
) -> dict[str, object]:
    """Build the run provenance block stored in ``run_metadata.json``."""
    project_root = Path(project_root).resolve()
    environment_lock = project_root / "environment-lock.yml"

    return {
        "repository": repository_state(project_root),
        "environment_lock": file_record(environment_lock),
        "canonical_inputs": file_manifest(canonical_inputs),
        "training_sources": file_manifest(training_sources),
        "training_master": file_record(master_path),
        "run_outputs": file_manifest(output_paths),
    }
