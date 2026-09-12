from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from unittest.mock import patch

from et_downscaling.run_provenance import (
    build_run_provenance,
    file_manifest,
    file_record,
    repository_state,
    sha256_file,
)


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "sample.bin"
    payload = b"ET Fundacion provenance\n"
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_file_record_and_manifest_include_size_and_digest(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    record = file_record(first)
    assert record["path"] == str(first.resolve())
    assert record["size_bytes"] == 5
    assert len(record["sha256"]) == 64

    manifest = file_manifest({"second": second, "first": first})
    assert list(manifest) == ["first", "second"]
    assert manifest["first"]["sha256"] == sha256_file(first)


def test_repository_state_matches_current_git_head():
    root = Path(__file__).resolve().parents[1]
    state = repository_state(root)
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()

    assert state["git_available"] is True
    assert state["commit"] == expected
    assert isinstance(state["status_porcelain"], list)
    assert state["dirty"] == bool(state["status_porcelain"])


def test_repository_state_preserves_porcelain_status_columns(tmp_path):
    responses = iter(
        [
            "abc123\r\n",
            "final-rf25-closure\r\n",
            " M modified.txt\r\nM  staged.txt\r\n?? untracked.txt\r\n",
        ]
    )
    with patch(
        "et_downscaling.run_provenance.subprocess.check_output",
        side_effect=lambda *args, **kwargs: next(responses),
    ):
        state = repository_state(tmp_path)

    assert state["commit"] == "abc123"
    assert state["branch"] == "final-rf25-closure"
    assert state["dirty"] is True
    assert state["status_porcelain"] == [
        " M modified.txt",
        "M  staged.txt",
        "?? untracked.txt",
    ]


def test_build_run_provenance_hashes_required_scientific_files(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "environment-lock.yml").write_text("name: test\n", encoding="utf-8")

    canonical = {}
    for name in ("basin", "stations"):
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        canonical[name] = path

    source = tmp_path / "source.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    master = tmp_path / "master.csv"
    master.write_text("y\n2\n", encoding="utf-8")
    output = tmp_path / "output.csv"
    output.write_text("z\n3\n", encoding="utf-8")

    provenance = build_run_provenance(
        project_root=project_root,
        canonical_inputs=canonical,
        training_sources={"source": source},
        master_path=master,
        output_paths={"output": output},
    )

    assert set(provenance["canonical_inputs"]) == {"basin", "stations"}
    assert provenance["training_sources"]["source"]["sha256"] == sha256_file(source)
    assert provenance["training_master"]["sha256"] == sha256_file(master)
    assert provenance["run_outputs"]["output"]["sha256"] == sha256_file(output)
    assert provenance["environment_lock"]["sha256"] == sha256_file(
        project_root / "environment-lock.yml"
    )
