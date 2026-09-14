"""Offline checks for the deterministic halo-sidecar migration."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = runpy.run_path(str(ROOT / "scripts/migrate_field_rf25_signatures.py"))


def test_extension_format_matches_existing_production_code():
    source = ast.parse((ROOT / "scripts/produce_st04_rf25_halo.py").read_bytes())
    function = next(node for node in ast.walk(source)
                    if isinstance(node, ast.FunctionDef) and node.name == "build_extension_signature")
    namespace = dict(hashlib=hashlib, METHOD=MIGRATION["METHOD"],
                     original_signature_builder=lambda model, aoa: MIGRATION["PRODUCTION_SIGNATURE"])
    namespace.update(zip(("support_lon", "support_lat", "support_distance_m"), MIGRATION["SUPPORT"]))
    exec(compile(ast.Module(body=[function], type_ignores=[]), "st04_signature_only", "exec"), namespace)
    expected = namespace["build_extension_signature"](None, None)
    actual = MIGRATION["extension_signature"](
        MIGRATION["PRODUCTION_SIGNATURE"], MIGRATION["METHOD"], *MIGRATION["SUPPORT"])
    assert actual == expected
    assert actual != MIGRATION["PRODUCTION_SIGNATURE"]


def test_signature_migration_preserves_metadata_and_legacy():
    original = {"scientific_signature": "old", "published_pixels": 0, "eligible": False,
                "conservation_error": float("nan")}
    encoded = json.dumps(original, sort_keys=True)
    updated = MIGRATION["migrated_metadata"](original, "old", "new")
    assert json.dumps(original, sort_keys=True) == encoded
    assert updated.pop("legacy_scientific_signature") == "old"
    assert updated["scientific_signature"] == "new"
    updated["scientific_signature"] = "old"
    assert json.dumps(updated, sort_keys=True) == encoded


@pytest.mark.parametrize("document", [
    {"scientific_signature": "unexpected"},
    {"scientific_signature": "old", "legacy_scientific_signature": "earlier"},
])
def test_unknown_or_already_migrated_metadata_is_rejected(document):
    with pytest.raises(ValueError):
        MIGRATION["migrated_metadata"](document, "old", "new")


def test_completed_migration_preserves_every_tiff_and_other_metadata():
    path = MIGRATION["PRODUCTS"] / MIGRATION["MANIFEST_NAME"]
    if not path.exists():
        pytest.skip("Local completed migration manifest unavailable")
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["all_tiffs_byte_identical"] is True
    before, after = manifest["tiff_sha256_before"], manifest["tiff_sha256_after"]
    assert before == after
    assert len(before) == 210
    assert sum(Path(name).name.startswith("RF25_halo7_") for name in before) == 70
    assert {p.relative_to(ROOT).as_posix() for p in MIGRATION["PRODUCTS"].rglob("*.tif")} == set(before)
    for name, digest in before.items():
        assert MIGRATION["sha256"](ROOT / name) == digest
    assert len(manifest["metadata_files"]) == 142
    assert {p.relative_to(ROOT).as_posix() for p in MIGRATION["PRODUCTS"].rglob("*.json")
            if p != path} == {entry["metadata_file"] for entry in manifest["metadata_files"]}
    for entry in manifest["metadata_files"]:
        metadata_path = ROOT / entry["metadata_file"]
        assert MIGRATION["sha256"](metadata_path) == entry["metadata_sha256_after"]
        current = json.loads(metadata_path.read_text())
        assert current.pop("legacy_scientific_signature") == entry["old_signature"]
        assert current["scientific_signature"] == entry["new_signature"]
        is_extension = any("RF25_halo7_ST04_" in name or "ST04_halo7.tif" in name
                           for name in entry["associated_rasters"])
        expected = manifest["st04_extension_signature"] if is_extension else MIGRATION["PRODUCTION_SIGNATURE"]
        assert entry["new_signature"] == expected
        original = entry["metadata_before"]
        if "scientific_signature" in original:
            current["scientific_signature"] = original["scientific_signature"]
        else:
            current.pop("scientific_signature")
            assert current.pop("canonical_production_scientific_signature") == MIGRATION["PRODUCTION_SIGNATURE"]
        assert json.dumps(current, sort_keys=True) == json.dumps(original, sort_keys=True)
        assert all(name in before for name in entry["associated_rasters"])
