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
    source = ast.parse((ROOT / "scripts/produce_mangrove_rf25_halo.py").read_bytes())
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

def test_final_field_validation_rasters_match_integrity_contract():
    contract_path = (
        ROOT
        / "config"
        / "field_validation_raster_integrity.json"
    )

    contract = json.loads(
        contract_path.read_text(encoding="utf-8")
    )

    expected = contract["tiff_sha256"]

    assert contract["version"] == 1
    assert contract["raster_count"] == 212
    assert contract["groups"]["local_halo7_products"] == 210
    assert contract["groups"]["st01_validation_extension"] == 2
    assert len(expected) == 212

    # There are 70 final RF25 halo products:
    # 14 periods x 5 monitoring stations.
    assert sum(
        Path(name).name.startswith("RF25_halo7_")
        for name in expected
    ) == 70

    for relative_path, expected_hash in expected.items():
        raster_path = ROOT / relative_path

        assert raster_path.exists(), relative_path
        assert MIGRATION["sha256"](raster_path) == expected_hash
