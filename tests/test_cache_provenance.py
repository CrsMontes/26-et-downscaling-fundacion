import json

import pytest

from et_downscaling.cache_provenance import (
    build_satellite_provenance,
    satellite_provenance_path,
    validate_satellite_provenance,
    write_satellite_provenance,
)


def test_s2_provenance_freezes_cs050(tmp_path):
    output = tmp_path / "satellite.csv"
    output.write_text("station_id\nST01\n", encoding="utf-8")

    path = write_satellite_provenance(output, "S2")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == build_satellite_provenance("S2")
    assert payload["schema_version"] == 3
    assert payload["s2_clear_threshold"] == 0.50
    assert payload["s2_daily_mosaic_sort_property"] == "system:index"
    assert payload["s2_preprocessing_version"].endswith("deterministic-mosaic-v2")
    assert validate_satellite_provenance(output, "S2") == payload



def test_ridge25_only_provenance_records_s1_not_queried(tmp_path):
    output = tmp_path / "satellite.csv"
    output.write_text("station_id\nST01\n", encoding="utf-8")

    path = write_satellite_provenance(
        output,
        "S2",
        ridge25_only=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["ridge25_only"] is True
    assert payload["sentinel1_queried"] is False
    assert validate_satellite_provenance(
        output,
        "S2",
        ridge25_only=True,
    ) == payload

    with pytest.raises(RuntimeError, match="does not match"):
        validate_satellite_provenance(
            output,
            "S2",
            ridge25_only=False,
        )

def test_missing_satellite_provenance_rejects_cache(tmp_path):
    output = tmp_path / "satellite.csv"
    output.write_text("station_id\nST01\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no provenance sidecar"):
        validate_satellite_provenance(output, "S2")


def test_mismatched_s2_threshold_rejects_cache(tmp_path):
    output = tmp_path / "satellite.csv"
    output.write_text("station_id\nST01\n", encoding="utf-8")
    path = satellite_provenance_path(output)
    payload = build_satellite_provenance("S2")
    payload["s2_clear_threshold"] = 0.60
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match"):
        validate_satellite_provenance(output, "S2")
