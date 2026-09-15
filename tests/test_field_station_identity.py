"""Physical-identity regressions; no acquisition, fitting, or raster generation."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

from et_downscaling.field_station_identity import (
    OLD_TO_NEW, STATIONS, VERSION, FIXED_KC, NDVI_STATION_IDS, MANGROVE_METHOD,
    attach_station_identity, validate_station_geometry, validate_station_table,
)
from et_downscaling.field_validation import load_field_inputs, prepare_field_daily, apply_scenarios
from et_downscaling.field_review_io import read_review_sheet

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ET_STATION_TEST_ROOT", ROOT))
BACKUP = Path(os.environ.get("ET_STATION_BASELINE", "E:/ET_backup_20260914"))
FIELD = Path("outputs/evaluation/field_validation")


def test_physical_identity_rules_and_crosswalk():
    assert OLD_TO_NEW == {"ST01": "ST02", "ST02": "ST03", "ST03": "ST04", "ST04": "ST01", "ST05": "ST05"}
    assert {key: value["station_uid"] for key, value in STATIONS.items()} == {
        "ST01": "mangrove", "ST02": "clean_pasture", "ST03": "oil_palm", "ST04": "banana", "ST05": "dry_forest"}
    assert FIXED_KC["historical"] == {"ST02": .85, "ST03": .95, "ST04": 1.1}
    assert FIXED_KC["fao_sensitivity"] == {"ST02": .75, "ST03": 1., "ST04": 1.1}
    assert NDVI_STATION_IDS == ("ST01", "ST05")
    assert [key for key, value in STATIONS.items() if value["reference_et"] == "ETo"] == ["ST02"]
    assert [key for key, value in STATIONS.items() if not value["inside_basin"]] == ["ST01"]
    assert [key for key, value in STATIONS.items() if value["coastal_era5_support"]] == ["ST01"]
    assert MANGROVE_METHOD == "st04_validation_extension_era5_nearest_valid_land_pixel_fill_v1"


def test_coordinate_and_attribute_swaps_are_rejected():
    geometry = json.loads((DATA_ROOT / "data/stations/fundacion_stations.geojson").read_text())
    validate_station_geometry(geometry)
    bad = deepcopy(geometry)
    bad["features"][0]["geometry"], bad["features"][1]["geometry"] = bad["features"][1]["geometry"], bad["features"][0]["geometry"]
    with pytest.raises(ValueError, match="coordinates"):
        validate_station_geometry(bad)
    bad = deepcopy(geometry)
    bad["features"][0]["properties"]["reference_et"] = "ETo"
    with pytest.raises(ValueError, match="reference_et"):
        validate_station_geometry(bad)


def test_legacy_and_mixed_namespace_joins_are_rejected():
    legacy = pd.DataFrame({"station_id": ["ST01"], "station": ["Pasture"]})
    with pytest.raises(ValueError, match="requires"):
        validate_station_table(legacy, require_uid=True)
    mixed = pd.DataFrame({"station_id": ["ST01"], "station_uid": ["clean_pasture"], "station_nomenclature_version": [VERSION]})
    with pytest.raises(ValueError, match="mismatch"):
        validate_station_table(mixed, require_uid=True)


def test_current_daily_harmonization_reproduces_physical_baseline():
    if not BACKUP.is_dir():
        pytest.skip("Read-only external checkpoint unavailable")
    baseline = pd.read_csv(BACKUP / FIELD / "field_daily_qc.csv", float_precision="round_trip")
    baseline["station_id"] = baseline.station_id.map(OLD_TO_NEW)
    reference = attach_station_identity(baseline[["station_id", "date", "ETo_mm_day", "ETr_mm_day"]].rename(columns={"date": "local_date"}))
    field, stations = load_field_inputs(DATA_ROOT)
    actual = prepare_field_daily(field, stations, reference)
    for frame in (baseline, actual):
        frame["date"] = pd.to_datetime(frame.date)
    keys = ["station_id", "date"]
    actual = actual.sort_values(keys).reset_index(drop=True)
    baseline = baseline.sort_values(keys).reset_index(drop=True)
    assert actual[keys].equals(baseline[keys])
    for column in ["qc_within_installation", "qc_nonmissing", "qc_positive", "qc_physical_range", "field_daily_valid", "reference_et", "canvas", "inside_basin"]:
        pd.testing.assert_series_equal(actual[column], baseline[column], check_dtype=False)
    for column in ["etgage_scaled_mm_day", "ETr_ETo_ratio", "etgage_eto_equivalent_mm_day"]:
        np.testing.assert_allclose(actual[column], baseline[column], rtol=1e-12, atol=1e-12, equal_nan=True)


def test_current_kc_selection_reproduces_saved_physical_values():
    path = DATA_ROOT / FIELD / "field_period_pairs.csv"
    if not path.is_file():
        pytest.skip("Recovered field pairs unavailable")
    saved = pd.read_csv(path, parse_dates=["period_start"])
    keys = ["station_id", "period_start"]
    base = saved.loc[saved.scenario.eq("historical_all_stations")].copy()
    computed = apply_scenarios(base)
    for scenario, old_name in [("historical", "historical_all_stations"), ("fao_sensitivity", "fao_sensitivity"), ("ndvi20_all", "ndvi20_all")]:
        actual = computed.loc[computed.scenario.eq(scenario)].sort_values(keys)
        expected = saved.loc[saved.scenario.eq(old_name)].sort_values(keys)
        for column in ["Kc_field_proxy", "ET_field_proxy_mm_period"]:
            np.testing.assert_allclose(actual[column], expected[column], rtol=1e-12, atol=1e-12, equal_nan=True)
        assert actual.Kc_source.tolist() == expected.Kc_source.tolist()
    assert base.loc[base.station_id.isin(NDVI_STATION_IDS), "Kc_source"].eq("current_rf25_local_20m_NDVI_proxy").all()
    assert base.loc[~base.station_id.isin(NDVI_STATION_IDS), "Kc_source"].eq("historical_fixed_proxy").all()


def test_workbook_common_sample_and_mangrove_products():
    period = read_review_sheet(DATA_ROOT / FIELD / "field_validation_review.xlsx", "PERIOD")
    columns = ["field_et_historical_mm_period", "modis_et_mm_period", "rf25_et_unreconciled_mm_period", "rf25_et_reconciled_mm_period"]
    common = period.loc[np.isfinite(period[columns]).all(axis=1) & period.publishable.fillna(False)]
    assert common.groupby("station_id").size().to_dict() == {"ST01": 8, "ST02": 5, "ST03": 7, "ST04": 7, "ST05": 10}
    assert common.station_id.isin(FIXED_KC["historical"]).sum() == 19
    assert period.loc[period.station_id.eq("ST01"), "validation_domain"].eq("validation_extension").all()
    assert period.loc[period.station_id.eq("ST04"), "validation_domain"].eq("fundacion_basin").all()
    halos = DATA_ROOT / FIELD / "local_halo7_products"
    assert len(list(halos.rglob("RF25_halo7_*.tif"))) == 70
    for path in halos.glob("*/ST01/metadata.json"):
        metadata = json.loads(path.read_text())
        assert metadata["station_id"] == "ST01"
        assert metadata["station_uid"] == "mangrove"
        assert (metadata["longitude"], metadata["latitude"]) == (-74.360002, 10.766952)
        assert metadata["scientific_signature"] == "ee61c62f352971afabbc6659909b48a0bc3c761ea81bb6c427ee7cd5113938d3"
    assert (DATA_ROOT / FIELD / "st01_validation_extension").is_dir()
    assert not (DATA_ROOT / FIELD / "st04_validation_extension").exists()
