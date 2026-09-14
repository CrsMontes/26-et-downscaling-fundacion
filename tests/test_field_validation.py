from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from et_downscaling.field_validation import (
    aggregate_field_periods, apply_scenarios, build_attrition, build_metrics,
    load_field_inputs, modis_periods, ndvi_kc, prepare_field_daily, sample_rf25_rasters,
)

ROOT = Path(__file__).resolve().parents[1]


def inputs(station_id="ST02", n=8):
    _, stations = load_field_inputs(ROOT)
    stations = stations.loc[stations.station_id.eq(station_id)].copy()
    dates = pd.date_range("2022-03-30", periods=n)
    field = pd.DataFrame({"station_id": station_id, "date": dates,
                          "etgage_daily_raw": 0.6, "within_installation_window": True})
    reference = pd.DataFrame({"station_id": station_id, "local_date": dates,
                              "ETo_mm_day": 4.0, "ETr_mm_day": 6.0})
    return field, stations, reference


def test_cm_to_mm_and_daily_etr_to_eto_before_aggregation():
    field, stations, reference = inputs()
    reference.loc[1, "ETr_mm_day"] = 12.0
    daily = prepare_field_daily(field, stations, reference)
    np.testing.assert_allclose(daily.etgage_scaled_mm_day, 6.0)
    assert daily.etgage_eto_equivalent_mm_day.iloc[0] == 4.0
    assert daily.etgage_eto_equivalent_mm_day.iloc[1] == 2.0
    aggregated = aggregate_field_periods(daily, modis_periods(field), stations)
    assert aggregated.field_reference_eto_mm_period.iloc[0] == 30.0
    # Dividing the period total by an average ratio gives a different result.
    assert aggregated.field_reference_eto_mm_period.iloc[0] != pytest.approx(48 / daily.ETr_ETo_ratio.mean())


def test_eto_station_is_not_converted_and_bad_reference_is_rejected():
    field, stations, reference = inputs("ST01")
    daily = prepare_field_daily(field, stations, reference)
    np.testing.assert_allclose(daily.etgage_eto_equivalent_mm_day, 6.0)
    reference.loc[0, "ETo_mm_day"] = 0
    with pytest.raises(ValueError, match="Missing/nonpositive"):
        prepare_field_daily(field, stations, reference)


def test_historical_qc_and_five_day_expansion_and_exclusive_end():
    field, stations, reference = inputs(n=9)
    field.loc[:3, "etgage_daily_raw"] = [0, -0.1, 1.3, np.nan]
    field.loc[8, "within_installation_window"] = False
    daily = prepare_field_daily(field, stations, reference)
    assert daily.field_daily_valid.sum() == 4
    periods = pd.DataFrame({"period_start": [pd.Timestamp("2022-03-30")], "number_days": [8]})
    result = aggregate_field_periods(daily, periods, stations)
    assert result.n_valid_field_days.iloc[0] == 4
    assert np.isnan(result.field_reference_eto_mm_period.iloc[0])
    field.loc[0, "etgage_daily_raw"] = 0.6
    daily = prepare_field_daily(field, stations, reference)
    result = aggregate_field_periods(daily, periods, stations)
    assert result.n_valid_field_days.iloc[0] == 5
    assert result.field_reference_eto_mm_period.iloc[0] == 32.0
    assert result.n_raw_field_days.iloc[0] == 8


def test_modis_year_end_uses_actual_duration():
    field = pd.DataFrame({"date": ["2021-12-31", "2022-01-01"]})
    periods = modis_periods(field)
    assert periods.number_days.tolist() == [5, 8]
    assert periods.period_start.dt.strftime("%Y-%m-%d").tolist() == ["2021-12-27", "2022-01-01"]


def test_ndvi_kc_is_rejected_not_clipped():
    result = ndvi_kc(pd.Series([0.5, 0.0, 2.0, np.nan, np.inf]))
    assert result.iloc[0] == pytest.approx(0.556)
    assert result.iloc[1:].isna().all()


def scenario_base():
    return pd.DataFrame({
        "station_id": ["ST01", "ST02", "ST03", "ST04", "ST05"],
        "period_start": pd.Timestamp("2022-03-30"), "field_period_valid": True,
        "field_reference_eto_mm_period": 40.0,
        "NDVI_local_20m": [0.5, np.nan, 0.6, 0.7, 0.8],
        "ET_MODIS_mm_period": [20.0, 30.0, 40.0, 50.0, np.nan],
        "ET_RF25_mm_period": [22.0, 32.0, np.nan, np.nan, np.nan],
        "RF25_status": ["available", "available", "aoa_excluded", "outside_basin", "missing_raster"],
    })


def test_fixed_kc_application_and_scenario_labels():
    pairs = apply_scenarios(scenario_base())
    historic = pairs.loc[pairs.scenario.eq("historical")].set_index("station_id")
    sensitivity = pairs.loc[pairs.scenario.eq("fao_sensitivity")].set_index("station_id")
    np.testing.assert_allclose(historic.loc[["ST01", "ST02", "ST03"], "ET_field_proxy_mm_period"], [34, 38, 44])
    np.testing.assert_allclose(sensitivity.loc[["ST01", "ST02", "ST03"], "ET_field_proxy_mm_period"], [30, 40, 44])
    assert historic.loc["ST04", "ET_field_proxy_mm_period"] == pytest.approx(40 * (1.457 * 0.7 - 0.1725))


def test_fair_comparisons_use_identical_keys_across_scenarios_and_products():
    pairs = apply_scenarios(scenario_base())
    metrics, stations = build_metrics(pairs)
    common = metrics.loc[metrics.comparison.eq("common_scenarios")]
    assert common.groupby("product").pair_keys_sha256.nunique().eq(1).all()
    assert common.loc[common["product"].eq("MODIS"), "n"].tolist() == [3, 3, 3]
    both = metrics.loc[metrics.comparison.eq("common_scenarios_products")]
    assert both.pair_keys_sha256.nunique() == 1
    assert both.n.eq(1).all()
    assert both.R2.isna().all() and both.KGE.isna().all()
    assert len(stations) == 5 * 3 * 2 * 3
    attrition = build_attrition(pairs).set_index(["scenario", "station_id"])
    assert attrition.loc[("ndvi20_all", "ST02"), "n_missing_ndvi"] == 1
    assert attrition.loc[("historical", "ST03"), "n_proxy_RF25_aoa_excluded"] == 1


def test_duplicate_days_and_wrong_station_cache_fail():
    field, stations, reference = inputs()
    with pytest.raises(ValueError, match="duplicate"):
        prepare_field_daily(pd.concat([field, field.iloc[:1]]), stations, reference)
    reference["station_id"] = "VF01"
    with pytest.raises(ValueError, match="Virtual10"):
        prepare_field_daily(field, stations, reference)


def test_versioned_field_stations_ignore_virtual_override(monkeypatch):
    monkeypatch.setenv("ET_STATIONS_GEOJSON", "does-not-exist.geojson")
    field, stations = load_field_inputs(ROOT)
    assert len(field) == 615
    assert field.groupby("station_id").size().eq(123).all()
    assert stations.station_id.tolist() == ["ST01", "ST02", "ST03", "ST04", "ST05"]


def test_canonical_training_rejects_field_station_master(tmp_path, monkeypatch):
    # Exercise the existing production population guard; do not change training code.
    spec = importlib.util.spec_from_file_location("field_test_training_builder", ROOT / "scripts/build_rf25_training_population.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    master = tmp_path / "master.csv"
    pd.DataFrame({"station_id": ["ST01"], "period_start": ["2022-03-30"], "modis_pixel_id": [123]}).to_csv(master, index=False)
    monkeypatch.setattr(module, "source_master_path", lambda: master)
    monkeypatch.setattr(module, "load_frozen_selection", lambda: (pd.DataFrame({"virtual_id": ["VF01"]}), pd.DataFrame()))
    with pytest.raises(RuntimeError, match="not the Virtual10 extraction"):
        module.main()


def test_sample_only_completed_published_rf25_and_record_aoa(
    tmp_path, monkeypatch
):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform

    import et_downscaling.field_rf25_products as products
    from et_downscaling.rf25_overlap_production import (
        RF25_EXACT_OVERLAP_PRODUCTION_VERSION as version,
    )

    directory = tmp_path / "2022-03-30"
    directory.mkdir()

    path = directory / f"ET_{version}_2022-03-30_20m.tif"

    bands = [
        "ET_mm_period",
        "AOA_inside",
        "stack_valid",
        "coarse_eligible",
    ]
    values = np.array(
        [
            [[24, -9999]],
            [[1, 0]],
            [[1, 1]],
            [[1, 0]],
        ],
        dtype="float32",
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=1,
        count=4,
        dtype="float32",
        crs="EPSG:32618",
        transform=from_origin(500000, 1200000, 20, 20),
        nodata=-9999,
    ) as dst:
        dst.write(values)
        for i, band_name in enumerate(bands, 1):
            dst.set_band_description(i, band_name)

    longitude, latitude = transform(
        "EPSG:32618",
        "EPSG:4326",
        [500010, 500030],
        [1199990, 1199990],
    )

    periods = pd.DataFrame(
        {
            "station_id": ["ST01", "ST02"],
            "longitude": longitude,
            "latitude": latitude,
            "inside_basin": True,
            "period_start": pd.Timestamp("2022-03-30"),
        }
    )

    # Raster exists but production metadata does not.
    sampled, _ = sample_rf25_rasters(periods, tmp_path)
    assert sampled.RF25_status.eq("incomplete_raster").all()

    metadata = {
        "period_start": "2022-03-30",
        "production_method_version": version,
        "max_abs_conservation_error_after_floor_mm": 1e-12,
    }
    metadata_path = (
        directory / f"production_metadata_{version}.json"
    )
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    before = path.read_bytes()

    # Metadata exists but no verified execution evidence exists.
    sampled, _ = sample_rf25_rasters(periods, tmp_path)
    assert sampled.RF25_status.eq("unverified_provenance").all()
    assert sampled.ET_RF25_mm_period.isna().all()

    # This test is about raster sampling semantics, not the evidence
    # verifier. Evidence verification is tested separately in
    # test_field_rf25_products.py.
    evidence_path = directory / "field_execution_evidence.json"
    evidence_path.write_text(
        json.dumps({"protocol": "test-only-placeholder"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        products,
        "product_status",
        lambda raster_root, date, contract: "verified_final_product",
    )

    sampled, provenance = sample_rf25_rasters(
        periods,
        tmp_path,
        {"test": "scientific-contract"},
    )

    assert sampled.RF25_status.tolist() == [
        "available",
        "aoa_excluded",
    ]
    assert sampled.ET_RF25_mm_period.iloc[0] == 24
    assert np.isnan(sampled.ET_RF25_mm_period.iloc[1])
    assert path.read_bytes() == before

    # Raster, metadata and execution-evidence hashes are recorded.
    assert len(provenance) == 3


