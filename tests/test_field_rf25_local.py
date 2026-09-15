import numpy as np
import pandas as pd
import pytest

from et_downscaling.field_rf25_local import dependency_closure, solve_closed_component
from et_downscaling.field_validation import (
    apply_validation_scenarios, build_validation_metrics, build_sequential_attrition,
)
from et_downscaling.overlap_reconciliation import OverlapEdges, solve_overlap_reconciliation
from et_downscaling.rf25_local_state import score_local_rf25


def test_closed_local_component_matches_global_solver_and_fixed_halo_does_not():
    edges = OverlapEdges(
        coarse_index=np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32),
        fine_index=np.array([0, 1, 1, 2, 2, 3, 4, 5], dtype=np.int32),
        overlap_area_m2=np.ones(8), represented_coarse=np.ones((1, 4), dtype=bool),
    )
    kc = np.array([[2., 3., 4., 5., 3., 4.]])
    usable = np.ones_like(kc, dtype=bool)
    modis = np.array([[10., 11., 12., 15.]])
    neighbors = {0: [1], 1: [0, 2], 2: [1], 3: []}
    closed = dependency_closure([0], neighbors.__getitem__, lambda p: True)
    assert set(closed) == {0, 1, 2}
    local, fine = solve_closed_component(edges, closed, kc, usable, modis)
    global_result = solve_overlap_reconciliation(kc, usable, modis, edges)
    np.testing.assert_allclose(local.et_final_nonnegative, global_result.et_final_nonnegative[fine], atol=1e-12)
    cropped, _ = solve_closed_component(edges, {0}, kc, usable, modis)
    assert abs(cropped.et_final_nonnegative[0] - local.et_final_nonnegative[0]) > 1e-4
    assert local.max_abs_error_after_nonnegative_mm < 0.01


def test_dependency_closure_keeps_ineligible_boundary_and_stops_there():
    visited = []
    def evaluate(parent):
        visited.append(parent)
        return parent < 2
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    result = dependency_closure([0], graph.__getitem__, evaluate)
    assert result == {0: True, 1: True, 2: False}
    assert visited == [0, 1, 2]


def field_rows():
    rows = []
    # Distinct failures exercise every sequential gate on ST02.
    for i in range(9):
        rows.append({
            "station_id": "ST02", "period_start": pd.Timestamp("2022-03-14") + pd.Timedelta(days=i*8),
            "field_period_valid": i != 0, "field_reference_eto_mm_period": 30. + i if i != 0 else np.nan,
            "NDVI_local_20m": np.nan if i == 1 else .6,
            "ET_MODIS_mm_period": np.nan if i == 2 else 28. + i,
            "predictor_stack_available": i != 3, "inside_AOA": i != 4,
            "parent_MODIS_eligible": i != 5, "reconciliation_successful": i != 6,
            "publication_eligible": i != 7,
            "ET_RF25_mm_period": 29. + i if i in (1, 8) else np.nan,
            "RF25_status": "available" if i in (1,8) else "excluded",
        })
    return pd.DataFrame(rows)


def test_four_sets_available_sample_is_not_restricted_by_ndvi_sensitivity():
    pairs = apply_validation_scenarios(field_rows())
    metrics, by_station = build_validation_metrics(pairs)
    main = metrics.loc[(metrics.scenario == "fixed_kc_main") & (metrics["sample"] == "available_sample")]
    assert main.loc[main.comparison == "MODIS_vs_field_proxy", "n"].item() == 7
    assert main.loc[main.comparison == "RF25_vs_field_proxy", "n"].item() == 2
    common = metrics.loc[(metrics["sample"] == "common_sample") & (metrics.comparison_family == "main_vs_sensitivities")]
    assert common.loc[common.comparison.eq("MODIS_vs_field_proxy"), "n"].eq(6).all()
    assert common.loc[common.comparison.ne("MODIS_vs_field_proxy"), "n"].eq(1).all()
    assert common.groupby("comparison").pair_keys_sha256.nunique().eq(1).all()
    assert set(pairs.scenario) == {"fixed_kc_main", "historical_all_stations", "fao_sensitivity", "ndvi20_all"}
    assert not by_station.empty


def test_attrition_is_nested_and_identifies_each_stage():
    pairs = apply_validation_scenarios(field_rows())
    attrition = build_sequential_attrition(pairs)
    ndvi = attrition.loc[attrition.scenario.eq("ndvi20_all") & attrition.station_id.eq("ST02")].sort_values("stage_order")
    assert ndvi.n_remaining.tolist() == [9,8,7,6,5,4,3,2,1]
    assert ndvi.n_lost_at_stage.tolist() == [0,1,1,1,1,1,1,1,1]
    total = attrition.loc[attrition.scenario.eq("ndvi20_all") & attrition.station_id.eq("ALL")]
    assert total.n_remaining.tolist() == ndvi.n_remaining.tolist()


def test_local_success_does_not_certify_global_publication():
    edges = OverlapEdges(np.array([0, 1]), np.array([0, 1]), np.ones(2), np.ones((1, 2), bool))
    kc, usable, modis = np.array([[1., 0.]]), np.ones((1, 2), bool), np.array([[10., 10.]])
    local, _ = solve_closed_component(edges, {0}, kc, usable, modis)
    assert local.et_final_nonnegative.item() == 10
    with pytest.raises(RuntimeError, match="Non-positive Kc mean"):
        solve_overlap_reconciliation(kc, usable, modis, edges)


def test_unknown_product_is_not_an_observed_stack_failure():
    rows = field_rows().iloc[[-1]].copy()
    rows["predictor_stack_available"] = pd.Series(pd.NA, index=rows.index, dtype="boolean")
    attrition = build_sequential_attrition(apply_validation_scenarios(rows))
    stack = attrition.loc[attrition.stage.eq("RF25_stack_complete")]
    assert stack.n_unknown_at_stage.eq(1).all()
    assert stack.n_failed_known_at_stage.eq(0).all()


def test_local_scoring_calls_frozen_model_and_final_aoa(monkeypatch):
    from types import SimpleNamespace
    from et_downscaling.rf25 import RF25_MODEL_FEATURES
    import et_downscaling.rf25_local_state as module
    calls = []
    class Model:
        def predict(self, frame):
            assert list(frame.columns) == RF25_MODEL_FEATURES
            calls.append(len(frame))
            return np.array([.7, .8])
    def score(matrix, parameters):
        return np.array([.2,.9]), np.array([True,False]), np.array([5,0])
    monkeypatch.setattr(module, "score_rf_weighted_aoa", score)
    cube = np.ones((1, 3, 25))
    cube[0,2,0] = np.nan
    result = score_local_rf25(cube, Model(), SimpleNamespace(feature_names=RF25_MODEL_FEATURES))
    assert calls == [2]
    assert result.stack_valid.tolist() == [[True, True, False]]
    assert result.aoa_inside.tolist() == [[True, False, False]]
    assert result.usable.tolist() == [[True, False, False]]
    assert result.kc_raw[0,0] == .7 and np.isnan(result.kc_raw[0,2])
