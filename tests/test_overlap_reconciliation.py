import numpy as np
from rasterio.transform import Affine

from et_downscaling.overlap_reconciliation import (
    MODIS_SINUSOIDAL_LOCAL_CRS,
    OverlapEdges,
    build_native_modis_grid,
    materialize_active_values,
    solve_overlap_reconciliation,
)


def test_modis_local_crs_uses_native_sphere():
    crs_text = MODIS_SINUSOIDAL_LOCAL_CRS.to_proj4()
    assert "+proj=sinu" in crs_text
    assert "+R=6371007.181" in crs_text


def test_native_grid_preserves_source_transform_phase():
    projection_info = {
        "crs": "SR-ORG:6974",
        "transform": [
            463.3127165279165,
            0.0,
            -20015109.354,
            0.0,
            -463.3127165279165,
            10007554.677,
        ],
    }

    grid = build_native_modis_grid(
        projection_info=projection_info,
        processing_bounds=(500000.0, 1150000.0, 501000.0, 1151000.0),
        analysis_crs="EPSG:32618",
    )

    native = Affine(*projection_info["transform"])
    relative = ~native * grid.transform

    assert grid.earth_engine_crs == "SR-ORG:6974"
    assert abs(relative.a - 1.0) < 1e-12
    assert abs(relative.e - 1.0) < 1e-12
    assert abs(relative.b) < 1e-12
    assert abs(relative.d) < 1e-12
    assert abs(relative.c - round(relative.c)) < 1e-9
    assert abs(relative.f - round(relative.f)) < 1e-9


def test_overlap_projection_conserves_two_non_nested_parents():
    # Four fine cells. Cell 1 overlaps both coarse parents; therefore it must
    # receive one single reconciled value, not two parent-specific values.
    kc_raw = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)
    usable = np.ones_like(kc_raw, dtype=bool)
    modis_et = np.array([[10.0, 20.0]], dtype=float)

    edges = OverlapEdges(
        coarse_index=np.array([0, 0, 1, 1, 1], dtype=np.int32),
        fine_index=np.array([0, 1, 1, 2, 3], dtype=np.int32),
        overlap_area_m2=np.array([400.0, 200.0, 200.0, 400.0, 400.0]),
        represented_coarse=np.array([[True, True]], dtype=bool),
    )

    result = solve_overlap_reconciliation(
        kc_raw=kc_raw,
        usable=usable,
        modis_et=modis_et,
        edges=edges,
        usable_support_fraction=0.90,
        tolerance_mm=0.01,
    )

    np.testing.assert_allclose(
        result.constraint_matrix @ result.et_final,
        result.target,
        rtol=0,
        atol=1e-10,
    )
    assert result.max_abs_final_error_mm < 1e-10
    assert result.negative_active_cells == 0
    assert result.negative_publishable_cells == 0

    full = materialize_active_values(
        fine_shape=kc_raw.shape,
        active_fine=result.active_fine,
        values=result.et_final_nonnegative,
        selected_active=result.publishable_active,
    )
    assert np.isfinite(full).all()


def test_support_rule_excludes_parent_below_90_percent():
    kc_raw = np.ones((1, 4), dtype=float)
    usable = np.array([[True, True, True, False]], dtype=bool)
    modis_et = np.array([[10.0, 20.0]], dtype=float)

    edges = OverlapEdges(
        coarse_index=np.array([0, 0, 1, 1], dtype=np.int32),
        fine_index=np.array([0, 1, 2, 3], dtype=np.int32),
        overlap_area_m2=np.array([400.0, 400.0, 400.0, 400.0]),
        represented_coarse=np.array([[True, True]], dtype=bool),
    )

    result = solve_overlap_reconciliation(
        kc_raw=kc_raw,
        usable=usable,
        modis_et=modis_et,
        edges=edges,
        usable_support_fraction=0.90,
        tolerance_mm=0.01,
    )

    assert result.eligible_coarse.tolist() == [0]
    assert result.eligible_coarse_mask.tolist() == [[True, False]]
    assert np.isclose(result.usable_fraction[0, 0], 1.0)
    assert np.isclose(result.usable_fraction[0, 1], 0.5)


def test_small_negative_floor_must_still_meet_conservation_tolerance():
    kc_raw = np.array([[1.0, 1.0]], dtype=float)
    usable = np.ones_like(kc_raw, dtype=bool)
    modis_et = np.array([[1.0]], dtype=float)

    # This simple aligned case remains non-negative and therefore validates
    # the post-floor QA path without changing the conserved result.
    edges = OverlapEdges(
        coarse_index=np.array([0, 0], dtype=np.int32),
        fine_index=np.array([0, 1], dtype=np.int32),
        overlap_area_m2=np.array([400.0, 400.0]),
        represented_coarse=np.array([[True]], dtype=bool),
    )

    result = solve_overlap_reconciliation(
        kc_raw=kc_raw,
        usable=usable,
        modis_et=modis_et,
        edges=edges,
        tolerance_mm=0.01,
    )

    assert result.max_abs_error_after_nonnegative_mm <= 0.01
    assert (result.et_final_nonnegative >= 0).all()
