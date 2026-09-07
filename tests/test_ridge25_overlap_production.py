import inspect
from dataclasses import replace

import numpy as np

from et_downscaling.aoa_ridge25 import AOAParameters
from et_downscaling.local_tiles import Tile
from et_downscaling.ridge25 import build_ridge25_model
from et_downscaling.ridge25_overlap_production import (
    CONSERVATION_SCOPE,
    OUTPUT_BANDS,
    PUBLISHED_RASTER_CONSERVATION,
    RAW_TILE_BANDS,
    RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION,
    _support_tiles,
    build_production_scientific_signature,
)


def test_exact_overlap_production_contract():
    assert RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION == (
        "ridge25_cs050_ge90_exact_overlap_support90_tol001_v3"
    )
    assert RAW_TILE_BANDS == [
        "Kc_raw",
        "dissimilarity_index",
        "stack_valid",
        "AOA_inside",
        "usable",
        "support_domain",
    ]
    assert CONSERVATION_SCOPE == (
        "full_reconciled_modis_support_before_publication_mask"
    )
    assert PUBLISHED_RASTER_CONSERVATION == (
        "not_guaranteed_after_publication_mask"
    )
    assert OUTPUT_BANDS == [
        "ET_mm_period",
        "Kc_raw",
        "dissimilarity_index",
        "stack_valid",
        "AOA_inside",
        "usable",
        "usable_fraction",
        "coarse_eligible",
        "ET_conservation_error_mm",
    ]


def test_exact_overlap_production_has_no_legacy_reconciliation():
    import et_downscaling.ridge25_overlap_production as production

    source = inspect.getsource(production)
    assert "solve_overlap_reconciliation" in source
    assert "reconcile_local_ridge25" not in source
    assert "coarse_to_fine_nearest" not in source
    assert "aggregate_average_to_grid" not in source


def test_support_tiles_add_halo(monkeypatch, tmp_path):
    tile = Tile(
        xmin=1000.0,
        ymin=2000.0,
        xmax=5000.0,
        ymax=6000.0,
        tile_id="r000_c000",
    )

    monkeypatch.setattr(
        "et_downscaling.ridge25_overlap_production.build_initial_tiles",
        lambda *_args, **_kwargs: ([tile], (1000.0, 2000.0, 5000.0, 6000.0)),
    )

    tiles, support_bounds, basin_bounds = _support_tiles(tmp_path, 4000)
    assert len(tiles) == 9
    assert support_bounds == (-3000.0, -2000.0, 9000.0, 10000.0)
    assert basin_bounds == (1000.0, 2000.0, 5000.0, 6000.0)


def test_scientific_signature_changes_with_fitted_state():
    rng = np.random.default_rng(42)
    predictors = rng.normal(size=(40, 25))
    target = rng.normal(size=40)
    model = build_ridge25_model().fit(predictors, target)
    aoa = AOAParameters(
        feature_names=tuple(f"f{index}" for index in range(25)),
        means=np.zeros(25),
        scales=np.ones(25),
        training_scaled=predictors.copy(),
        mean_training_distance=1.5,
        threshold=0.61,
        training_di=np.zeros(40),
    )

    signature = build_production_scientific_signature(model, aoa)
    changed_aoa = replace(aoa, threshold=0.62)
    changed_signature = build_production_scientific_signature(
        model,
        changed_aoa,
    )

    assert len(signature) == 64
    assert signature != changed_signature
