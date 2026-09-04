import inspect

from et_downscaling.local_tiles import Tile
from et_downscaling.ridge25_overlap_production import (
    OUTPUT_BANDS,
    RAW_TILE_BANDS,
    RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION,
    _support_tiles,
)


def test_exact_overlap_production_contract():
    assert RIDGE25_EXACT_OVERLAP_PRODUCTION_VERSION == (
        "ridge25_exact_overlap_support90_tol001_v2"
    )
    assert RAW_TILE_BANDS == [
        "Kc_raw",
        "dissimilarity_index",
        "stack_valid",
        "AOA_inside",
        "usable",
        "support_domain",
    ]
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
