import inspect

from et_downscaling.local_tiles import Tile
from et_downscaling.ridge25_local_production import (
    OUTPUT_BANDS,
    RIDGE25_LOCAL_PRODUCTION_VERSION,
    _processing_grid,
)


def test_final_local_production_contract():
    assert (
        RIDGE25_LOCAL_PRODUCTION_VERSION
        == "ridge25_local_aoa_support90_tol001_v1"
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


def test_processing_grid_contains_one_kilometre_buffer():
    tile = Tile(
        xmin=1000.0,
        ymin=2000.0,
        xmax=5000.0,
        ymax=6000.0,
        tile_id="test",
    )

    (
        xmin,
        ymin,
        xmax,
        ymax,
        width,
        height,
        _,
    ) = _processing_grid(tile)

    assert xmin == 0.0
    assert ymin == 1000.0
    assert xmax == 6000.0
    assert ymax == 7000.0
    assert width == 300
    assert height == 300


def test_new_production_does_not_use_legacy_ee_reconciliation():
    import et_downscaling.ridge25_local_production as production

    source = inspect.getsource(
        production
    )

    assert "score_local_ridge25" in source
    assert "reconcile_local_ridge25" in source

    assert (
        "build_ridge25_constrained_et"
        not in source
    )

    assert (
        "build_ee_ridge25_prediction"
        not in source
    )



def test_modis_parent_ownership_uses_pixel_centers():
    from rasterio.transform import from_origin

    from et_downscaling.config import ANALYSIS_CRS
    from et_downscaling.local_tiles import Tile
    from et_downscaling.ridge25_local_production import (
        _modis_parent_owned_by_tile,
    )

    tile = Tile(
        xmin=0.0,
        ymin=0.0,
        xmax=4000.0,
        ymax=4000.0,
        tile_id="ownership_test",
    )

    transform = from_origin(
        -1000.0,
        5000.0,
        1000.0,
        1000.0,
    )

    owned = _modis_parent_owned_by_tile(
        tile=tile,
        modis_shape=(6, 6),
        modis_transform=transform,
        modis_crs=ANALYSIS_CRS,
    )

    assert int(owned.sum()) == 16


def test_basin_production_does_not_use_recursive_subdivision():
    import inspect

    import et_downscaling.ridge25_local_production as production

    source = inspect.getsource(
        production.download_ridge25_basin
    )

    assert "download_tile_adaptive" not in source
    assert "split_tile" not in source
    assert "download_tile(" in source
