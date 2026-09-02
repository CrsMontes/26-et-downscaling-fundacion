import inspect

from et_downscaling import ridge25_spatial


def test_ridge25_reconciliation_has_no_sentinel1_argument():
    parameters = inspect.signature(
        ridge25_spatial.build_ridge25_constrained_et
    ).parameters

    assert "s1_predictors" not in parameters


def test_ridge25_product_accepts_in_memory_model():
    parameters = inspect.signature(
        ridge25_spatial.build_ridge25_product
    ).parameters

    assert "model" in parameters
    assert "model_path" not in parameters


def test_ridge25_spatial_source_has_no_s1_or_chirps_dependency():
    source = inspect.getsource(ridge25_spatial)

    assert "get_sentinel1_collection" not in source
    assert "CHIRPS_COLLECTION_ID" not in source
    assert "joblib.load" not in source


def test_reconciliation_is_three_pass_production_only():
    source = inspect.getsource(
        ridge25_spatial.build_ridge25_constrained_et
    )

    assert "MODIS_RECONCILIATION_PASSES" in source
    assert "fit(" not in source
