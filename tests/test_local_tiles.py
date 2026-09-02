from et_downscaling.local_tiles import (
    PREDICTION_SCALE_M,
    Tile,
    _normalize_tile_size,
)


def test_tile_sizes_are_aligned_to_prediction_grid():
    assert _normalize_tile_size(4000) % PREDICTION_SCALE_M == 0
    assert _normalize_tile_size(4011) % PREDICTION_SCALE_M == 0


def test_tile_pixel_dimensions_are_exact():
    tile = Tile(
        xmin=1000,
        ymin=2000,
        xmax=5000,
        ymax=6000,
        tile_id="test",
    )

    assert tile.width_px == 200
    assert tile.height_px == 200


def test_local_tile_module_does_not_use_drive_or_persistent_assets():
    import inspect
    import et_downscaling.local_tiles as local_tiles

    source = inspect.getsource(local_tiles)

    assert "Export.image.toDrive" not in source
    assert "Export.image.toAsset" not in source
    assert "Export.table.toDrive" not in source
    assert "Export.table.toAsset" not in source
    assert "getDownloadURL" in source

def test_only_resource_errors_trigger_adaptive_split():
    from et_downscaling.local_tiles import _is_splittable_error

    assert _is_splittable_error(
        RuntimeError("User memory limit exceeded.")
    )
    assert _is_splittable_error(
        RuntimeError("Request size too large.")
    )
    assert not _is_splittable_error(
        RuntimeError("Missing Sentinel-2 observations.")
    )

