from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from et_downscaling.overlap_reconciliation import NativeModisGrid
from et_downscaling import ridge25_overlap_production as production


def test_native_modis_writer_preserves_grid_phase_and_does_not_resample(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        production,
        "_analysis_geometry",
        lambda project_root: mapping(box(100, 100, 900, 900)),
    )

    transform = from_origin(0, 1500, 500, 500)
    grid = NativeModisGrid(
        earth_engine_crs="EPSG:32618",
        local_crs=CRS.from_epsg(32618),
        transform=transform,
        shape=(3, 3),
    )
    modis_et = np.arange(9, dtype=float).reshape(3, 3) + 10.0

    result = production._write_native_modis_basin_raster(
        project_root=tmp_path,
        period_start="2022-03-30",
        modis_et=modis_et,
        modis_grid=grid,
        output_directory=tmp_path / "rasters_modis",
    )

    with rasterio.open(result["raster"]) as src:
        assert src.count == 1
        assert src.descriptions == ("ET_MODIS_mm_period",)
        assert src.crs == CRS.from_epsg(32618)
        assert np.isclose(abs(src.res[0]), 500.0)
        assert np.isclose(abs(src.res[1]), 500.0)
        values = src.read(1, masked=True).compressed()
        assert set(values.tolist()).issubset(set(modis_et.ravel().astype(np.float32).tolist()))
