from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from et_downscaling.final_outputs import (
    FINAL_PERIODS,
    build_minimal_final_outputs,
    copy_native_modis_raster,
    derive_et_only_raster,
)


def _write_scientific_raster(path: Path, et: np.ndarray):
    data = np.zeros((9, *et.shape), dtype=np.float32)
    data[0] = et
    data[1:] = 1.0
    profile = {
        "driver": "GTiff",
        "height": et.shape[0],
        "width": et.shape[1],
        "count": 9,
        "dtype": "float32",
        "crs": "EPSG:32618",
        "transform": from_origin(500000, 1200000, 20, 20),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        descriptions = [
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
        for index, description in enumerate(descriptions, start=1):
            dst.set_band_description(index, description)


def _write_modis_raster(path: Path, values: np.ndarray):
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext +no_defs",
        "transform": from_origin(-1000000, 1000000, 463.31271653, 463.31271653),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(np.float32), 1)
        dst.set_band_description(1, "ET_MODIS_mm_period")


def test_et_only_is_exact_band_one_copy(tmp_path):
    source = tmp_path / "scientific.tif"
    destination = tmp_path / "ET.tif"
    et = np.array([[1.0, 2.0], [3.0, -9999.0]], dtype=np.float32)
    _write_scientific_raster(source, et)

    derive_et_only_raster(source, destination)

    with rasterio.open(source) as src, rasterio.open(destination) as dst:
        assert dst.count == 1
        assert dst.descriptions == ("ET_mm_period",)
        assert dst.crs == src.crs
        assert dst.transform == src.transform
        assert dst.nodata == src.nodata
        np.testing.assert_array_equal(dst.read(1), src.read(1))


def test_native_modis_copy_preserves_grid_and_values(tmp_path):
    source = tmp_path / "MODIS_source.tif"
    destination = tmp_path / "MODIS_copy.tif"
    values = np.array([[12.0, 15.0], [20.0, -9999.0]], dtype=np.float32)
    _write_modis_raster(source, values)

    copy_native_modis_raster(source, destination)

    with rasterio.open(source) as src, rasterio.open(destination) as dst:
        assert dst.descriptions == ("ET_MODIS_mm_period",)
        assert dst.crs == src.crs
        assert dst.transform == src.transform
        assert dst.nodata == src.nodata
        np.testing.assert_array_equal(dst.read(1), src.read(1))


def test_minimal_final_outputs_contains_only_accepted_files(tmp_path):
    project_root = tmp_path / "repo"
    workspace_current = tmp_path / "ET_fundacion_workspace" / "current"
    run_dir = workspace_current / "runs" / "20260907T162048Z_2020_2024"
    raster_root = workspace_current / "rasters"
    notebook_dir = project_root / "notebooks"
    notebook_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (notebook_dir / "final_results_visualization.ipynb").write_text(
        "{}\n", encoding="utf-8"
    )

    production_outputs = {}
    for index, period in enumerate(FINAL_PERIODS, start=1):
        period_dir = raster_root / period
        period_dir.mkdir(parents=True)
        raster = period_dir / f"scientific_{period}.tif"
        et = np.array(
            [[index, index + 1], [index + 2, -9999.0]],
            dtype=np.float32,
        )
        _write_scientific_raster(raster, et)
        metadata = period_dir / f"production_{period}.json"
        metadata.write_text(
            json.dumps(
                {
                    "eligible_modis_parents": 10 + index,
                    "publishable_active_fraction_of_active_support": 0.9,
                    "max_abs_conservation_error_after_floor_mm": 0.001,
                }
            ),
            encoding="utf-8",
        )
        modis_dir = workspace_current / "rasters_modis" / period
        modis_dir.mkdir(parents=True)
        modis_raster = modis_dir / f"MODIS_ET_{period}_native.tif"
        _write_modis_raster(
            modis_raster,
            np.array([[10 + index, 11 + index]], dtype=np.float32),
        )
        production_outputs[period] = {
            "raster": str(raster),
            "production_metadata": str(metadata),
            "tile_manifest": str(period_dir / "manifest.csv"),
            "modis_raster": str(modis_raster),
        }

    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "production_outputs": production_outputs,
                "provenance": {
                    "repository": {
                        "commit": "d0ced516d24f42f724081c63c9ab3c8f4c2687a8"
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    final_dir = tmp_path / "ET_fundacion_workspace" / "final"
    outputs = build_minimal_final_outputs(
        workspace_current=workspace_current,
        project_root=project_root,
        output_dir=final_dir,
        run_dir=run_dir,
    )

    assert outputs["source_run"] == run_dir.resolve()
    assert sorted(path.name for path in final_dir.iterdir()) == sorted(
        [
            "ET_2020-03-13_20m.tif",
            "ET_2021-11-25_20m.tif",
            "ET_2022-03-30_20m.tif",
            "final_results_visualization.ipynb",
            "modis",
            "raster_summary.csv",
        ]
    )
    assert sorted(path.name for path in (final_dir / "modis").iterdir()) == sorted(
        [f"MODIS_ET_{period}_native.tif" for period in FINAL_PERIODS]
    )

    summary = np.genfromtxt(
        final_dir / "raster_summary.csv",
        delimiter=",",
        dtype=str,
        encoding="utf-8",
    )
    assert summary.shape[0] == 7  # header + 3 published + 3 common-support rows
