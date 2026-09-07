"""Build the minimal local final-output view from a completed production run.

This module does not fit models, query Earth Engine, reconcile ET, or alter the
scientific multiband products. It only derives one-band ET convenience rasters
from band 1 of the frozen scientific rasters and summarizes their published and
common spatial support.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio


FINAL_PERIODS = (
    "2020-03-13",
    "2021-11-25",
    "2022-03-30",
)


def load_json(path: Path) -> dict:
    """Load a UTF-8 JSON object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_latest_complete_run(
    runs_dir: Path,
    periods: Iterable[str] = FINAL_PERIODS,
) -> tuple[Path, dict]:
    """Return the newest run containing scientific rasters for all periods."""
    runs_dir = Path(runs_dir)
    required = tuple(periods)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {runs_dir}")

    candidates = sorted(
        (path for path in runs_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in candidates:
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = load_json(metadata_path)
        production = metadata.get("production_outputs", {})
        if all(period in production for period in required):
            raster_paths = [
                Path(production[period].get("raster", ""))
                for period in required
            ]
            if all(path.is_file() for path in raster_paths):
                return run_dir, metadata

    raise FileNotFoundError(
        "No completed run contains scientific rasters for all final periods: "
        + ", ".join(required)
    )


def validate_scientific_raster(path: Path) -> None:
    """Validate the frozen scientific-raster contract needed for extraction."""
    path = Path(path)
    with rasterio.open(path) as src:
        if src.count != 9:
            raise ValueError(
                f"Expected 9-band scientific raster, found {src.count}: {path}"
            )
        if src.descriptions[0] != "ET_mm_period":
            raise ValueError(
                "Band 1 must be ET_mm_period in the scientific raster: "
                f"{path}"
            )
        if src.crs is None or src.crs.to_epsg() != 32618:
            raise ValueError(f"Expected EPSG:32618 scientific raster: {path}")
        xres, yres = src.res
        if not np.isclose(xres, 20.0) or not np.isclose(yres, 20.0):
            raise ValueError(f"Expected 20 m scientific raster: {path}")


def derive_et_only_raster(source: Path, destination: Path) -> Path:
    """Copy band 1 exactly into a one-band GeoTIFF without recalculating ET."""
    source = Path(source)
    destination = Path(destination)
    validate_scientific_raster(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source) as src:
        profile = src.profile.copy()
        profile.update(count=1)

        with rasterio.open(destination, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                dst.write(src.read(1, window=window), 1, window=window)
            dst.set_band_description(1, "ET_mm_period")
            dataset_tags = src.tags()
            if dataset_tags:
                dst.update_tags(**dataset_tags)
            band_tags = src.tags(1)
            if band_tags:
                dst.update_tags(1, **band_tags)

    # Verify exact numeric identity after writing.
    with rasterio.open(source) as src, rasterio.open(destination) as dst:
        if src.crs != dst.crs or src.transform != dst.transform:
            raise RuntimeError("ET-only raster georeferencing changed during copy.")
        if src.width != dst.width or src.height != dst.height:
            raise RuntimeError("ET-only raster dimensions changed during copy.")
        if src.nodata != dst.nodata:
            raise RuntimeError("ET-only raster NoData value changed during copy.")
        for _, window in src.block_windows(1):
            if not np.array_equal(
                src.read(1, window=window),
                dst.read(1, window=window),
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"ET-only values differ from scientific band 1: {destination}"
                )

    return destination


def validate_native_modis_raster(path: Path) -> None:
    """Validate the one-band native-grid MODIS ET convenience product."""
    path = Path(path)
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"Expected one-band native MODIS raster: {path}")
        if src.descriptions[0] != "ET_MODIS_mm_period":
            raise ValueError(f"Unexpected native MODIS band description: {path}")
        if src.crs is None:
            raise ValueError(f"Native MODIS raster has no CRS: {path}")
        if not np.isfinite(abs(src.res[0])) or not np.isfinite(abs(src.res[1])):
            raise ValueError(f"Native MODIS raster has invalid pixel size: {path}")


def copy_native_modis_raster(source: Path, destination: Path) -> Path:
    """Copy a native-grid MODIS ET raster exactly without resampling."""
    source = Path(source)
    destination = Path(destination)
    validate_native_modis_raster(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    with rasterio.open(source) as src, rasterio.open(destination) as dst:
        if src.crs != dst.crs or src.transform != dst.transform:
            raise RuntimeError("Native MODIS georeferencing changed during copy.")
        if src.width != dst.width or src.height != dst.height:
            raise RuntimeError("Native MODIS dimensions changed during copy.")
        if src.nodata != dst.nodata:
            raise RuntimeError("Native MODIS NoData changed during copy.")
        if not np.array_equal(src.read(1), dst.read(1), equal_nan=True):
            raise RuntimeError("Native MODIS values changed during copy.")
    return destination


def resolve_native_modis_source(
    workspace_current: Path,
    production_outputs: dict[str, dict],
    period: str,
) -> Path:
    """Resolve a native MODIS product from run metadata or local backfill."""
    recorded = str(production_outputs.get(period, {}).get("modis_raster", "")).strip()
    if recorded:
        path = Path(recorded)
        if path.is_file():
            return path
    fallback = (
        Path(workspace_current)
        / "rasters_modis"
        / period
        / f"MODIS_ET_{period}_native.tif"
    )
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"Native MODIS ET raster not found for {period}. "
        "Run scripts/export_modis_coarse_rasters.py to backfill it."
    )


def _valid_mask(src: rasterio.io.DatasetReader) -> np.ndarray:
    return ~np.ma.getmaskarray(src.read(1, masked=True))


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty ET support.")
    return {
        "ET_min_mm_period": float(np.min(values)),
        "ET_p01_mm_period": float(np.percentile(values, 1)),
        "ET_p05_mm_period": float(np.percentile(values, 5)),
        "ET_median_mm_period": float(np.median(values)),
        "ET_mean_mm_period": float(np.mean(values)),
        "ET_p95_mm_period": float(np.percentile(values, 95)),
        "ET_p99_mm_period": float(np.percentile(values, 99)),
        "ET_max_mm_period": float(np.max(values)),
    }


def build_raster_summary(
    et_rasters: dict[str, Path],
    production_outputs: dict[str, dict],
    run_id: str,
    source_git_commit: str,
) -> pd.DataFrame:
    """Summarize each published raster and their three-date common support."""
    periods = tuple(et_rasters)
    if not periods:
        raise ValueError("No ET rasters supplied for summary.")

    masks: dict[str, np.ndarray] = {}
    published_counts: dict[str, int] = {}
    pixel_area_km2: float | None = None
    rows: list[dict] = []

    for period in periods:
        path = Path(et_rasters[period])
        with rasterio.open(path) as src:
            mask = _valid_mask(src)
            masks[period] = mask
            values = np.asarray(src.read(1)[mask], dtype=np.float64)
            published_counts[period] = int(values.size)
            if pixel_area_km2 is None:
                pixel_area_km2 = abs(src.res[0] * src.res[1]) / 1_000_000.0

        metadata_path = Path(
            production_outputs[period].get("production_metadata", "")
        )
        production_metadata = load_json(metadata_path) if metadata_path.is_file() else {}

        row = {
            "run_id": run_id,
            "source_git_commit": source_git_commit,
            "date": period,
            "scope": "published",
            "valid_pixels": int(values.size),
            "area_km2": float(values.size * pixel_area_km2),
            "fraction_of_published_support": 1.0,
            **_stats(values),
            "eligible_modis_parents": production_metadata.get(
                "eligible_modis_parents"
            ),
            "publishable_active_fraction_of_active_support": production_metadata.get(
                "publishable_active_fraction_of_active_support"
            ),
            "max_abs_conservation_error_after_floor_mm": production_metadata.get(
                "max_abs_conservation_error_after_floor_mm"
            ),
        }
        rows.append(row)

    reference_shape = next(iter(masks.values())).shape
    if any(mask.shape != reference_shape for mask in masks.values()):
        raise ValueError("Final ET rasters do not share one raster grid.")

    common_mask = np.logical_and.reduce([masks[period] for period in periods])
    common_count = int(common_mask.sum())
    if common_count == 0:
        raise ValueError("The final ET rasters have no common published support.")

    for period in periods:
        path = Path(et_rasters[period])
        with rasterio.open(path) as src:
            values = np.asarray(src.read(1)[common_mask], dtype=np.float64)
        rows.append(
            {
                "run_id": run_id,
                "source_git_commit": source_git_commit,
                "date": period,
                "scope": "common_all_dates",
                "valid_pixels": common_count,
                "area_km2": float(common_count * pixel_area_km2),
                "fraction_of_published_support": (
                    common_count / published_counts[period]
                ),
                **_stats(values),
                "eligible_modis_parents": pd.NA,
                "publishable_active_fraction_of_active_support": pd.NA,
                "max_abs_conservation_error_after_floor_mm": pd.NA,
            }
        )

    return pd.DataFrame(rows)


def build_minimal_final_outputs(
    workspace_current: Path,
    project_root: Path,
    output_dir: Path | None = None,
    run_dir: Path | None = None,
    periods: Iterable[str] = FINAL_PERIODS,
) -> dict[str, Path]:
    """Create the minimal local final directory from a completed run."""
    workspace_current = Path(workspace_current).resolve()
    project_root = Path(project_root).resolve()
    periods = tuple(periods)

    if run_dir is None:
        run_dir, run_metadata = find_latest_complete_run(
            workspace_current / "runs",
            periods,
        )
    else:
        run_dir = Path(run_dir).resolve()
        run_metadata = load_json(run_dir / "run_metadata.json")

    production_outputs = run_metadata.get("production_outputs", {})
    missing = [period for period in periods if period not in production_outputs]
    if missing:
        raise ValueError(
            "Selected run does not contain all final raster periods: "
            + ", ".join(missing)
        )

    provenance = run_metadata.get("provenance", {})
    source_git_commit = str(
        provenance.get("repository", {}).get("commit", "unknown")
    )
    run_id = run_dir.name

    if output_dir is None:
        output_dir = workspace_current.parent / "final"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    et_rasters: dict[str, Path] = {}
    modis_rasters: dict[str, Path] = {}
    for period in periods:
        source = Path(production_outputs[period]["raster"])
        if not source.is_file():
            raise FileNotFoundError(f"Scientific raster not found: {source}")
        destination = output_dir / f"ET_{period}_20m.tif"
        derive_et_only_raster(source, destination)
        et_rasters[period] = destination

        modis_source = resolve_native_modis_source(
            workspace_current=workspace_current,
            production_outputs=production_outputs,
            period=period,
        )
        modis_destination = output_dir / "modis" / f"MODIS_ET_{period}_native.tif"
        copy_native_modis_raster(modis_source, modis_destination)
        modis_rasters[period] = modis_destination

    summary = build_raster_summary(
        et_rasters=et_rasters,
        production_outputs=production_outputs,
        run_id=run_id,
        source_git_commit=source_git_commit,
    )
    summary_path = output_dir / "raster_summary.csv"
    summary.to_csv(summary_path, index=False)

    notebook_source = project_root / "notebooks" / "final_results_visualization.ipynb"
    notebook_destination = output_dir / notebook_source.name
    if not notebook_source.is_file():
        raise FileNotFoundError(
            f"Final visualization notebook not found: {notebook_source}"
        )
    shutil.copy2(notebook_source, notebook_destination)

    return {
        **{f"et_{period}": path for period, path in et_rasters.items()},
        **{f"modis_{period}": path for period, path in modis_rasters.items()},
        "raster_summary": summary_path,
        "notebook": notebook_destination,
        "source_run": run_dir,
    }
