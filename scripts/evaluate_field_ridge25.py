"""Evaluate Ridge-25 spatial-OOF downscaling against field-derived ET.

This is the field-evaluation companion to the final exact-overlap production
path. It keeps the scientific separation between reference-ET auditing and the
fine-resolution ET comparison, while replacing the legacy iterative raster
reconciliation with the accepted real-area overlap operator.

Scientific interpretation
-------------------------
1. ETgage is treated as an external observation of reference ET, not as a
   direct observation of actual ET.
2. Field-derived actual ET is computed as ETgage-derived ETo-equivalent × Kc.
   Fixed external Kc values are used for ST01-ST03; NDVI-derived Kc for ST04-ST05
   is retained only as a sensitivity analysis because NDVI is also a Ridge-25
   predictor.
3. Fine-resolution predictions are spatial out-of-fold (OOF): the complete
   spatial block containing the field station is excluded from model fitting.
4. The AOA is rebuilt from the fold-training population only.
5. For each station-period, raw Ridge-25/AOA tiles are built over the station
   tile plus one full 4 km tile ring. The raw fields are mosaicked first and
   reconciled once with native-grid MODIS ET using exact 20 m↔MODIS overlap
   areas. No coarse-to-fine nearest correction and no iterative raster
   reconciliation are used.
6. Publication follows the frozen rule: complete stack + fold-specific AOA +
   Kc_raw >= 0 + >=90% exact MODIS support. Negative ET after the unconstrained
   projection is floored once to zero, and the result is accepted only if the
   exact-overlap conservation error remains <=0.01 mm.
7. This is a local external field comparison at field sites. It is not an
   independent validation of the complete 20 m raster domain and it does not
   claim that ETgage directly measures actual ET.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from rasterio.transform import from_origin, rowcol
from rasterio.warp import transform as transform_coordinates
from sklearn.metrics import r2_score

from et_downscaling.aoa_ridge25 import build_unweighted_aoa
from et_downscaling.config import (
    ANALYSIS_CRS,
    OUTPUT_PERIOD_LABEL,
    build_training_output_filename,
    get_optical_output_label,
)
from et_downscaling.local_reconciliation import (
    RIDGE25_RECONCILIATION_TOLERANCE_MM,
    RIDGE25_USABLE_SUPPORT_FRACTION,
)
from et_downscaling.local_tiles import Tile, build_initial_tiles
from et_downscaling.modeling import train_and_validate_ridge25
from et_downscaling.overlap_reconciliation import (
    build_overlap_edges,
    materialize_active_values,
    solve_overlap_reconciliation,
)
from et_downscaling.production import PREDICTION_SCALE_M
from et_downscaling.ridge25 import RIDGE25_MODEL_FEATURES, build_ridge25_model
from et_downscaling.ridge25_overlap_production import (
    OUTPUT_BANDS,
    RAW_TILE_BANDS,
    _build_raw_tile,
    _download_native_modis,
    _fine_diagnostics,
)
from et_downscaling.workspace import get_workspace_paths


FIELD_EVALUATION_VERSION = "ridge25_spatial_oof_exact_overlap_local_v1"
FIELD_SCALE_FACTOR = 10.0
MIN_VALID_DAYS_PER_PERIOD = 5
VALID_DAILY_ET_RANGE_MM = (0.05, 12.0)
LOCAL_TILE_SIZE_M = 4000
LOCAL_SUPPORT_RINGS = 1

FIXED_KC = {
    "ST01": 0.85,
    "ST02": 0.95,
    "ST03": 1.10,
}

NDVI_KC_SLOPE = 1.457
NDVI_KC_INTERCEPT = -0.1725
NDVI_KC_VALID_RANGE = (0.10, 1.50)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Ridge-25 spatial-OOF exact-overlap downscaling at ETgage sites."
        )
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Google Cloud project with Earth Engine access.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional number of field station-period rows for a smoke test.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard the evaluation checkpoint and rebuild field rows.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "si", "sí"})
    )


def git_head(root: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def calculate_metrics(observed, predicted) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "observed": pd.to_numeric(observed, errors="coerce"),
            "predicted": pd.to_numeric(predicted, errors="coerce"),
        }
    ).dropna()

    n = len(frame)
    if n < 2:
        return {
            "n": int(n),
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "BIAS": np.nan,
            "r": np.nan,
            "KGE": np.nan,
        }

    y = frame["observed"].to_numpy(float)
    p = frame["predicted"].to_numpy(float)
    error = p - y

    y_sd = float(np.std(y, ddof=0))
    p_sd = float(np.std(p, ddof=0))
    y_mean = float(np.mean(y))
    p_mean = float(np.mean(p))

    if y_sd > 0 and p_sd > 0:
        correlation = float(np.corrcoef(y, p)[0, 1])
    else:
        correlation = np.nan

    if (
        np.isfinite(correlation)
        and y_sd > 0
        and p_sd > 0
        and y_mean != 0
    ):
        alpha = p_sd / y_sd
        beta = p_mean / y_mean
        kge = float(
            1.0
            - np.sqrt(
                (correlation - 1.0) ** 2
                + (alpha - 1.0) ** 2
                + (beta - 1.0) ** 2
            )
        )
    else:
        kge = np.nan

    return {
        "n": int(n),
        "R2": float(r2_score(y, p)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "BIAS": float(np.mean(error)),
        "r": correlation,
        "KGE": kge,
    }


def load_station_metadata(root: Path) -> pd.DataFrame:
    path = root / "data" / "stations" / "fundacion_stations.geojson"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = []
    for feature in payload["features"]:
        properties = dict(feature["properties"])
        longitude, latitude = feature["geometry"]["coordinates"]
        properties["longitude"] = float(longitude)
        properties["latitude"] = float(latitude)
        rows.append(properties)
    metadata = pd.DataFrame(rows)
    metadata["station_id"] = metadata["station_id"].astype(str)
    for column in ["inside_basin", "installation_conforms_manual"]:
        metadata[column] = to_bool(metadata[column])
    return metadata


def load_inputs(root: Path):
    workspace = get_workspace_paths(root).ensure()
    optical_label = get_optical_output_label("S2")
    master_path = (
        workspace.master
        / optical_label
        / build_training_output_filename("S2")
    )
    reference_path = (
        workspace.master
        / optical_label
        / f"reference_et_daily_{OUTPUT_PERIOD_LABEL}.csv"
    )
    field_path = root / "data" / "field" / "field_etgage.csv"

    for path in [master_path, reference_path, field_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    master = pd.read_csv(master_path, dtype={"station_id": "string"})
    reference = pd.read_csv(reference_path, dtype={"station_id": "string"})
    field = pd.read_csv(field_path, dtype={"station_id": "string"})
    metadata = load_station_metadata(root)
    return workspace, master, reference, field, metadata, master_path, reference_path


def prepare_field_daily(
    field: pd.DataFrame,
    reference: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    field = field.copy()
    reference = reference.copy()

    field["date"] = pd.to_datetime(field["date"], errors="raise")
    reference["local_date"] = pd.to_datetime(
        reference["local_date"], errors="raise"
    )

    field = field.merge(
        reference[
            ["station_id", "local_date", "ETo_mm_day", "ETr_mm_day"]
        ],
        left_on=["station_id", "date"],
        right_on=["station_id", "local_date"],
        how="left",
        validate="many_to_one",
    )

    field = field.merge(
        metadata[
            [
                "station_id",
                "canvas",
                "reference_et",
                "installation_conforms_manual",
                "inside_basin",
                "longitude",
                "latitude",
            ]
        ],
        on="station_id",
        how="left",
        validate="many_to_one",
    )

    field["within_installation_window"] = to_bool(
        field["within_installation_window"]
    )
    field["etgage_daily_raw"] = pd.to_numeric(
        field["etgage_daily_raw"], errors="coerce"
    )
    field["etgage_reference_mm_day"] = (
        field["etgage_daily_raw"] * FIELD_SCALE_FACTOR
    )

    field["field_daily_valid"] = (
        field["within_installation_window"]
        & field["etgage_daily_raw"].notna()
        & field["etgage_reference_mm_day"].between(
            *VALID_DAILY_ET_RANGE_MM
        )
    )

    valid = field.loc[field["field_daily_valid"]].copy()
    if valid[["ETo_mm_day", "ETr_mm_day"]].isna().any().any():
        raise RuntimeError("Missing ETo/ETr for valid ETgage days.")

    high_reference = valid["reference_et"].astype(str).str.upper().eq("ETR")
    valid["pm_reference_mm_day"] = np.where(
        high_reference,
        valid["ETr_mm_day"],
        valid["ETo_mm_day"],
    )

    ratio = valid["ETr_mm_day"] / valid["ETo_mm_day"]
    valid["etgage_eto_equivalent_mm_day"] = np.where(
        high_reference,
        valid["etgage_reference_mm_day"] / ratio,
        valid["etgage_reference_mm_day"],
    )

    return field, valid


def build_reference_audit(valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station_id, group in valid.groupby("station_id"):
        etgage = group["etgage_reference_mm_day"].to_numpy(float)
        pm = group["pm_reference_mm_day"].to_numpy(float)
        rows.append(
            {
                "station_id": station_id,
                "station": group["station"].iloc[0],
                "canvas": group["canvas"].iloc[0],
                "reference_et": group["reference_et"].iloc[0],
                "installation_conforms_manual": bool(
                    group["installation_conforms_manual"].iloc[0]
                ),
                "inside_basin": bool(group["inside_basin"].iloc[0]),
                "n_days": int(len(group)),
                "ETgage_mean_mm_day": float(np.mean(etgage)),
                "PM_mean_mm_day": float(np.mean(pm)),
                "bias_PM_minus_ETgage_mm_day": float(np.mean(pm - etgage)),
                "bias_pct": float(
                    100.0 * (np.mean(pm) / np.mean(etgage) - 1.0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)


def aggregate_field_periods(
    valid: pd.DataFrame,
    master: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    master = master.copy()
    master["period_start"] = pd.to_datetime(master["period_start"], errors="raise")
    master["number_days"] = pd.to_numeric(master["number_days"], errors="raise")

    required = {
        "station_id",
        "period_start",
        "number_days",
        "ET_mm_period",
        "ETo_mm_period",
        "NDVI_mean",
    }
    missing = sorted(required - set(master.columns))
    if missing:
        raise RuntimeError(
            "Master is missing field-evaluation columns: " + ", ".join(missing)
        )

    metadata_indexed = metadata.set_index("station_id")
    rows = []

    for station_id, daily in valid.groupby("station_id"):
        periods = master.loc[
            master["station_id"].astype(str) == str(station_id)
        ].copy()
        periods = periods.sort_values("period_start")
        periods["period_end"] = periods["period_start"] + pd.to_timedelta(
            periods["number_days"], unit="D"
        )

        for period in periods.itertuples(index=False):
            group = daily.loc[
                (daily["date"] >= period.period_start)
                & (daily["date"] < period.period_end)
            ].copy()
            if len(group) < MIN_VALID_DAYS_PER_PERIOD:
                continue

            reference_mean = float(
                group["etgage_eto_equivalent_mm_day"].mean()
            )
            field_reference_period = reference_mean * int(period.number_days)

            kc = FIXED_KC.get(str(station_id), np.nan)
            kc_source = "FAO-56 fixed"
            analysis_role = "fixed_kc_main"

            if not np.isfinite(kc):
                ndvi = float(period.NDVI_mean)
                candidate = NDVI_KC_SLOPE * ndvi + NDVI_KC_INTERCEPT
                if NDVI_KC_VALID_RANGE[0] <= candidate <= NDVI_KC_VALID_RANGE[1]:
                    kc = candidate
                    kc_source = "NDVI-derived sensitivity"
                    analysis_role = "ndvi_kc_sensitivity"
                else:
                    kc = np.nan
                    kc_source = "NDVI-derived outside valid range"
                    analysis_role = "excluded_invalid_kc"

            meta = metadata_indexed.loc[str(station_id)]
            rows.append(
                {
                    "station_id": str(station_id),
                    "station": str(meta["station"]),
                    "period_start": period.period_start,
                    "number_days": int(period.number_days),
                    "n_valid_field_days": int(len(group)),
                    "field_reference_eto_mm_period": field_reference_period,
                    "Kc_field_conversion": (
                        float(kc) if np.isfinite(kc) else np.nan
                    ),
                    "Kc_source": kc_source,
                    "analysis_role": analysis_role,
                    "field_derived_et_mm_period": (
                        field_reference_period * float(kc)
                        if np.isfinite(kc)
                        else np.nan
                    ),
                    "ET_MODIS_mm_period": (
                        float(period.ET_mm_period)
                        if np.isfinite(period.ET_mm_period)
                        else np.nan
                    ),
                    "ETo_model_mm_period": (
                        float(period.ETo_mm_period)
                        if np.isfinite(period.ETo_mm_period)
                        else np.nan
                    ),
                    "NDVI_footprint_mean": (
                        float(period.NDVI_mean)
                        if np.isfinite(period.NDVI_mean)
                        else np.nan
                    ),
                    "inside_basin": bool(meta["inside_basin"]),
                    "installation_conforms_manual": bool(
                        meta["installation_conforms_manual"]
                    ),
                    "longitude": float(meta["longitude"]),
                    "latitude": float(meta["latitude"]),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["station_id", "period_start"])
        .reset_index(drop=True)
    )


def build_fold_resources(result):
    population = result.population.copy()
    population["station_id"] = population["station_id"].astype(str)
    population["spatial_block"] = population["spatial_block"].astype(str)
    population["period_start"] = pd.to_datetime(population["period_start"])

    station_blocks = population.groupby("station_id")["spatial_block"].unique()
    invalid = station_blocks[station_blocks.apply(len) != 1]
    if not invalid.empty:
        raise RuntimeError(
            "A station maps to more than one spatial block: "
            + str(invalid.to_dict())
        )
    station_to_block = {
        str(station): str(values[0])
        for station, values in station_blocks.items()
    }

    saved_oof = result.spatial_oof.copy()
    saved_oof["station_id"] = saved_oof["station_id"].astype(str)
    saved_oof["spatial_block"] = saved_oof["spatial_block"].astype(str)
    saved_oof["period_start"] = pd.to_datetime(saved_oof["period_start"])

    resources = {}
    for block in sorted(population["spatial_block"].unique()):
        train = population.loc[population["spatial_block"] != block].copy()
        test = population.loc[population["spatial_block"] == block].copy()

        model = build_ridge25_model()
        model.fit(train[RIDGE25_MODEL_FEATURES], train["Kc_target"])

        check = test[
            ["station_id", "period_start", *RIDGE25_MODEL_FEATURES]
        ].merge(
            saved_oof.loc[
                saved_oof["spatial_block"] == block,
                ["station_id", "period_start", "prediction"],
            ],
            on=["station_id", "period_start"],
            how="inner",
            validate="one_to_one",
        )
        local_prediction = model.predict(check[RIDGE25_MODEL_FEATURES])
        max_difference = float(
            np.max(
                np.abs(
                    local_prediction - check["prediction"].to_numpy(float)
                )
            )
        )
        if max_difference > 1e-10:
            raise RuntimeError(
                f"Spatial OOF fold reproduction failed for {block}: "
                f"max diff={max_difference:.3e}"
            )

        aoa = build_unweighted_aoa(train, group_column="spatial_block")
        resources[block] = {
            "model": model,
            "aoa": aoa,
            "training_rows": int(len(train)),
            "test_rows": int(len(test)),
            "oof_max_abs_difference": max_difference,
        }
        print(
            f"Block {block}: OOF reproduction PASS; "
            f"train={len(train)}, test={len(test)}, "
            f"AOA threshold={aoa.threshold:.6f}"
        )

    return station_to_block, resources


def station_xy(row: pd.Series) -> tuple[float, float]:
    x, y = transform_coordinates(
        "EPSG:4326",
        ANALYSIS_CRS,
        [float(row["longitude"])],
        [float(row["latitude"])],
    )
    return float(x[0]), float(y[0])


def find_station_tile(tiles, x: float, y: float):
    matches = [
        tile
        for tile in tiles
        if tile.xmin <= x < tile.xmax and tile.ymin < y <= tile.ymax
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one 4 km tile at station coordinate; found {len(matches)}."
        )
    return matches[0]


def support_tiles_around(
    center_tile: Tile,
    rings: int = LOCAL_SUPPORT_RINGS,
) -> tuple[list[Tile], tuple[float, float, float, float]]:
    if rings < 1:
        raise ValueError("Exact-overlap field support requires at least one ring.")

    tile_size = float(center_tile.width_m)
    if not math.isclose(tile_size, float(center_tile.height_m)):
        raise RuntimeError("Field support expects square 4 km tiles.")

    tiles: list[Tile] = []
    for dy in range(-rings, rings + 1):
        for dx in range(-rings, rings + 1):
            xmin = center_tile.xmin + dx * tile_size
            ymin = center_tile.ymin + dy * tile_size
            xmax = xmin + tile_size
            ymax = ymin + tile_size
            tiles.append(
                Tile(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    tile_id=(
                        f"sx{int(round(xmin))}_sy{int(round(ymin))}"
                    ),
                    level=0,
                )
            )

    tiles = sorted(tiles, key=lambda item: (-item.ymax, item.xmin))
    bounds = (
        min(tile.xmin for tile in tiles),
        min(tile.ymin for tile in tiles),
        max(tile.xmax for tile in tiles),
        max(tile.ymax for tile in tiles),
    )
    return tiles, bounds


def _safe_block(block: str) -> str:
    return block.replace("/", "_").replace("\\", "_")


def raw_tile_cache_path(
    output_directory: Path,
    block: str,
    date_text: str,
    tile_id: str,
) -> Path:
    path = (
        output_directory
        / "exact_overlap_raw_tile_cache"
        / _safe_block(block)
        / date_text
    )
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{tile_id}.npz"


def product_cache_path(
    output_directory: Path,
    block: str,
    date_text: str,
    center_tile_id: str,
) -> Path:
    path = (
        output_directory
        / "exact_overlap_product_cache"
        / _safe_block(block)
        / date_text
    )
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{center_tile_id}.npz"


def get_raw_tile(
    output_directory: Path,
    date_text: str,
    block: str,
    fold_resource: dict,
    tile: Tile,
    timeout_seconds: int,
) -> tuple[np.ndarray, dict]:
    path = raw_tile_cache_path(
        output_directory,
        block,
        date_text,
        tile.tile_id,
    )

    if path.is_file():
        payload = np.load(path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("evaluation_version") == FIELD_EVALUATION_VERSION:
            return np.asarray(payload["data"], dtype=np.float64), metadata
        path.unlink(missing_ok=True)

    data, metadata = _build_raw_tile(
        period_start=date_text,
        model=fold_resource["model"],
        aoa_parameters=fold_resource["aoa"],
        tile=tile,
        timeout_seconds=timeout_seconds,
    )
    data = np.asarray(data, dtype=np.float64)
    expected = (len(RAW_TILE_BANDS), tile.height_px, tile.width_px)
    if data.shape != expected:
        raise RuntimeError(
            f"Unexpected raw exact-overlap tile shape {data.shape}; "
            f"expected {expected}."
        )

    metadata = dict(metadata)
    metadata["evaluation_version"] = FIELD_EVALUATION_VERSION
    metadata["spatial_block"] = block

    np.savez_compressed(
        path,
        data=data.astype(np.float32),
        metadata_json=np.array(json.dumps(metadata, default=str)),
    )
    return data, metadata


def mosaic_raw_tiles(
    completed: list[tuple[Tile, np.ndarray]],
    support_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, object]:
    xmin, ymin, xmax, ymax = support_bounds
    scale = float(PREDICTION_SCALE_M)
    width = int(round((xmax - xmin) / scale))
    height = int(round((ymax - ymin) / scale))
    transform = from_origin(xmin, ymax, scale, scale)

    mosaic = np.full(
        (len(RAW_TILE_BANDS), height, width),
        np.nan,
        dtype=np.float64,
    )

    for tile, data in completed:
        col_off = int(round((tile.xmin - xmin) / scale))
        row_off = int(round((ymax - tile.ymax) / scale))
        rs = slice(row_off, row_off + tile.height_px)
        cs = slice(col_off, col_off + tile.width_px)
        mosaic[:, rs, cs] = data

    return mosaic, transform


def build_local_exact_product(
    output_directory: Path,
    date_text: str,
    block: str,
    fold_resource: dict,
    center_tile: Tile,
    timeout_seconds: int,
) -> tuple[np.ndarray, dict]:
    cache = product_cache_path(
        output_directory,
        block,
        date_text,
        center_tile.tile_id,
    )
    if cache.is_file():
        payload = np.load(cache, allow_pickle=False)
        metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("evaluation_version") == FIELD_EVALUATION_VERSION:
            return np.asarray(payload["data"], dtype=np.float64), metadata
        cache.unlink(missing_ok=True)

    support_tiles, support_bounds = support_tiles_around(center_tile)
    raw_tiles: list[tuple[Tile, np.ndarray]] = []
    for tile in support_tiles:
        data, _ = get_raw_tile(
            output_directory=output_directory,
            date_text=date_text,
            block=block,
            fold_resource=fold_resource,
            tile=tile,
            timeout_seconds=timeout_seconds,
        )
        raw_tiles.append((tile, data))

    raw, support_transform = mosaic_raw_tiles(raw_tiles, support_bounds)
    fine_shape = raw.shape[1:]

    kc_raw = raw[RAW_TILE_BANDS.index("Kc_raw")]
    dissimilarity = raw[RAW_TILE_BANDS.index("dissimilarity_index")]
    stack_valid = raw[RAW_TILE_BANDS.index("stack_valid")] > 0.5
    aoa_inside = raw[RAW_TILE_BANDS.index("AOA_inside")] > 0.5
    usable = raw[RAW_TILE_BANDS.index("usable")] > 0.5
    domain = raw[RAW_TILE_BANDS.index("support_domain")] > 0.5

    modis_et, modis_grid = _download_native_modis(
        period_start=date_text,
        support_bounds=support_bounds,
        timeout_seconds=timeout_seconds,
    )

    edges = build_overlap_edges(
        domain=domain,
        fine_transform=support_transform,
        fine_crs=ANALYSIS_CRS,
        modis_et=modis_et,
        modis_transform=modis_grid.transform,
        modis_crs=modis_grid.local_crs,
        progress_every=0,
    )

    result = solve_overlap_reconciliation(
        kc_raw=kc_raw,
        usable=usable,
        modis_et=modis_et,
        edges=edges,
        usable_support_fraction=RIDGE25_USABLE_SUPPORT_FRACTION,
        tolerance_mm=RIDGE25_RECONCILIATION_TOLERANCE_MM,
    )

    et_support = materialize_active_values(
        fine_shape=fine_shape,
        active_fine=result.active_fine,
        values=result.et_final_nonnegative,
        selected_active=result.publishable_active,
    )

    minimum_support, all_eligible, maximum_error = _fine_diagnostics(
        edges=edges,
        result=result,
        fine_size=kc_raw.size,
    )
    minimum_support = minimum_support.reshape(fine_shape)
    all_eligible = all_eligible.reshape(fine_shape)
    maximum_error = maximum_error.reshape(fine_shape)

    full_arrays = np.stack(
        [
            et_support,
            kc_raw,
            dissimilarity,
            stack_valid.astype(np.float64),
            aoa_inside.astype(np.float64),
            usable.astype(np.float64),
            minimum_support,
            all_eligible,
            maximum_error,
        ],
        axis=0,
    )

    scale = float(PREDICTION_SCALE_M)
    support_xmin, _, _, support_ymax = support_bounds
    col_off = int(round((center_tile.xmin - support_xmin) / scale))
    row_off = int(round((support_ymax - center_tile.ymax) / scale))
    rs = slice(row_off, row_off + center_tile.height_px)
    cs = slice(col_off, col_off + center_tile.width_px)
    center_data = np.asarray(full_arrays[:, rs, cs], dtype=np.float64)

    adjustment = result.et_final_nonnegative - result.et_initial
    finite_pair = np.isfinite(result.et_initial) & np.isfinite(
        result.et_final_nonnegative
    )
    if int(finite_pair.sum()) > 1:
        correlation = float(
            np.corrcoef(
                result.et_initial[finite_pair],
                result.et_final_nonnegative[finite_pair],
            )[0, 1]
        )
    else:
        correlation = np.nan

    metadata = {
        "evaluation_version": FIELD_EVALUATION_VERSION,
        "period_start": date_text,
        "spatial_block": block,
        "center_tile_id": center_tile.tile_id,
        "support_rings": LOCAL_SUPPORT_RINGS,
        "support_tile_count": len(support_tiles),
        "support_bounds": list(support_bounds),
        "reconciliation": "single_exact_overlap_after_local_raw_mosaic",
        "usable_support_fraction": RIDGE25_USABLE_SUPPORT_FRACTION,
        "conservation_tolerance_mm": RIDGE25_RECONCILIATION_TOLERANCE_MM,
        "eligible_modis_parents": int(result.eligible_coarse.size),
        "active_fine_cells": int(result.active_fine.size),
        "publishable_active_cells_support": int(
            result.publishable_active.sum()
        ),
        "negative_active_before_floor": int(result.negative_active_cells),
        "negative_publishable_before_floor": int(
            result.negative_publishable_cells
        ),
        "max_abs_conservation_error_before_floor_mm": float(
            result.max_abs_final_error_mm
        ),
        "max_abs_conservation_error_after_floor_mm": float(
            result.max_abs_error_after_nonnegative_mm
        ),
        "mae_adjustment_mm": float(np.mean(np.abs(adjustment))),
        "rmse_adjustment_mm": float(np.sqrt(np.mean(adjustment**2))),
        "pearson_final_vs_initial": correlation,
    }

    np.savez_compressed(
        cache,
        data=center_data.astype(np.float32),
        metadata_json=np.array(json.dumps(metadata, default=str)),
    )
    return center_data, metadata


def clean_value(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= -9990:
        return np.nan
    return value


def sample_product(
    data: np.ndarray,
    tile: Tile,
    x: float,
    y: float,
) -> dict[str, float]:
    transform = from_origin(
        tile.xmin,
        tile.ymax,
        PREDICTION_SCALE_M,
        PREDICTION_SCALE_M,
    )
    row_index, col_index = rowcol(transform, x, y)
    if not (
        0 <= row_index < data.shape[1]
        and 0 <= col_index < data.shape[2]
    ):
        raise RuntimeError("Station sample falls outside selected tile array.")

    values = {
        band: clean_value(data[index, row_index, col_index])
        for index, band in enumerate(OUTPUT_BANDS)
    }
    values["pixel_row"] = int(row_index)
    values["pixel_col"] = int(col_index)
    return values


def expected_nonpublication_status(error: RuntimeError) -> str | None:
    message = str(error)
    if "No MODIS parents pass the support rule" in message:
        return "no_eligible_modis_parent"
    if "No fully represented MODIS parents were found" in message:
        return "insufficient_exact_overlap_support"
    if "Non-positive Kc mean reached an eligible parent" in message:
        return "nonpositive_parent_kc"
    if "Setting negative ET to zero violates the MODIS conservation tolerance" in message:
        return "post_floor_conservation_failed"
    return None


def build_metrics_tables(
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    published = pairs.loc[
        pairs["ET_downscaled_oof_mm_period"].notna()
        & pairs["field_derived_et_mm_period"].notna()
        & pairs["ET_MODIS_mm_period"].notna()
    ].copy()

    subsets = {
        "fixed_kc_main_ST01_ST03": published.loc[
            published["station_id"].isin(FIXED_KC)
        ],
        "ST01_installation_conforming": published.loc[
            published["station_id"].eq("ST01")
        ],
        "all_available_sensitivity": published,
    }
    models = {
        "MODIS_parent": "ET_MODIS_mm_period",
        "Ridge25_exact_overlap_spatial_OOF": "ET_downscaled_oof_mm_period",
    }

    metric_rows = []
    for subset_name, subset in subsets.items():
        for model_name, column in models.items():
            metric_rows.append(
                {
                    "subset": subset_name,
                    "model": model_name,
                    **calculate_metrics(
                        subset["field_derived_et_mm_period"],
                        subset[column],
                    ),
                }
            )

    by_station_rows = []
    for station_id, subset in published.groupby("station_id"):
        row = {
            "station_id": station_id,
            "station": subset["station"].iloc[0],
            "n": int(len(subset)),
            "Kc_source": subset["Kc_source"].iloc[0],
            "installation_conforms_manual": bool(
                subset["installation_conforms_manual"].iloc[0]
            ),
        }
        for label, column in models.items():
            metrics = calculate_metrics(
                subset["field_derived_et_mm_period"],
                subset[column],
            )
            for metric, value in metrics.items():
                if metric == "n":
                    continue
                row[f"{label}_{metric}"] = value

        modis_rmse = row.get("MODIS_parent_RMSE", np.nan)
        fine_rmse = row.get(
            "Ridge25_exact_overlap_spatial_OOF_RMSE",
            np.nan,
        )
        if np.isfinite(modis_rmse) and np.isfinite(fine_rmse):
            row["delta_RMSE_MODIS_minus_downscaled"] = (
                modis_rmse - fine_rmse
            )
        by_station_rows.append(row)

    return pd.DataFrame(metric_rows), pd.DataFrame(by_station_rows)


def main() -> None:
    args = parse_arguments()
    root = project_root()
    (
        workspace,
        master,
        reference,
        field,
        metadata,
        master_path,
        reference_path,
    ) = load_inputs(root)

    print("Rebuilding frozen Ridge-25 training/validation population...")
    result = train_and_validate_ridge25(
        master,
        verify_reference_2020_2024=True,
    )
    print("Training rows:", len(result.population))
    print("Spatial R2:", f"{result.spatial_metrics['R2']:.6f}")
    print("Spatial RMSE:", f"{result.spatial_metrics['RMSE']:.6f}")

    _, valid_daily = prepare_field_daily(field, reference, metadata)
    reference_audit = build_reference_audit(valid_daily)
    candidates = aggregate_field_periods(valid_daily, master, metadata)

    station_to_block, fold_resources = build_fold_resources(result)
    candidates["spatial_block"] = candidates["station_id"].map(
        station_to_block
    )

    missing_blocks = candidates.loc[
        candidates["spatial_block"].isna(), "station_id"
    ].unique()
    if len(missing_blocks):
        raise RuntimeError(
            "Field stations absent from Ridge-25 training population: "
            + ", ".join(map(str, missing_blocks))
        )

    output_directory = (
        workspace.diagnostics / "field_ridge25_oof_exact_overlap"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "field_ridge25_oof_checkpoint.csv"

    if args.restart:
        checkpoint_path.unlink(missing_ok=True)

    completed = pd.DataFrame()
    if checkpoint_path.is_file():
        completed = pd.read_csv(
            checkpoint_path,
            dtype={"station_id": "string"},
            parse_dates=["period_start"],
        )

    keys_done = set()
    if not completed.empty:
        keys_done = set(
            zip(
                completed["station_id"].astype(str),
                completed["period_start"].dt.strftime("%Y-%m-%d"),
            )
        )

    work = candidates.copy()
    if args.max_rows is not None:
        work = work.head(args.max_rows).copy()

    initial_tiles, _ = build_initial_tiles(
        root,
        tile_size_m=LOCAL_TILE_SIZE_M,
    )

    print("Initializing Earth Engine...")
    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()

    rows = completed.to_dict("records") if not completed.empty else []

    for index, row in work.iterrows():
        station_id = str(row["station_id"])
        date_text = pd.Timestamp(row["period_start"]).strftime("%Y-%m-%d")
        key = (station_id, date_text)
        if key in keys_done:
            continue

        output = row.to_dict()
        output["period_start"] = pd.Timestamp(row["period_start"])
        output["ET_downscaled_oof_mm_period"] = np.nan
        output["production_status"] = "not_processed"
        output["evaluation_version"] = FIELD_EVALUATION_VERSION

        if not bool(row["inside_basin"]):
            output["production_status"] = "station_outside_basin"
            rows.append(output)
        else:
            block = str(row["spatial_block"])
            fold_resource = fold_resources[block]
            x, y = station_xy(row)
            tile = find_station_tile(initial_tiles, x, y)

            print(
                f"[{index + 1}/{len(work)}] {station_id} {date_text} "
                f"block={block} tile={tile.tile_id} exact-overlap"
            )

            try:
                data, local_metadata = build_local_exact_product(
                    output_directory=output_directory,
                    date_text=date_text,
                    block=block,
                    fold_resource=fold_resource,
                    center_tile=tile,
                    timeout_seconds=args.timeout_seconds,
                )
            except RuntimeError as error:
                status = expected_nonpublication_status(error)
                if status is None:
                    raise

                output["tile_id"] = tile.tile_id
                output["fold_training_rows"] = fold_resource["training_rows"]
                output["fold_test_rows"] = fold_resource["test_rows"]
                output["fold_oof_max_abs_difference"] = fold_resource[
                    "oof_max_abs_difference"
                ]
                output["fold_aoa_threshold"] = float(
                    fold_resource["aoa"].threshold
                )
                output["production_status"] = status
                output["production_error"] = str(error)
                rows.append(output)

            else:
                sample = sample_product(data, tile, x, y)
                output.update(sample)

                output["tile_id"] = tile.tile_id
                output["fold_training_rows"] = fold_resource["training_rows"]
                output["fold_test_rows"] = fold_resource["test_rows"]
                output["fold_oof_max_abs_difference"] = fold_resource[
                    "oof_max_abs_difference"
                ]
                output["fold_aoa_threshold"] = float(
                    fold_resource["aoa"].threshold
                )
                output["support_rings"] = local_metadata["support_rings"]
                output["support_tile_count"] = local_metadata[
                    "support_tile_count"
                ]
                output["eligible_modis_parents_local"] = local_metadata[
                    "eligible_modis_parents"
                ]
                output["negative_publishable_before_floor_local"] = (
                    local_metadata["negative_publishable_before_floor"]
                )
                output[
                    "max_abs_conservation_error_after_floor_mm_local"
                ] = local_metadata[
                    "max_abs_conservation_error_after_floor_mm"
                ]
                output["mae_adjustment_mm_local"] = local_metadata[
                    "mae_adjustment_mm"
                ]
                output["rmse_adjustment_mm_local"] = local_metadata[
                    "rmse_adjustment_mm"
                ]
                output["pearson_final_vs_initial_local"] = local_metadata[
                    "pearson_final_vs_initial"
                ]

                et_value = sample["ET_mm_period"]
                output["ET_downscaled_oof_mm_period"] = et_value
                output["production_status"] = (
                    "published"
                    if np.isfinite(et_value)
                    else "not_published"
                )
                rows.append(output)

        checkpoint = pd.DataFrame(rows)
        checkpoint["period_start"] = pd.to_datetime(
            checkpoint["period_start"]
        )
        checkpoint = checkpoint.sort_values(
            ["station_id", "period_start"]
        ).drop_duplicates(
            ["station_id", "period_start"],
            keep="last",
        )
        checkpoint.to_csv(checkpoint_path, index=False)
        keys_done.add(key)

    pairs = pd.read_csv(
        checkpoint_path,
        dtype={"station_id": "string"},
        parse_dates=["period_start"],
    )
    metrics, by_station = build_metrics_tables(pairs)

    reference_path_out = output_directory / "field_reference_et_audit.csv"
    candidate_path = output_directory / "field_period_candidates.csv"
    pairs_path = output_directory / "field_ridge25_oof_pairs.csv"
    metrics_path = output_directory / "field_ridge25_oof_metrics.csv"
    station_path = output_directory / "field_ridge25_oof_by_station.csv"

    reference_audit.to_csv(reference_path_out, index=False)
    candidates.to_csv(candidate_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    by_station.to_csv(station_path, index=False)

    metadata_output = {
        "git_head": git_head(root),
        "evaluation_version": FIELD_EVALUATION_VERSION,
        "method": (
            "Ridge25 spatial OOF + fold-specific AOA + one local raw mosaic "
            "+ exact 20m-MODIS overlap reconciliation"
        ),
        "master": str(master_path),
        "daily_reference": str(reference_path),
        "training_rows": int(len(result.population)),
        "candidate_field_periods": int(len(candidates)),
        "processed_rows": int(len(pairs)),
        "published_rows": int(
            pairs["ET_downscaled_oof_mm_period"].notna().sum()
        ),
        "production_status_counts": {
            str(key): int(value)
            for key, value in pairs["production_status"]
            .fillna("missing")
            .value_counts()
            .items()
        },
        "fixed_kc_candidate_rows": int(
            candidates["station_id"].isin(FIXED_KC).sum()
        ),
        "ndvi_kc_sensitivity_candidate_rows": int(
            (~candidates["station_id"].isin(FIXED_KC)).sum()
        ),
        "local_support_tile_size_m": LOCAL_TILE_SIZE_M,
        "local_support_rings": LOCAL_SUPPORT_RINGS,
        "local_support_tile_count": int((2 * LOCAL_SUPPORT_RINGS + 1) ** 2),
        "usable_support_fraction": RIDGE25_USABLE_SUPPORT_FRACTION,
        "conservation_tolerance_mm": RIDGE25_RECONCILIATION_TOLERANCE_MM,
        "interpretation": (
            "Local external field comparison using the accepted exact-overlap "
            "operator on an interior production-style support window; not "
            "independent validation of the complete 20 m raster domain. ETgage "
            "is reference ET and field actual ET remains Kc-derived."
        ),
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata_output, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print("FIELD RIDGE-25 EXACT-OVERLAP OOF EVALUATION COMPLETE")
    print("=" * 92)
    print("Candidate periods:", len(candidates))
    print("Processed rows:", len(pairs))
    print(
        "Published fine rows:",
        int(pairs["ET_downscaled_oof_mm_period"].notna().sum()),
    )
    print("Production status:")
    print(
        pairs["production_status"]
        .fillna("missing")
        .value_counts()
        .to_string()
    )
    print()
    print("REFERENCE ET AUDIT")
    print(reference_audit.to_string(index=False))
    print()
    print("ET METRICS")
    print(metrics.to_string(index=False))
    print()
    print("BY STATION")
    print(by_station.to_string(index=False))
    print()
    print("Output:", output_directory)


if __name__ == "__main__":
    main()