"""Compare equal-weight and Ridge-coefficient-weighted AOA on real 20 m pixels.

The script is deliberately diagnostic: it does not modify the accepted AOA,
production version, model, or final rasters. For every requested date it
retrieves each 20 m predictor tile once, then scores BOTH AOA definitions
locally from the same predictor cube. It also evaluates the exact-overlap
90% MODIS support rule without solving a new ET reconciliation.

Default dates span dry, wet, and the previously audited 2022-04-07 case.
All outputs stay in the local ET_FUNDACION_WORKSPACE. Google Drive and
persistent Earth Engine assets are never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE_ENV_VAR = "ET_FUNDACION_WORKSPACE"
DEFAULT_DATES = ["2020-03-13", "2021-11-25", "2022-04-07"]
DIAGNOSTIC_VERSION = "aoa_equal_vs_ridgecoef_sensitivity_v1"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score equal-weight and |Ridge coefficient|-weighted AOA on the "
            "same real 20 m predictor pixels."
        )
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Google Cloud Project ID with Earth Engine access.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help=(
            "External workspace root. Defaults to ET_FUNDACION_WORKSPACE or "
            "the repository sibling ET_fundacion_workspace/current."
        ),
    )
    parser.add_argument(
        "--run-directory",
        default=None,
        help="Specific saved Ridge-25 run. Defaults to the newest completed run.",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=DEFAULT_DATES,
        help="MODIS period starts (YYYY-MM-DD).",
    )
    parser.add_argument("--tile-size-m", type=int, default=4000)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore diagnostic tile caches and recompute them.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_workspace(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    override = os.environ.get(WORKSPACE_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return project_root().parent / "ET_fundacion_workspace" / "current"


def resolve_run_directory(workspace: Path, value: str | None) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Run directory not found: {path}")
        return path
    candidates = sorted(
        path for path in (workspace / "runs").iterdir()
        if path.is_dir() and (path / "tables" / "ridge25_training_population.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No completed Ridge-25 runs found under {workspace / 'runs'}")
    return candidates[-1]


def resolve_project_id(value: str | None) -> str:
    project_id = value.strip() if value else ""
    if not project_id:
        project_id = input("Google Cloud Project ID: ").strip()
    if not project_id:
        raise ValueError("Google Cloud Project ID cannot be empty.")
    return project_id


def build_weighted_aoa(population, features, coefficients, AOAParameters):
    """Return AOAParameters implementing z_j * |beta_j| distances.

    The existing scorer standardizes as (x - mean) / scale. Setting the stored
    scale to sample_sd / weight therefore evaluates exactly z * weight without
    changing production code. This object is used only by this sensitivity.
    """
    from sklearn.metrics import pairwise_distances

    matrix = population[list(features)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Training predictors contain non-finite values.")

    means = matrix.mean(axis=0)
    sample_scales = matrix.std(axis=0, ddof=1)
    weights = np.abs(np.asarray(coefficients, dtype=float))
    if np.any(sample_scales <= 0) or np.any(~np.isfinite(sample_scales)):
        raise ValueError("Invalid AOA sample standard deviations.")
    if np.any(weights <= 0) or np.any(~np.isfinite(weights)):
        raise ValueError(
            "All Ridge coefficient weights must be finite and positive for this sensitivity."
        )

    weighted_scaled = ((matrix - means) / sample_scales) * weights
    distances = pairwise_distances(weighted_scaled, metric="euclidean")
    np.fill_diagonal(distances, np.nan)
    mean_training_distance = float(np.nanmean(np.nanmean(distances, axis=1)))

    groups = population["spatial_block"].astype(str).to_numpy()
    training_di = np.full(len(population), np.nan, dtype=float)
    for index in range(len(population)):
        candidate = groups != groups[index]
        if not candidate.any():
            raise ValueError("Each AOA row requires another spatial block.")
        training_di[index] = (
            float(np.nanmin(distances[index, candidate])) / mean_training_distance
        )

    q1 = float(np.quantile(training_di, 0.25))
    q3 = float(np.quantile(training_di, 0.75))
    threshold = min(float(np.max(training_di)), q3 + 1.5 * (q3 - q1))

    # score_unweighted_aoa computes (x - means) / scales.  This effective
    # scale makes that expression equal to standardized_x * weight.
    effective_scales = sample_scales / weights
    return AOAParameters(
        feature_names=tuple(features),
        means=means,
        scales=effective_scales,
        training_scaled=weighted_scaled,
        mean_training_distance=mean_training_distance,
        threshold=threshold,
        training_di=training_di,
    ), weights


def scientific_signature(model_arrays, equal_aoa, weighted_aoa) -> str:
    digest = hashlib.sha256()
    digest.update(DIAGNOSTIC_VERSION.encode("utf-8"))
    digest.update(b"\0")
    for array in (*model_arrays, equal_aoa.means, equal_aoa.scales,
                  equal_aoa.training_scaled, [equal_aoa.threshold],
                  weighted_aoa.means, weighted_aoa.scales,
                  weighted_aoa.training_scaled, [weighted_aoa.threshold]):
        values = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(values.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def tile_offsets(tile, bounds, scale):
    xmin, _ymin, _xmax, ymax = bounds
    col0 = int(round((tile.xmin - xmin) / scale))
    row0 = int(round((ymax - tile.ymax) / scale))
    return row0, col0


def load_or_build_tile(
    *,
    tile,
    period_start,
    cache_directory,
    signature,
    restart,
    model,
    equal_aoa,
    weighted_aoa,
    features,
    timeout_seconds,
    ee,
    ANALYSIS_CRS,
    PREDICTION_SCALE_M,
    PROCESSING_BUFFER_M,
    build_ridge25_production_stack,
    score_unweighted_aoa,
    _download_ee_bytes,
    _read_downloaded_array,
    _processing_grid,
):
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = cache_directory / f"{tile.tile_id}.npz"
    meta_path = cache_directory / f"{tile.tile_id}.json"

    if not restart and path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("diagnostic_version") == DIAGNOSTIC_VERSION
                and meta.get("scientific_signature") == signature
                and meta.get("period_start") == period_start
                and meta.get("tile_id") == tile.tile_id
            ):
                with np.load(path) as payload:
                    return {name: payload[name] for name in payload.files}
        except Exception:
            pass
        path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    core = ee.Geometry.Rectangle(
        [tile.xmin, tile.ymin, tile.xmax, tile.ymax],
        proj=ANALYSIS_CRS,
        geodesic=False,
    )
    context = build_ridge25_production_stack(
        period_start_text=period_start,
        basin_geometry=core,
    )
    (
        source_xmin,
        _source_ymin,
        _source_xmax,
        source_ymax,
        source_width,
        source_height,
        requested_transform,
    ) = _processing_grid(tile)

    predictor_image = context["stack"].select(features).toFloat()
    parameters = {
        "bands": list(features),
        "crs": ANALYSIS_CRS,
        "crs_transform": [
            PREDICTION_SCALE_M,
            0,
            source_xmin,
            0,
            -PREDICTION_SCALE_M,
            source_ymax,
        ],
        "dimensions": [source_width, source_height],
        "format": "GEO_TIFF",
    }
    payload = _download_ee_bytes(predictor_image, parameters, timeout_seconds)
    predictor_bands, transform, crs = _read_downloaded_array(payload)

    expected_shape = (len(features), source_height, source_width)
    if predictor_bands.shape != expected_shape:
        raise RuntimeError(
            f"Downloaded predictor shape {predictor_bands.shape}; expected {expected_shape}."
        )
    if crs is None or crs.to_string() != ANALYSIS_CRS:
        raise RuntimeError("Downloaded predictor stack CRS mismatch.")
    if not transform.almost_equals(requested_transform):
        raise RuntimeError("Downloaded predictor transform differs from requested 20 m grid.")

    cube = np.moveaxis(
        predictor_bands.filled(np.nan), 0, -1
    ).astype(np.float64)
    flat = cube.reshape(-1, len(features))
    stack_valid = np.isfinite(flat).all(axis=1)

    kc = np.full(flat.shape[0], np.nan, dtype=float)
    if stack_valid.any():
        kc[stack_valid] = model.predict(flat[stack_valid])

    equal_di, equal_inside = score_unweighted_aoa(flat, equal_aoa)
    weighted_di, weighted_inside = score_unweighted_aoa(flat, weighted_aoa)

    source_shape = (source_height, source_width)
    kc = kc.reshape(source_shape)
    stack_valid = stack_valid.reshape(source_shape)
    equal_di = equal_di.reshape(source_shape)
    equal_inside = equal_inside.reshape(source_shape)
    weighted_di = weighted_di.reshape(source_shape)
    weighted_inside = weighted_inside.reshape(source_shape)

    buffer_pixels_float = float(PROCESSING_BUFFER_M) / float(PREDICTION_SCALE_M)
    buffer_pixels = int(round(buffer_pixels_float))
    if not math.isclose(buffer_pixels_float, buffer_pixels, abs_tol=1e-9):
        raise RuntimeError("Production buffer is not aligned with the 20 m grid.")
    rs = slice(buffer_pixels, buffer_pixels + tile.height_px)
    cs = slice(buffer_pixels, buffer_pixels + tile.width_px)

    result = {
        "kc_raw": kc[rs, cs].astype(np.float32),
        "stack_valid": stack_valid[rs, cs].astype(np.uint8),
        "equal_di": equal_di[rs, cs].astype(np.float32),
        "equal_inside": equal_inside[rs, cs].astype(np.uint8),
        "weighted_di": weighted_di[rs, cs].astype(np.float32),
        "weighted_inside": weighted_inside[rs, cs].astype(np.uint8),
    }
    temporary = path.with_suffix(".part.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **result)
    temporary.replace(path)
    meta_path.write_text(
        json.dumps(
            {
                "diagnostic_version": DIAGNOSTIC_VERSION,
                "scientific_signature": signature,
                "period_start": period_start,
                "tile_id": tile.tile_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def exact_support_summary(
    *,
    usable,
    kc_raw,
    basin_mask,
    edges,
    modis_et,
    support_threshold,
):
    coarse = np.asarray(edges.coarse_index, dtype=np.int64)
    fine = np.asarray(edges.fine_index, dtype=np.int64)
    area = np.asarray(edges.overlap_area_m2, dtype=float)
    represented = np.asarray(edges.represented_coarse, dtype=bool).ravel()

    coarse_count = modis_et.size
    coarse_area = np.bincount(coarse, weights=area, minlength=coarse_count)
    usable_flat = np.asarray(usable, dtype=bool).ravel()
    usable_area = np.bincount(
        coarse,
        weights=area * usable_flat[fine].astype(float),
        minlength=coarse_count,
    )
    fraction = np.full(coarse_count, np.nan, dtype=float)
    positive = coarse_area > 0
    fraction[positive] = usable_area[positive] / coarse_area[positive]

    finite_kc = usable_flat[fine] & np.isfinite(np.asarray(kc_raw).ravel()[fine])
    valid_kc_area = np.bincount(
        coarse[finite_kc], weights=area[finite_kc], minlength=coarse_count
    )
    eligible = (
        represented
        & np.isfinite(np.asarray(modis_et, dtype=float).ravel())
        & (valid_kc_area > 0)
        & (fraction >= support_threshold)
    )

    basin_flat = np.asarray(basin_mask, dtype=bool).ravel()
    basin_coarse = np.zeros(coarse_count, dtype=bool)
    basin_edges = basin_flat[fine]
    if basin_edges.any():
        basin_coarse[np.unique(coarse[basin_edges])] = True

    touched = np.bincount(fine, minlength=usable_flat.size) > 0
    bad_parent_count = np.bincount(
        fine,
        weights=(~eligible[coarse]).astype(np.int32),
        minlength=usable_flat.size,
    )
    all_parents_eligible = touched & (bad_parent_count == 0)
    publishable = usable_flat & all_parents_eligible

    relevant = represented & basin_coarse & np.isfinite(np.asarray(modis_et).ravel())
    return {
        "represented_valid_parents_touching_basin": int(relevant.sum()),
        "support90_eligible_parents_touching_basin": int((eligible & basin_coarse).sum()),
        "support90_eligible_parent_fraction_touching_basin": float(
            (eligible & basin_coarse).sum() / relevant.sum()
        ) if relevant.any() else float("nan"),
        "median_usable_fraction_relevant_parents": float(
            np.nanmedian(fraction[relevant])
        ) if relevant.any() else float("nan"),
        "p10_usable_fraction_relevant_parents": float(
            np.nanquantile(fraction[relevant], 0.10)
        ) if relevant.any() else float("nan"),
        "publishable_basin_pixels_pre_reconciliation": int(
            (publishable & basin_flat).sum()
        ),
    }


def main() -> None:
    args = parse_arguments()
    os.environ.setdefault("ET_START_DATE", "2020-01-01")
    os.environ.setdefault("ET_END_DATE_EXCLUSIVE", "2025-01-01")

    # Heavy/project imports occur only after the period environment is fixed.
    import ee
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    from et_downscaling.aoa_ridge25 import (
        AOAParameters,
        build_unweighted_aoa,
        score_unweighted_aoa,
    )
    from et_downscaling.config import ANALYSIS_CRS
    from et_downscaling.local_reconciliation import RIDGE25_USABLE_SUPPORT_FRACTION
    from et_downscaling.local_tiles import _analysis_geometry
    from et_downscaling.overlap_reconciliation import build_overlap_edges
    from et_downscaling.production import PREDICTION_SCALE_M, PROCESSING_BUFFER_M
    from et_downscaling.ridge25 import (
        RIDGE25_MODEL_FEATURES,
        build_ridge25_model,
        extract_ridge25_parameters,
    )
    from et_downscaling.ridge25_local_production import (
        _download_ee_bytes,
        _read_downloaded_array,
    )
    from et_downscaling.ridge25_overlap_production import (
        _download_native_modis,
        _processing_grid,
        _support_tiles,
    )
    from et_downscaling.ridge25_production import build_ridge25_production_stack

    root = project_root()
    workspace = resolve_workspace(args.workspace)
    run_directory = resolve_run_directory(workspace, args.run_directory)
    tables = run_directory / "tables"
    project_id = resolve_project_id(args.project)

    population = pd.read_csv(
        tables / "ridge25_training_population.csv",
        dtype={"station_id": str},
    )
    saved_parameters = pd.read_csv(tables / "ridge25_model_parameters.csv")
    features = list(RIDGE25_MODEL_FEATURES)
    if saved_parameters["feature"].astype(str).tolist() != features:
        raise RuntimeError("Saved model feature order differs from Ridge-25 specification.")

    model = build_ridge25_model()
    model.fit(
        population[features].to_numpy(dtype=float),
        population["Kc_target"].to_numpy(dtype=float),
    )
    means, scales, coefficients, intercept = extract_ridge25_parameters(model, features)
    for name, fitted, saved in (
        ("scaler_mean", means, saved_parameters["scaler_mean"].to_numpy(dtype=float)),
        ("scaler_scale", scales, saved_parameters["scaler_scale"].to_numpy(dtype=float)),
        (
            "ridge_coefficient_standardized",
            coefficients,
            saved_parameters["ridge_coefficient_standardized"].to_numpy(dtype=float),
        ),
    ):
        if not np.allclose(fitted, saved, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Refitted Ridge-25 {name} does not reproduce saved parameters.")

    equal_aoa = build_unweighted_aoa(population, group_column="spatial_block")
    weighted_aoa, weights = build_weighted_aoa(
        population,
        features,
        coefficients,
        AOAParameters,
    )
    signature = scientific_signature(
        (means, scales, coefficients, [intercept]), equal_aoa, weighted_aoa
    )

    output_root = workspace / "diagnostics" / "closure_tests" / "aoa_map_sensitivity"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("ET FUNDACION - AOA MAP SENSITIVITY")
    print("=" * 88)
    print("Run:", run_directory.name)
    print("Dates:", ", ".join(args.dates))
    print("Equal-weight threshold:", f"{equal_aoa.threshold:.6f}")
    print("Coefficient-weighted threshold:", f"{weighted_aoa.threshold:.6f}")
    print("Earth Engine project:", project_id)
    print("Google Drive: DISABLED")
    print("Persistent EE assets: DISABLED")
    print()

    ee.Initialize(project=project_id)
    ee.Number(1).getInfo()

    support_tiles, support_bounds, _basin_grid_bounds = _support_tiles(
        project_root=root,
        tile_size_m=args.tile_size_m,
    )
    scale = float(PREDICTION_SCALE_M)
    xmin, ymin, xmax, ymax = support_bounds
    width = int(round((xmax - xmin) / scale))
    height = int(round((ymax - ymin) / scale))
    support_transform = from_origin(xmin, ymax, scale, scale)
    basin_geometry = _analysis_geometry(root)
    basin_mask = rasterize(
        [(basin_geometry, 1)],
        out_shape=(height, width),
        transform=support_transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)

    all_rows: list[dict[str, object]] = []
    for period_start in args.dates:
        print("-" * 88)
        print("Period:", period_start)
        date_directory = output_root / period_start
        tile_cache = date_directory / "tiles"
        date_directory.mkdir(parents=True, exist_ok=True)

        kc_raw = np.full((height, width), np.nan, dtype=np.float32)
        stack_valid = np.zeros((height, width), dtype=bool)
        equal_inside = np.zeros((height, width), dtype=bool)
        weighted_inside = np.zeros((height, width), dtype=bool)
        equal_di = np.full((height, width), np.nan, dtype=np.float32)
        weighted_di = np.full((height, width), np.nan, dtype=np.float32)
        domain = np.zeros((height, width), dtype=bool)

        for index, tile in enumerate(support_tiles, start=1):
            print(f"  [{index}/{len(support_tiles)}] {tile.tile_id}")
            state = load_or_build_tile(
                tile=tile,
                period_start=period_start,
                cache_directory=tile_cache,
                signature=signature,
                restart=args.restart,
                model=model,
                equal_aoa=equal_aoa,
                weighted_aoa=weighted_aoa,
                features=features,
                timeout_seconds=args.timeout_seconds,
                ee=ee,
                ANALYSIS_CRS=ANALYSIS_CRS,
                PREDICTION_SCALE_M=PREDICTION_SCALE_M,
                PROCESSING_BUFFER_M=PROCESSING_BUFFER_M,
                build_ridge25_production_stack=build_ridge25_production_stack,
                score_unweighted_aoa=score_unweighted_aoa,
                _download_ee_bytes=_download_ee_bytes,
                _read_downloaded_array=_read_downloaded_array,
                _processing_grid=_processing_grid,
            )
            row0, col0 = tile_offsets(tile, support_bounds, scale)
            rs = slice(row0, row0 + tile.height_px)
            cs = slice(col0, col0 + tile.width_px)
            kc_raw[rs, cs] = state["kc_raw"]
            stack_valid[rs, cs] = state["stack_valid"].astype(bool)
            equal_inside[rs, cs] = state["equal_inside"].astype(bool)
            weighted_inside[rs, cs] = state["weighted_inside"].astype(bool)
            equal_di[rs, cs] = state["equal_di"]
            weighted_di[rs, cs] = state["weighted_di"]
            domain[rs, cs] = True

        equal_usable = stack_valid & equal_inside & np.isfinite(kc_raw) & (kc_raw >= 0)
        weighted_usable = stack_valid & weighted_inside & np.isfinite(kc_raw) & (kc_raw >= 0)

        print("  Downloading native-grid MODIS for support90 comparison...")
        modis_et, modis_grid = _download_native_modis(
            period_start=period_start,
            support_bounds=support_bounds,
            timeout_seconds=args.timeout_seconds,
        )
        print("  Building exact fine/coarse overlap operator once...")
        edges = build_overlap_edges(
            domain=domain,
            fine_transform=support_transform,
            fine_crs=ANALYSIS_CRS,
            modis_et=modis_et,
            modis_transform=modis_grid.transform,
            modis_crs=modis_grid.local_crs,
            progress_every=1000,
        )

        equal_support = exact_support_summary(
            usable=equal_usable,
            kc_raw=kc_raw,
            basin_mask=basin_mask,
            edges=edges,
            modis_et=modis_et,
            support_threshold=RIDGE25_USABLE_SUPPORT_FRACTION,
        )
        weighted_support = exact_support_summary(
            usable=weighted_usable,
            kc_raw=kc_raw,
            basin_mask=basin_mask,
            edges=edges,
            modis_et=modis_et,
            support_threshold=RIDGE25_USABLE_SUPPORT_FRACTION,
        )

        basin_stack = basin_mask & stack_valid
        basin_negative = basin_stack & np.isfinite(kc_raw) & (kc_raw < 0)
        valid_stack_count = int(basin_stack.sum())
        disagreement = basin_stack & (equal_inside != weighted_inside)

        for method, inside, di, usable, support in (
            ("equal_weight", equal_inside, equal_di, equal_usable, equal_support),
            (
                "abs_standardized_ridge_coefficient",
                weighted_inside,
                weighted_di,
                weighted_usable,
                weighted_support,
            ),
        ):
            inside_basin_stack = basin_stack & inside
            di_values = di[basin_stack & np.isfinite(di)]
            negative_inside = basin_negative & inside
            row = {
                "period_start": period_start,
                "method": method,
                "basin_pixels": int(basin_mask.sum()),
                "stack_valid_basin_pixels": valid_stack_count,
                "stack_valid_basin_fraction": float(valid_stack_count / basin_mask.sum()),
                "AOA_inside_basin_stack_pixels": int(inside_basin_stack.sum()),
                "AOA_inside_fraction_of_stack_valid": float(
                    inside_basin_stack.sum() / valid_stack_count
                ) if valid_stack_count else float("nan"),
                "usable_basin_pixels_before_support90": int((basin_mask & usable).sum()),
                "usable_fraction_of_stack_valid": float(
                    (basin_mask & usable).sum() / valid_stack_count
                ) if valid_stack_count else float("nan"),
                "negative_Kc_basin_stack_pixels": int(basin_negative.sum()),
                "negative_Kc_inside_AOA_pixels": int(negative_inside.sum()),
                "negative_Kc_inside_AOA_fraction": float(
                    negative_inside.sum() / basin_negative.sum()
                ) if basin_negative.any() else 0.0,
                "DI_median": float(np.nanmedian(di_values)) if di_values.size else float("nan"),
                "DI_p90": (
                    float(np.nanquantile(di_values, 0.90))
                    if di_values.size
                    else float("nan")
                ),
                "DI_p95": (
                    float(np.nanquantile(di_values, 0.95))
                    if di_values.size
                    else float("nan")
                ),
                "AOA_classification_disagreement_pixels": int(disagreement.sum()),
                "AOA_classification_disagreement_fraction_of_stack_valid": float(
                    disagreement.sum() / valid_stack_count
                ) if valid_stack_count else float("nan"),
                **support,
            }
            all_rows.append(row)

        date_rows = pd.DataFrame([row for row in all_rows if row["period_start"] == period_start])
        date_rows.to_csv(date_directory / "aoa_map_sensitivity_summary.csv", index=False)
        print(date_rows.to_string(index=False))
        print()

    summary = pd.DataFrame(all_rows)
    summary_path = output_root / "aoa_map_sensitivity_all_dates.csv"
    summary.to_csv(summary_path, index=False)

    weight_table = pd.DataFrame(
        {
            "feature": features,
            "abs_standardized_ridge_coefficient": weights,
            "squared_distance_contribution": weights ** 2,
        }
    )
    weight_table["squared_distance_contribution_fraction"] = (
        weight_table["squared_distance_contribution"]
        / weight_table["squared_distance_contribution"].sum()
    )
    weight_table.sort_values(
        "abs_standardized_ridge_coefficient", ascending=False
    ).to_csv(output_root / "coefficient_weights.csv", index=False)

    metadata = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "scientific_signature": signature,
        "run_directory": str(run_directory),
        "dates": list(args.dates),
        "tile_size_m": int(args.tile_size_m),
        "equal_weight_threshold": float(equal_aoa.threshold),
        "coefficient_weighted_threshold": float(weighted_aoa.threshold),
        "weight_definition": "absolute standardized Ridge-25 coefficient",
        "distance_weight_application": "standardized predictor multiplied directly by weight",
        "support_rule": float(RIDGE25_USABLE_SUPPORT_FRACTION),
        "reconciliation_solved": False,
        "google_drive_used": False,
        "persistent_earth_engine_asset_created": False,
        "decision_rule": (
            "This sensitivity must not select AOA by field R2, spatial OOF error, "
            "or larger mapped area. Compare stability, support90 impact, and the "
            "scientific interpretability of the coefficient-weighted domain."
        ),
    }
    (output_root / "aoa_map_sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("=" * 88)
    print("AOA MAP SENSITIVITY COMPLETE")
    print("=" * 88)
    print("Summary:", summary_path)
    print("Tile caches:", output_root)
    print("No production raster or production AOA was modified.")


if __name__ == "__main__":
    main()
