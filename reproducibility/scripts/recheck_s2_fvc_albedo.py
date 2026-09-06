"""Recheck Sentinel-2 FVC and albedo against the final Ridge feature set.

This diagnostic is deliberately narrow. It uses the current Sentinel-2
preprocessing, recalibrates FVC with the historical two-stage percentile
method, evaluates fold-specific training-only FVC transformations, and compares
Ridge with and without FVC/albedo. It does not tune Ridge and does not alter the
production feature specification or FVC configuration file.

Earth Engine work is performed only with ``--execute``. Annual temporary chunks
are resumable and are removed after successful merged outputs are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from et_downscaling.config import (
    ANALYSIS_CRS,
    S2_CLEAR_THRESHOLD,
    S2_DAILY_MOSAIC_SORT_PROPERTY,
    S2_PREPROCESSING_VERSION,
    S2_QA_BAND,
    build_training_output_filename,
)
from et_downscaling.modeling import (
    OPTICAL_COVERAGE_THRESHOLD_PCT,
    TARGET_COLUMN,
    build_ridge25_model,
    calculate_metrics,
    canonicalize_master,
)
from et_downscaling.ridge25 import RIDGE25_MODEL_FEATURES, RIDGE25_OPTICAL_FEATURES
from et_downscaling.workspace import get_workspace_paths


START_DATE = "2020-01-01"
END_DATE_EXCLUSIVE = "2025-01-01"
FVC_CALIBRATION_COVERAGE_THRESHOLD_PCT = 80.0
FVC_WITHIN_FOOTPRINT_LOW_PERCENTILE = 5
FVC_WITHIN_FOOTPRINT_HIGH_PERCENTILE = 95
FVC_GLOBAL_LOW_QUANTILE = 0.05
FVC_GLOBAL_HIGH_QUANTILE = 0.95
DIAGNOSTIC_VERSION = "s2-fvc-albedo-recheck-v1"
KEY_COLUMNS = ["station_id", "period_start"]
MODEL_CONFIGURATIONS = (
    "ridge25_base",
    "ridge25_plus_albedo",
    "ridge25_plus_fvc",
    "ridge25_plus_albedo_fvc",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def output_root() -> Path:
    workspace = get_workspace_paths(project_root()).ensure()
    return workspace.diagnostics / "s2_fvc_albedo_recheck"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Recalibrate Sentinel-2 FVC and re-evaluate FVC/albedo as "
            "incremental Ridge predictors."
        )
    )
    parser.add_argument("--project")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard resumable diagnostic chunks before execution.",
    )
    return parser.parse_args(argv)


def configuration_signature(
    stage: str,
    start: str,
    end_exclusive: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    signature = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "stage": stage,
        "start_date": start,
        "end_date_exclusive": end_exclusive,
        "s2_qa_band": S2_QA_BAND,
        "s2_clear_threshold": float(S2_CLEAR_THRESHOLD),
        "s2_daily_mosaic_sort_property": S2_DAILY_MOSAIC_SORT_PROPERTY,
        "s2_preprocessing_version": S2_PREPROCESSING_VERSION,
        "fvc_calibration_coverage_threshold_pct": (
            FVC_CALIBRATION_COVERAGE_THRESHOLD_PCT
        ),
    }
    if extra:
        signature.update(extra)
    return signature


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def annual_intervals() -> list[tuple[str, str]]:
    return [
        (f"{year}-01-01", f"{year + 1}-01-01")
        for year in range(2020, 2025)
    ]


def calculate_endmembers(candidates: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Apply the historical two-stage FVC endmember rule locally."""
    required = {
        "station_id",
        "optical_coverage_pct",
        "nonwater_pixel_count",
        "ndvi_p05_nonwater",
        "ndvi_p95_nonwater",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(
            "FVC candidate table is missing columns: " + ", ".join(missing)
        )

    data = candidates.copy()
    numeric = [
        "optical_coverage_pct",
        "nonwater_pixel_count",
        "ndvi_p05_nonwater",
        "ndvi_p95_nonwater",
    ]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data[["ndvi_p05_nonwater", "ndvi_p95_nonwater"]] = data[
        ["ndvi_p05_nonwater", "ndvi_p95_nonwater"]
    ].mask(lambda values: values <= -9990)

    eligible = data.loc[
        data["optical_coverage_pct"].ge(FVC_CALIBRATION_COVERAGE_THRESHOLD_PCT)
        & data["nonwater_pixel_count"].gt(0)
        & data["ndvi_p05_nonwater"].notna()
        & data["ndvi_p95_nonwater"].notna()
    ].copy()

    if eligible.empty:
        raise RuntimeError("No Sentinel-2 observations are eligible for FVC calibration.")

    low = float(
        eligible["ndvi_p05_nonwater"].quantile(
            FVC_GLOBAL_LOW_QUANTILE,
            interpolation="linear",
        )
    )
    high = float(
        eligible["ndvi_p95_nonwater"].quantile(
            FVC_GLOBAL_HIGH_QUANTILE,
            interpolation="linear",
        )
    )

    if not (-1.0 <= low < high <= 1.0):
        raise RuntimeError(f"Invalid recalibrated FVC endmembers: low={low}, high={high}")

    return (
        {
            "low": low,
            "high": high,
            "n": int(len(eligible)),
            "n_stations": int(eligible["station_id"].astype(str).nunique()),
        },
        eligible,
    )


def load_training_master() -> pd.DataFrame:
    workspace = get_workspace_paths(project_root()).ensure()
    path = (
        workspace.master
        / "S2"
        / build_training_output_filename("S2")
    )
    if not path.is_file():
        raise FileNotFoundError(
            "The current Sentinel-2 training master is required only for target, "
            "meteorology and fold definitions. Run the main pipeline at least once.\n"
            f"Missing: {path}"
        )
    return pd.read_csv(path, dtype={"station_id": str})


def build_calibration_manifest(candidates: pd.DataFrame, master: pd.DataFrame) -> dict[str, object]:
    canonical = canonicalize_master(master)
    fold_lookup = canonical[KEY_COLUMNS + ["spatial_block", "year"]].copy()
    fold_lookup["period_start"] = pd.to_datetime(
        fold_lookup["period_start"]
    ).dt.strftime("%Y-%m-%d")
    data = candidates.merge(
        fold_lookup,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if data[["spatial_block", "year"]].isna().any().any():
        raise RuntimeError("FVC calibration candidates could not be assigned to all folds.")

    global_result, _ = calculate_endmembers(data)
    spatial: dict[str, dict[str, object]] = {}
    spatial_group_to_variant: dict[str, str] = {}

    for fold_number, group in enumerate(
        sorted(data["spatial_block"].astype(str).unique()),
        start=1,
    ):
        variant = f"spatial_fold_{fold_number}"
        training = data.loc[data["spatial_block"].astype(str).ne(group)].copy()
        if training["spatial_block"].astype(str).eq(group).any():
            raise RuntimeError("Spatial validation candidates entered FVC calibration.")
        result, _ = calculate_endmembers(training)
        spatial[variant] = {
            "excluded_spatial_block": group,
            **result,
        }
        spatial_group_to_variant[group] = variant

    temporal: dict[str, dict[str, object]] = {}
    year_to_variant: dict[str, str] = {}
    for year in sorted(int(value) for value in data["year"].unique()):
        variant = f"temporal_{year}"
        training = data.loc[pd.to_numeric(data["year"]).ne(year)].copy()
        if pd.to_numeric(training["year"]).eq(year).any():
            raise RuntimeError("Temporal validation candidates entered FVC calibration.")
        result, _ = calculate_endmembers(training)
        temporal[variant] = {
            "excluded_year": year,
            **result,
        }
        year_to_variant[str(year)] = variant

    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "source": "Sentinel-2",
        "analysis_start": START_DATE,
        "analysis_end_exclusive": END_DATE_EXCLUSIVE,
        "method": "two_stage_global_percentile",
        "water_rule": "NDWI <= 0",
        "coverage_threshold_pct": FVC_CALIBRATION_COVERAGE_THRESHOLD_PCT,
        "within_footprint_percentiles": [
            FVC_WITHIN_FOOTPRINT_LOW_PERCENTILE,
            FVC_WITHIN_FOOTPRINT_HIGH_PERCENTILE,
        ],
        "across_observation_quantiles": [
            FVC_GLOBAL_LOW_QUANTILE,
            FVC_GLOBAL_HIGH_QUANTILE,
        ],
        "global_2020_2024": global_result,
        "spatial_training_only": spatial,
        "temporal_training_only": temporal,
        "spatial_group_to_variant": spatial_group_to_variant,
        "year_to_variant": year_to_variant,
        "production_fvc_config_changed": False,
        "production_ridge_feature_list_changed": False,
    }


def calibration_variants(manifest: dict[str, object]) -> dict[str, tuple[float, float]]:
    global_record = manifest["global_2020_2024"]
    variants = {
        "global_2020_2024": (
            float(global_record["low"]),
            float(global_record["high"]),
        )
    }
    for family in ("spatial_training_only", "temporal_training_only"):
        for name, record in manifest[family].items():
            variants[name] = (float(record["low"]), float(record["high"]))
    return variants


def _safe_ratio(numerator, denominator, name):
    import ee

    return (
        ee.Image(numerator)
        .divide(denominator)
        .updateMask(ee.Image(denominator).abs().gt(1e-6))
        .rename(name)
        .toFloat()
    )


def build_candidate_collection(modis_inputs, s2_collection, start: str, end_exclusive: str):
    import ee

    from et_downscaling.availability_diagnostic import _period_context, _period_values
    from et_downscaling.sentinel2 import build_s2_medoid

    images, image_indexes, footprints, footprint_indexes = _period_context(
        modis_inputs,
        start,
        end_exclusive,
    )

    def process_image(image_index):
        period_start, period_end, _ = _period_values(images.get(image_index))

        def process_footprint(footprint_index):
            footprint = ee.Feature(footprints.get(footprint_index))
            geometry = footprint.geometry()
            period = (
                ee.ImageCollection(s2_collection)
                .filterDate(period_start, period_end)
                .filterBounds(geometry)
            )
            medoid = build_s2_medoid(period, geometry)
            valid = (
                medoid.select(["Green", "Red", "NIR"])
                .mask()
                .reduce(ee.Reducer.min())
                .rename("valid")
                .uint8()
            )
            coverage_raw = valid.unmask(0).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                crs=ANALYSIS_CRS,
                scale=20,
                maxPixels=10_000_000,
                tileScale=4,
            ).get("valid")
            coverage = ee.Number(
                ee.Algorithms.If(
                    ee.Algorithms.IsEqual(coverage_raw, None),
                    0,
                    ee.Number(coverage_raw).multiply(100),
                )
            )

            nir = medoid.select("NIR")
            red = medoid.select("Red")
            green = medoid.select("Green")
            ndvi = _safe_ratio(nir.subtract(red), nir.add(red), "NDVI")
            ndwi = _safe_ratio(green.subtract(nir), green.add(nir), "NDWI")
            nonwater_ndvi = ndvi.updateMask(valid).updateMask(ndwi.lte(0))
            stats = ee.Dictionary(
                nonwater_ndvi.reduceRegion(
                    reducer=ee.Reducer.percentile(
                        [
                            FVC_WITHIN_FOOTPRINT_LOW_PERCENTILE,
                            FVC_WITHIN_FOOTPRINT_HIGH_PERCENTILE,
                        ],
                        ["p05", "p95"],
                    ).combine(ee.Reducer.count(), sharedInputs=True),
                    geometry=geometry,
                    crs=ANALYSIS_CRS,
                    scale=20,
                    maxPixels=10_000_000,
                    tileScale=4,
                )
            )
            p05 = stats.get("NDVI_p05", -9999)
            p95 = stats.get("NDVI_p95", -9999)
            count = ee.Number(stats.get("NDVI_count", 0))
            valid_for_calibration = (
                coverage.gte(FVC_CALIBRATION_COVERAGE_THRESHOLD_PCT)
                .And(count.gt(0))
                .int()
            )

            return ee.Feature(
                None,
                {
                    "station_id": footprint.get("station_id"),
                    "period_start": period_start.format("yyyy-MM-dd"),
                    "optical_products": period.size(),
                    "optical_unique_dates": ee.List(
                        period.aggregate_array("date_key")
                    ).distinct().size(),
                    "optical_coverage_pct": coverage,
                    "nonwater_pixel_count": count,
                    "ndvi_p05_nonwater": p05,
                    "ndvi_p95_nonwater": p95,
                    "valid_for_fvc_calibration": valid_for_calibration,
                },
            )

        return footprint_indexes.map(process_footprint)

    return ee.FeatureCollection(image_indexes.map(process_image).flatten())


def build_predictor_collection(
    modis_inputs,
    s2_collection,
    start: str,
    end_exclusive: str,
    calibration_manifest: dict[str, object],
):
    import ee

    from et_downscaling.availability_diagnostic import _period_context, _period_values
    from et_downscaling.optical import get_optical_coverage
    from et_downscaling.sentinel2 import add_s2_indices, build_s2_medoid

    images, image_indexes, footprints, footprint_indexes = _period_context(
        modis_inputs,
        start,
        end_exclusive,
    )
    variants = calibration_variants(calibration_manifest)
    optical_bands = [feature.removesuffix("_mean") for feature in RIDGE25_OPTICAL_FEATURES]
    fvc_band_names = [f"FVC_{name}" for name in variants]
    reduce_bands = optical_bands + ["Albedo"] + fvc_band_names
    output_names = list(RIDGE25_OPTICAL_FEATURES) + ["Albedo_mean"] + [
        f"{band}_mean" for band in fvc_band_names
    ]

    def process_image(image_index):
        period_start, period_end, _ = _period_values(images.get(image_index))

        def process_footprint(footprint_index):
            footprint = ee.Feature(footprints.get(footprint_index))
            geometry = footprint.geometry()
            period = (
                ee.ImageCollection(s2_collection)
                .filterDate(period_start, period_end)
                .filterBounds(geometry)
            )
            medoid = build_s2_medoid(period, geometry)
            stack = add_s2_indices(medoid)
            ndvi = stack.select("NDVI")

            for name, (low, high) in variants.items():
                fvc = (
                    ndvi.subtract(low)
                    .divide(high - low)
                    .clamp(0.0, 1.0)
                    .rename(f"FVC_{name}")
                    .toFloat()
                )
                stack = stack.addBands(fvc)

            raw_values = ee.Dictionary(
                stack.select(reduce_bands).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    crs=ANALYSIS_CRS,
                    scale=20,
                    maxPixels=10_000_000,
                    tileScale=4,
                )
            )
            values = ee.Dictionary.fromLists(
                output_names,
                [raw_values.get(band, -9999) for band in reduce_bands],
            )
            coverage = get_optical_coverage(period, geometry, "S2").multiply(100)

            return ee.Feature(
                None,
                {
                    "station_id": footprint.get("station_id"),
                    "period_start": period_start.format("yyyy-MM-dd"),
                    "optical_products": period.size(),
                    "optical_unique_dates": ee.List(
                        period.aggregate_array("date_key")
                    ).distinct().size(),
                    "optical_union_coverage_pct": coverage,
                },
            ).set(values)

        return footprint_indexes.map(process_footprint)

    return ee.FeatureCollection(image_indexes.map(process_image).flatten())


def export_chunk(
    stage: str,
    start: str,
    end_exclusive: str,
    feature_collection,
    selectors: list[str],
    force: bool,
    extra_signature: dict[str, object] | None = None,
) -> Path:
    from et_downscaling.export import export_feature_collection

    root = output_root()
    chunk_dir = root / "_chunks" / stage
    chunk_dir.mkdir(parents=True, exist_ok=True)
    year = pd.Timestamp(start).year
    csv_path = chunk_dir / f"{stage}_{year}.csv"
    manifest_path = csv_path.with_suffix(".manifest.json")
    expected = configuration_signature(
        stage,
        start,
        end_exclusive,
        extra=extra_signature,
    )

    if force:
        csv_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    if csv_path.is_file() and manifest_path.is_file():
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            actual = {}
        if (
            all(actual.get(key) == value for key, value in expected.items())
            and actual.get("csv_sha256") == sha256_file(csv_path)
        ):
            print(f"Reusing validated diagnostic chunk: {csv_path}")
            return csv_path

    relative = csv_path.relative_to(get_workspace_paths(project_root()).ensure().root)
    exported = Path(
        export_feature_collection(
            feature_collection,
            relative,
            selectors,
        )
    )
    manifest = {
        **expected,
        "rows": int(len(pd.read_csv(exported))),
        "csv_sha256": sha256_file(exported),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return exported


def merge_chunks(paths: list[Path], output_path: Path) -> pd.DataFrame:
    table = pd.concat(
        [pd.read_csv(path, dtype={"station_id": str}) for path in paths],
        ignore_index=True,
    )
    table["period_start"] = pd.to_datetime(table["period_start"]).dt.strftime("%Y-%m-%d")
    table = table.sort_values(KEY_COLUMNS).reset_index(drop=True)
    if table.duplicated(KEY_COLUMNS).any():
        raise RuntimeError(f"Duplicate station-period keys in {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def clean_numeric_predictors(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = table.copy()
    result[columns] = result[columns].apply(pd.to_numeric, errors="coerce")
    result[columns] = result[columns].mask(result[columns] <= -9990)
    return result


def build_screening_population(
    master: pd.DataFrame,
    predictors: pd.DataFrame,
    calibration_manifest: dict[str, object],
) -> pd.DataFrame:
    predictor_columns = (
        list(RIDGE25_OPTICAL_FEATURES)
        + ["Albedo_mean", "optical_union_coverage_pct"]
        + [f"FVC_{name}_mean" for name in calibration_variants(calibration_manifest)]
    )
    predictor_table = clean_numeric_predictors(predictors, predictor_columns)

    data = master.copy()
    data["station_id"] = data["station_id"].astype(str)
    data["period_start"] = pd.to_datetime(data["period_start"]).dt.strftime("%Y-%m-%d")
    replacement = predictor_table[KEY_COLUMNS + predictor_columns]
    data = data.drop(columns=[column for column in predictor_columns if column in data], errors="ignore")
    for feature in RIDGE25_OPTICAL_FEATURES:
        diagnostic_name = f"s2_{feature}"
        if diagnostic_name in data.columns:
            data = data.drop(columns=[diagnostic_name])
    data = data.merge(replacement, on=KEY_COLUMNS, how="left", validate="one_to_one")
    data["s2_coverage_pct"] = data["optical_union_coverage_pct"]
    canonical = canonicalize_master(data)

    eligible = (
        canonical["modis_good"].eq(1)
        & canonical["target_complete"].eq(1)
        & canonical[TARGET_COLUMN].notna()
        & canonical["s2_coverage_pct"].ge(OPTICAL_COVERAGE_THRESHOLD_PCT)
        & canonical[RIDGE25_MODEL_FEATURES].notna().all(axis=1)
    )
    population = canonical.loc[eligible].copy().sort_values(KEY_COLUMNS).reset_index(drop=True)

    extra_columns = ["Albedo_mean"] + [
        f"FVC_{name}_mean" for name in calibration_variants(calibration_manifest)
    ]
    population[extra_columns] = population[extra_columns].apply(pd.to_numeric, errors="coerce")
    if population[extra_columns].isna().any().any():
        missing = population[extra_columns].isna().sum()
        missing = missing.loc[missing.gt(0)].to_dict()
        raise RuntimeError(
            "Derived FVC/albedo predictors do not share the complete Ridge population: "
            f"{missing}"
        )
    return population


def _build_matrix(rows: pd.DataFrame, configuration: str, fvc_column: str) -> pd.DataFrame:
    matrix = rows[RIDGE25_MODEL_FEATURES].copy()
    if configuration in {"ridge25_plus_albedo", "ridge25_plus_albedo_fvc"}:
        matrix["Albedo_mean"] = rows["Albedo_mean"].to_numpy(dtype=float)
    if configuration in {"ridge25_plus_fvc", "ridge25_plus_albedo_fvc"}:
        matrix["FVC_recalibrated_mean"] = rows[fvc_column].to_numpy(dtype=float)
    return matrix


def run_screening(
    population: pd.DataFrame,
    calibration_manifest: dict[str, object],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    oof_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    spatial_mapping = calibration_manifest["spatial_group_to_variant"]
    temporal_mapping = calibration_manifest["year_to_variant"]

    for split_type, group_column, mapping in (
        ("spatial", "spatial_block", spatial_mapping),
        ("temporal", "year", temporal_mapping),
    ):
        groups = population[group_column].astype(str)
        for group in sorted(groups.unique()):
            test_mask = groups.eq(group).to_numpy()
            train_mask = ~test_mask
            variant = mapping[str(group)]
            fvc_column = f"FVC_{variant}_mean"

            for configuration in MODEL_CONFIGURATIONS:
                model = build_ridge25_model()
                train = population.loc[train_mask]
                test = population.loc[test_mask]
                model.fit(
                    _build_matrix(train, configuration, fvc_column),
                    train[TARGET_COLUMN],
                )
                prediction = model.predict(
                    _build_matrix(test, configuration, fvc_column)
                )
                metrics = calculate_metrics(test[TARGET_COLUMN], prediction)
                fold_rows.append(
                    {
                        "split_type": split_type,
                        "group": str(group),
                        "configuration": configuration,
                        "fvc_variant": variant,
                        **metrics,
                    }
                )
                frame = test[
                    KEY_COLUMNS + ["spatial_block", "year", TARGET_COLUMN]
                ].copy()
                frame["split_type"] = split_type
                frame["group"] = str(group)
                frame["configuration"] = configuration
                frame["fvc_variant"] = variant
                frame["prediction"] = prediction
                frame["error"] = prediction - test[TARGET_COLUMN].to_numpy(dtype=float)
                oof_rows.append(frame)

    oof = pd.concat(oof_rows, ignore_index=True)
    folds = pd.DataFrame(fold_rows)

    overall_rows = []
    for (split_type, configuration), group in oof.groupby(
        ["split_type", "configuration"],
        sort=True,
    ):
        overall_rows.append(
            {
                "split_type": split_type,
                "configuration": configuration,
                **calculate_metrics(group[TARGET_COLUMN], group["prediction"]),
            }
        )
    overall = pd.DataFrame(overall_rows)

    station_rows = []
    for (split_type, configuration, station_id), group in oof.groupby(
        ["split_type", "configuration", "station_id"],
        sort=True,
    ):
        station_rows.append(
            {
                "split_type": split_type,
                "configuration": configuration,
                "station_id": station_id,
                **calculate_metrics(group[TARGET_COLUMN], group["prediction"]),
            }
        )
    by_station = pd.DataFrame(station_rows)

    q10 = float(population[TARGET_COLUMN].quantile(0.10))
    q90 = float(population[TARGET_COLUMN].quantile(0.90))
    extreme_rows = []
    for (split_type, configuration), group in oof.groupby(
        ["split_type", "configuration"],
        sort=True,
    ):
        subsets = {
            "low_q10": group.loc[group[TARGET_COLUMN].le(q10)],
            "middle_80pct": group.loc[
                group[TARGET_COLUMN].gt(q10) & group[TARGET_COLUMN].lt(q90)
            ],
            "high_q10": group.loc[group[TARGET_COLUMN].ge(q90)],
        }
        for label, subset in subsets.items():
            extreme_rows.append(
                {
                    "split_type": split_type,
                    "configuration": configuration,
                    "target_subset": label,
                    "target_q10": q10,
                    "target_q90": q90,
                    **calculate_metrics(subset[TARGET_COLUMN], subset["prediction"]),
                }
            )
    extremes = pd.DataFrame(extreme_rows)

    paired_rows = []
    base = oof.loc[oof["configuration"].eq("ridge25_base")][
        ["split_type", *KEY_COLUMNS, TARGET_COLUMN, "prediction"]
    ].rename(columns={"prediction": "prediction_base"})
    for configuration in MODEL_CONFIGURATIONS[1:]:
        candidate = oof.loc[oof["configuration"].eq(configuration)][
            ["split_type", *KEY_COLUMNS, "prediction"]
        ].rename(columns={"prediction": "prediction_candidate"})
        paired = base.merge(
            candidate,
            on=["split_type", *KEY_COLUMNS],
            how="inner",
            validate="one_to_one",
        )
        paired["abs_error_base"] = (
            paired["prediction_base"] - paired[TARGET_COLUMN]
        ).abs()
        paired["abs_error_candidate"] = (
            paired["prediction_candidate"] - paired[TARGET_COLUMN]
        ).abs()
        paired["delta_abs_error_candidate_minus_base"] = (
            paired["abs_error_candidate"] - paired["abs_error_base"]
        )
        for split_type, group in paired.groupby("split_type", sort=True):
            delta = group["delta_abs_error_candidate_minus_base"]
            paired_rows.append(
                {
                    "split_type": split_type,
                    "configuration": configuration,
                    "n": int(len(group)),
                    "mean_delta_abs_error": float(delta.mean()),
                    "median_delta_abs_error": float(delta.median()),
                    "proportion_improved": float(delta.lt(0).mean()),
                    "proportion_worsened": float(delta.gt(0).mean()),
                }
            )
    paired_summary = pd.DataFrame(paired_rows)

    return oof, overall, folds, by_station, extremes, paired_summary


def execute(project_id: str, force: bool) -> dict[str, Path]:
    import ee

    from et_downscaling.availability_diagnostic import (
        get_dynamic_modis_inputs,
        get_dynamic_s2_collection,
    )

    root = output_root()
    root.mkdir(parents=True, exist_ok=True)
    if force:
        shutil.rmtree(root / "_chunks", ignore_errors=True)

    print("Initializing Earth Engine...")
    ee.Initialize(project=project_id)
    modis_inputs = get_dynamic_modis_inputs(START_DATE, END_DATE_EXCLUSIVE)
    s2_collection = get_dynamic_s2_collection(
        modis_inputs["station_footprints"],
        START_DATE,
        END_DATE_EXCLUSIVE,
    )

    candidate_selectors = [
        "station_id",
        "period_start",
        "optical_products",
        "optical_unique_dates",
        "optical_coverage_pct",
        "nonwater_pixel_count",
        "ndvi_p05_nonwater",
        "ndvi_p95_nonwater",
        "valid_for_fvc_calibration",
    ]
    candidate_chunks = []
    print("\n=== FVC CALIBRATION CANDIDATES ===")
    for start, end_exclusive in annual_intervals():
        print(start, "to", end_exclusive)
        collection = build_candidate_collection(
            modis_inputs,
            s2_collection,
            start,
            end_exclusive,
        )
        candidate_chunks.append(
            export_chunk(
                "candidates",
                start,
                end_exclusive,
                collection,
                candidate_selectors,
                force=False,
            )
        )

    candidates_path = root / "s2_fvc_calibration_candidates_2020_2024.csv"
    candidates = merge_chunks(candidate_chunks, candidates_path)

    master = load_training_master()
    calibration = build_calibration_manifest(candidates, master)
    calibration["candidate_csv_sha256"] = sha256_file(candidates_path)
    calibration_path = root / "s2_fvc_calibration_manifest.json"
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    variants = calibration_variants(calibration)
    predictor_selectors = [
        "station_id",
        "period_start",
        "optical_products",
        "optical_unique_dates",
        "optical_union_coverage_pct",
        *RIDGE25_OPTICAL_FEATURES,
        "Albedo_mean",
        *[f"FVC_{name}_mean" for name in variants],
    ]
    predictor_chunks = []
    calibration_signature = canonical_json_sha256(
        calibration_variants(calibration)
    )
    print("\n=== RECALIBRATED FVC + ALBEDO PREDICTORS ===")
    for start, end_exclusive in annual_intervals():
        print(start, "to", end_exclusive)
        collection = build_predictor_collection(
            modis_inputs,
            s2_collection,
            start,
            end_exclusive,
            calibration,
        )
        predictor_chunks.append(
            export_chunk(
                "predictors",
                start,
                end_exclusive,
                collection,
                predictor_selectors,
                force=False,
                extra_signature={
                    "fvc_calibration_signature": calibration_signature,
                },
            )
        )

    predictors_path = root / "s2_fvc_albedo_predictors_2020_2024.csv"
    predictors = merge_chunks(predictor_chunks, predictors_path)
    key_check = candidates[KEY_COLUMNS].merge(
        predictors[KEY_COLUMNS],
        on=KEY_COLUMNS,
        how="outer",
        indicator=True,
    )
    if not key_check["_merge"].eq("both").all():
        raise RuntimeError(
            "Candidate and predictor exports returned different station-period universes."
        )

    population = build_screening_population(master, predictors, calibration)
    (
        oof,
        overall,
        folds,
        by_station,
        extremes,
        paired,
    ) = run_screening(population, calibration)

    outputs = {
        "candidates": candidates_path,
        "calibration_manifest": calibration_path,
        "predictors": predictors_path,
        "oof_predictions": root / "model_screening_oof_predictions.csv",
        "metrics_overall": root / "model_screening_metrics_overall.csv",
        "metrics_by_fold": root / "model_screening_metrics_by_fold.csv",
        "metrics_by_station": root / "model_screening_metrics_by_station.csv",
        "metrics_by_target_subset": root / "model_screening_metrics_by_target_subset.csv",
        "paired_error_deltas": root / "model_screening_paired_error_deltas.csv",
        "screening_manifest": root / "model_screening_manifest.json",
    }
    oof.to_csv(outputs["oof_predictions"], index=False)
    overall.to_csv(outputs["metrics_overall"], index=False)
    folds.to_csv(outputs["metrics_by_fold"], index=False)
    by_station.to_csv(outputs["metrics_by_station"], index=False)
    extremes.to_csv(outputs["metrics_by_target_subset"], index=False)
    paired.to_csv(outputs["paired_error_deltas"], index=False)

    screening_manifest = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "source": "Sentinel-2",
        "training_population_rows": int(len(population)),
        "training_population_rule": (
            "MODIS good + complete target + S2 union coverage >= 90% + complete Ridge-25 predictors"
        ),
        "ridge_alpha": 1.0,
        "base_predictor_count": len(RIDGE25_MODEL_FEATURES),
        "configurations": list(MODEL_CONFIGURATIONS),
        "fvc_fold_specific_calibration": True,
        "albedo_formula_changed": False,
        "production_feature_list_changed": False,
        "winner_declared": False,
        "candidate_csv_sha256": sha256_file(candidates_path),
        "predictor_csv_sha256": sha256_file(predictors_path),
    }
    outputs["screening_manifest"].write_text(
        json.dumps(screening_manifest, indent=2),
        encoding="utf-8",
    )

    shutil.rmtree(root / "_chunks", ignore_errors=True)

    print("\n=== FVC ENDMEMBERS ===")
    print(json.dumps(calibration["global_2020_2024"], indent=2))
    print("\n=== MODEL SCREENING ===")
    print(overall.to_string(index=False))
    print("\n=== PAIRED ABSOLUTE-ERROR DELTAS VS RIDGE-25 ===")
    print(paired.to_string(index=False))
    print("\nOutputs:", root)
    return outputs


def main(argv=None) -> int:
    args = parse_arguments(argv)
    plan = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "analysis_start": START_DATE,
        "analysis_end_exclusive": END_DATE_EXCLUSIVE,
        "s2_clear_threshold": float(S2_CLEAR_THRESHOLD),
        "s2_daily_mosaic_sort_property": S2_DAILY_MOSAIC_SORT_PROPERTY,
        "s2_preprocessing_version": S2_PREPROCESSING_VERSION,
        "fvc_calibration_threshold_pct": FVC_CALIBRATION_COVERAGE_THRESHOLD_PCT,
        "model_configurations": list(MODEL_CONFIGURATIONS),
        "production_configuration_modified": False,
        "output_root": str(output_root()),
    }
    print(json.dumps(plan, indent=2))

    if not args.execute:
        print("Dry run only. Add --execute --project <PROJECT_ID> to run Earth Engine.")
        return 0
    if not args.project:
        raise ValueError("--execute requires --project")

    execute(args.project, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
