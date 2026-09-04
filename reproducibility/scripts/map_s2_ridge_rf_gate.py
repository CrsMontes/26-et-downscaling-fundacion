"""Cartographic stress test of fixed S2 Ridge and RF candidates.

The only remote operation is a resumable extraction of the same 20 predictor
bands over the five station MODIS-parent footprints for two preregistered
periods. Model fitting, DI/AOA, reconciliation, summaries, and maps are local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.base import clone

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import screen_feature_families as families
import screen_optical_algorithms as algorithms


PERIODS = {
    "extreme_dry": "2020-03-13",
    "extreme_wet": "2021-11-25",
}
FEATURES = families.S2_COMMON + families.ETO_DRIVERS + families.SEASONALITY
MODEL_NAMES = ("ridge", "random_forest")
STATION_LABELS = {
    "ST01": "Clean pasture", "ST02": "Oil palm", "ST03": "Banana",
    "ST04": "Mangrove", "ST05": "Dry forest",
}
MISSING = -9999.0


def project_root():
    return Path(__file__).resolve().parents[2]


def output_root():
    return project_root() / "outputs/diagnostics/2020_2024/s2_ridge_rf_cartographic_gate"


def canonical_hash(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_training():
    data, folds, _, _ = families.load_fixed_population(project_root())
    if len(data) != 550 or data[FEATURES].isna().any().any():
        raise RuntimeError("The fixed 550-row GE90 training matrix is invalid")
    if len(FEATURES) != 20 or len(set(FEATURES)) != 20:
        raise RuntimeError("The gate requires exactly 20 unique predictors")
    return data, folds


def fit_models(training):
    templates = algorithms.build_algorithms()
    fitted = {}
    for name in MODEL_NAMES:
        model = clone(templates[name])
        model.fit(training[FEATURES], training.Kc_target)
        fitted[name] = model
    return fitted


def model_weights(model_name, model):
    if model_name == "random_forest":
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        values = np.abs(np.asarray(model.named_steps["regressor"].coef_, dtype=float))
    if len(values) != len(FEATURES) or np.any(values < 0) or values.sum() <= 0:
        raise RuntimeError(f"Invalid AOA weights for {model_name}")
    return values / values.sum()


def build_aoa_spec(training, model_name, model):
    """Use identical scaling, DI, and threshold rule with model-specific weights."""
    values = training[FEATURES].to_numpy(float)
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0, ddof=1)
    if np.any(standard_deviations == 0):
        raise RuntimeError("Zero-variance AOA predictor")
    weights = model_weights(model_name, model)
    weighted = ((values - means) / standard_deviations) * weights
    distances = cdist(weighted, weighted)
    mean_training_distance = distances[np.triu_indices(len(values), 1)].mean()
    groups = training.spatial_block.astype(str).to_numpy()
    train_di = np.array([
        distances[index, groups != groups[index]].min() / mean_training_distance
        for index in range(len(values))
    ])
    q1, q3 = np.quantile(train_di, [0.25, 0.75])
    upper_fence = q3 + 1.5 * (q3 - q1)
    threshold = train_di[train_di <= upper_fence].max()
    return {
        "model": model_name, "features": FEATURES,
        "means": means.tolist(), "standard_deviations": standard_deviations.tolist(),
        "weights": weights.tolist(), "weighted_training_reference": weighted.tolist(),
        "mean_training_distance": float(mean_training_distance),
        "train_di": train_di.tolist(), "threshold": float(threshold),
        "threshold_method": "observed_non_outlier_max",
        "training_rows": len(training), "spatial_groups": int(pd.Series(groups).nunique()),
    }


def calculate_di(predictors, specification):
    values = predictors[FEATURES].to_numpy(float)
    weighted = (
        (values - np.asarray(specification["means"]))
        / np.asarray(specification["standard_deviations"])
    ) * np.asarray(specification["weights"])
    nearest = cdist(weighted, np.asarray(specification["weighted_training_reference"])).min(axis=1)
    di = nearest / specification["mean_training_distance"]
    return di, di <= specification["threshold"]


def cache_path(period_start):
    return output_root() / "predictor_cache_v2_training_context" / f"s2_20m_{period_start.replace('-', '')}.csv"


def extraction_manifest(period_start):
    return {
        "period_start": period_start, "features": FEATURES,
        "feature_count": 20, "grid_crs": "EPSG:32618", "grid_scale_m": 20,
        "domain": "five_station_MODIS_parent_footprints",
        "coarse_context": "local station-period ETo-driver values held constant within each MODIS parent footprint",
        "missing_sentinel": MISSING, "model_training_performed_remotely": False,
    }


def validate_cache(path, period_start):
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        return False
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = extraction_manifest(period_start)
    if saved != expected:
        return False
    data = pd.read_csv(path)
    required = set(FEATURES + ["station_id", "x", "y", "predictor_valid", "ET_MODIS_mm_period"])
    return required.issubset(data.columns) and len(data) > 0


def build_remote_samples(period_start):
    import ee
    from et_downscaling.availability_diagnostic import get_dynamic_modis_inputs
    from et_downscaling.production import (
        build_harmonic_predictors, build_modis_period_context,
        build_s2_common_predictors, get_fine_projection,
    )

    inputs = get_dynamic_modis_inputs(
        period_start,
        (pd.Timestamp(period_start) + pd.Timedelta(days=9)).date().isoformat(),
    )
    footprints = ee.FeatureCollection(inputs["station_footprints"])
    geometry = footprints.geometry()
    projection = get_fine_projection()
    context = build_modis_period_context(period_start, geometry)
    optical, _ = build_s2_common_predictors(
        footprints, context["period_start"], context["period_end"], geometry, projection
    )
    optical = optical.rename(families.S2_COMMON)
    store = pd.read_csv(
        project_root() / "outputs/diagnostics/2020_2024/experimental_feature_store/feature_store.csv",
        dtype={"station_id": str},
    )
    period_context = store.loc[
        pd.to_datetime(store.period_start).dt.strftime("%Y-%m-%d").eq(period_start),
        ["station_id", *families.ETO_DRIVERS],
    ]
    if set(period_context.station_id) != set(STATION_LABELS):
        raise RuntimeError(f"Incomplete station-period ETo context for {period_start}")
    context_images = []
    for row in period_context.itertuples(index=False):
        footprint = ee.Feature(
            footprints.filter(ee.Filter.eq("station_id", row.station_id)).first()
        )
        values = [float(getattr(row, feature)) for feature in families.ETO_DRIVERS]
        context_images.append(
            ee.Image.constant(values).rename(families.ETO_DRIVERS)
            .clip(footprint.geometry()).toFloat()
        )
    meteorology = ee.ImageCollection.fromImages(context_images).mosaic()
    harmonics = build_harmonic_predictors(period_start, geometry)
    stack = optical.addBands(meteorology).addBands(harmonics).select(FEATURES).reproject(projection)
    valid = stack.mask().reduce(ee.Reducer.min()).rename("predictor_valid").uint8()
    coords = ee.Image.pixelCoordinates(projection).rename(["x", "y"])
    modis_et = context["modis_et"].reproject(projection).rename("ET_MODIS_mm_period")
    export_image = stack.unmask(MISSING).addBands(valid).addBands(coords).addBands(modis_et.unmask(MISSING))
    return export_image.sampleRegions(
        collection=footprints, properties=["station_id"], scale=20,
        projection=projection, geometries=False, tileScale=4,
    )


def extract_caches(project):
    # Set the approved experiment period before importing modules whose
    # collection helpers read the analysis window at import time.
    os.environ["ET_START_DATE"] = "2020-01-01"
    os.environ["ET_END_DATE_EXCLUSIVE"] = "2025-01-01"
    import ee
    from et_downscaling.export import export_feature_collection

    ee.Initialize(project=project)
    records = []
    selectors = ["station_id", *FEATURES, "predictor_valid", "x", "y", "ET_MODIS_mm_period"]
    for period_start in PERIODS.values():
        path = cache_path(period_start)
        if validate_cache(path, period_start):
            records.append({"period_start": period_start, "status": "reused", "path": str(path)})
            continue
        if path.exists() or path.with_suffix(".manifest.json").exists():
            raise RuntimeError(f"Incomplete or incompatible predictor cache: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        relative = path.relative_to(project_root() / "outputs")
        result = Path(export_feature_collection(
            build_remote_samples(period_start), str(relative), selectors
        ))
        result.with_suffix(".manifest.json").write_text(
            json.dumps(extraction_manifest(period_start), indent=2), encoding="utf-8"
        )
        if not validate_cache(result, period_start):
            raise RuntimeError(f"Downloaded predictor cache failed validation: {result}")
        records.append({"period_start": period_start, "status": "downloaded", "path": str(result)})
    return records


def summarize_distribution(values):
    values = pd.Series(values).dropna().astype(float)
    quantiles = values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": len(values), "minimum": values.min(), "p01": quantiles.loc[0.01],
        "p05": quantiles.loc[0.05], "p25": quantiles.loc[0.25],
        "median": quantiles.loc[0.5], "p75": quantiles.loc[0.75],
        "p95": quantiles.loc[0.95], "p99": quantiles.loc[0.99],
        "maximum": values.max(), "mean": values.mean(), "sd": values.std(),
    }


def process_period(period_label, period_start, training, models, aoa_specs):
    raw = pd.read_csv(cache_path(period_start), dtype={"station_id": str})
    raw["period_label"] = period_label
    raw["period_start"] = period_start
    valid = raw.predictor_valid.eq(1) & raw[FEATURES].ne(MISSING).all(axis=1)
    for name, model in models.items():
        raw[f"Kc_raw_{name}"] = np.nan
        prediction = model.predict(raw.loc[valid, FEATURES])
        raw.loc[valid, f"Kc_raw_{name}"] = prediction
        di, inside = calculate_di(raw.loc[valid, FEATURES], aoa_specs[name])
        raw[f"DI_{name}"] = np.nan
        raw[f"AOA_{name}"] = False
        raw.loc[valid, f"DI_{name}"] = di
        raw.loc[valid, f"AOA_{name}"] = inside
    raw["common_AOA"] = valid & raw.AOA_ridge & raw.AOA_random_forest
    raw["AOA_disagreement"] = valid & raw.AOA_ridge.ne(raw.AOA_random_forest)

    target_min, target_max = training.Kc_target.min(), training.Kc_target.max()
    distribution_rows, parent_rows, conservation_rows = [], [], []
    for name in MODEL_NAMES:
        kc = f"Kc_raw_{name}"
        own = valid & raw[f"AOA_{name}"]
        common = raw.common_AOA
        for domain_name, mask in (("valid", valid), ("own_AOA", own), ("common_AOA", common)):
            summary = summarize_distribution(raw.loc[mask, kc])
            summary.update({
                "period_label": period_label, "period_start": period_start,
                "model": name, "domain": domain_name,
                "total_grid_pixels": len(raw), "coverage_fraction": float(mask.mean()),
                "negative_count": int((raw.loc[mask, kc] < 0).sum()),
                "outside_training_range_count": int(
                    ((raw.loc[mask, kc] < target_min) | (raw.loc[mask, kc] > target_max)).sum()
                ),
            })
            distribution_rows.append(summary)

        filled_column = f"Kc_filled_{name}"
        et_column = f"ET_reconciled_{name}"
        raw[filled_column] = np.nan
        raw[et_column] = np.nan
        for station_id, index in raw.groupby("station_id").groups.items():
            index = pd.Index(index)
            station_valid = valid.loc[index]
            valid_fraction = float(station_valid.mean())
            values = raw.loc[index, kc]
            if valid_fraction < 0.90 or not station_valid.any():
                continue
            filled = values.fillna(values.loc[station_valid].mean())
            modis_values = raw.loc[index, "ET_MODIS_mm_period"].replace(MISSING, np.nan).dropna()
            if modis_values.empty:
                continue
            modis_et = float(modis_values.median())
            scale = modis_et / filled.mean()
            reconciled = filled * scale
            raw.loc[index, filled_column] = filled
            raw.loc[index, et_column] = reconciled
            common_station = raw.loc[index, "common_AOA"]
            common_values = values.loc[common_station]
            if len(common_values):
                q = common_values.quantile([0.05, 0.25, 0.75, 0.95])
                parent_rows.append({
                    "period_label": period_label, "period_start": period_start,
                    "model": name, "station_id": station_id,
                    "coverage": STATION_LABELS[station_id], "n_common_AOA": len(common_values),
                    "Kc_sd": common_values.std(), "Kc_IQR": q.loc[0.75] - q.loc[0.25],
                    "Kc_p95_minus_p05": q.loc[0.95] - q.loc[0.05],
                })
            rho = spearmanr(filled, reconciled).statistic
            conservation_rows.append({
                "period_label": period_label, "period_start": period_start,
                "model": name, "station_id": station_id,
                "valid_fraction": valid_fraction,
                "filled_count": int((~station_valid).sum()),
                "modis_ET_mm_period": modis_et,
                "reaggregated_ET_mm_period": reconciled.mean(),
                "conservation_error_mm": reconciled.mean() - modis_et,
                "Kc_ET_spearman": rho,
            })
    raw["Kc_difference_ridge_minus_rf"] = raw.Kc_raw_ridge - raw.Kc_raw_random_forest
    return raw, distribution_rows, parent_rows, conservation_rows


def make_maps(data, period_label):
    columns = [
        ("Kc_raw_ridge", "Ridge Kc raw", "viridis"),
        ("Kc_raw_random_forest", "RF Kc raw", "viridis"),
        ("Kc_difference_ridge_minus_rf", "Ridge - RF", "coolwarm"),
        ("DI_ridge", "Ridge DI", "magma"), ("DI_random_forest", "RF DI", "magma"),
        ("AOA_ridge", "Ridge AOA", "gray"),
        ("AOA_random_forest", "RF AOA", "gray"),
        ("ET_reconciled_ridge", "Ridge reconciled ET", "viridis"),
        ("ET_reconciled_random_forest", "RF reconciled ET", "viridis"),
    ]
    figure, axes = plt.subplots(len(STATION_LABELS), len(columns), figsize=(21, 15))
    for row, (station_id, label) in enumerate(STATION_LABELS.items()):
        station = data.loc[data.station_id.eq(station_id)]
        for column_index, (column, title, cmap) in enumerate(columns):
            axis = axes[row, column_index]
            artist = axis.scatter(station.x, station.y, c=station[column], s=5, cmap=cmap)
            axis.set_aspect("equal"); axis.set_xticks([]); axis.set_yticks([])
            if row == 0: axis.set_title(title)
            if column_index == 0: axis.set_ylabel(f"{station_id}\n{label}")
            figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.02)
    figure.suptitle(f"S2 Ridge vs RF cartographic stress test: {period_label}")
    figure.tight_layout()
    path = output_root() / "maps" / f"comparison_{period_label}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project")
    parser.add_argument("--extract", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    training, _ = load_training()
    models = fit_models(training)
    specs = {name: build_aoa_spec(training, name, model) for name, model in models.items()}
    missing = [date for date in PERIODS.values() if not validate_cache(cache_path(date), date)]
    records = [
        {"period_start": date_value, "status": "reused", "path": str(cache_path(date_value))}
        for date_value in PERIODS.values()
    ]
    if missing:
        if not args.extract:
            raise RuntimeError(f"Missing predictor caches: {missing}; rerun with --extract --project PROJECT")
        if not args.project:
            raise ValueError("--project is required with --extract")
        records = extract_caches(args.project)

    output_root().mkdir(parents=True, exist_ok=True)
    joblib.dump(models, output_root() / "fitted_models_experimental.joblib")
    (output_root() / "aoa_specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    grids, distributions, parents, conservation, maps = [], [], [], [], []
    for label, date_value in PERIODS.items():
        grid, rows, parent_rows, conservation_rows = process_period(
            label, date_value, training, models, specs
        )
        grids.append(grid); distributions.extend(rows); parents.extend(parent_rows)
        conservation.extend(conservation_rows); maps.append(str(make_maps(grid, label)))
    combined = pd.concat(grids, ignore_index=True)
    combined.to_csv(output_root() / "pixel_predictions.csv", index=False)
    pd.DataFrame(distributions).to_csv(output_root() / "prediction_distributions.csv", index=False)
    pd.DataFrame(parents).to_csv(output_root() / "within_modis_variability.csv", index=False)
    pd.DataFrame(conservation).to_csv(output_root() / "reconciliation_diagnostics.csv", index=False)
    coverage = combined.groupby("period_label", as_index=False).agg(
        grid_pixels=("predictor_valid", "size"), valid_pixels=("predictor_valid", "sum"),
        ridge_AOA_pixels=("AOA_ridge", "sum"), rf_AOA_pixels=("AOA_random_forest", "sum"),
        common_AOA_pixels=("common_AOA", "sum"), AOA_disagreement_pixels=("AOA_disagreement", "sum"),
    )
    coverage.to_csv(output_root() / "aoa_coverage.csv", index=False)
    disagreement = combined.groupby(["period_label", "station_id"], as_index=False).agg(
        grid_pixels=("predictor_valid", "size"), valid_pixels=("predictor_valid", "sum"),
        ridge_AOA_pixels=("AOA_ridge", "sum"), rf_AOA_pixels=("AOA_random_forest", "sum"),
        common_AOA_pixels=("common_AOA", "sum"), disagreement_pixels=("AOA_disagreement", "sum"),
    )
    disagreement["coverage"] = disagreement.station_id.map(STATION_LABELS)
    disagreement.to_csv(output_root() / "aoa_coverage_by_station.csv", index=False)
    differences = []
    for period_label, group in combined.loc[combined.common_AOA].groupby("period_label"):
        row = summarize_distribution(group.Kc_difference_ridge_minus_rf)
        row.update({"period_label": period_label, "domain": "common_AOA"})
        differences.append(row)
    pd.DataFrame(differences).to_csv(output_root() / "ridge_minus_rf_distribution.csv", index=False)
    paired_periods = combined.loc[combined.common_AOA].pivot_table(
        index=["station_id", "x", "y"], columns="period_label",
        values=["Kc_raw_ridge", "Kc_raw_random_forest"],
    ).dropna()
    stability = []
    for station_id, group in paired_periods.groupby(level="station_id"):
        for model in MODEL_NAMES:
            column = f"Kc_raw_{model}"
            stability.append({
                "station_id": station_id, "coverage": STATION_LABELS[station_id],
                "model": model, "n_common_AOA_both_periods": len(group),
                "dry_wet_spearman": group[(column, "extreme_dry")].corr(
                    group[(column, "extreme_wet")], method="spearman"
                ),
                "mean_wet_minus_dry_Kc": (
                    group[(column, "extreme_wet")] - group[(column, "extreme_dry")]
                ).mean(),
            })
    pd.DataFrame(stability).to_csv(output_root() / "dry_wet_spatial_stability.csv", index=False)
    manifest = {
        "experiment": "S2_Ridge_RF_cartographic_stress_test",
        "periods": PERIODS,
        "period_interpretation": "contrasting extremes for cartographic stress testing, not climatological representatives",
        "training_rows": 550, "features": FEATURES, "feature_count": 20,
        "grid_m": 20, "domain": "five station MODIS-parent footprints",
        "common_comparison_domain": "valid predictors intersect Ridge AOA intersect RF AOA",
        "aoa": {name: {"weight_source": "absolute standardized coefficients" if name == "ridge" else "normalized feature_importances_", "threshold": specs[name]["threshold"]} for name in MODEL_NAMES},
        "remote_extraction_records": records, "maps": maps,
        "tuning_performed": False, "production_specification_changed": False,
        "model_frozen": False,
    }
    (output_root() / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(coverage.to_string(index=False))
    print(pd.DataFrame(distributions).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
