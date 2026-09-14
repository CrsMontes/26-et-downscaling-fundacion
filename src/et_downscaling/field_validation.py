"""Standalone field ET proxy comparison; never a training-data source.

Daily/period rules: 480f50b and 46b7d30. Local NDVI support: 726b914.
The current RF25 optical functions supply NDVI; existing published rasters
supply RF25 ET. No model fitting, AOA scoring, or reconciliation occurs here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import calculate_metrics

SCENARIOS = ("historical", "fao_sensitivity", "ndvi20_all")
FIXED_KC = {
    "historical": {"ST01": 0.85, "ST02": 0.95, "ST03": 1.10},
    "fao_sensitivity": {"ST01": 0.75, "ST02": 1.00, "ST03": 1.10},
    "ndvi20_all": {},
}
KEYS = ["station_id", "period_start"]
PRODUCTS = {"MODIS": "ET_MODIS_mm_period", "RF25": "ET_RF25_mm_period"}
RAW_TO_MM = 10.0
MIN_VALID_DAYS = 5
DAILY_RANGE = (0.05, 12.0)
NDVI_SLOPE, NDVI_INTERCEPT = 1.457, -0.1725
KC_RANGE = (0.10, 1.50)
VERSION = "field_proxy_rf25_v1"
VALIDATION_SCENARIOS = ("fixed_kc_main", "historical_all_stations", "fao_sensitivity", "ndvi20_all")


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique(table: pd.DataFrame, keys: list[str], name: str) -> None:
    if table[keys].isna().any().any() or table.duplicated(keys).any():
        raise ValueError(f"{name}: missing or duplicate keys {keys}.")


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "si", "sí"})


def load_field_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read versioned real stations, deliberately ignoring Virtual10 overrides."""
    field = pd.read_csv(root / "data/field/field_etgage.csv", dtype={"station_id": str})
    geojson = json.loads((root / "data/stations/fundacion_stations.geojson").read_text(encoding="utf-8"))
    rows = []
    for feature in geojson["features"]:
        if feature["geometry"]["type"] != "Point":
            raise ValueError("Field stations must have Point geometry.")
        longitude, latitude = feature["geometry"]["coordinates"]
        rows.append({**feature["properties"], "longitude": longitude, "latitude": latitude})
    stations = pd.DataFrame(rows)
    unique(stations, ["station_id"], "stations")
    if set(field.station_id) != set(stations.station_id):
        raise ValueError("Field observations and versioned station IDs differ.")
    return field, stations


def prepare_field_daily(field: pd.DataFrame, stations: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Historical QC and daily reference harmonization, with explicit bad-input errors."""
    daily = field.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    unique(daily, ["station_id", "date"], "field")
    reference = reference.copy()
    reference["local_date"] = pd.to_datetime(reference["local_date"], errors="raise")
    unique(reference, ["station_id", "local_date"], "daily reference")
    columns = ["station_id", "reference_et", "canvas", "installation_conforms_manual", "inside_basin"]
    daily = daily.merge(stations[columns], on="station_id", validate="many_to_one", how="left")
    daily["reference_et"] = daily.reference_et.str.upper()
    if not daily.reference_et.isin(["ETO", "ETR"]).all():
        raise ValueError("Every field station must explicitly specify ETo or ETr.")
    daily["etgage_daily_raw"] = pd.to_numeric(daily.etgage_daily_raw, errors="coerce")
    daily["etgage_scaled_mm_day"] = daily.etgage_daily_raw * RAW_TO_MM
    daily["qc_within_installation"] = to_bool(daily.within_installation_window)
    daily["qc_nonmissing"] = daily.etgage_daily_raw.notna()
    daily["qc_positive"] = daily.etgage_daily_raw.gt(0)
    daily["qc_physical_range"] = daily.etgage_scaled_mm_day.between(*DAILY_RANGE)
    daily["field_daily_valid"] = daily[["qc_within_installation", "qc_nonmissing", "qc_positive", "qc_physical_range"]].all(axis=1)
    daily = daily.merge(
        reference[["station_id", "local_date", "ETo_mm_day", "ETr_mm_day"]],
        left_on=["station_id", "date"], right_on=["station_id", "local_date"],
        how="left", validate="many_to_one",
    )
    for column in ("ETo_mm_day", "ETr_mm_day"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    modeled = daily[["ETo_mm_day", "ETr_mm_day"]]
    bad = daily.field_daily_valid & ~(np.isfinite(modeled).all(axis=1) & modeled.gt(0).all(axis=1))
    if bad.any():
        examples = daily.loc[bad, ["station_id", "date"]].head().to_dict("records")
        raise ValueError(f"Missing/nonpositive modeled ETo/ETr on {int(bad.sum())} valid field days: {examples}. Do not use Virtual10 reference rows for ST01-ST05.")
    daily["ETr_ETo_ratio"] = daily.ETr_mm_day / daily.ETo_mm_day
    daily["etgage_eto_equivalent_mm_day"] = np.where(
        daily.reference_et.eq("ETR"),
        daily.etgage_scaled_mm_day / daily.ETr_ETo_ratio,
        daily.etgage_scaled_mm_day,
    )
    daily.loc[~daily.field_daily_valid, "etgage_eto_equivalent_mm_day"] = np.nan
    return daily


def modis_periods(field: pd.DataFrame) -> pd.DataFrame:
    """Jan-1 anchored 8-day windows, with the actual shortened year-end window."""
    dates = pd.to_datetime(field.date, errors="raise")
    first, last = dates.min(), dates.max()
    rows = []
    for year in range(first.year, last.year + 1):
        year_end = pd.Timestamp(year=year + 1, month=1, day=1)
        for start in pd.date_range(f"{year}-01-01", year_end - pd.Timedelta(days=1), freq="8D"):
            end = min(start + pd.Timedelta(days=8), year_end)
            if end > first and start <= last:
                rows.append({"period_start": start, "number_days": (end - start).days})
    return pd.DataFrame(rows)


def aggregate_field_periods(daily: pd.DataFrame, periods: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Keep rejected periods for attrition reporting; proxy only when >=5 days."""
    rows = []
    for station in stations.itertuples(index=False):
        records = daily.loc[daily.station_id.eq(station.station_id)]
        for period in periods.itertuples(index=False):
            start = pd.Timestamp(period.period_start)
            days = int(period.number_days)
            if days <= 0 or days > 8:
                raise ValueError("MODIS period length must be between 1 and 8 days.")
            group = records.loc[records.date.ge(start) & records.date.lt(start + pd.Timedelta(days=days))]
            valid = group.loc[group.field_daily_valid]
            enough = len(valid) >= MIN_VALID_DAYS
            rows.append({
                "station_id": station.station_id, "station": station.station,
                "longitude": station.longitude, "latitude": station.latitude,
                "inside_basin": station.inside_basin,
                "installation_conforms_manual": station.installation_conforms_manual,
                "reference_et": station.reference_et, "canvas": station.canvas,
                "period_start": start, "number_days": days,
                "n_raw_field_days": len(group), "n_valid_field_days": len(valid),
                "field_period_valid": enough,
                "field_reference_eto_mm_period": float(valid.etgage_eto_equivalent_mm_day.mean() * days) if enough else np.nan,
            })
    result = pd.DataFrame(rows)
    unique(result, KEYS, "field periods")
    return result


def ndvi_kc(ndvi: pd.Series) -> pd.Series:
    candidate = pd.to_numeric(ndvi, errors="coerce") * NDVI_SLOPE + NDVI_INTERCEPT
    return candidate.where(np.isfinite(candidate) & candidate.between(*KC_RANGE))


def apply_scenarios(periods: pd.DataFrame) -> pd.DataFrame:
    """Always construct all scenarios so fair-row masks do not depend on CLI selection."""
    rows = []
    for scenario in SCENARIOS:
        table = periods.copy()
        table["scenario"] = scenario
        fixed = table.station_id.map(FIXED_KC[scenario])
        use_ndvi = fixed.isna()
        table["Kc_field_proxy"] = fixed.where(~use_ndvi, ndvi_kc(table.NDVI_local_20m))
        label = "historical_fixed_proxy" if scenario == "historical" else "requested_fao_sensitivity_proxy"
        table["Kc_source"] = np.where(use_ndvi, "current_rf25_local_20m_NDVI_proxy", label)
        table["field_proxy_status"] = np.select(
            [~table.field_period_valid,
             use_ndvi & ~np.isfinite(table.NDVI_local_20m),
             table.Kc_field_proxy.isna()],
            ["insufficient_valid_days", "missing_ndvi", "kc_outside_valid_range"],
            default="available",
        )
        table["ET_field_proxy_mm_period"] = table.field_reference_eto_mm_period * table.Kc_field_proxy
        rows.append(table)
    result = pd.concat(rows, ignore_index=True)
    for product, column in PRODUCTS.items():
        result[f"paired_{product}"] = np.isfinite(result.ET_field_proxy_mm_period) & np.isfinite(result[column])
        result[f"common_scenarios_{product}"] = result.groupby(KEYS)[f"paired_{product}"].transform("all")
    result["common_scenarios_products"] = result.common_scenarios_MODIS & result.common_scenarios_RF25
    return result


def paired_metrics(table: pd.DataFrame, column: str) -> dict:
    observed = table.ET_field_proxy_mm_period.to_numpy(float)
    predicted = table[column].to_numpy(float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[finite], predicted[finite]
    if len(observed) < 2:
        error = predicted - observed
        return {"n": len(observed), "R2": np.nan, "RMSE": float(abs(error[0])) if len(error) else np.nan,
                "MAE": float(abs(error[0])) if len(error) else np.nan,
                "BIAS": float(error[0]) if len(error) else np.nan, "KGE": np.nan}
    result = calculate_metrics(observed, predicted)
    if observed.std() == 0:
        result["R2"] = np.nan  # Undefined, rather than sklearn's finite replacement.
    return result


def build_metrics(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Available pairs, common scenarios per product, and common scenarios/products."""
    rows = []
    scopes = [("ALL", pairs)] + list(pairs.groupby("station_id", sort=True))
    for station_id, scope in scopes:
        for scenario, group in scope.groupby("scenario", sort=False):
            for product, column in PRODUCTS.items():
                masks = {"available_pairs": group[f"paired_{product}"],
                         "common_scenarios": group[f"common_scenarios_{product}"],
                         "common_scenarios_products": group.common_scenarios_products}
                for comparison, mask in masks.items():
                    subset = group.loc[mask].sort_values(KEYS)
                    keys = subset[KEYS].assign(period_start=lambda x: x.period_start.dt.strftime("%Y-%m-%d"))
                    digest = hashlib.sha256(keys.to_csv(index=False).encode()).hexdigest()
                    rows.append({"station_id": station_id, "scenario": scenario, "product": product,
                                 "comparison": comparison, "pair_keys_sha256": digest,
                                 **paired_metrics(subset, column)})
    result = pd.DataFrame(rows)
    return result.loc[result.station_id.eq("ALL")].reset_index(drop=True), result.loc[result.station_id.ne("ALL")].reset_index(drop=True)


def build_attrition(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, station_id), group in pairs.groupby(["scenario", "station_id"], sort=False):
        valid = group.field_period_valid
        proxy = group.field_proxy_status.eq("available")
        row = {"scenario": scenario, "station_id": station_id, "n_candidate_periods": len(group),
               "n_valid_field_periods": int(valid.sum()), "n_field_proxy": int(proxy.sum()),
               "n_insufficient_valid_days": int((~valid).sum()),
               "n_missing_ndvi": int((valid & group.field_proxy_status.eq("missing_ndvi")).sum()),
               "n_invalid_ndvi_kc": int((valid & group.field_proxy_status.eq("kc_outside_valid_range")).sum()),
               "n_MODIS_pairs": int(group.paired_MODIS.sum()), "n_RF25_pairs": int(group.paired_RF25.sum()),
               "n_common_MODIS": int(group.common_scenarios_MODIS.sum()),
               "n_common_RF25": int(group.common_scenarios_RF25.sum()),
               "n_common_products": int(group.common_scenarios_products.sum()),
               "n_proxy_missing_MODIS": int((proxy & ~group.paired_MODIS).sum())}
        # Exclusive RF25 reason counts among valid field periods, and among usable proxies.
        for status in ("missing_raster", "incomplete_raster", "outside_extent", "outside_basin",
                       "aoa_excluded", "invalid_stack", "ineligible_support", "unpublished", "available"):
            mask = group.RF25_status.eq(status)
            row[f"n_valid_field_RF25_{status}"] = int((valid & mask).sum())
            row[f"n_proxy_RF25_{status}"] = int((proxy & mask).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def apply_validation_scenarios(periods: pd.DataFrame) -> pd.DataFrame:
    """Four declared scientific sets; legacy helper remains covered by its tests."""
    pairs = apply_scenarios(periods)
    historical = pairs.scenario.eq("historical")
    main = pairs.loc[historical & pairs.station_id.isin(FIXED_KC["historical"])].copy()
    main["scenario"] = "fixed_kc_main"
    pairs.loc[historical, "scenario"] = "historical_all_stations"
    pairs = pd.concat([main, pairs], ignore_index=True)
    pairs["scenario_role"] = np.where(pairs.scenario.eq("fixed_kc_main"), "main_conservative", "sensitivity")
    pairs["shares_sentinel2_with_rf25"] = pairs.Kc_source.eq("current_rf25_local_20m_NDVI_proxy")
    # Do not carry obsolete common-mask names into the canonical output.
    pairs = pairs.drop(columns=[c for c in pairs if c.startswith("common_scenarios")])
    pairs["available_sample_MODIS"] = np.isfinite(pairs.ET_field_proxy_mm_period) & np.isfinite(pairs.ET_MODIS_mm_period)
    pairs["available_sample_RF25"] = pairs.available_sample_MODIS & np.isfinite(pairs.ET_RF25_mm_period)
    families = {
        "main_vs_sensitivities": VALIDATION_SCENARIOS,
        "all_station_sensitivities": VALIDATION_SCENARIOS[1:],
    }
    for family, scenarios in families.items():
        subset = pairs.loc[pairs.scenario.isin(scenarios)]
        for product in PRODUCTS:
            complete = subset.groupby(KEYS)[f"available_sample_{product}"].agg(["sum", "size"])
            common = complete.index[(complete["sum"] == len(scenarios)) & (complete["size"] == len(scenarios))]
            pairs[f"common_sample_{family}_{product}"] = pd.MultiIndex.from_frame(pairs[KEYS]).isin(common) & pairs.scenario.isin(scenarios)
        pairs[f"common_sample_{family}"] = pairs[f"common_sample_{family}_RF25"]
    return pairs


def build_validation_metrics(pairs: pd.DataFrame):
    rows = []
    comparisons = {
        "MODIS_vs_field_proxy": ("ET_field_proxy_mm_period", "ET_MODIS_mm_period"),
        "RF25_vs_field_proxy": ("ET_field_proxy_mm_period", "ET_RF25_mm_period"),
        "MODIS_vs_RF25": ("ET_MODIS_mm_period", "ET_RF25_mm_period"),
    }
    for station, scope in [("ALL", pairs), *list(pairs.groupby("station_id", sort=True))]:
        for scenario, group in scope.groupby("scenario", sort=False):
            for comparison, (observed, predicted) in comparisons.items():
                available = group.available_sample_MODIS if comparison == "MODIS_vs_field_proxy" else group.available_sample_RF25
                selections = [("available_sample", "own_domain", available)]
                for family in ("main_vs_sensitivities", "all_station_sensitivities"):
                    if family == "all_station_sensitivities" and scenario == "fixed_kc_main":
                        continue
                    product = "MODIS" if comparison == "MODIS_vs_field_proxy" else "RF25"
                    selections.append(("common_sample", family, group[f"common_sample_{family}_{product}"]))
                for sample, family, mask in selections:
                    selected = group.loc[mask].sort_values(KEYS)
                    frame = selected.copy()
                    frame["ET_field_proxy_mm_period"] = selected[observed]
                    metrics = paired_metrics(frame, predicted)
                    keys = selected[KEYS].assign(period_start=lambda t: t.period_start.dt.strftime("%Y-%m-%d"))
                    rows.append({"station_id": station, "scenario": scenario,
                                 "comparison": comparison, "sample": sample, "comparison_family": family,
                                 "pair_keys_sha256": hashlib.sha256(keys.to_csv(index=False).encode()).hexdigest(), **metrics})
    metrics = pd.DataFrame(rows)
    return metrics.loc[metrics.station_id.eq("ALL")], metrics.loc[metrics.station_id.ne("ALL")]


def build_sequential_attrition(pairs: pd.DataFrame):
    """Nested gates in the stated scientific order; counts never increase."""
    rows = []
    for (scenario, station), group in pairs.groupby(["scenario", "station_id"], sort=False):
        mask = pd.Series(True, index=group.index)
        gates = {
            "candidate_field_periods": mask.copy(),
            "ge5_valid_ETgage_days": group.field_period_valid,
            "field_proxy_available": np.isfinite(group.ET_field_proxy_mm_period),
            "MODIS_available": np.isfinite(group.ET_MODIS_mm_period),
            "RF25_stack_complete": group.predictor_stack_available.fillna(False),
            "inside_AOA": group.inside_AOA.fillna(False),
            "parent_MODIS_eligible": group.parent_MODIS_eligible.fillna(False),
            "reconciliation_successful": group.reconciliation_successful.fillna(False),
            "final_comparison_pair": group.publication_eligible.fillna(False) & np.isfinite(group.ET_RF25_mm_period),
        }
        previous = len(group)
        for order, (stage, gate) in enumerate(gates.items()):
            unknown = int((mask & group[stage_column].isna()).sum()) if (stage_column := {
                "RF25_stack_complete": "predictor_stack_available", "inside_AOA": "inside_AOA",
                "parent_MODIS_eligible": "parent_MODIS_eligible", "reconciliation_successful": "reconciliation_successful",
                "final_comparison_pair": "publication_eligible",
            }.get(stage)) else 0
            mask &= gate.astype(bool)
            count = int(mask.sum())
            rows.append({"scenario": scenario, "station_id": station, "stage_order": order,
                         "stage": stage, "n_remaining": count, "n_lost_at_stage": previous - count,
                         "n_unknown_at_stage": unknown, "n_failed_known_at_stage": previous - count - unknown})
            previous = count
    result = pd.DataFrame(rows)
    totals = result.groupby(["scenario", "stage_order", "stage"], sort=False, as_index=False)[
        ["n_remaining", "n_lost_at_stage", "n_unknown_at_stage", "n_failed_known_at_stage"]].sum()
    totals["station_id"] = "ALL"
    return pd.concat([result, totals], ignore_index=True)


def sample_rf25_rasters(periods: pd.DataFrame, raster_root: Path, contract: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Read final published ET and diagnostic bands; never fill or regenerate ET."""
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import Window
    from .rf25_overlap_production import RF25_EXACT_OVERLAP_PRODUCTION_VERSION as version
    from .field_rf25_products import product_status, product_paths
    from .config import ANALYSIS_CRS

    result = periods.copy()
    result["ET_RF25_mm_period"] = np.nan
    result["RF25_status"] = "missing_raster"
    result["RF25_AOA_inside"] = np.nan
    result["RF25_raster"] = ""
    for column in ("predictor_stack_available", "inside_AOA", "parent_MODIS_eligible",
                   "reconciliation_successful", "publication_eligible"):
        result[column] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    provenance = {}
    for start, group in result.groupby("period_start", sort=True):
        date_text = pd.Timestamp(start).strftime("%Y-%m-%d")
        directory = Path(raster_root) / date_text
        path = directory / f"ET_{version}_{date_text}_20m.tif"
        if not path.is_file():
            continue
        metadata_path = directory / f"production_metadata_{version}.json"
        result.loc[group.index, "RF25_raster"] = str(path.resolve())
        if not metadata_path.is_file():
            result.loc[group.index, "RF25_status"] = "incomplete_raster"
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("period_start") != date_text or metadata.get("production_method_version") != version:
            raise ValueError(f"RF25 production metadata does not match {path}.")
        provenance[str(path.resolve())] = sha256(path)
        provenance[str(metadata_path.resolve())] = sha256(metadata_path)
        status = product_status(Path(raster_root), date_text, contract)
        if status != "verified_final_product":
            result.loc[group.index, "RF25_status"] = status
            continue
        evidence = product_paths(Path(raster_root), date_text)[2]
        provenance[str(evidence.resolve())] = sha256(evidence)
        error = metadata.get("max_abs_conservation_error_after_floor_mm", np.nan)
        if not np.isfinite(error) or error > 0.01:
            raise ValueError(f"RF25 global reconciliation is not certified: {path}")
        with rasterio.open(path) as source:
            required = {"ET_mm_period", "AOA_inside", "stack_valid", "coarse_eligible"}
            if not required.issubset(source.descriptions) or source.crs is None:
                raise ValueError(f"Final RF25 bands/CRS are missing: {path}")
            if (source.crs.to_string() != ANALYSIS_CRS or source.transform.a != 20
                    or source.transform.e != -20 or source.transform.b != 0 or source.transform.d != 0
                    or source.transform.c % 20 != 0 or source.transform.f % 20 != 0):
                raise ValueError(f"RF25 canonical 20 m grid is missing: {path}")
            for index, row in group.iterrows():
                x, y = transform("EPSG:4326", source.crs, [row.longitude], [row.latitude])
                r, c = source.index(x[0], y[0])
                if not (0 <= r < source.height and 0 <= c < source.width):
                    result.loc[index, "RF25_status"] = "outside_extent"
                    continue
                values = source.read(window=Window(c, r, 1, 1), masked=True).astype(float).filled(np.nan)[:, 0, 0]
                bands = dict(zip(source.descriptions, values))
                for column, band in (("predictor_stack_available", "stack_valid"),
                                     ("inside_AOA", "AOA_inside"), ("parent_MODIS_eligible", "coarse_eligible")):
                    result.loc[index, column] = bool(bands[band] == 1) if np.isfinite(bands[band]) else pd.NA
                result.loc[index, "RF25_AOA_inside"] = bands["AOA_inside"]
                et = bands["ET_mm_period"]
                result.loc[index, "reconciliation_successful"] = True
                result.loc[index, "publication_eligible"] = bool(np.isfinite(et))
                if np.isfinite(et):
                    if et < 0 or bands["AOA_inside"] != 1 or bands["stack_valid"] != 1 or bands["coarse_eligible"] != 1:
                        raise ValueError(f"Published RF25 ET violates raster flags: {path}, {row.station_id}")
                    result.loc[index, "ET_RF25_mm_period"] = et
                    status = "available"
                elif bands["stack_valid"] == 0:
                    status = "invalid_stack"
                elif bands["AOA_inside"] == 0:
                    status = "aoa_excluded"
                elif bands["coarse_eligible"] == 0:
                    status = "ineligible_support"
                elif all(not np.isfinite(bands[name]) for name in required):
                    status = "outside_basin"
                else:
                    status = "unpublished"
                result.loc[index, "RF25_status"] = status
    return result, provenance


def ee_table(collection) -> pd.DataFrame:
    return pd.DataFrame(feature["properties"] for feature in collection.getInfo()["features"])


def field_footprints(stations: pd.DataFrame):
    """Use current native MODIS geometry, explicitly with the versioned field points."""
    import ee
    from .modis import (assign_station_footprints, build_modis_grid, build_modis_pixel_id,
                        get_modis_collection, get_modis_projection, get_modis_scale)
    samples = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([row.longitude, row.latitude]),
                   {"station_id": row.station_id, "station": row.station})
        for row in stations.itertuples(index=False)
    ])
    projection = get_modis_projection(get_modis_collection())
    scale = get_modis_scale(projection)
    pixel_id = build_modis_pixel_id(projection)
    grid = build_modis_grid(samples, pixel_id, projection, scale)
    return assign_station_footprints(samples, grid, pixel_id, projection, scale)


def acquire_reference(field: pd.DataFrame, stations: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Acquire only field-date ERA5 inputs, caching resumable station-month partitions."""
    import ee
    from .meteorology_export import build_era5_station_supports, build_station_support_table, build_era5_hourly_table
    from .reference_et_local import build_daily_reference_et
    cache.mkdir(parents=True, exist_ok=True)
    support_path = cache / "station_support.csv"
    if support_path.is_file():
        support = pd.read_csv(support_path)
    else:
        footprints = field_footprints(stations)
        era5_support = build_era5_station_supports(footprints)
        support = ee_table(build_station_support_table(footprints, era5_support))
        support.to_csv(support_path, index=False)
    dates = pd.to_datetime(field.date)
    first, end = dates.min(), dates.max() + pd.Timedelta(days=1)
    chunks = []
    for station in support.to_dict("records"):
        for month in pd.period_range(first, end - pd.Timedelta(days=1), freq="M"):
            start_local, end_local = max(first, month.start_time), min(end, (month + 1).start_time)
            path = cache / f"era5_{station['station_id']}_{month}.csv"
            print(f"Field ERA5: {station['station_id']} {month}", flush=True)
            if path.is_file():
                chunk = pd.read_csv(path)
            else:
                start_utc = (start_local + pd.Timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
                end_utc = (end_local + pd.Timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
                chunk = ee_table(build_era5_hourly_table(ee.Feature(None, station), start_utc, end_utc))
                unique(chunk, ["station_id", "timestamp_utc"], "ERA5 partition")
                expected = int((end_local - start_local).total_seconds() / 3600)
                if len(chunk) != expected:
                    raise ValueError(f"Incomplete hourly ERA5 partition: {path}, {len(chunk)} != {expected}")
                chunk.to_csv(path, index=False)
            chunks.append(chunk)
    hourly = pd.concat(chunks, ignore_index=True)
    unique(hourly, ["station_id", "timestamp_utc"], "field ERA5")
    return build_daily_reference_et(hourly, support)


def acquire_satellite(periods: pd.DataFrame, stations: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Sample native MODIS ET and the current-final S2 predictor NDVI at 20 m."""
    import ee
    from .production import (build_modis_period_context, build_study_feature_collection,
                             get_fine_projection, PROCESSING_BUFFER_M)
    from .rf25_production import build_s2_rf25_predictors

    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for period in periods.itertuples(index=False):
        start = pd.Timestamp(period.period_start).strftime("%Y-%m-%d")
        path = cache / f"satellite_{start}.csv"
        print(f"Field MODIS / current RF25 NDVI: {start}", flush=True)
        if path.is_file():
            table = pd.read_csv(path, dtype={"station_id": str})
        else:
            features = []
            for station in stations.itertuples(index=False):
                point = ee.Geometry.Point([station.longitude, station.latitude])
                geometry = point.buffer(100).buffer(PROCESSING_BUFFER_M)
                context = build_modis_period_context(start, geometry)
                optical, _ = build_s2_rf25_predictors(
                    build_study_feature_collection(geometry), context["period_start"],
                    context["period_end"], geometry, get_fine_projection(),
                )
                ndvi = optical.select("NDVI_mean").reduceRegion(
                    reducer=ee.Reducer.first(), geometry=point,
                    crs=get_fine_projection(), maxPixels=100,
                ).get("NDVI_mean")
                modis = context["modis_et"].reduceRegion(
                    reducer=ee.Reducer.first(), geometry=point,
                    crs=context["modis_projection"], maxPixels=100,
                ).get("ET_mm_period")
                features.append(ee.Feature(None, {
                    "station_id": station.station_id, "period_start": start,
                    "number_days": context["number_days"],
                    "NDVI_local_20m": ndvi, "ET_MODIS_mm_period": modis,
                }))
            table = ee_table(ee.FeatureCollection(features))
            for column in ("NDVI_local_20m", "ET_MODIS_mm_period"):
                if column not in table:
                    table[column] = np.nan
            table.to_csv(path, index=False)
        if set(table.station_id) != set(stations.station_id):
            raise ValueError(f"Satellite cache is not the five field stations: {path}")
        if not pd.to_numeric(table.number_days).eq(period.number_days).all():
            raise ValueError(f"Satellite and field period durations disagree: {path}")
        rows.append(table)
    return pd.concat(rows, ignore_index=True)
