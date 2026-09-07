"""Build the final field-comparison table and explicit analysis scenarios.

The output keeps the technical sample and the scientific subsets separate:

1. all_with_AOA
   All station-periods that satisfy the accepted fine-product rules including
   the fold-specific AOA.

2. all_without_AOA_pure_extrapolations
   The accepted sample plus only station pixels that become valid when AOA is
   removed and that are themselves outside AOA. Other filters are unchanged.

3. fixed_Kc_with_AOA
   ST01-ST03 subset of all_with_AOA. This is a subset, not a separate sample.

4. fixed_Kc_without_AOA_pure_extrapolations
   ST01-ST03 subset of the AOA sensitivity sample. This is supplementary.

ST04-ST05 use local 20 m Sentinel-2 NDVI only to construct the field ET proxy
sensitivity. The field proxy is not an independent observation of actual ET.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import ee
import numpy as np
import pandas as pd

from et_downscaling.config import ANALYSIS_CRS
from et_downscaling.ridge25_production import build_ridge25_production_stack


FIXED_KC = {
    "ST01": 0.85,
    "ST02": 0.95,
    "ST03": 1.10,
}
NDVI_KC_SLOPE = 1.457
NDVI_KC_INTERCEPT = -0.1725
NDVI_KC_VALID_RANGE = (0.10, 1.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    return parser.parse_args()


def load_field_module(root: Path):
    path = root / "scripts" / "evaluate_field_ridge25.py"
    spec = importlib.util.spec_from_file_location("field_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_local_ndvi(rows: pd.DataFrame, project: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=["station_id", "period_start", "NDVI_local_20m"]
        )

    print()
    print("Initializing Earth Engine for local 20 m NDVI only...")
    ee.Initialize(project=project)
    ee.Number(1).getInfo()

    features = []
    for row in rows.itertuples(index=False):
        date_text = pd.Timestamp(row.period_start).strftime("%Y-%m-%d")
        point = ee.Geometry.Point([float(row.longitude), float(row.latitude)])
        local_geometry = point.buffer(100)

        context = build_ridge25_production_stack(
            period_start_text=date_text,
            basin_geometry=local_geometry,
        )
        value = (
            context["optical"]
            .select("NDVI_mean")
            .reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                crs=ANALYSIS_CRS,
                scale=20,
                maxPixels=100,
            )
            .get("NDVI_mean")
        )
        features.append(
            ee.Feature(
                None,
                {
                    "station_id": str(row.station_id),
                    "period_start": date_text,
                    "NDVI_local_20m": value,
                },
            )
        )

    payload = ee.FeatureCollection(features).getInfo()
    sampled = pd.DataFrame(
        feature["properties"] for feature in payload["features"]
    )
    sampled["period_start"] = pd.to_datetime(sampled["period_start"])
    sampled["NDVI_local_20m"] = pd.to_numeric(
        sampled["NDVI_local_20m"], errors="coerce"
    )
    return sampled


def scenario_summary(
    table: pd.DataFrame,
    scenario: str,
    membership_column: str,
    ridge_column: str,
) -> dict:
    subset = table.loc[table[membership_column]].copy()
    complete = subset.dropna(
        subset=[
            "ET_field_proxy_mm_period",
            "ET_MODIS_parent_mm_period",
            ridge_column,
        ]
    )
    station_counts = (
        subset.groupby("station_id")
        .size()
        .astype(int)
        .to_dict()
    )
    return {
        "scenario": scenario,
        "n_technical": int(len(subset)),
        "n_complete_field_proxy": int(len(complete)),
        "n_inside_basin": int(pd.to_numeric(subset["inside_basin"], errors="coerce").fillna(0).astype(bool).sum()),
        "n_external": int((~pd.to_numeric(subset["inside_basin"], errors="coerce").fillna(0).astype(bool)).sum()),
        "n_fixed_kc": int(subset["station_id"].isin(FIXED_KC).sum()),
        "n_ndvi_kc_sensitivity": int(subset["station_id"].isin(["ST04", "ST05"]).sum()),
        "n_station_pixels_outside_AOA": int(
            pd.to_numeric(subset["AOA_inside"], errors="coerce").lt(0.5).sum()
        ),
        "station_counts": "; ".join(
            f"{key}:{value}" for key, value in sorted(station_counts.items())
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    module = load_field_module(root)

    (
        workspace,
        master,
        reference,
        field,
        metadata,
        _master_path,
        _reference_path,
    ) = module.load_inputs(root)

    sensitivity_path = (
        workspace.diagnostics
        / "field_ridge25_aoa_sensitivity"
        / "field_ridge25_with_vs_without_aoa.csv"
    )
    if not sensitivity_path.is_file():
        raise FileNotFoundError(sensitivity_path)

    aoa = pd.read_csv(sensitivity_path)
    aoa["station_id"] = aoa["station_id"].astype(str)
    aoa["period_start"] = pd.to_datetime(aoa["period_start"])

    accepted_with_aoa = aoa["status_with_AOA"].eq("valid")
    pure_extrapolation = (
        aoa["status_without_AOA"].eq("valid")
        & pd.to_numeric(aoa["AOA_inside"], errors="coerce").lt(0.5)
    )
    selected_without_aoa = accepted_with_aoa | pure_extrapolation

    selected = aoa.loc[selected_without_aoa].copy()
    selected["comparison_role"] = np.where(
        pure_extrapolation.loc[selected.index],
        "pure_AOA_extrapolation",
        "inside_AOA_accepted",
    )
    selected = (
        selected.drop_duplicates(["station_id", "period_start"])
        .sort_values(["station_id", "period_start"])
        .reset_index(drop=True)
    )

    # Rebuild the field-period aggregation only to attach the harmonized
    # reference-ET period and field metadata.
    _, valid_daily = module.prepare_field_daily(field, reference, metadata)
    candidates = module.aggregate_field_periods(
        valid_daily, master, metadata
    )
    candidates["station_id"] = candidates["station_id"].astype(str)
    candidates["period_start"] = pd.to_datetime(candidates["period_start"])

    field_columns = [
        "station_id",
        "station",
        "period_start",
        "number_days",
        "n_valid_field_days",
        "field_reference_eto_mm_period",
        "ET_MODIS_mm_period",
        "inside_basin",
        "installation_conforms_manual",
        "longitude",
        "latitude",
    ]
    table = selected.merge(
        candidates[field_columns],
        on=["station_id", "period_start"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_field"),
    )

    table["Kc_field_proxy"] = np.nan
    table["Kc_source"] = ""

    for station_id, kc in FIXED_KC.items():
        mask = table["station_id"].eq(station_id)
        table.loc[mask, "Kc_field_proxy"] = kc
        table.loc[mask, "Kc_source"] = "fixed_literature_proxy"

    ndvi_mask = table["station_id"].isin(["ST04", "ST05"])
    ndvi_request = table.loc[
        ndvi_mask,
        ["station_id", "period_start", "longitude", "latitude"],
    ].copy()

    ndvi_samples = sample_local_ndvi(ndvi_request, args.project)
    table = table.merge(
        ndvi_samples,
        on=["station_id", "period_start"],
        how="left",
        validate="one_to_one",
    )

    candidate_kc = (
        NDVI_KC_SLOPE * table["NDVI_local_20m"]
        + NDVI_KC_INTERCEPT
    )
    valid_ndvi_kc = candidate_kc.between(*NDVI_KC_VALID_RANGE)
    table.loc[
        ndvi_mask & valid_ndvi_kc, "Kc_field_proxy"
    ] = candidate_kc.loc[ndvi_mask & valid_ndvi_kc]
    table.loc[
        ndvi_mask & valid_ndvi_kc, "Kc_source"
    ] = "local_20m_NDVI_sensitivity"
    table.loc[
        ndvi_mask & ~valid_ndvi_kc, "Kc_source"
    ] = "local_20m_NDVI_invalid_or_missing"

    table["ET_field_proxy_mm_period"] = (
        pd.to_numeric(
            table["field_reference_eto_mm_period"], errors="coerce"
        )
        * table["Kc_field_proxy"]
    )
    table["ET_MODIS_parent_mm_period"] = pd.to_numeric(
        table["ET_MODIS_mm_period"], errors="coerce"
    )
    table["ET_Ridge_without_AOA_mm_period"] = pd.to_numeric(
        table["ET_without_AOA_mm_period"], errors="coerce"
    )
    table["ET_Ridge_with_AOA_mm_period"] = pd.to_numeric(
        table["ET_with_AOA_mm_period"], errors="coerce"
    )

    table["included_all_with_AOA"] = (
        table["status_with_AOA"].eq("valid")
    )
    table["included_all_without_AOA_pure_extrapolations"] = (
        table["included_all_with_AOA"]
        | table["comparison_role"].eq("pure_AOA_extrapolation")
    )
    table["included_fixed_Kc_with_AOA"] = (
        table["included_all_with_AOA"]
        & table["station_id"].isin(FIXED_KC)
    )
    table["included_fixed_Kc_without_AOA_pure_extrapolations"] = (
        table["included_all_without_AOA_pure_extrapolations"]
        & table["station_id"].isin(FIXED_KC)
    )

    output_columns = [
        "station_id",
        "station",
        "period_start",
        "evaluation_domain",
        "inside_basin",
        "installation_conforms_manual",
        "number_days",
        "n_valid_field_days",
        "comparison_role",
        "field_reference_eto_mm_period",
        "NDVI_local_20m",
        "Kc_field_proxy",
        "Kc_source",
        "ET_field_proxy_mm_period",
        "ET_MODIS_parent_mm_period",
        "ET_Ridge_without_AOA_mm_period",
        "ET_Ridge_with_AOA_mm_period",
        "AOA_inside",
        "dissimilarity_index",
        "support_without_AOA",
        "support_with_AOA",
        "status_without_AOA",
        "status_with_AOA",
        "included_all_with_AOA",
        "included_all_without_AOA_pure_extrapolations",
        "included_fixed_Kc_with_AOA",
        "included_fixed_Kc_without_AOA_pure_extrapolations",
    ]
    final_table = table[
        [column for column in output_columns if column in table.columns]
    ].copy()

    scenarios = [
        (
            "all_with_AOA",
            "included_all_with_AOA",
            "ET_Ridge_with_AOA_mm_period",
            "All covers; accepted fold-specific AOA; ST04-ST05 remain Kc-NDVI sensitivity.",
        ),
        (
            "all_without_AOA_pure_extrapolations",
            "included_all_without_AOA_pure_extrapolations",
            "ET_Ridge_without_AOA_mm_period",
            "Accepted sample plus only pure station-pixel AOA extrapolations; all other filters unchanged.",
        ),
        (
            "fixed_Kc_with_AOA",
            "included_fixed_Kc_with_AOA",
            "ET_Ridge_with_AOA_mm_period",
            "ST01-ST03 subset of all_with_AOA. The count is a subset, not additional excluded observations.",
        ),
        (
            "fixed_Kc_without_AOA_pure_extrapolations",
            "included_fixed_Kc_without_AOA_pure_extrapolations",
            "ET_Ridge_without_AOA_mm_period",
            "Supplementary ST01-ST03 AOA sensitivity.",
        ),
    ]

    scenario_rows = []
    metric_rows = []
    for scenario, membership, ridge_column, description in scenarios:
        summary = scenario_summary(
            final_table, scenario, membership, ridge_column
        )
        summary["description"] = description
        scenario_rows.append(summary)

        subset = final_table.loc[final_table[membership]].dropna(
            subset=[
                "ET_field_proxy_mm_period",
                "ET_MODIS_parent_mm_period",
                ridge_column,
            ]
        )
        for model_name, prediction in [
            ("MODIS_parent", "ET_MODIS_parent_mm_period"),
            ("Ridge25_exact_overlap_spatial_OOF", ridge_column),
        ]:
            metric_rows.append(
                {
                    "scenario": scenario,
                    "model": model_name,
                    **module.calculate_metrics(
                        subset["ET_field_proxy_mm_period"],
                        subset[prediction],
                    ),
                }
            )

    scenario_table = pd.DataFrame(scenario_rows)
    metrics_table = pd.DataFrame(metric_rows)

    out_dir = (
        workspace.diagnostics / "field_ridge25_final_scenarios"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = out_dir / "field_comparison_scenarios.csv"
    scenario_path = out_dir / "field_scenario_definitions.csv"
    metrics_path = out_dir / "field_scenario_metrics.csv"

    final_table.to_csv(comparison_path, index=False)
    scenario_table.to_csv(scenario_path, index=False)
    metrics_table.to_csv(metrics_path, index=False)

    print()
    print("=" * 120)
    print("FINAL FIELD SCENARIOS")
    print("=" * 120)
    print(scenario_table.to_string(index=False))
    print()
    print("=" * 120)
    print("FIELD METRICS")
    print("=" * 120)
    print(
        metrics_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print()
    print("Saved:")
    print(" -", comparison_path)
    print(" -", scenario_path)
    print(" -", metrics_path)


if __name__ == "__main__":
    main()
