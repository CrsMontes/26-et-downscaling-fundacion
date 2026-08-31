"""Small, reproducible QA for fine fill and multi-parent MODIS conservation.

The QA is intentionally bounded to native MODIS parents around selected
stations for one period. It does not process the basin, export rasters, or
train a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd

from et_downscaling.model_spec import COMMON_MODEL_FEATURES, PRODUCTION_MODEL_FILENAME
from et_downscaling.model_transfer import build_ee_regressor
from et_downscaling.modis import (
    assign_station_footprints,
    build_modis_grid,
    build_modis_pixel_id,
)
from et_downscaling.production import (
    MODIS_CONSERVATION_TOLERANCE_MM,
    PREDICTION_SCALE_M,
    _mean_at_modis_support,
    build_modis_constrained_et,
    build_production_stack,
)
from et_downscaling.stations import load_station_dataframe


MISSING = -9999.0
FILL_EQUALITY_TOLERANCE = 1e-5
UNCHANGED_TOLERANCE = 1e-7


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="QA fine fill and MODIS conservation on a bounded window."
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--period-start", default="2022-05-25")
    parser.add_argument(
        "--station-ids",
        nargs="+",
        default=["ST02", "ST03"],
    )
    parser.add_argument("--buffer-m", type=float, default=1200.0)
    return parser.parse_args()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def station_collection(station_ids: list[str]) -> ee.FeatureCollection:
    stations = load_station_dataframe().set_index("station_id")
    missing = sorted(set(station_ids) - set(stations.index))
    if missing:
        raise ValueError(f"Unknown station IDs: {missing}")
    selected = stations.loc[station_ids]
    return ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point([row.longitude, row.latitude]),
            {"station_id": station_id, "station": row.station},
        )
        for station_id, row in selected.iterrows()
    ])


def as_fine(image: ee.Image, name: str, fine_projection: ee.Projection) -> ee.Image:
    return (
        ee.Image(image)
        .rename(name)
        .unmask(MISSING)
        .reproject(fine_projection)
        .toFloat()
    )


def feature_rows(payload: dict) -> list[dict]:
    rows = []
    for feature in payload["features"]:
        row = dict(feature["properties"])
        geometry = feature.get("geometry")
        if geometry and geometry.get("type") == "Point":
            row["longitude"], row["latitude"] = geometry["coordinates"]
        rows.append(row)
    return rows


def describe_fill(
    samples: pd.DataFrame,
    parent_qa: pd.DataFrame,
) -> tuple[list[dict], bool]:
    summaries = []
    all_pass = True
    for station_id, group in samples.groupby("station_id"):
        raw_valid = group["raw_valid"].eq(1)
        filled_valid = group["filled_valid"].eq(1)
        new_fill = group["new_fill"].eq(1)
        valid_unchanged = group.loc[raw_valid & filled_valid]
        filled_gaps = group.loc[new_fill]

        unchanged_error = float(
            (valid_unchanged.Kc_filled - valid_unchanged.Kc_raw)
            .abs().max()
        )
        fill_error = float(
            (filled_gaps.Kc_filled - filled_gaps.Kc_parent_mean)
            .abs().max()
        ) if len(filled_gaps) else float("nan")

        reconciled = group.loc[filled_valid & group.ET_reconciled.ne(MISSING)].copy()
        reconciled["effective_scale"] = (
            reconciled.ET_reconciled / reconciled.Kc_filled
        )
        raw_reconciled = reconciled.loc[reconciled.raw_valid.eq(1)]
        spearman_all = float(
            reconciled[["Kc_filled", "ET_reconciled"]]
            .corr(method="spearman").iloc[0, 1]
        )
        spearman_raw = float(
            raw_reconciled[["Kc_raw", "ET_reconciled"]]
            .corr(method="spearman").iloc[0, 1]
        )

        parent = parent_qa.loc[parent_qa.station_id.eq(station_id)].iloc[0]
        parent_pass = bool(
            parent.eligible == 1
            and parent.fine_fill_fraction > 0
            and len(filled_gaps) > 0
            and unchanged_error <= UNCHANGED_TOLERANCE
            and fill_error <= FILL_EQUALITY_TOLERANCE
            and abs(parent.conservation_error_mm)
            <= MODIS_CONSERVATION_TOLERANCE_MM
        )
        all_pass = all_pass and parent_pass
        summaries.append({
            "station_id": station_id,
            "modis_pixel_id": int(parent.modis_pixel_id),
            "eligible": int(parent.eligible),
            "fine_fill_fraction_coarse": float(parent.fine_fill_fraction),
            "sample_rows": int(len(group)),
            "raw_valid_rows": int(raw_valid.sum()),
            "filled_valid_rows": int(filled_valid.sum()),
            "new_fill_rows": int(new_fill.sum()),
            "sample_new_fill_fraction": float(new_fill.mean()),
            "maximum_abs_change_on_raw_valid_Kc": unchanged_error,
            "maximum_abs_fill_minus_parent_mean_Kc": fill_error,
            "Kc_raw_mean_valid": float(group.loc[raw_valid, "Kc_raw"].mean()),
            "Kc_filled_mean_valid": float(group.loc[filled_valid, "Kc_filled"].mean()),
            "Kc_filled_vs_ET_spearman": spearman_all,
            "Kc_raw_vs_ET_spearman_on_raw_valid": spearman_raw,
            "effective_scale_min": float(reconciled.effective_scale.min()),
            "effective_scale_max": float(reconciled.effective_scale.max()),
            "conservation_error_mm": float(parent.conservation_error_mm),
            "pass": parent_pass,
        })
    return summaries, all_pass


def main() -> None:
    args = parse_arguments()
    root = get_project_root()
    output_directory = root / "outputs" / "processed" / "qa" / "conservative_reconciliation"
    output_directory.mkdir(parents=True, exist_ok=True)

    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()
    stations = station_collection(args.station_ids)
    geometry = stations.geometry().buffer(args.buffer_m)
    context = build_production_stack(args.period_start, geometry)

    model_path = (
        root / "outputs" / "processed" / "models" / "S2"
        / PRODUCTION_MODEL_FILENAME
    )
    model = joblib.load(model_path)
    classifier, tree_strings = build_ee_regressor(model, COMMON_MODEL_FEATURES)
    kc_raw = context["stack"].classify(classifier, "Kc_raw").toFloat()
    outputs = build_modis_constrained_et(
        kc_raw=kc_raw,
        optical_predictors=context["optical"],
        s1_predictors=context["s1"],
        model_stack=context["stack"],
        modis_et=context["modis_et"],
        modis_projection=context["modis_projection"],
        fine_projection=context["fine_projection"],
        basin_geometry=geometry,
    )

    modis_projection = context["modis_projection"]
    modis_scale = modis_projection.nominalScale()
    pixel_id = build_modis_pixel_id(modis_projection)
    station_grid = build_modis_grid(
        stations, pixel_id, modis_projection, modis_scale
    )
    station_footprints = assign_station_footprints(
        stations, station_grid, pixel_id, modis_projection, modis_scale
    )

    # The canonical station grid spans a two-MODIS-pixel buffer around the
    # selected points and supplies several parents without a basin-scale graph.
    qa_grid = station_grid
    coarse_qa_image = ee.Image.cat([
        outputs["optical_valid_fraction"],
        outputs["s1_valid_fraction"],
        outputs["model_stack_valid_fraction"],
        outputs["fine_fill_fraction"],
        outputs["eligible"].rename("eligible"),
        outputs["conservation_error"].rename("conservation_error_mm"),
        context["modis_et"].rename("ET_MODIS_mm_period"),
        outputs["et_reaggregated"].rename("ET_reaggregated_mm_period"),
    ])

    def attach_coarse_qa(feature):
        feature = ee.Feature(feature)
        values = coarse_qa_image.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=feature.geometry().centroid(1),
            maxPixels=100,
        )
        return feature.set(values).set("period_start", args.period_start)

    all_parent_payload = qa_grid.map(attach_coarse_qa).getInfo()
    all_parent_qa = pd.DataFrame(feature_rows(all_parent_payload))
    if "eligible" not in all_parent_qa:
        raise RuntimeError(
            "The native MODIS grid did not materialize eligibility values. "
            f"Available properties: {sorted(all_parent_qa.columns)}"
        )
    all_parent_qa = all_parent_qa.loc[
        all_parent_qa["eligible"].notna()
    ].copy()
    all_parent_qa["eligible"] = all_parent_qa["eligible"].astype(int)
    eligible_parent_qa = all_parent_qa.loc[all_parent_qa.eligible.eq(1)].copy()
    eligible_parent_qa["abs_conservation_error_mm"] = (
        eligible_parent_qa.conservation_error_mm.abs()
    )

    station_parent_payload = station_footprints.map(attach_coarse_qa).getInfo()
    station_parent_qa = pd.DataFrame(feature_rows(station_parent_payload))
    station_parent_qa["eligible"] = station_parent_qa["eligible"].astype(int)

    kc_parent_mean = _mean_at_modis_support(
        kc_raw,
        modis_projection,
        "Kc_parent_mean",
    )
    raw_valid = kc_raw.mask().unmask(0).rename("raw_valid")
    filled_valid = outputs["kc_filled"].mask().unmask(0).rename("filled_valid")
    new_fill = raw_valid.eq(0).And(filled_valid.eq(1)).rename("new_fill")
    fine_image = ee.Image.cat([
        as_fine(kc_raw, "Kc_raw", context["fine_projection"]),
        as_fine(outputs["kc_filled"], "Kc_filled", context["fine_projection"]),
        as_fine(kc_parent_mean, "Kc_parent_mean", context["fine_projection"]),
        as_fine(outputs["et_final"], "ET_reconciled", context["fine_projection"]),
        raw_valid.reproject(context["fine_projection"]).toFloat(),
        filled_valid.reproject(context["fine_projection"]).toFloat(),
        new_fill.reproject(context["fine_projection"]).toFloat(),
    ])
    fine_payload = fine_image.sampleRegions(
        collection=station_footprints,
        properties=["station_id", "station", "modis_pixel_id"],
        projection=context["fine_projection"],
        scale=PREDICTION_SCALE_M,
        tileScale=8,
        geometries=True,
    ).getInfo()
    fine_samples = pd.DataFrame(feature_rows(fine_payload))

    fill_summary, fill_pass = describe_fill(fine_samples, station_parent_qa)
    conservation_max = float(
        eligible_parent_qa.abs_conservation_error_mm.max()
    )
    conservation_pass = bool(
        len(eligible_parent_qa) >= 2
        and conservation_max <= MODIS_CONSERVATION_TOLERANCE_MM
    )
    status = "PASS" if fill_pass and conservation_pass else "FAIL"

    compact_date = args.period_start.replace("-", "")
    all_parent_path = output_directory / f"multi_parent_conservation_{compact_date}.csv"
    station_parent_path = output_directory / f"fill_parent_qa_{compact_date}.csv"
    fine_path = output_directory / f"fine_fill_pixels_{compact_date}.csv"
    report_path = output_directory / f"conservative_reconciliation_{compact_date}.json"
    all_parent_qa.to_csv(all_parent_path, index=False)
    station_parent_qa.to_csv(station_parent_path, index=False)
    fine_samples.to_csv(fine_path, index=False)

    report = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "earth_engine_project": args.project,
        "period_start": args.period_start,
        "station_ids": args.station_ids,
        "buffer_m": args.buffer_m,
        "rf_trees": len(tree_strings),
        "fine_fill": {
            "status": "PASS" if fill_pass else "FAIL",
            "fill_equality_tolerance": FILL_EQUALITY_TOLERANCE,
            "unchanged_tolerance": UNCHANGED_TOLERANCE,
            "parents": fill_summary,
        },
        "multi_parent_conservation": {
            "status": "PASS" if conservation_pass else "FAIL",
            "eligible_parent_count": int(len(eligible_parent_qa)),
            "tolerance_mm": MODIS_CONSERVATION_TOLERANCE_MM,
            "maximum_abs_error_mm": conservation_max,
            "p95_abs_error_mm": float(
                eligible_parent_qa.abs_conservation_error_mm.quantile(0.95)
            ),
            "mean_abs_error_mm": float(
                eligible_parent_qa.abs_conservation_error_mm.mean()
            ),
        },
        "files": {
            "multi_parent": str(all_parent_path.relative_to(root)),
            "fill_parents": str(station_parent_path.relative_to(root)),
            "fine_pixels": str(fine_path.relative_to(root)),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("Saved:", report_path, flush=True)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
