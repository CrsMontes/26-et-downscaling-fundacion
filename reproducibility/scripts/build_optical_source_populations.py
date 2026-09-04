"""Build strictly paired Phase 3A populations and validation folds locally."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from et_downscaling.model_spec import SPATIAL_BLOCK_SIZE_KM
from et_downscaling.optical_source_experiment import COMMON_PREDICTORS, THRESHOLDS
from et_downscaling.reference_et_local import build_daily_reference_et


KEYS = ["station_id", "period_start"]
MISSING_SENTINEL_MAX = -9990.0


def project_root():
    return Path(__file__).resolve().parents[2]


def output_root():
    return project_root() / "outputs" / "diagnostics" / "2020_2024" / "optical_source_experiment"


def require_unique(table, name):
    if table.duplicated(KEYS).any():
        raise ValueError(f"{name} has duplicate station-period keys")


def add_spatial_blocks(table):
    result = table.copy()
    latitude = pd.to_numeric(result["station_latitude"], errors="raise")
    longitude = pd.to_numeric(result["station_longitude"], errors="raise")
    km_lat = 111.32
    km_lon = 111.32 * np.cos(np.radians(latitude.mean()))
    block_x = np.floor(longitude * km_lon / SPATIAL_BLOCK_SIZE_KM).astype(int)
    block_y = np.floor(latitude * km_lat / SPATIAL_BLOCK_SIZE_KM).astype(int)
    result["spatial_block"] = block_x.astype(str) + "_" + block_y.astype(str)
    return result


def aggregate_eto(periods, era5, support):
    daily = build_daily_reference_et(era5, support)
    daily["station_id"] = daily.station_id.astype(str)
    daily["local_date"] = pd.to_datetime(daily.local_date).dt.date
    rows = []
    for row in periods.itertuples(index=False):
        start = row.period_start.date()
        end = start + timedelta(days=int(row.period_days))
        selected = daily[
            daily.station_id.eq(str(row.station_id))
            & daily.local_date.ge(start) & daily.local_date.lt(end)
        ]
        complete = (
            len(selected) == int(row.period_days)
            and selected.era5_daily_complete.eq(1).all()
            and selected.ETo_mm_day.notna().all()
        )
        rows.append({
            "station_id": str(row.station_id), "period_start": row.period_start,
            "eto_days_total": len(selected), "eto_days_expected": int(row.period_days),
            "eto_complete": int(complete),
            "ETo_mm_period": selected.ETo_mm_day.sum(min_count=1) if complete else np.nan,
        })
    return pd.DataFrame(rows), daily


def build_fold_tables(populations):
    assignments = []
    definitions = []
    for threshold, population in populations.items():
        for split_type, group_column in (
            ("spatial", "spatial_block"), ("temporal", "year")
        ):
            groups = sorted(population[group_column].unique(), key=str)
            mapping = {group: fold for fold, group in enumerate(groups, start=1)}
            for row in population.itertuples(index=False):
                assignments.append({
                    "threshold_pct": threshold, "split_type": split_type,
                    "station_id": row.station_id,
                    "period_start": row.period_start.strftime("%Y-%m-%d"),
                    "group": str(getattr(row, group_column)),
                    "fold": mapping[getattr(row, group_column)],
                })
            for group, fold in mapping.items():
                test_count = int(population[group_column].eq(group).sum())
                definitions.append({
                    "threshold_pct": threshold, "split_type": split_type,
                    "fold": fold, "test_group": str(group),
                    "train_rows": len(population) - test_count,
                    "test_rows": test_count,
                })
    return pd.DataFrame(assignments), pd.DataFrame(definitions)


def main():
    root = output_root()
    raw = root / "raw"
    availability = project_root() / "outputs" / "diagnostics" / "2020_2024" / "availability" / "raw"
    optical = pd.read_csv(raw / "paired_optical_common.csv", dtype={"station_id": str})
    modis = pd.read_csv(availability / "modis_station_period.csv", dtype={"station_id": str})
    s2_availability = pd.read_csv(availability / "s2_station_period.csv", dtype={"station_id": str})
    hls_availability = pd.read_csv(availability / "hls_station_period.csv", dtype={"station_id": str})
    era5 = pd.read_csv(raw / "era5_hourly.csv", dtype={"station_id": str})
    support = pd.read_csv(
        project_root() / "outputs" / "raw" / "meteorology" / "station_support.csv",
        dtype={"station_id": str},
    )

    for table in (optical, modis, s2_availability, hls_availability):
        table["station_id"] = table.station_id.astype(str)
        table["period_start"] = pd.to_datetime(table.period_start, errors="raise")
    for table, name in (
        (optical, "optical"), (modis, "modis"),
        (s2_availability, "s2 availability"), (hls_availability, "hls availability"),
    ):
        require_unique(table, name)
        if len(table) != 1150:
            raise RuntimeError(f"Expected 1150 {name} rows, found {len(table)}")

    periods = optical[["station_id", "period_start", "period_days"]].copy()
    eto, daily = aggregate_eto(periods, era5, support)
    master = optical.merge(
        modis[KEYS + ["ET_mm_period", "modis_good"]], on=KEYS,
        how="left", validate="one_to_one",
    ).merge(eto, on=KEYS, how="left", validate="one_to_one")
    master = master.merge(
        s2_availability[KEYS + ["continuous_valid_coverage_pct"]].rename(
            columns={"continuous_valid_coverage_pct": "s2_coverage_pct"}
        ), on=KEYS, validate="one_to_one",
    ).merge(
        hls_availability[KEYS + ["combined_continuous_valid_coverage_pct"]].rename(
            columns={"combined_continuous_valid_coverage_pct": "hls_coverage_pct"}
        ), on=KEYS, validate="one_to_one",
    )
    master = master.merge(
        support[["station_id", "station_longitude", "station_latitude"]],
        on="station_id", how="left", validate="many_to_one",
    )
    feature_columns = {
        source: [f"{source}_{name}_mean" for name in COMMON_PREDICTORS]
        for source in ("s2", "hls")
    }
    for columns in feature_columns.values():
        master[columns] = master[columns].apply(pd.to_numeric, errors="coerce")
        master[columns] = master[columns].mask(master[columns] <= MISSING_SENTINEL_MAX)
    master["s2_predictors_complete"] = master[feature_columns["s2"]].notna().all(axis=1).astype(int)
    master["hls_predictors_complete"] = master[feature_columns["hls"]].notna().all(axis=1).astype(int)
    master["target_complete"] = (
        master.modis_good.eq(1) & master.eto_complete.eq(1)
        & master.ET_mm_period.notna() & master.ETo_mm_period.gt(0)
    ).astype(int)
    master["Kc_target"] = np.where(
        master.target_complete.eq(1), master.ET_mm_period / master.ETo_mm_period, np.nan
    )
    master = add_spatial_blocks(master)
    master["year"] = master.period_start.dt.year

    populations = {}
    audit_rows = []
    for threshold in THRESHOLDS:
        availability_mask = (
            master.s2_coverage_pct.ge(threshold) & master.hls_coverage_pct.ge(threshold)
            & master.modis_good.eq(1)
        )
        final_mask = (
            availability_mask & master.target_complete.eq(1)
            & master.s2_predictors_complete.eq(1)
            & master.hls_predictors_complete.eq(1)
        )
        master[f"paired_candidate_ge_{threshold}"] = final_mask.astype(int)
        population = master.loc[final_mask].copy().sort_values(KEYS).reset_index(drop=True)
        populations[threshold] = population
        audit_rows.append({
            "threshold_pct": threshold,
            "availability_maximum": int(availability_mask.sum()),
            "final_population": len(population),
            "reduction": int(availability_mask.sum() - len(population)),
            "missing_eto_or_target": int((availability_mask & master.target_complete.ne(1)).sum()),
            "missing_s2_predictors": int((availability_mask & master.s2_predictors_complete.ne(1)).sum()),
            "missing_hls_predictors": int((availability_mask & master.hls_predictors_complete.ne(1)).sum()),
        })

    assignments, definitions = build_fold_tables(populations)
    population_dir = root / "population"
    folds_dir = root / "folds"
    metadata_dir = root / "metadata"
    for directory in (population_dir, folds_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    master.to_csv(population_dir / "paired_master.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(population_dir / "population_audit.csv", index=False)
    for threshold, population in populations.items():
        population.to_csv(population_dir / f"paired_population_ge{threshold}.csv", index=False)
    assignments.to_csv(folds_dir / "fold_assignments.csv", index=False)
    definitions.to_csv(folds_dir / "fold_definitions.csv", index=False)
    daily.to_csv(raw / "daily_reference_eto.csv", index=False)
    manifest = {
        "rows": len(master), "populations": {str(k): len(v) for k, v in populations.items()},
        "spatial_block_size_km": SPATIAL_BLOCK_SIZE_KM,
        "spatial_groups": int(master.spatial_block.nunique()),
        "temporal_groups": sorted(master.year.unique().tolist()),
        "strictly_paired": True, "training_performed": False,
        "aoa_di_performed": False, "maps_generated": False,
    }
    (metadata_dir / "population_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print("Spatial groups:", manifest["spatial_groups"])
    print("training_performed = false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
