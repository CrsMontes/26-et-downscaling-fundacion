"""Local construction of the ET downscaling training master."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import (
    END_DATE,
    OPTICAL_QA_THRESHOLDS_PCT,
    START_DATE,
    normalize_optical_source,
)
from .reference_et_local import build_daily_reference_et, prepare_hourly_era5
from .schema import (
    COMMON_SATELLITE_MODEL_STAT_COLUMNS,
    get_satellite_stat_columns,
    get_source_model_candidate_stat_columns,
)

MISSING_SENTINEL_MAX = -9990.0


def _resolve_optical_source(
    satellite: pd.DataFrame,
) -> str:
    if "optical_source" not in satellite.columns:
        raise ValueError(
            "Required column 'optical_source' is missing."
        )

    values = (
        satellite["optical_source"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if len(values) != 1:
        raise ValueError(
            "Satellite table must contain exactly one optical "
            f"source. Found: {values}"
        )

    return normalize_optical_source(
        values[0]
    )


def _normalize_station_id(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    if "station_id" not in result.columns:
        raise ValueError("Required column 'station_id' is missing.")
    result["station_id"] = result["station_id"].astype(str)
    return result


def _replace_numeric_sentinels(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    for column in result.columns:
        if column in {"station_id"}:
            continue
        converted = pd.to_numeric(result[column], errors="coerce")
        numeric_fraction = converted.notna().mean() if len(result) else 0.0
        if numeric_fraction > 0.8:
            result[column] = converted
            result.loc[result[column] <= MISSING_SENTINEL_MAX, column] = np.nan
    return result


def _check_unique(table: pd.DataFrame, keys, name: str) -> None:
    duplicated = table.duplicated(keys, keep=False)
    if duplicated.any():
        examples = table.loc[duplicated, keys].head().to_dict("records")
        raise ValueError(f"{name} contains duplicate keys {keys}: {examples}")


def _prepare_chirps(chirps_daily: pd.DataFrame) -> pd.DataFrame:
    result = _normalize_station_id(chirps_daily)
    required = {"station_id", "date", "precipitation_mm"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Missing CHIRPS columns: {sorted(missing)}")
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.date
    result["precipitation_mm"] = pd.to_numeric(result["precipitation_mm"], errors="coerce")
    result.loc[result["precipitation_mm"] <= MISSING_SENTINEL_MAX, "precipitation_mm"] = np.nan
    _check_unique(result, ["station_id", "date"], "CHIRPS daily table")
    return result


def _aggregate_period_inputs(
    periods: pd.DataFrame,
    era5_hourly: pd.DataFrame,
    daily_reference_et: pd.DataFrame,
    chirps_daily: pd.DataFrame,
) -> pd.DataFrame:
    hourly = prepare_hourly_era5(era5_hourly)
    daily_reference = _normalize_station_id(daily_reference_et)
    daily_reference["local_date"] = pd.to_datetime(
        daily_reference["local_date"], errors="coerce"
    ).dt.date
    chirps = _prepare_chirps(chirps_daily)

    rows = []
    for row in periods.itertuples(index=False):
        station_id = str(row.station_id)
        start = row.period_start
        number_days = int(row.number_days)
        end = start + timedelta(days=number_days)

        utc_start = pd.Timestamp(start, tz="UTC")
        utc_end = pd.Timestamp(end, tz="UTC")
        station_hourly = hourly.loc[
            (hourly["station_id"] == station_id)
            & (hourly["timestamp_utc"] >= utc_start)
            & (hourly["timestamp_utc"] < utc_end)
        ]
        expected_hours = number_days * 24

        station_daily = daily_reference.loc[
            (daily_reference["station_id"] == station_id)
            & (daily_reference["local_date"] >= start)
            & (daily_reference["local_date"] < end)
        ]

        station_chirps_period = chirps.loc[
            (chirps["station_id"] == station_id)
            & (chirps["date"] >= start)
            & (chirps["date"] < end)
        ]
        previous_start = start - timedelta(days=30)
        station_chirps_previous = chirps.loc[
            (chirps["station_id"] == station_id)
            & (chirps["date"] >= previous_start)
            & (chirps["date"] < start)
        ]

        raw_complete_hours = int(station_hourly["raw_values_complete"].sum())
        era5_hours_total = len(station_hourly)
        reference_days_total = len(station_daily)
        reference_days_complete = int(station_daily["era5_daily_complete"].sum())
        chirps_days_period = len(station_chirps_period)
        chirps_valid_days_period = int(station_chirps_period["precipitation_mm"].notna().sum())
        chirps_days_prev30 = len(station_chirps_previous)
        chirps_valid_days_prev30 = int(station_chirps_previous["precipitation_mm"].notna().sum())

        era5_temporal_complete = (
            era5_hours_total == expected_hours
            and raw_complete_hours == expected_hours
        )
        reference_et_complete = (
            reference_days_total == number_days
            and reference_days_complete == number_days
            and station_daily[["ETo_mm_day", "ETr_mm_day"]].notna().all().all()
        )
        chirps_complete = (
            chirps_days_period == number_days
            and chirps_valid_days_period == number_days
            and chirps_days_prev30 == 30
            and chirps_valid_days_prev30 == 30
        )

        rows.append(
            {
                "station_id": station_id,
                "period_start": start,
                "Tair_mean_C": station_hourly["Tair_C"].mean(),
                "Tair_max_C": station_hourly["Tair_C"].max(),
                "VPD_mean_kPa": station_hourly["VPD_kPa"].mean(),
                "VPD_max_kPa": station_hourly["VPD_kPa"].max(),
                "SolarRad_MJ_m2_day": station_hourly["SolarRad_MJ_m2_hour"].sum(min_count=1) / number_days,
                "Wind_mean_ms": station_hourly["Wind_ms"].mean(),
                "Precip_period_mm": station_chirps_period["precipitation_mm"].sum(min_count=1),
                "Precip_prev30d_mm": station_chirps_previous["precipitation_mm"].sum(min_count=1),
                "ETo_mm_period": station_daily["ETo_mm_day"].sum(min_count=1),
                "ETr_mm_period": station_daily["ETr_mm_day"].sum(min_count=1),
                "ETo_mm_day": station_daily["ETo_mm_day"].sum(min_count=1) / number_days,
                "ETr_mm_day": station_daily["ETr_mm_day"].sum(min_count=1) / number_days,
                "era5_hours_total": era5_hours_total,
                "era5_hours_expected": expected_hours,
                "era5_valid_hours": raw_complete_hours,
                "reference_et_days_total": reference_days_total,
                "reference_et_days_expected": number_days,
                "reference_et_complete": int(reference_et_complete),
                "chirps_days_period": chirps_days_period,
                "chirps_days_expected": number_days,
                "chirps_valid_days_period": chirps_valid_days_period,
                "chirps_days_prev30": chirps_days_prev30,
                "chirps_days_prev30_expected": 30,
                "chirps_valid_days_prev30": chirps_valid_days_prev30,
                "era5_temporal_complete": int(era5_temporal_complete),
                "chirps_complete": int(chirps_complete),
                "meteo_complete": int(
                    era5_temporal_complete and chirps_complete and reference_et_complete
                ),
            }
        )

    return pd.DataFrame(rows)


def build_training_master(
    satellite: pd.DataFrame,
    era5_hourly: pd.DataFrame,
    chirps_daily: pd.DataFrame,
    station_support: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the complete local master and daily reference-ET QA table."""
    satellite = _normalize_station_id(satellite)
    satellite = _replace_numeric_sentinels(satellite)
    support = _normalize_station_id(station_support)

    optical_source = (
        _resolve_optical_source(
            satellite
        )
    )

    extraction_stat_columns = (
        get_satellite_stat_columns(
            optical_source
        )
    )

    source_model_stat_columns = (
        get_source_model_candidate_stat_columns(
            optical_source
        )
    )

    common_model_stat_columns = list(
        COMMON_SATELLITE_MODEL_STAT_COLUMNS
    )

    required_satellite = {
        "station_id",
        "station",
        "period_start",
        "number_days",
        "ET_mm_period",
        "modis_good",
        "optical_source",
        "optical_union_coverage_pct",
        "s1_union_coverage_pct",
        "s1_valid",
        *extraction_stat_columns,
    }
    missing = required_satellite - set(satellite.columns)
    if missing:
        raise ValueError(f"Missing satellite columns: {sorted(missing)}")

    satellite["period_start"] = pd.to_datetime(
        satellite["period_start"], errors="coerce"
    ).dt.date
    if satellite["period_start"].isna().any():
        raise ValueError("Invalid satellite period_start values were found.")

    satellite["number_days"] = pd.to_numeric(
        satellite["number_days"], errors="raise"
    ).astype(int)
    _check_unique(satellite, ["station_id", "period_start"], "Satellite table")

    daily_reference = build_daily_reference_et(
        era5_hourly=era5_hourly,
        station_support=support,
    )
    analysis_start = date.fromisoformat(START_DATE)
    analysis_end = date.fromisoformat(END_DATE)
    daily_reference = daily_reference.loc[
        (daily_reference["local_date"] >= analysis_start)
        & (daily_reference["local_date"] < analysis_end)
    ].copy()

    periods = satellite[["station_id", "period_start", "number_days"]].copy()
    period_meteorology = _aggregate_period_inputs(
        periods=periods,
        era5_hourly=era5_hourly,
        daily_reference_et=daily_reference,
        chirps_daily=chirps_daily,
    )
    _check_unique(period_meteorology, ["station_id", "period_start"], "Period meteorology")

    master = satellite.merge(
        period_meteorology,
        on=["station_id", "period_start"],
        how="left",
        validate="one_to_one",
    )

    support_metadata_columns = [
        column
        for column in [
            "station_id",
            "station_longitude",
            "station_latitude",
            "footprint_centroid_longitude",
            "footprint_centroid_latitude",
            "footprint_mean_elevation_m",
            "elevation_sampling_method",
            "elevation_grid_scale_m",
            "era5_support_m",
            "era5_sampling_method",
            "era5_sampling_longitude",
            "era5_sampling_latitude",
            "era5_sampling_distance_m",
            "chirps_support_m",
        ]
        if column in support.columns
    ]
    master = master.merge(
        support[support_metadata_columns].drop_duplicates("station_id"),
        on="station_id",
        how="left",
        validate="many_to_one",
    )

    master["Kc_target"] = np.where(
        master["ETo_mm_period"] > 0,
        master["ET_mm_period"] / master["ETo_mm_period"],
        np.nan,
    )
    master["ET_reconstructed_mm_period"] = (
        master["Kc_target"] * master["ETo_mm_period"]
    )
    master["ET_reconstruction_error_mm"] = (
        master["ET_reconstructed_mm_period"] - master["ET_mm_period"]
    )

    master["satellite_extraction_complete"] = (
        master[
            extraction_stat_columns
        ]
        .notna()
        .all(axis=1)
        .astype(int)
    )

    master["common_satellite_predictors_complete"] = (
        master[
            common_model_stat_columns
        ]
        .notna()
        .all(axis=1)
        .astype(int)
    )

    master["source_satellite_predictors_complete"] = (
        master[
            source_model_stat_columns
        ]
        .notna()
        .all(axis=1)
        .astype(int)
    )

    master["target_complete"] = (
        master["ET_mm_period"].notna()
        & master["ETo_mm_period"].notna()
        & master["Kc_target"].notna()
        & (master["modis_good"] == 1)
    ).astype(int)

    for threshold in OPTICAL_QA_THRESHOLDS_PCT:
        label = str(int(threshold))
        optical_flag = master["optical_union_coverage_pct"] >= threshold
        master[f"optical_ge_{label}"] = optical_flag.astype(int)
        common_candidate = (
            (master["target_complete"] == 1)
            & (
                master[
                    "common_satellite_predictors_complete"
                ]
                == 1
            )
            & (master["meteo_complete"] == 1)
            & (master["s1_valid"] == 1)
            & optical_flag
        )

        source_candidate = (
            (master["target_complete"] == 1)
            & (
                master[
                    "source_satellite_predictors_complete"
                ]
                == 1
            )
            & (master["meteo_complete"] == 1)
            & (master["s1_valid"] == 1)
            & optical_flag
        )

        master[
            f"training_candidate_common_ge_{label}"
        ] = common_candidate.astype(int)

        master[
            f"training_candidate_source_ge_{label}"
        ] = source_candidate.astype(int)

    master = master.sort_values(["station_id", "period_start"]).reset_index(drop=True)
    daily_reference = daily_reference.sort_values(["station_id", "local_date"]).reset_index(drop=True)
    return master, daily_reference
