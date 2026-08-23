"""Local standardized reference evapotranspiration calculations.

The equations reproduce the project's ASCE-EWRI daily standardized
Penman-Monteith implementation. Earth Engine is not used here.

ERA5-Land remains a coarse atmospheric support. NASADEM is used only to
characterize mean MODIS-footprint elevation; it does not define ETo/ETr
spatial resolution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOCAL_UTC_OFFSET_HOURS = -5
SHORT_REFERENCE_CN = 900.0
SHORT_REFERENCE_CD = 0.34
TALL_REFERENCE_CN = 1600.0
TALL_REFERENCE_CD = 0.38
REFERENCE_ALBEDO = 0.23
STEFAN_BOLTZMANN_MJ_K4_M2_DAY = 4.903e-9
SOLAR_CONSTANT_MJ_M2_MIN = 0.0820
WIND_MEASUREMENT_HEIGHT_M = 10.0
MIN_RS_RSO = 0.30
MAX_RS_RSO = 1.00
MISSING_SENTINEL_MAX = -9990.0

ERA5_RAW_COLUMNS = [
    "temperature_2m_K",
    "dewpoint_temperature_2m_K",
    "u_wind_10m_ms",
    "v_wind_10m_ms",
    "surface_solar_radiation_downwards_hourly_J_m2",
]


def saturation_vapor_pressure_kpa(temperature_c):
    temperature_c = np.asarray(temperature_c, dtype=float)
    return 0.6108 * np.exp(
        17.27 * temperature_c / (temperature_c + 237.3)
    )


def wind_speed_to_2m(
    wind_speed,
    measurement_height_m=WIND_MEASUREMENT_HEIGHT_M,
):
    factor = 4.87 / np.log(67.8 * measurement_height_m - 5.42)
    return np.asarray(wind_speed, dtype=float) * factor


def atmospheric_pressure_kpa(elevation_m):
    elevation_m = np.asarray(elevation_m, dtype=float)
    return 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26


def extraterrestrial_radiation_mj_m2_day(latitude_deg, day_of_year):
    latitude_rad = np.deg2rad(np.asarray(latitude_deg, dtype=float))
    day_of_year = np.asarray(day_of_year, dtype=float)
    annual_angle = day_of_year * 2.0 * np.pi / 365.0
    inverse_relative_distance = 1.0 + 0.033 * np.cos(annual_angle)
    solar_declination = 0.409 * np.sin(annual_angle - 1.39)
    sunset_argument = -np.tan(latitude_rad) * np.tan(solar_declination)
    sunset_argument = np.clip(sunset_argument, -1.0, 1.0)
    sunset_hour_angle = np.arccos(sunset_argument)
    radiation_term = (
        sunset_hour_angle * np.sin(latitude_rad) * np.sin(solar_declination)
        + np.cos(latitude_rad)
        * np.cos(solar_declination)
        * np.sin(sunset_hour_angle)
    )
    return (
        (24.0 * 60.0 / np.pi)
        * SOLAR_CONSTANT_MJ_M2_MIN
        * inverse_relative_distance
        * radiation_term
    )


def _coerce_raw_era5(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    missing = set(["station_id", "timestamp_utc", *ERA5_RAW_COLUMNS]) - set(result.columns)
    if missing:
        raise ValueError(f"Missing ERA5 columns: {sorted(missing)}")

    for column in ERA5_RAW_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result.loc[result[column] <= MISSING_SENTINEL_MAX, column] = np.nan

    result["timestamp_utc"] = pd.to_datetime(
        result["timestamp_utc"], utc=True, errors="coerce"
    )
    if result["timestamp_utc"].isna().any():
        raise ValueError("Invalid ERA5 UTC timestamps were found.")

    result["station_id"] = result["station_id"].astype(str)
    return result


def prepare_hourly_era5(table: pd.DataFrame) -> pd.DataFrame:
    """Convert raw ERA5-Land fields to analysis units locally."""
    result = _coerce_raw_era5(table)
    result["Tair_C"] = result["temperature_2m_K"] - 273.15
    result["Tdew_C"] = result["dewpoint_temperature_2m_K"] - 273.15
    result["Wind_ms"] = np.sqrt(
        result["u_wind_10m_ms"] ** 2 + result["v_wind_10m_ms"] ** 2
    )
    # Preserve the original ERA5-Land values in the raw export,
    # but enforce the physical lower bound for downward solar
    # radiation during local processing. Direct GEE re-querying
    # confirmed that the few negative values originate in the
    # source hourly band and their effect on ETo/ETr is negligible.
    result["SolarRad_MJ_m2_hour"] = (
        result[
            "surface_solar_radiation_downwards_hourly_J_m2"
        ]
        .clip(lower=0.0)
        * 1e-6
    )
    result["ea_kPa"] = saturation_vapor_pressure_kpa(result["Tdew_C"])
    result["VPD_kPa"] = np.maximum(
        saturation_vapor_pressure_kpa(result["Tair_C"]) - result["ea_kPa"],
        0.0,
    )

    local_time = result["timestamp_utc"] + pd.to_timedelta(
        LOCAL_UTC_OFFSET_HOURS, unit="h"
    )
    result["local_datetime"] = local_time
    result["local_date"] = local_time.dt.date
    result["raw_values_complete"] = result[ERA5_RAW_COLUMNS].notna().all(axis=1).astype(int)
    return result


def _calculate_daily_reference_et_values(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.copy()
    date_values = pd.to_datetime(result["local_date"])
    day_of_year = date_values.dt.dayofyear.to_numpy(dtype=float)
    latitude = result["footprint_centroid_latitude"].to_numpy(dtype=float)
    elevation = result["footprint_mean_elevation_m"].to_numpy(dtype=float)

    tmin = result["Tmin_day_C"].to_numpy(dtype=float)
    tmax = result["Tmax_day_C"].to_numpy(dtype=float)
    tmean = result["Tmean_day_C"].to_numpy(dtype=float)
    ea = result["ea_day_kPa"].to_numpy(dtype=float)
    wind2 = result["Wind2m_mean_ms"].to_numpy(dtype=float)
    rs = result["Rs_day_MJ_m2"].to_numpy(dtype=float)

    es = (
        saturation_vapor_pressure_kpa(tmin)
        + saturation_vapor_pressure_kpa(tmax)
    ) / 2.0
    vpd = np.maximum(es - ea, 0.0)
    es_tmean = saturation_vapor_pressure_kpa(tmean)
    delta = 4098.0 * es_tmean / (tmean + 237.3) ** 2
    pressure = atmospheric_pressure_kpa(elevation)
    gamma = 0.000665 * pressure
    ra = extraterrestrial_radiation_mj_m2_day(latitude, day_of_year)
    rso = (0.75 + 2e-5 * elevation) * ra
    rns = rs * (1.0 - REFERENCE_ALBEDO)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs_rso_raw = rs / rso
    rs_rso_used = np.clip(rs_rso_raw, MIN_RS_RSO, MAX_RS_RSO)
    cloudiness = 1.35 * rs_rso_used - 0.35
    humidity = 0.34 - 0.14 * np.sqrt(ea)
    temperature_factor = (
        STEFAN_BOLTZMANN_MJ_K4_M2_DAY
        * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4)
        / 2.0
    )
    rnl = temperature_factor * humidity * cloudiness
    rn = rns - rnl

    def calculate_reference_et(cn, cd):
        energy_term = 0.408 * delta * rn
        aerodynamic_term = gamma * cn * wind2 * vpd / (tmean + 273.0)
        denominator = delta + gamma * (1.0 + cd * wind2)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (energy_term + aerodynamic_term) / denominator

    result["es_day_kPa"] = es
    result["VPD_day_kPa"] = vpd
    result["Delta_kPa_C"] = delta
    result["Pressure_kPa"] = pressure
    result["Gamma_kPa_C"] = gamma
    result["Ra_day_MJ_m2"] = ra
    result["Rso_day_MJ_m2"] = rso
    result["Rs_Rso_raw"] = rs_rso_raw
    result["Rs_Rso_used"] = rs_rso_used
    result["Rns_day_MJ_m2"] = rns
    result["Rnl_day_MJ_m2"] = rnl
    result["Rn_day_MJ_m2"] = rn
    result["ETo_mm_day"] = calculate_reference_et(
        SHORT_REFERENCE_CN, SHORT_REFERENCE_CD
    )
    result["ETr_mm_day"] = calculate_reference_et(
        TALL_REFERENCE_CN, TALL_REFERENCE_CD
    )
    return result


def build_daily_reference_et(
    era5_hourly: pd.DataFrame,
    station_support: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate hourly ERA5-Land to Colombia local days and calculate ETo/ETr."""
    hourly = prepare_hourly_era5(era5_hourly)
    support = station_support.copy()
    support["station_id"] = support["station_id"].astype(str)

    required_support = {
        "station_id",
        "footprint_centroid_latitude",
        "footprint_mean_elevation_m",
    }
    missing_support = required_support - set(support.columns)
    if missing_support:
        raise ValueError(f"Missing station-support columns: {sorted(missing_support)}")

    grouped = hourly.groupby(["station_id", "local_date"], sort=True, observed=True)
    daily = grouped.agg(
        era5_hours_total=("timestamp_utc", "size"),
        era5_valid_hours=("raw_values_complete", "sum"),
        Tmin_day_C=("Tair_C", "min"),
        Tmax_day_C=("Tair_C", "max"),
        Tair_hourly_mean_C=("Tair_C", "mean"),
        ea_day_kPa=("ea_kPa", "mean"),
        Wind10m_mean_ms=("Wind_ms", "mean"),
        Rs_day_MJ_m2=("SolarRad_MJ_m2_hour", lambda values: values.sum(min_count=1)),
    ).reset_index()

    daily["Tmean_day_C"] = (daily["Tmin_day_C"] + daily["Tmax_day_C"]) / 2.0
    daily["Wind2m_mean_ms"] = wind_speed_to_2m(daily["Wind10m_mean_ms"])
    daily["era5_hours_expected"] = 24
    daily["era5_daily_complete"] = (
        (daily["era5_hours_total"] == 24)
        & (daily["era5_valid_hours"] == 24)
    ).astype(int)

    support_columns = [
        "station_id",
        "footprint_centroid_latitude",
        "footprint_mean_elevation_m",
    ]
    daily = daily.merge(
        support[support_columns].drop_duplicates("station_id"),
        on="station_id",
        how="left",
        validate="many_to_one",
    )
    if daily[["footprint_centroid_latitude", "footprint_mean_elevation_m"]].isna().any().any():
        raise ValueError("Station support could not be joined to all ERA5 daily rows.")

    return _calculate_daily_reference_et_values(daily)
