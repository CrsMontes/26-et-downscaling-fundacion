"""Build reusable 2020-2024 period meteorology from local raw inputs.

This script reuses the existing production-compatible aggregation logic.
No Earth Engine access or model training is performed.
"""

from pathlib import Path

import pandas as pd

from et_downscaling.local_training import _aggregate_period_inputs


def main():
    root = Path("outputs/diagnostics/2020_2024")

    optical_path = (
        root
        / "optical_source_experiment"
        / "raw"
        / "paired_optical_common.csv"
    )

    era5_path = (
        root
        / "optical_source_experiment"
        / "raw"
        / "era5_hourly.csv"
    )

    daily_path = (
        root
        / "optical_source_experiment"
        / "raw"
        / "daily_reference_eto.csv"
    )

    chirps_path = (
        root
        / "meteorology_experiment"
        / "raw"
        / "chirps_daily_20191202_20241231.csv"
    )

    optical = pd.read_csv(optical_path)

    required = [
        "station_id",
        "period_start",
        "period_days",
    ]

    missing = [
        column
        for column in required
        if column not in optical.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing period columns: {missing}"
        )

    periods = (
        optical[
            required
        ]
        .drop_duplicates(
            ["station_id", "period_start"]
        )
        .rename(
            columns={
                "period_days": "number_days"
            }
        )
        .copy()
    )

    periods["station_id"] = (
        periods["station_id"].astype(str)
    )

    periods["period_start"] = (
        pd.to_datetime(
            periods["period_start"]
        )
        .dt.date
    )

    era5 = pd.read_csv(era5_path)
    daily = pd.read_csv(daily_path)
    chirps = pd.read_csv(chirps_path)

    meteorology = _aggregate_period_inputs(
        periods=periods,
        era5_hourly=era5,
        daily_reference_et=daily,
        chirps_daily=chirps,
    )

    if len(meteorology) != len(periods):
        raise RuntimeError(
            f"Row mismatch: "
            f"{len(meteorology)} != {len(periods)}"
        )

    if meteorology.duplicated(
        ["station_id", "period_start"]
    ).any():
        raise RuntimeError(
            "Duplicate station-period rows."
        )

    output_dir = (
        root
        / "meteorology_experiment"
        / "processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "period_meteorology.csv"
    )

    meteorology.to_csv(
        output_path,
        index=False,
    )

    print("\nPERIOD METEOROLOGY 2020-2024")
    print("============================")
    print("Rows:", len(meteorology))
    print(
        "Meteo complete:",
        int(meteorology["meteo_complete"].sum()),
    )
    print(
        "ERA5 temporal complete:",
        int(
            meteorology[
                "era5_temporal_complete"
            ].sum()
        ),
    )
    print(
        "Reference ET complete:",
        int(
            meteorology[
                "reference_et_complete"
            ].sum()
        ),
    )
    print(
        "CHIRPS complete:",
        int(
            meteorology[
                "chirps_complete"
            ].sum()
        ),
    )

    print("\nVARIABLE RANGES")

    variables = [
        "Tair_mean_C",
        "Tair_max_C",
        "VPD_mean_kPa",
        "VPD_max_kPa",
        "SolarRad_MJ_m2_day",
        "Wind_mean_ms",
        "Precip_period_mm",
        "Precip_prev30d_mm",
        "ETo_mm_period",
        "ETo_mm_day",
    ]

    print(
        meteorology[variables]
        .describe()
        .T[
            ["min", "mean", "std", "max"]
        ]
        .to_string()
    )

    print()
    print("Output:", output_path)
    print("earth_engine_access = false")
    print("training_performed = false")


if __name__ == "__main__":
    main()
