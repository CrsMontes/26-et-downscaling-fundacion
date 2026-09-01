import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ee
import numpy as np
import pandas as pd

from et_downscaling.config import (
    END_DATE,
    OUTPUT_PERIOD_LABEL,
    START_DATE,
)
from et_downscaling.hls import (
    build_hls_medoid,
    get_hls_collection,
    get_local_hls_mgrs_tiles,
)
from et_downscaling.modis import (
    build_modis_inputs,
)


COVERAGE_THRESHOLD_PCT = 80.0
SOURCE_SCALE_M = 30
WITHIN_LOW_PERCENTILE = 5
WITHIN_HIGH_PERCENTILE = 95
GLOBAL_LOW_QUANTILE = 0.05
GLOBAL_HIGH_QUANTILE = 0.95

OUTPUT_DIRECTORY = (
    Path("outputs")
    / "diagnostics"
    / "hls_fvc_recalibration"
    / OUTPUT_PERIOD_LABEL
)

OBSERVATIONS_FILENAME = (
    "hls_fvc_recalibration_observations.csv"
)

SUMMARY_FILENAME = (
    "hls_fvc_recalibration_summary.json"
)

CANDIDATE_CONFIG_FILENAME = (
    "fvc_endmembers_candidate.json"
)

CURRENT_CONFIG_PATH = (
    Path("config")
    / "fvc_endmembers.json"
)


# ============================================================
# Command-line arguments
# ============================================================


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Recalibrate HLS FVC NDVI endmembers using the "
            "current production HLS preprocessing and verified "
            "local MGRS filtering."
        )
    )

    parser.add_argument(
        "--project",
        default=None,
        help=(
            "Google Cloud Project ID for Earth Engine. If omitted, "
            "the script asks interactively."
        ),
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Discard the diagnostic checkpoint and recompute all selected "
            "MODIS periods. The production FVC config is never "
            "overwritten by this script."
        ),
    )

    return parser.parse_args()


# ============================================================
# Earth Engine initialization
# ============================================================


def initialize_earth_engine(project_id=None):
    if project_id is None:
        project_id = input(
            "Google Cloud Project ID: "
        ).strip()

    if not project_id:
        raise ValueError(
            "Google Cloud Project ID cannot be empty."
        )

    ee.Initialize(
        project=project_id
    )

    ee.Number(1).getInfo()

    print(
        "Earth Engine initialized with project:",
        project_id,
    )

    return project_id


# ============================================================
# MODIS period table
# ============================================================


def get_modis_periods(
    modis_collection,
):
    timestamps = (
        ee.ImageCollection(
            modis_collection
        )
        .aggregate_array(
            "system:time_start"
        )
        .getInfo()
    )

    periods = []

    for timestamp_ms in timestamps:
        period_start_dt = (
            datetime.fromtimestamp(
                float(timestamp_ms) / 1000.0,
                tz=timezone.utc,
            )
        )

        nominal_end = (
            period_start_dt
            + timedelta(days=8)
        )

        next_year_start = datetime(
            period_start_dt.year + 1,
            1,
            1,
            tzinfo=timezone.utc,
        )

        period_end_dt = min(
            nominal_end,
            next_year_start,
        )

        periods.append(
            (
                period_start_dt.strftime(
                    "%Y-%m-%d"
                ),
                period_end_dt.strftime(
                    "%Y-%m-%d"
                ),
            )
        )

    return sorted(
        set(periods)
    )


# ============================================================
# Calibration masks and indices
# ============================================================


def build_ndvi_ndwi(
    medoid,
):
    medoid = ee.Image(
        medoid
    )

    ndvi = (
        medoid.normalizedDifference(
            [
                "NIR",
                "Red",
            ]
        )
        .rename(
            "NDVI"
        )
        .toFloat()
    )

    ndwi = (
        medoid.normalizedDifference(
            [
                "Green",
                "NIR",
            ]
        )
        .rename(
            "NDWI"
        )
        .toFloat()
    )

    return (
        ndvi.addBands(
            ndwi
        )
    )



def get_common_valid_mask(
    medoid,
):
    return (
        ee.Image(
            medoid
        )
        .select(
            [
                "Green",
                "Red",
                "NIR",
            ]
        )
        .mask()
        .reduce(
            ee.Reducer.min()
        )
        .rename(
            "valid"
        )
        .uint8()
    )


# ============================================================
# One station-period calibration feature
# ============================================================


def build_station_period_feature(
    footprint,
    hls_collection,
    local_mgrs_tiles,
    period_start,
    period_end,
):
    footprint = ee.Feature(
        footprint
    )

    geometry = (
        footprint.geometry()
    )

    period_collection = (
        ee.ImageCollection(
            hls_collection
        )
        .filterDate(
            period_start,
            period_end,
        )
    )

    local_collection = (
        period_collection
        .filter(
            ee.Filter.inList(
                "hls_mgrs_tile",
                local_mgrs_tiles,
            )
        )
        .filterBounds(
            geometry
        )
    )

    product_count = (
        local_collection.size()
    )

    medoid = ee.Image(
        build_hls_medoid(
            local_collection,
            geometry,
        )
    )

    valid_mask = (
        get_common_valid_mask(
            medoid
        )
    )

    coverage_raw = (
        valid_mask
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=SOURCE_SCALE_M,
            maxPixels=1e7,
            tileScale=4,
        )
        .get(
            "valid"
        )
    )

    coverage_pct = ee.Number(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(
                coverage_raw,
                None,
            ),
            0.0,
            ee.Number(
                coverage_raw
            ).multiply(
                100.0
            ),
        )
    )

    indices = (
        build_ndvi_ndwi(
            medoid
        )
    )

    # Historical diagnosed calibration rule:
    # exclude pixels with NDWI > 0 before deriving NDVI
    # endmember candidates.
    nonwater_mask = (
        indices
        .select(
            "NDWI"
        )
        .lte(0)
    )

    ndvi_nonwater = (
        indices
        .select(
            "NDVI"
        )
        .updateMask(
            valid_mask
        )
        .updateMask(
            nonwater_mask
        )
    )

    percentile_reducer = (
        ee.Reducer.percentile(
            [
                WITHIN_LOW_PERCENTILE,
                WITHIN_HIGH_PERCENTILE,
            ],
            [
                "p05",
                "p95",
            ],
        )
        .combine(
            reducer2=(
                ee.Reducer.count()
            ),
            sharedInputs=True,
        )
    )

    ndvi_stats = ee.Dictionary(
        ndvi_nonwater.reduceRegion(
            reducer=percentile_reducer,
            geometry=geometry,
            scale=SOURCE_SCALE_M,
            maxPixels=1e7,
            tileScale=4,
        )
    )

    return ee.Feature(
        None,
        {
            "source": "HLS",
            "station": footprint.get(
                "station"
            ),
            "station_id": footprint.get(
                "station_id"
            ),
            "period_start": period_start,
            "period_end": period_end,
            "source_scale_m": SOURCE_SCALE_M,
            "products": product_count,
            "hls_mgrs_tiles": ";".join(
                local_mgrs_tiles
            ),
            "coverage_pct": coverage_pct,
            "NDVI_p05": ndvi_stats.get(
                "NDVI_p05"
            ),
            "NDVI_p95": ndvi_stats.get(
                "NDVI_p95"
            ),
            "nonwater_pixel_count": ndvi_stats.get(
                "NDVI_count"
            ),
        },
    )


# ============================================================
# Checkpoint utilities
# ============================================================


def load_checkpoint(
    checkpoint_path,
):
    if not checkpoint_path.is_file():
        return pd.DataFrame()

    dataframe = pd.read_csv(
        checkpoint_path,
        dtype={
            "station_id": "string",
        },
    )

    return dataframe



def save_checkpoint(
    dataframe,
    checkpoint_path,
):
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        checkpoint_path,
        index=False,
    )


# ============================================================
# Two-stage global calibration
# ============================================================


def calculate_endmembers(
    observations,
):
    observations = (
        observations.copy()
    )

    for column in [
        "coverage_pct",
        "NDVI_p05",
        "NDVI_p95",
        "nonwater_pixel_count",
    ]:
        observations[column] = (
            pd.to_numeric(
                observations[column],
                errors="coerce",
            )
        )

    eligible = observations.loc[
        (
            observations[
                "coverage_pct"
            ]
            >= COVERAGE_THRESHOLD_PCT
        )
        & observations[
            "NDVI_p05"
        ].notna()
        & observations[
            "NDVI_p95"
        ].notna()
        & (
            observations[
                "nonwater_pixel_count"
            ]
            > 0
        )
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "No eligible HLS station-period observations "
            "were available for FVC calibration."
        )

    ndvi_low = float(
        eligible[
            "NDVI_p05"
        ].quantile(
            GLOBAL_LOW_QUANTILE
        )
    )

    ndvi_high = float(
        eligible[
            "NDVI_p95"
        ].quantile(
            GLOBAL_HIGH_QUANTILE
        )
    )

    if not (
        np.isfinite(ndvi_low)
        and np.isfinite(ndvi_high)
        and -1.0 <= ndvi_low < ndvi_high <= 1.0
    ):
        raise RuntimeError(
            "Invalid recalibrated HLS FVC endmembers: "
            f"low={ndvi_low}, high={ndvi_high}."
        )

    station_count = int(
        eligible[
            "station_id"
        ].nunique()
    )

    return {
        "ndvi_low_endmember": ndvi_low,
        "ndvi_high_endmember": ndvi_high,
        "n_observations": int(
            len(eligible)
        ),
        "n_stations": station_count,
    }, eligible


# ============================================================
# Candidate config
# ============================================================


def build_candidate_config(
    hls_calibration,
):
    if not CURRENT_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            "Current FVC config was not found: "
            f"{CURRENT_CONFIG_PATH}"
        )

    with CURRENT_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        calibration = json.load(
            file
        )

    if (
        "sources" not in calibration
        or "S2" not in calibration[
            "sources"
        ]
    ):
        raise RuntimeError(
            "Current FVC config does not contain the S2 "
            "calibration that must be preserved."
        )

    calibration[
        "sources"
    ][
        "HLS"
    ] = hls_calibration

    calibration[
        "calibration_scope"
    ] = (
        "diagnostic_candidate_after_hls_mgrs_correction"
    )

    calibration[
        "recalibration_provenance"
    ] = {
        "source": "HLS",
        "reason": (
            "Recomputed after correcting HLS local spatial "
            "selection with verified MGRS tiles."
        ),
        "generated_utc": (
            datetime.now(
                timezone.utc
            )
            .replace(
                microsecond=0
            )
            .isoformat()
        ),
        "production_config_overwritten": False,
    }

    return calibration


# ============================================================
# Main
# ============================================================


def main():
    args = parse_arguments()

    initialize_earth_engine(
        args.project
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        OUTPUT_DIRECTORY
        / OBSERVATIONS_FILENAME
    )

    if (
        args.restart
        and checkpoint_path.exists()
    ):
        checkpoint_path.unlink()

    modis_inputs = (
        build_modis_inputs()
    )

    modis_collection = (
        ee.ImageCollection(
            modis_inputs[
                "collection"
            ]
        )
    )

    station_footprints = (
        ee.FeatureCollection(
            modis_inputs[
                "station_footprints"
            ]
        )
    )

    hls_collection = (
        get_hls_collection(
            station_footprints
        )
    )

    periods = get_modis_periods(
        modis_collection
    )

    if not periods:
        raise RuntimeError("No MODIS periods found for the configured interval.")

    footprint_list = (
        station_footprints.toList(
            station_footprints.size()
        )
    )

    station_count = int(
        station_footprints.size().getInfo()
    )

    if station_count != 5:
        raise RuntimeError(
            "Expected 5 station footprints, found "
            f"{station_count}."
        )

    station_inputs = []

    print()
    print(
        "Resolving verified local HLS MGRS tiles..."
    )

    for station_index in range(
        station_count
    ):
        footprint = ee.Feature(
            footprint_list.get(
                station_index
            )
        )

        station_name = str(
            footprint.get(
                "station"
            ).getInfo()
        )

        local_mgrs_tiles = (
            get_local_hls_mgrs_tiles(
                footprint.geometry()
            )
            .getInfo()
        )

        if not local_mgrs_tiles:
            raise RuntimeError(
                "No local HLS MGRS tiles were resolved for "
                f"station '{station_name}'."
            )

        local_mgrs_tiles = sorted(
            {
                str(tile)
                for tile in local_mgrs_tiles
            }
        )

        print(
            f"  {station_name}: "
            f"{', '.join(local_mgrs_tiles)}"
        )

        station_inputs.append(
            {
                "footprint": footprint,
                "mgrs_tiles": local_mgrs_tiles,
            }
        )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    if checkpoint.empty:
        completed_periods = set()
    else:
        counts = (
            checkpoint.groupby(
                [
                    "period_start",
                    "period_end",
                ]
            )
            .size()
        )

        completed_periods = {
            period
            for period, count in counts.items()
            if int(count) == station_count
        }

    print()
    print(
        "HLS FVC RECALIBRATION"
    )
    print(
        "====================="
    )
    print(
        "Period:",
        START_DATE,
        "to",
        END_DATE,
    )
    print(
        "MODIS periods:",
        len(periods),
    )
    print(
        "Station footprints:",
        station_count,
    )
    print(
        "Existing complete periods:",
        len(completed_periods),
    )
    print(
        "Production FVC config will NOT be overwritten."
    )
    print()

    rows = (
        checkpoint.to_dict(
            orient="records"
        )
        if not checkpoint.empty
        else []
    )

    for period_index, (
        period_start,
        period_end,
    ) in enumerate(
        periods,
        start=1,
    ):
        period_key = (
            period_start,
            period_end,
        )

        if period_key in completed_periods:
            print(
                f"{period_index:03d}/{len(periods)} "
                f"using checkpoint | {period_start} -> {period_end}"
            )
            continue

        print(
            f"{period_index:03d}/{len(periods)} "
            f"processing | {period_start} -> {period_end}"
        )

        period_features = []

        for station_input in station_inputs:
            period_features.append(
                build_station_period_feature(
                    footprint=(
                        station_input[
                            "footprint"
                        ]
                    ),
                    hls_collection=hls_collection,
                    local_mgrs_tiles=(
                        station_input[
                            "mgrs_tiles"
                        ]
                    ),
                    period_start=period_start,
                    period_end=period_end,
                )
            )

        info = (
            ee.FeatureCollection(
                period_features
            )
            .getInfo()
        )

        new_rows = [
            feature[
                "properties"
            ]
            for feature in info[
                "features"
            ]
        ]

        rows = [
            row
            for row in rows
            if (
                row.get(
                    "period_start"
                ),
                row.get(
                    "period_end"
                ),
            )
            != period_key
        ]

        rows.extend(
            new_rows
        )

        checkpoint = pd.DataFrame(
            rows
        )

        checkpoint[
            "station_id"
        ] = (
            checkpoint[
                "station_id"
            ]
            .astype(
                "string"
            )
        )

        checkpoint = (
            checkpoint.sort_values(
                [
                    "period_start",
                    "station_id",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        save_checkpoint(
            checkpoint,
            checkpoint_path,
        )

    observations = load_checkpoint(
        checkpoint_path
    )

    expected_rows = (
        len(periods)
        * station_count
    )

    if len(observations) != expected_rows:
        raise RuntimeError(
            "Incomplete HLS calibration table: expected "
            f"{expected_rows} rows, found {len(observations)}."
        )

    duplicate_count = int(
        observations.duplicated(
            [
                "station_id",
                "period_start",
            ]
        ).sum()
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate station-period rows found in HLS "
            f"calibration table: {duplicate_count}."
        )

    hls_calibration, eligible = (
        calculate_endmembers(
            observations
        )
    )

    candidate_config = (
        build_candidate_config(
            hls_calibration
        )
    )

    candidate_path = (
        OUTPUT_DIRECTORY
        / CANDIDATE_CONFIG_FILENAME
    )

    with candidate_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            candidate_config,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write(
            "\n"
        )

    current_hls = None

    with CURRENT_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        current_config = json.load(
            file
        )

    current_hls = (
        current_config.get(
            "sources",
            {}
        ).get(
            "HLS"
        )
    )

    summary = {
        "calibration_name": (
            "hls_fvc_recalibration_after_mgrs_correction"
        ),
        "calibration_period": {
            "start": START_DATE,
            "end": END_DATE,
        },
        "method": (
            "two_stage_global_percentile"
        ),
        "coverage_threshold_pct": (
            COVERAGE_THRESHOLD_PCT
        ),
        "water_exclusion": (
            "exclude NDWI > 0"
        ),
        "within_footprint_period": {
            "low_candidate_percentile": (
                WITHIN_LOW_PERCENTILE
            ),
            "high_candidate_percentile": (
                WITHIN_HIGH_PERCENTILE
            ),
        },
        "across_all_valid_observations": {
            "low_quantile": (
                GLOBAL_LOW_QUANTILE
            ),
            "high_quantile": (
                GLOBAL_HIGH_QUANTILE
            ),
        },
        "total_station_period_rows": int(
            len(observations)
        ),
        "eligible_station_period_rows": int(
            len(eligible)
        ),
        "eligible_station_count": int(
            eligible[
                "station_id"
            ].nunique()
        ),
        "current_hls_calibration": (
            current_hls
        ),
        "candidate_hls_calibration": (
            hls_calibration
        ),
        "production_config_overwritten": False,
    }

    summary_path = (
        OUTPUT_DIRECTORY
        / SUMMARY_FILENAME
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write(
            "\n"
        )

    print()
    print(
        "HLS FVC RECALIBRATION RESULT"
    )
    print(
        "============================"
    )
    print(
        "Total station-period rows:",
        len(observations),
    )
    print(
        "Eligible rows (coverage >= 80%):",
        len(eligible),
    )
    print(
        "Eligible stations:",
        eligible[
            "station_id"
        ].nunique(),
    )
    print(
        "Candidate low NDVI endmember:",
        round(
            hls_calibration[
                "ndvi_low_endmember"
            ],
            9,
        ),
    )
    print(
        "Candidate high NDVI endmember:",
        round(
            hls_calibration[
                "ndvi_high_endmember"
            ],
            9,
        ),
    )

    if current_hls is not None:
        print()
        print(
            "Current HLS calibration:"
        )
        print(
            json.dumps(
                current_hls,
                indent=2,
            )
        )

    print()
    print(
        "Diagnostic observations:",
        checkpoint_path,
    )
    print(
        "Summary:",
        summary_path,
    )
    print(
        "Candidate config:",
        candidate_path,
    )
    print()
    print(
        "Production config was not changed."
    )


if __name__ == "__main__":
    main()
