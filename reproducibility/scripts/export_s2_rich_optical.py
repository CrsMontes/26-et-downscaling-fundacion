"""Export the rich Sentinel-2 optical predictor table for the Phase 3 experiment.

This diagnostic reuses the existing Sentinel-2 preprocessing, medoid,
predictor definitions, MODIS footprints, and temporal periods.

No model training is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee
import numpy as np
import pandas as pd

from et_downscaling.candidate_paths import get_candidate_study_paths

from et_downscaling.availability_diagnostic import (
    annual_partitions,
    get_dynamic_modis_inputs,
    get_dynamic_s2_collection,
)
from et_downscaling.export import export_feature_collection
from et_downscaling.optical import build_optical_predictors
from et_downscaling.optical_source_experiment import (
    _base_properties,
    _period_context,
    _period_values,
)
from et_downscaling.schema import get_optical_extraction_bands


KEY_COLUMNS = [
    "station_id",
    "modis_pixel_id",
    "period_start",
]

COMMON_PREDICTORS = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
    "NDVI",
    "EVI",
    "SAVI",
    "NDWI",
    "NDMI",
]

RICH_S2_PREDICTORS = get_optical_extraction_bands("S2")

BASE_SELECTORS = [
    "station",
    "station_id",
    "modis_pixel_id",
    "period_start",
    "period_end",
    "period_end_exclusive",
    "period_days",
    "footprint_area_m2",
    "s2_products",
    "s2_unique_dates",
]

EXPORT_SELECTORS = BASE_SELECTORS + [
    f"s2_{name}_mean"
    for name in RICH_S2_PREDICTORS
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Export the rich Sentinel-2 predictor table without "
            "HLS, Sentinel-1, ERA5, or model training."
        )
    )
    parser.add_argument(
        "--start-date",
        required=True,
    )
    parser.add_argument(
        "--end-date-exclusive",
        required=True,
    )
    parser.add_argument(
        "--period-label",
        required=True,
    )
    parser.add_argument(
        "--project",
        required=True,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute Earth Engine downloads. Otherwise show the plan only.",
    )
    return parser.parse_args()


def project_root():
    return Path(__file__).resolve().parents[2]


def experiment_root(period_label):
    if period_label != "2020_2024":
        raise ValueError("Only the approved 2020_2024 period is allowed")
    return get_candidate_study_paths(project_root()).optical_root


def mean_properties(image, geometry):
    defaults = ee.Dictionary.fromLists(
        [
            f"s2_{name}_mean"
            for name in RICH_S2_PREDICTORS
        ],
        ee.List.repeat(
            -9999,
            len(RICH_S2_PREDICTORS),
        ),
    )

    values = ee.Dictionary(
        ee.Image(image)
        .select(RICH_S2_PREDICTORS)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=20,
            maxPixels=1_000_000,
            tileScale=4,
        )
    )

    renamed = ee.Dictionary.fromLists(
        [
            f"s2_{name}_mean"
            for name in RICH_S2_PREDICTORS
        ],
        [
            values.get(name, -9999)
            for name in RICH_S2_PREDICTORS
        ],
    )

    return defaults.combine(
        renamed,
        True,
    )


def build_s2_rich_table(
    modis_inputs,
    s2_collection,
    partition_start,
    partition_end,
):
    images, image_indexes, footprints, footprint_indexes = (
        _period_context(
            modis_inputs,
            partition_start,
            partition_end,
        )
    )

    def process_image(image_index):
        period_start, period_end, period_days = (
            _period_values(
                images.get(image_index)
            )
        )

        def process_footprint(footprint_index):
            footprint = ee.Feature(
                footprints.get(footprint_index)
            )
            geometry = footprint.geometry()

            s2_period = (
                ee.ImageCollection(s2_collection)
                .filterDate(
                    period_start,
                    period_end,
                )
                .filterBounds(
                    geometry
                )
            )

            predictors = build_optical_predictors(
                s2_period,
                geometry,
                "S2",
            )

            properties = _base_properties(
                footprint,
                period_start,
                period_end,
                period_days,
            )

            properties.update(
                {
                    "modis_pixel_id": footprint.get(
                        "modis_pixel_id"
                    ),
                    "footprint_area_m2": geometry.area(
                        maxError=1
                    ),
                    "s2_products": s2_period.size(),
                    "s2_unique_dates": (
                        ee.List(
                            s2_period.aggregate_array(
                                "date_key"
                            )
                        )
                        .distinct()
                        .size()
                    ),
                }
            )

            return (
                ee.Feature(
                    None,
                    properties,
                )
                .set(
                    mean_properties(
                        predictors,
                        geometry,
                    )
                )
            )

        return footprint_indexes.map(
            process_footprint
        )

    return ee.FeatureCollection(
        image_indexes
        .map(process_image)
        .flatten()
    )


def merge_chunks(paths, output_path):
    frames = [
        pd.read_csv(path)
        for path in paths
    ]

    table = pd.concat(
        frames,
        ignore_index=True,
    )

    table = table.sort_values(
        [
            "station_id",
            "period_start",
        ]
    ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        output_path,
        index=False,
    )

    return table


def validate_against_common(
    rich_table,
    common_path,
):
    if not common_path.exists():
        raise FileNotFoundError(
            f"Common optical table not found: {common_path}"
        )

    common = pd.read_csv(
        common_path
    )

    if common.duplicated(
        KEY_COLUMNS
    ).any():
        raise RuntimeError(
            "Duplicate keys found in paired optical common table."
        )

    if rich_table.duplicated(
        KEY_COLUMNS
    ).any():
        raise RuntimeError(
            "Duplicate keys found in rich S2 table."
        )

    key_check = (
        common[KEY_COLUMNS]
        .merge(
            rich_table[KEY_COLUMNS],
            on=KEY_COLUMNS,
            how="outer",
            indicator=True,
        )
    )

    unmatched = int(
        (
            key_check["_merge"]
            != "both"
        ).sum()
    )

    if unmatched:
        raise RuntimeError(
            f"Rich S2 and common tables differ by {unmatched} keys."
        )

    predictor_columns = [
        f"s2_{name}_mean"
        for name in COMMON_PREDICTORS
    ]

    comparison = (
        common[
            KEY_COLUMNS
            + predictor_columns
        ]
        .merge(
            rich_table[
                KEY_COLUMNS
                + predictor_columns
            ],
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
            suffixes=(
                "_common",
                "_rich",
            ),
        )
    )

    parity = {}

    for column in predictor_columns:
        common_values = pd.to_numeric(
            comparison[
                f"{column}_common"
            ],
            errors="coerce",
        )
        rich_values = pd.to_numeric(
            comparison[
                f"{column}_rich"
            ],
            errors="coerce",
        )

        valid = (
            common_values.notna()
            & rich_values.notna()
            & (common_values != -9999)
            & (rich_values != -9999)
        )

        if valid.any():
            max_abs_difference = float(
                np.max(
                    np.abs(
                        common_values[valid]
                        - rich_values[valid]
                    )
                )
            )
        else:
            max_abs_difference = None

        parity[column] = (
            max_abs_difference
        )

    finite_differences = [
        value
        for value in parity.values()
        if value is not None
    ]

    overall_max = (
        max(finite_differences)
        if finite_differences
        else None
    )

    return {
        "common_rows": int(len(common)),
        "rich_rows": int(len(rich_table)),
        "unmatched_keys": unmatched,
        "common_predictor_max_abs_difference": parity,
        "overall_max_abs_difference": overall_max,
    }


def main():
    args = parse_arguments()

    partitions = annual_partitions(
        args.start_date,
        args.end_date_exclusive,
    )

    print(
        f"Phase 3A S2-rich plan: "
        f"{len(partitions)} annual S2 downloads"
    )
    print(
        "Optical predictors:",
        len(RICH_S2_PREDICTORS),
    )
    print(
        ", ".join(
            RICH_S2_PREDICTORS
        )
    )
    print(
        "HLS download = false"
    )
    print(
        "ERA5 download = false"
    )
    print(
        "Sentinel-1 download = false"
    )
    print(
        "training_performed = false"
    )

    if not args.execute:
        print(
            "Dry plan only: Earth Engine was not initialized."
        )
        return

    root = experiment_root(
        args.period_label
    )

    chunks_dir = (
        root
        / "raw"
        / "_chunks"
        / "s2_rich"
    )

    chunks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ee.Initialize(
        project=args.project
    )

    modis_inputs = (
        get_dynamic_modis_inputs(
            args.start_date,
            args.end_date_exclusive,
        )
    )

    _, _, footprints, _ = (
        _period_context(
            modis_inputs,
            args.start_date,
            args.end_date_exclusive,
        )
    )

    s2_collection = (
        get_dynamic_s2_collection(
            ee.FeatureCollection(
                footprints
            ),
            args.start_date,
            args.end_date_exclusive,
        )
    )

    chunk_paths = []

    for partition_start, partition_end in partitions:
        output_path = (
            chunks_dir
            / (
                f"s2_rich_"
                f"{partition_start}_"
                f"{partition_end}.csv"
            )
        )

        if output_path.exists():
            print(
                f"Reusing: {output_path.name}"
            )
            chunk_paths.append(
                output_path
            )
            continue

        print(
            f"Exporting: "
            f"{partition_start} -> {partition_end}"
        )

        table = build_s2_rich_table(
            modis_inputs,
            s2_collection,
            partition_start,
            partition_end,
        )

        export_feature_collection(
            table,
            str(output_path),
            EXPORT_SELECTORS,
        )

        chunk_paths.append(
            output_path
        )

    final_output = (
        root
        / "raw"
        / "s2_rich_optical.csv"
    )

    rich_table = merge_chunks(
        chunk_paths,
        final_output,
    )

    common_path = (
        root
        / "raw"
        / "paired_optical_common.csv"
    )

    validation = (
        validate_against_common(
            rich_table,
            common_path,
        )
    )

    metadata_dir = (
        root
        / "metadata"
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "start_date": args.start_date,
        "end_date_exclusive": (
            args.end_date_exclusive
        ),
        "period_label": (
            args.period_label
        ),
        "earth_engine_project": (
            args.project
        ),
        "source": "S2",
        "predictor_support_m": 20,
        "predictors": (
            RICH_S2_PREDICTORS
        ),
        "downloads_expected": (
            len(partitions)
        ),
        "rows": int(
            len(rich_table)
        ),
        "validation_against_common": (
            validation
        ),
        "hls_download_performed": False,
        "era5_download_performed": False,
        "sentinel1_download_performed": False,
        "training_performed": False,
        "aoa_di_performed": False,
    }

    manifest_path = (
        metadata_dir
        / "s2_rich_extraction_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print()
    print(
        "S2 rich rows:",
        len(rich_table),
    )
    print(
        "Unmatched common keys:",
        validation[
            "unmatched_keys"
        ],
    )
    print(
        "Common predictor max abs difference:",
        validation[
            "overall_max_abs_difference"
        ],
    )
    print(
        "Output:",
        final_output,
    )
    print(
        "training_performed = false"
    )


if __name__ == "__main__":
    main()
