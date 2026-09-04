"""Export reusable Sentinel-1 predictors for R077 and R142.

The diagnostic preserves the two relevant acquisition geometries separately:
- R077 ASCENDING
- R142 DESCENDING

One output row is produced for every station x MODIS period.
No model training or production-model modification is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ee
import pandas as pd

from et_downscaling.candidate_paths import get_candidate_study_paths

from et_downscaling.availability_diagnostic import (
    annual_partitions,
    get_dynamic_modis_inputs,
    get_dynamic_s1_collection,
)
from et_downscaling.export import export_feature_collection
from et_downscaling.optical_source_experiment import (
    _base_properties,
    _period_context,
    _period_values,
)
from et_downscaling.sentinel1 import (
    build_s1_median,
    get_s1_coverage,
)


S1_GEOMETRIES = {
    "r077": {
        "pass": "ASCENDING",
        "relative_orbit": 77,
    },
    "r142": {
        "pass": "DESCENDING",
        "relative_orbit": 142,
    },
}

S1_PREDICTOR_BANDS = [
    "VV_dB",
    "VH_dB",
    "VV_minus_VH_dB",
    "Angle_deg",
]

KEY_COLUMNS = [
    "station_id",
    "modis_pixel_id",
    "period_start",
]

BASE_SELECTORS = [
    "station",
    "station_id",
    "modis_pixel_id",
    "period_start",
    "period_end",
    "period_end_exclusive",
    "period_days",
    "footprint_area_m2",
]

GEOMETRY_SELECTORS = []

for prefix in S1_GEOMETRIES:
    GEOMETRY_SELECTORS.extend(
        [
            f"{prefix}_pass",
            f"{prefix}_relative_orbit",
            f"{prefix}_products",
            f"{prefix}_unique_dates",
            f"{prefix}_acquisition_dates",
            f"{prefix}_has_acquisition",
            f"{prefix}_coverage_pct",
            f"{prefix}_has_valid_coverage",
        ]
    )

    GEOMETRY_SELECTORS.extend(
        [
            f"{prefix}_{band}_mean"
            for band in S1_PREDICTOR_BANDS
        ]
    )

EXPORT_SELECTORS = (
    BASE_SELECTORS
    + GEOMETRY_SELECTORS
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Export paired R077/R142 Sentinel-1 predictors "
            "for the 2020-2024 diagnostic experiment."
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
        help=(
            "Execute Earth Engine downloads. "
            "Without this flag only the plan is shown."
        ),
    )

    return parser.parse_args()


def project_root():
    return Path(__file__).resolve().parents[2]


def experiment_root(period_label):
    if period_label != "2020_2024":
        raise ValueError("Only the approved 2020_2024 period is allowed")
    return get_candidate_study_paths(project_root()).s1_root


def mean_properties(
    image,
    geometry,
    prefix,
):
    values = ee.Dictionary(
        ee.Image(image)
        .select(S1_PREDICTOR_BANDS)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=10,
            maxPixels=1_000_000,
            tileScale=4,
        )
    )

    return ee.Dictionary.fromLists(
        [
            f"{prefix}_{band}_mean"
            for band in S1_PREDICTOR_BANDS
        ],
        [
            values.get(
                band,
                -9999,
            )
            for band in S1_PREDICTOR_BANDS
        ],
    )


def build_s1_wide_table(
    modis_inputs,
    s1_collection,
    partition_start,
    partition_end,
):
    (
        images,
        image_indexes,
        footprints,
        footprint_indexes,
    ) = _period_context(
        modis_inputs,
        partition_start,
        partition_end,
    )

    def process_image(image_index):
        (
            period_start,
            period_end,
            period_days,
        ) = _period_values(
            images.get(image_index)
        )

        def process_footprint(footprint_index):
            footprint = ee.Feature(
                footprints.get(
                    footprint_index
                )
            )

            geometry = (
                footprint.geometry()
            )

            properties = _base_properties(
                footprint,
                period_start,
                period_end,
                period_days,
            )

            properties.update(
                {
                    "modis_pixel_id": (
                        footprint.get(
                            "modis_pixel_id"
                        )
                    ),
                    "footprint_area_m2": (
                        geometry.area(
                            maxError=1
                        )
                    ),
                }
            )

            feature = ee.Feature(
                None,
                properties,
            )

            for (
                prefix,
                configuration,
            ) in S1_GEOMETRIES.items():

                period = (
                    ee.ImageCollection(
                        s1_collection
                    )
                    .filterDate(
                        period_start,
                        period_end,
                    )
                    .filterBounds(
                        geometry
                    )
                    .filter(
                        ee.Filter.eq(
                            "orbitProperties_pass",
                            configuration["pass"],
                        )
                    )
                    .filter(
                        ee.Filter.eq(
                            "relativeOrbitNumber_start",
                            configuration[
                                "relative_orbit"
                            ],
                        )
                    )
                )

                dates = (
                    ee.List(
                        period.aggregate_array(
                            "date_key"
                        )
                    )
                    .distinct()
                    .sort()
                )

                median = build_s1_median(
                    period,
                    geometry,
                )

                coverage = (
                    get_s1_coverage(
                        median,
                        geometry,
                    )
                    .multiply(100)
                )

                feature = (
                    feature
                    .set(
                        {
                            f"{prefix}_pass": (
                                configuration[
                                    "pass"
                                ]
                            ),
                            f"{prefix}_relative_orbit": (
                                configuration[
                                    "relative_orbit"
                                ]
                            ),
                            f"{prefix}_products": (
                                period.size()
                            ),
                            f"{prefix}_unique_dates": (
                                dates.size()
                            ),
                            f"{prefix}_acquisition_dates": (
                                dates.join(";")
                            ),
                            f"{prefix}_has_acquisition": (
                                period
                                .size()
                                .gt(0)
                                .int()
                            ),
                            f"{prefix}_coverage_pct": (
                                coverage
                            ),
                            f"{prefix}_has_valid_coverage": (
                                coverage
                                .gt(0)
                                .int()
                            ),
                        }
                    )
                    .set(
                        mean_properties(
                            median,
                            geometry,
                            prefix,
                        )
                    )
                )

            return feature

        return footprint_indexes.map(
            process_footprint
        )

    return ee.FeatureCollection(
        image_indexes
        .map(process_image)
        .flatten()
    )


def merge_chunks(
    chunk_paths,
    output_path,
):
    frames = [
        pd.read_csv(path)
        for path in chunk_paths
    ]

    table = pd.concat(
        frames,
        ignore_index=True,
    )

    table = (
        table
        .sort_values(
            [
                "station_id",
                "period_start",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        output_path,
        index=False,
    )

    return table


def validate_output(
    table,
    period_label,
):
    if table.duplicated(
        KEY_COLUMNS
    ).any():
        raise RuntimeError(
            "Duplicate station-period keys "
            "found in Sentinel-1 output."
        )

    common_path = (
        get_candidate_study_paths(project_root()).optical_root
        / "raw"
        / "paired_optical_common.csv"
    )

    common = pd.read_csv(
        common_path
    )

    key_check = (
        common[
            KEY_COLUMNS
        ]
        .merge(
            table[
                KEY_COLUMNS
            ],
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
            f"S1/common key mismatch: "
            f"{unmatched} unmatched keys."
        )

    summary = {}

    for prefix in S1_GEOMETRIES:

        predictor_columns = [
            f"{prefix}_{band}_mean"
            for band in S1_PREDICTOR_BANDS
        ]

        products = pd.to_numeric(
            table[
                f"{prefix}_products"
            ],
            errors="coerce",
        )

        coverage = pd.to_numeric(
            table[
                f"{prefix}_coverage_pct"
            ],
            errors="coerce",
        )

        predictor_table = (
            table[
                predictor_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        predictor_complete = (
            predictor_table
            .notna()
            .all(axis=1)
            & predictor_table
            .ne(-9999)
            .all(axis=1)
        )

        summary[prefix] = {
            "rows": int(
                len(table)
            ),
            "rows_with_acquisition": int(
                products.gt(0).sum()
            ),
            "rows_with_valid_coverage": int(
                coverage.gt(0).sum()
            ),
            "rows_ge80_coverage": int(
                coverage.ge(80).sum()
            ),
            "rows_ge90_coverage": int(
                coverage.ge(90).sum()
            ),
            "rows_ge99_coverage": int(
                coverage.ge(99).sum()
            ),
            "predictor_complete_rows": int(
                predictor_complete.sum()
            ),
            "mean_coverage_pct": float(
                coverage.mean()
            ),
        }

    return {
        "rows": int(
            len(table)
        ),
        "unmatched_common_keys": (
            unmatched
        ),
        "geometry_summary": (
            summary
        ),
    }


def main():
    args = parse_arguments()

    partitions = annual_partitions(
        args.start_date,
        args.end_date_exclusive,
    )

    print(
        f"S1 geometry plan: "
        f"{len(partitions)} annual downloads"
    )

    print(
        "Geometries:"
    )

    for (
        prefix,
        configuration,
    ) in S1_GEOMETRIES.items():
        print(
            f"  {prefix}: "
            f"{configuration['pass']} "
            f"R{configuration['relative_orbit']:03d}"
        )

    print(
        "Predictors: "
        + ", ".join(
            S1_PREDICTOR_BANDS
        )
    )

    print(
        "S2 download = false"
    )
    print(
        "HLS download = false"
    )
    print(
        "ERA5 download = false"
    )
    print(
        "training_performed = false"
    )

    if not args.execute:
        print(
            "Dry plan only: "
            "Earth Engine was not initialized."
        )
        return

    root = experiment_root(
        args.period_label
    )

    chunks_dir = (
        root
        / "raw"
        / "_chunks"
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

    (
        _,
        _,
        footprints,
        _,
    ) = _period_context(
        modis_inputs,
        args.start_date,
        args.end_date_exclusive,
    )

    s1_collection = (
        get_dynamic_s1_collection(
            ee.FeatureCollection(
                footprints
            ),
            args.start_date,
            args.end_date_exclusive,
        )
    )

    chunk_paths = []

    for (
        partition_start,
        partition_end,
    ) in partitions:

        output_path = (
            chunks_dir
            / (
                f"s1_geometries_"
                f"{partition_start}_"
                f"{partition_end}.csv"
            )
        )

        if output_path.exists():
            print(
                f"Reusing: "
                f"{output_path.name}"
            )

            chunk_paths.append(
                output_path
            )

            continue

        print(
            f"Exporting: "
            f"{partition_start} -> "
            f"{partition_end}"
        )

        table = build_s1_wide_table(
            modis_inputs,
            s1_collection,
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
        / "s1_geometry_predictors.csv"
    )

    table = merge_chunks(
        chunk_paths,
        final_output,
    )

    validation = validate_output(
        table,
        args.period_label,
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
        "start_date": (
            args.start_date
        ),
        "end_date_exclusive": (
            args.end_date_exclusive
        ),
        "period_label": (
            args.period_label
        ),
        "earth_engine_project": (
            args.project
        ),
        "geometries": (
            S1_GEOMETRIES
        ),
        "predictor_support_m": 10,
        "predictors": (
            S1_PREDICTOR_BANDS
        ),
        "validation": (
            validation
        ),
        "s2_download_performed": False,
        "hls_download_performed": False,
        "era5_download_performed": False,
        "training_performed": False,
        "aoa_di_performed": False,
    }

    with (
        metadata_dir
        / "s1_geometry_extraction_manifest.json"
    ).open(
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
        "S1 rows:",
        validation["rows"],
    )

    print(
        "Unmatched common keys:",
        validation[
            "unmatched_common_keys"
        ],
    )

    for (
        prefix,
        summary,
    ) in validation[
        "geometry_summary"
    ].items():

        print()
        print(
            prefix.upper()
        )

        for (
            key,
            value,
        ) in summary.items():
            print(
                f"  {key}: {value}"
            )

    print()
    print(
        "Output:",
        final_output,
    )

    print(
        "training_performed = false"
    )


if __name__ == "__main__":
    main()
