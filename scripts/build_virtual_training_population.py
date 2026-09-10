"""Build the final Virtual10 RF-25 training population from source data.

The script reads the selected Virtual10 MODIS supports, rebuilds all required
2020-2024 satellite and meteorological source tables, constructs the local
training master, applies the final GE90/25-predictor eligibility contract, and
writes the RF-25 training population. It does not fit a model.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve
from http.client import RemoteDisconnected

import numpy as np
import pandas as pd

from et_downscaling.virtual_station import resolve_virtual_workspace


DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE_EXCLUSIVE = "2025-01-01"
TRAINING_DESIGN_NAME = "virtual10_random_ge90_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final Virtual10 RF-25 training population."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Google Cloud project ID with Earth Engine access.",
    )
    parser.add_argument(
        "--selection-dir",
        default=None,
        help=(
            "Directory containing virtual_modis_footprints.geojson. Defaults to outputs/training/selection/."
        ),
    )
    parser.add_argument(
        "--workspace-root",
        default=None,
        help=(
            "Generated-output root. Defaults to repository-local outputs/."
        ),
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
    )
    parser.add_argument(
        "--end-date-exclusive",
        default=DEFAULT_END_DATE_EXCLUSIVE,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild completed extraction checkpoints.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_workspace_root(value: str | None) -> Path:
    return resolve_virtual_workspace(value, project_root())



def resolve_selection_dir(
    workspace_root: Path,
    value: str | None,
) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return (
        workspace_root
        / "training"
        / "selection"
    )



def load_geojson_as_ee_feature_collection(path: Path, ee):
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = []

    for item in payload.get("features", []):
        geometry = item.get("geometry")
        properties = dict(item.get("properties", {}))
        if geometry is None:
            continue
        features.append(
            ee.Feature(
                ee.Geometry(geometry),
                properties,
            )
        )

    if not features:
        raise ValueError(f"No features found in {path}")

    return ee.FeatureCollection(features)


def download_feature_collection_csv(
    ee,
    feature_collection,
    output_path: Path,
    selectors,
    max_attempts: int = 5,
    retry_sleep_seconds: int = 10,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    for attempt in range(1, max_attempts + 1):
        try:
            temporary_path.unlink(missing_ok=True)
            url = ee.FeatureCollection(
                feature_collection
            ).getDownloadURL(
                filetype="CSV",
                selectors=list(selectors),
                filename=output_path.stem,
            )
            urlretrieve(url, temporary_path)
            temporary_path.replace(output_path)
            return output_path

        except HTTPError as error:
            temporary_path.unlink(missing_ok=True)
            retryable = error.code in {500, 502, 503, 504}
            if not retryable or attempt == max_attempts:
                raise
            print(f"Earth Engine returned HTTP {error.code}; retrying...")

        except (
            URLError,
            RemoteDisconnected,
            ConnectionResetError,
            TimeoutError,
        ) as error:
            temporary_path.unlink(missing_ok=True)
            if attempt == max_attempts:
                raise
            print("Network error:", error)
            print("Retrying...")

        time.sleep(retry_sleep_seconds)

    raise RuntimeError("Earth Engine CSV download failed unexpectedly.")


def merge_csv_chunks(
    chunk_paths: list[Path],
    output_path: Path,
) -> int:
    header = None
    total_rows = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = None

        for chunk_path in chunk_paths:
            with chunk_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as input_file:
                reader = csv.DictReader(input_file)

                if reader.fieldnames is None:
                    continue

                if header is None:
                    header = reader.fieldnames
                    writer = csv.DictWriter(
                        output_file,
                        fieldnames=header,
                    )
                    writer.writeheader()
                elif reader.fieldnames != header:
                    raise ValueError(
                        f"CSV schema mismatch in {chunk_path}"
                    )

                for row in reader:
                    writer.writerow(row)
                    total_rows += 1

    return total_rows


def get_quarter_ranges(
    year: int,
    analysis_start: date,
    analysis_end: date,
):
    bounds = [
        ("Q1", date(year, 1, 1), date(year, 4, 1)),
        ("Q2", date(year, 4, 1), date(year, 7, 1)),
        ("Q3", date(year, 7, 1), date(year, 10, 1)),
        ("Q4", date(year, 10, 1), date(year + 1, 1, 1)),
    ]

    result = []
    for name, start, end in bounds:
        effective_start = max(start, analysis_start)
        effective_end = min(end, analysis_end)
        if effective_start < effective_end:
            result.append(
                (
                    name,
                    effective_start.isoformat(),
                    effective_end.isoformat(),
                )
            )
    return result


def get_era5_utc_windows(
    analysis_start: date,
    analysis_end: date,
):
    raw_start = datetime.combine(
        analysis_start,
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    raw_end = datetime.combine(
        analysis_end,
        datetime.min.time(),
        tzinfo=timezone.utc,
    ) + timedelta(hours=5)

    windows = []
    year = raw_start.year

    while datetime(year, 1, 1, tzinfo=timezone.utc) < raw_end:
        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        effective_start = max(raw_start, year_start)
        effective_end = min(raw_end, year_end)

        if effective_start < effective_end:
            windows.append(
                (
                    year,
                    effective_start.isoformat().replace("+00:00", "Z"),
                    effective_end.isoformat().replace("+00:00", "Z"),
                    int(
                        (
                            effective_end - effective_start
                        ).total_seconds()
                        // 3600
                    ),
                )
            )
        year += 1

    return windows


def calculate_metrics(
    observed: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    residual = predicted_array - observed_array

    r2 = float(r2_score(observed_array, predicted_array))
    rmse = float(
        np.sqrt(
            mean_squared_error(
                observed_array,
                predicted_array,
            )
        )
    )
    mae = float(
        mean_absolute_error(
            observed_array,
            predicted_array,
        )
    )
    bias = float(np.mean(residual))

    if (
        np.std(observed_array, ddof=0) == 0
        or np.mean(observed_array) == 0
    ):
        kge = float("nan")
    else:
        correlation = float(
            np.corrcoef(
                observed_array,
                predicted_array,
            )[0, 1]
        )
        alpha = float(
            np.std(predicted_array, ddof=0)
            / np.std(observed_array, ddof=0)
        )
        beta = float(
            np.mean(predicted_array)
            / np.mean(observed_array)
        )
        kge = float(
            1
            - np.sqrt(
                (correlation - 1) ** 2
                + (alpha - 1) ** 2
                + (beta - 1) ** 2
            )
        )

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "BIAS": bias,
        "KGE": kge,
    }



def load_frozen_selection(
    selection_dir: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load the frozen Virtual10 support selection and exact GE90 period whitelist."""
    selected_path = (
        selection_dir
        / "selected_supports.csv"
    )
    check_root = (
        selection_dir
        / "availability_checks"
    )

    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    if not check_root.is_dir():
        raise FileNotFoundError(check_root)

    selected = pd.read_csv(
        selected_path,
        dtype={"virtual_id": str},
    ).sort_values("virtual_id").reset_index(drop=True)

    if len(selected) != 10:
        raise RuntimeError(
            f"Frozen Virtual10 selection contains {len(selected)} supports; expected 10."
        )
    if selected["virtual_id"].nunique() != 10:
        raise RuntimeError(
            "Frozen Virtual10 selection contains duplicate virtual IDs."
        )
    if selected["modis_pixel_id"].nunique() != 10:
        raise RuntimeError(
            "Frozen Virtual10 selection contains duplicate MODIS pixel IDs."
        )
    if selected["spatial_block_utm10km"].nunique() != 10:
        raise RuntimeError(
            "Frozen Virtual10 selection does not contain 10 distinct fixed UTM blocks."
        )

    period_whitelist: dict[str, list[str]] = {}

    for row in selected.itertuples(index=False):
        virtual_id = str(row.virtual_id)
        candidate_order = int(row.candidate_order)
        modis_pixel_id = int(row.modis_pixel_id)

        check_path = (
            check_root
            / f"{candidate_order:05d}_{modis_pixel_id}.csv"
        )
        if not check_path.is_file():
            raise FileNotFoundError(check_path)

        check = pd.read_csv(check_path)
        check["period_start"] = pd.to_datetime(
            check["period_start"],
            errors="raise",
        ).dt.strftime("%Y-%m-%d")

        ge90 = pd.to_numeric(
            check["ge90"],
            errors="raise",
        ).astype(int)

        periods = (
            check.loc[
                ge90.eq(1),
                "period_start",
            ]
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        if len(periods) != int(row.ge90_total):
            raise RuntimeError(
                f"{virtual_id}: frozen GE90 check has {len(periods)} periods "
                f"but selected_supports.csv records {int(row.ge90_total)}."
            )

        by_year = (
            pd.Series(
                pd.to_datetime(periods)
            )
            .dt.year
            .value_counts()
            .to_dict()
        )
        for year in range(2020, 2025):
            expected = int(
                getattr(
                    row,
                    f"ge90_{year}",
                )
            )
            actual = int(
                by_year.get(year, 0)
            )
            if actual != expected:
                raise RuntimeError(
                    f"{virtual_id}: frozen {year} GE90 mismatch: "
                    f"check={actual}, selected table={expected}."
                )

        period_whitelist[virtual_id] = periods

    return selected, period_whitelist


def build_canonical_virtual_footprints(
    *,
    ee,
    selected: pd.DataFrame,
    modis_collection,
    modis_projection,
    modis_scale,
):
    """Rebuild the selected supports from their frozen centroid points.

    The selected_supports.csv table is the canonical sampling record. Native
    MODIS polygons are reconstructed server-side from those centroid points,
    eliminating the GeoJSON polygon round-trip that caused the failed QA.
    """
    from et_downscaling.modis import (
        assign_station_footprints,
        build_modis_grid,
        build_modis_pixel_id,
    )

    point_features = []
    for row in selected.itertuples(index=False):
        properties = {
            "station_id": str(row.virtual_id),
            "station": str(row.virtual_id),
            "virtual_site": 1,
            "selected_modis_pixel_id": int(row.modis_pixel_id),
            "candidate_order": int(row.candidate_order),
            "spatial_block": str(row.spatial_block_utm10km),
            "spatial_block_utm10km": str(row.spatial_block_utm10km),
            "longitude": float(row.longitude),
            "latitude": float(row.latitude),
        }
        point_features.append(
            ee.Feature(
                ee.Geometry.Point(
                    [
                        float(row.longitude),
                        float(row.latitude),
                    ]
                ),
                properties,
            )
        )

    points = ee.FeatureCollection(
        point_features
    )

    pixel_id = build_modis_pixel_id(
        modis_projection
    )
    grid = build_modis_grid(
        points,
        pixel_id,
        modis_projection,
        modis_scale,
    )
    assigned = assign_station_footprints(
        points,
        grid,
        pixel_id,
        modis_projection,
        modis_scale,
    )

    def enrich(feature):
        feature = ee.Feature(feature)
        station_id = feature.get("station_id")
        source = ee.Feature(
            points.filter(
                ee.Filter.eq(
                    "station_id",
                    station_id,
                )
            ).first()
        )
        return (
            feature
            .copyProperties(source)
            .set(
                "modis_pixel_id",
                feature.get(
                    "modis_pixel_id"
                ),
            )
            .set(
                "footprint_area_m2",
                feature.get(
                    "footprint_area_m2"
                ),
            )
        )

    footprints = ee.FeatureCollection(
        assigned.map(enrich)
    )

    info = footprints.getInfo()
    actual_by_id = {
        str(feature["properties"]["station_id"]): int(
            feature["properties"]["modis_pixel_id"]
        )
        for feature in info.get("features", [])
    }
    expected_by_id = {
        str(row.virtual_id): int(row.modis_pixel_id)
        for row in selected.itertuples(index=False)
    }

    if actual_by_id != expected_by_id:
        rows = []
        for virtual_id in sorted(expected_by_id):
            rows.append(
                f"{virtual_id}: expected={expected_by_id[virtual_id]} "
                f"assigned={actual_by_id.get(virtual_id)}"
            )
        raise RuntimeError(
            "Canonical MODIS reconstruction does not reproduce the frozen "
            "selection:\n"
            + "\n".join(rows)
        )

    return footprints, info


def write_canonical_footprint_geojson(
    *,
    ee,
    footprints,
    output_path: Path,
) -> None:
    """Write a provenance-only WGS84 copy of the reconstructed footprints."""
    def to_wgs84(feature):
        feature = ee.Feature(feature)
        return ee.Feature(
            feature.geometry().transform(
                "EPSG:4326",
                1,
            ),
            feature.toDictionary(),
        )

    payload = (
        ee.FeatureCollection(
            footprints.map(to_wgs84)
        )
        .getInfo()
    )
    output_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": payload.get(
                    "features",
                    [],
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def assert_satellite_matches_frozen_selection(
    *,
    satellite: pd.DataFrame,
    selected: pd.DataFrame,
    frozen_periods: dict[str, list[str]],
) -> None:
    """Hard QA gate: training extraction must reproduce frozen support-periods."""
    data = satellite.copy()
    data["station_id"] = (
        data["station_id"]
        .astype(str)
    )
    data["period_start"] = pd.to_datetime(
        data["period_start"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    duplicate = data.duplicated(
        ["station_id", "period_start"]
    )
    if duplicate.any():
        raise RuntimeError(
            "Canonical satellite extraction contains duplicate station-period keys."
        )

    expected_pairs = {
        (virtual_id, period)
        for virtual_id, periods in frozen_periods.items()
        for period in periods
    }
    actual_pairs = set(
        zip(
            data["station_id"],
            data["period_start"],
            strict=True,
        )
    )

    if actual_pairs != expected_pairs:
        only_expected = sorted(
            expected_pairs - actual_pairs
        )
        only_actual = sorted(
            actual_pairs - expected_pairs
        )
        raise RuntimeError(
            "Canonical satellite extraction does not reproduce the frozen GE90 "
            "period whitelist. "
            f"Missing={len(only_expected)}, unexpected={len(only_actual)}. "
            f"First missing={only_expected[:10]}, first unexpected={only_actual[:10]}"
        )

    expected_modis = {
        str(row.virtual_id): int(row.modis_pixel_id)
        for row in selected.itertuples(index=False)
    }

    modis_values = pd.to_numeric(
        data["modis_pixel_id"],
        errors="raise",
    ).astype("int64")
    data = data.assign(
        _modis_pixel_id_int=modis_values
    )

    mismatches = []
    for virtual_id, expected_id in expected_modis.items():
        actual_ids = sorted(
            data.loc[
                data["station_id"].eq(
                    virtual_id
                ),
                "_modis_pixel_id_int",
            ]
            .unique()
            .tolist()
        )
        if actual_ids != [expected_id]:
            mismatches.append(
                f"{virtual_id}: expected={expected_id}, actual={actual_ids}"
            )

    if mismatches:
        raise RuntimeError(
            "Canonical satellite extraction uses wrong MODIS support IDs:\n"
            + "\n".join(mismatches)
        )

    if "optical_union_coverage_pct" in data.columns:
        coverage = pd.to_numeric(
            data["optical_union_coverage_pct"],
            errors="coerce",
        )
        bad = data.loc[
            coverage.lt(89.999999),
            [
                "station_id",
                "period_start",
                "optical_union_coverage_pct",
            ],
        ]
        if not bad.empty:
            raise RuntimeError(
                "Frozen GE90 period(s) recomputed below 90% after canonical "
                "footprint reconstruction. First rows:\n"
                + bad.head(20).to_string(index=False)
            )

def main() -> None:
    args = parse_args()

    os.environ["ET_START_DATE"] = args.start_date
    os.environ[
        "ET_END_DATE_EXCLUSIVE"
    ] = args.end_date_exclusive

    # Period-sensitive project imports occur after environment configuration.
    import ee

    from et_downscaling.config import (
        OUTPUT_PERIOD_LABEL,
        build_satellite_output_filename,
        build_training_output_filename,
    )
    from et_downscaling.dataset import (
        build_availability_table,
        build_observations_with_stats,
        build_output_table,
        get_extraction_observations,
    )
    from et_downscaling.local_training import (
        build_training_master,
    )
    from et_downscaling.meteorology_export import (
        ERA5_EXPORT_SELECTORS,
        STATION_SUPPORT_SELECTORS,
        build_era5_hourly_table,
        build_era5_station_supports,
        build_station_support_table,
        get_station_support,
    )
    from et_downscaling.modis import (
        get_modis_collection,
        get_modis_projection,
        get_modis_scale,
    )
    from et_downscaling.training import prepare_rf25_population
    from et_downscaling.optical import (
        get_optical_collection,
    )
    from et_downscaling.schema import (
        get_satellite_export_selectors,
    )

    root = project_root()
    workspace_root = resolve_workspace_root(args.workspace_root)
    current_root = workspace_root / "current"
    selection_dir = resolve_selection_dir(
        workspace_root,
        args.selection_dir,
    )

    training_root = (
        workspace_root
        / "training"
    )
    raw_root = training_root / "raw"
    satellite_root = raw_root / "satellite" / "S2"
    meteorology_root = raw_root / "meteorology"
    master_root = training_root / "master"
    results_root = workspace_root / "evaluation" / "results"

    for path in (
        satellite_root,
        meteorology_root,
        master_root,
        results_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    forbidden = {
        current_root.resolve(),
        (workspace_root / "final").resolve(),
    }
    for path in (
        satellite_root,
        meteorology_root,
        master_root,
        results_root,
    ):
        resolved = path.resolve()
        if any(
            resolved == item
            or item in resolved.parents
            for item in forbidden
        ):
            raise RuntimeError(
                f"Experimental path resolves inside stable outputs: {resolved}"
            )

    selected_supports, frozen_ge90_periods = (
        load_frozen_selection(
            selection_dir
        )
    )

    print("=" * 88)
    print("ET FUNDACION - VIRTUAL10 RF-25 TRAINING DATA")
    print("=" * 88)
    print("Analysis period:", args.start_date, "to", args.end_date_exclusive)
    print("Training root:", training_root)
    print("Frozen selection source: selected_supports.csv + availability_checks/")
    print("MODIS polygons reconstructed canonically from frozen centroid points: YES")
    print("Google Drive used: NO")
    print()

    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()

    modis_collection = get_modis_collection()
    modis_projection = get_modis_projection(
        modis_collection
    )
    modis_scale = get_modis_scale(
        modis_projection
    )

    (
        virtual_footprints,
        virtual_info,
    ) = build_canonical_virtual_footprints(
        ee=ee,
        selected=selected_supports,
        modis_collection=modis_collection,
        modis_projection=modis_projection,
        modis_scale=modis_scale,
    )

    write_canonical_footprint_geojson(
        ee=ee,
        footprints=virtual_footprints,
        output_path=(
            selection_dir
            / "virtual_modis_footprints_canonical.geojson"
        ),
    )

    virtual_ids = (
        selected_supports[
            "virtual_id"
        ]
        .astype(str)
        .sort_values()
        .tolist()
    )
    modis_ids = (
        selected_supports[
            "modis_pixel_id"
        ]
        .astype("int64")
        .tolist()
    )

    print("Virtual supports:", ", ".join(virtual_ids))
    print("Unique frozen MODIS footprints: 10")
    print(
        "Frozen GE90 periods:",
        sum(
            len(periods)
            for periods in frozen_ge90_periods.values()
        ),
    )
    print()

    analysis_start = date.fromisoformat(
        args.start_date
    )
    analysis_end = date.fromisoformat(
        args.end_date_exclusive
    )
    years = list(
        range(
            analysis_start.year,
            (analysis_end - timedelta(days=1)).year + 1,
        )
    )

    # ------------------------------------------------------------------
    # 1. Satellite / MODIS footprint extraction
    # ------------------------------------------------------------------
    satellite_final = (
        satellite_root
        / build_satellite_output_filename("S2")
    )
    satellite_chunk_root = (
        satellite_root
        / "_chunks"
        / OUTPUT_PERIOD_LABEL
    )

    if satellite_final.is_file() and not args.force:
        print("Using existing virtual satellite master:", satellite_final)
        satellite_check = pd.read_csv(
            satellite_final,
            dtype={"station_id": str},
        )
        assert_satellite_matches_frozen_selection(
            satellite=satellite_check,
            selected=selected_supports,
            frozen_periods=frozen_ge90_periods,
        )
        print(
            "Frozen-selection / existing-extraction QA: PASS"
        )
    else:
        print("=== VIRTUAL SATELLITE EXTRACTION ===")

        if args.force and satellite_chunk_root.exists():
            shutil.rmtree(satellite_chunk_root)
        satellite_chunk_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        base_modis_inputs = {
            "collection": modis_collection,
            "projection": modis_projection,
            "scale": modis_scale,
            "station_footprints": virtual_footprints,
        }

        optical_collection = get_optical_collection(
            virtual_footprints,
            "S2",
        )
        export_selectors = (
            get_satellite_export_selectors("S2")
        )

        chunk_paths: list[Path] = []

        for station_index, station_id in enumerate(
            virtual_ids,
            start=1,
        ):
            for year in years:
                for (
                    quarter_name,
                    quarter_start,
                    quarter_end,
                ) in get_quarter_ranges(
                    year,
                    analysis_start,
                    analysis_end,
                ):
                    filename = (
                        f"station_{station_index:02d}_"
                        f"{year}_{quarter_name}.csv"
                    )
                    chunk_path = (
                        satellite_chunk_root
                        / filename
                    )

                    if chunk_path.exists() and not args.force:
                        print("Using satellite checkpoint:", chunk_path.name)
                        chunk_paths.append(chunk_path)
                        continue

                    print(
                        "Satellite |",
                        station_id,
                        "|",
                        year,
                        quarter_name,
                    )

                    station_footprint = (
                        virtual_footprints.filter(
                            ee.Filter.eq(
                                "station_id",
                                station_id,
                            )
                        )
                    )
                    partition_inputs = dict(
                        base_modis_inputs
                    )
                    partition_inputs[
                        "collection"
                    ] = modis_collection.filterDate(
                        quarter_start,
                        quarter_end,
                    )
                    partition_inputs[
                        "station_footprints"
                    ] = station_footprint

                    availability = build_availability_table(
                        modis_inputs=partition_inputs,
                        optical_collection=optical_collection,
                        s1_collection=None,
                        optical_source="S2",
                    )
                    observations = (
                        get_extraction_observations(
                            availability
                        )
                        .filter(
                            ee.Filter.inList(
                                "period_start",
                                frozen_ge90_periods[
                                    station_id
                                ],
                            )
                        )
                    )
                    with_stats = build_observations_with_stats(
                        valid_observations=observations,
                        optical_collection=optical_collection,
                        s1_collection=None,
                        optical_source="S2",
                    )
                    output = build_output_table(
                        with_stats
                    )

                    download_feature_collection_csv(
                        ee=ee,
                        feature_collection=output["all"],
                        output_path=chunk_path,
                        selectors=export_selectors,
                    )
                    chunk_paths.append(chunk_path)

        if not chunk_paths:
            raise RuntimeError(
                "No virtual satellite partitions were created."
            )

        satellite_rows = merge_csv_chunks(
            chunk_paths,
            satellite_final,
        )

        satellite_check = pd.read_csv(
            satellite_final,
            dtype={"station_id": str},
        )
        assert_satellite_matches_frozen_selection(
            satellite=satellite_check,
            selected=selected_supports,
            frozen_periods=frozen_ge90_periods,
        )

        print("Virtual satellite rows:", satellite_rows)
        print(
            "Frozen-selection / extracted-period QA: PASS"
        )
        print("Saved:", satellite_final)

    # ------------------------------------------------------------------
    # 2. ERA5-Land support and hourly extraction
    # ------------------------------------------------------------------
    support_path = (
        meteorology_root
        / "station_support.csv"
    )
    era5_final = (
        meteorology_root
        / f"era5_hourly_{OUTPUT_PERIOD_LABEL}.csv"
    )
    era5_chunk_root = (
        meteorology_root
        / "_chunks"
        / OUTPUT_PERIOD_LABEL
        / "era5"
    )

    print()
    print("=== VIRTUAL METEOROLOGY EXTRACTION ===")

    era5_supports = build_era5_station_supports(
        virtual_footprints
    )
    support_table = build_station_support_table(
        virtual_footprints,
        era5_supports,
    )

    if args.force or not support_path.is_file():
        download_feature_collection_csv(
            ee=ee,
            feature_collection=support_table,
            output_path=support_path,
            selectors=STATION_SUPPORT_SELECTORS,
        )
    else:
        print("Using existing station support:", support_path)

    if era5_final.is_file() and not args.force:
        print("Using existing ERA5 hourly master:", era5_final)
    else:
        if args.force and era5_chunk_root.exists():
            shutil.rmtree(era5_chunk_root)
        era5_chunk_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        era5_chunks: list[Path] = []

        for station_index, station_id in enumerate(
            virtual_ids,
            start=1,
        ):
            station_support = get_station_support(
                support_table,
                station_id,
            )

            for (
                year,
                utc_start,
                utc_end,
                expected_hours,
            ) in get_era5_utc_windows(
                analysis_start,
                analysis_end,
            ):
                filename = (
                    f"station_{station_index:02d}_"
                    f"{year}.csv"
                )
                chunk_path = (
                    era5_chunk_root
                    / filename
                )

                if chunk_path.exists() and not args.force:
                    print("Using ERA5 checkpoint:", chunk_path.name)
                    era5_chunks.append(chunk_path)
                    continue

                print(
                    "ERA5 |",
                    station_id,
                    "|",
                    year,
                )
                table = build_era5_hourly_table(
                    station_support,
                    utc_start,
                    utc_end,
                )

                download_feature_collection_csv(
                    ee=ee,
                    feature_collection=table,
                    output_path=chunk_path,
                    selectors=ERA5_EXPORT_SELECTORS,
                )

                rows = sum(
                    1
                    for _ in chunk_path.open(
                        "r",
                        encoding="utf-8",
                    )
                ) - 1
                if rows != expected_hours:
                    raise RuntimeError(
                        f"ERA5 row count mismatch for {station_id} {year}: "
                        f"{rows} != {expected_hours}"
                    )

                era5_chunks.append(chunk_path)

        era5_rows = merge_csv_chunks(
            era5_chunks,
            era5_final,
        )

        expected_era5_rows = (
            sum(
                window[3]
                for window in get_era5_utc_windows(
                    analysis_start,
                    analysis_end,
                )
            )
            * len(virtual_ids)
        )
        if era5_rows != expected_era5_rows:
            raise RuntimeError(
                f"ERA5 total row count mismatch: "
                f"{era5_rows} != {expected_era5_rows}"
            )

        print("ERA5 rows:", era5_rows)
        print("Saved:", era5_final)

    # ------------------------------------------------------------------
    # 3. Build the virtual master locally
    # ------------------------------------------------------------------
    print()
    print("=== VIRTUAL TRAINING MASTER ===")

    satellite = pd.read_csv(
        satellite_final,
        dtype={"station_id": str},
    )
    station_support = pd.read_csv(
        support_path,
        dtype={"station_id": str},
    )
    era5_hourly = pd.read_csv(
        era5_final,
        dtype={"station_id": str},
    )

    virtual_master, virtual_daily = (
        build_training_master(
            satellite=satellite,
            era5_hourly=era5_hourly,
            chirps_daily=None,
            station_support=station_support,
        )
    )

    virtual_master_path = (
        master_root
        / "virtual_training_master.csv"
    )
    virtual_daily_path = (
        master_root
        / "virtual_reference_et_daily.csv"
    )
    virtual_master.to_csv(
        virtual_master_path,
        index=False,
    )
    virtual_daily.to_csv(
        virtual_daily_path,
        index=False,
    )

    virtual_population = prepare_rf25_population(virtual_master)

    if len(selected_supports) != 10:
        raise RuntimeError("Selected-support table must contain exactly 10 Virtual10 supports.")
    if selected_supports["spatial_block_utm10km"].nunique() != 10:
        raise RuntimeError("Virtual10 supports must occupy 10 distinct fixed UTM blocks.")

    block_map = dict(
        zip(
            selected_supports["virtual_id"].astype(str),
            selected_supports["spatial_block_utm10km"].astype(str),
            strict=True,
        )
    )
    virtual_population["spatial_block"] = (
        virtual_population["station_id"].astype(str).map(block_map)
    )
    if virtual_population["spatial_block"].isna().any():
        missing_ids = sorted(
            virtual_population.loc[virtual_population["spatial_block"].isna(), "station_id"]
            .astype(str).unique().tolist()
        )
        raise RuntimeError("Missing frozen UTM block for support(s): " + ", ".join(missing_ids))
    if not virtual_population["station_id"].astype(str).str.startswith("VF").all():
        raise RuntimeError("Virtual10 population contains a real-station ID.")
    if virtual_population["station_id"].nunique() != 10:
        raise RuntimeError("Virtual10 population does not retain 10 supports after GE90.")
    if virtual_population["spatial_block"].nunique() != 10:
        raise RuntimeError("Virtual10 population does not retain 10 spatial blocks after GE90.")

    output_path = results_root / "virtual10_training_population.csv"
    virtual_population.to_csv(output_path, index=False)
    metadata = {
        "design": "Virtual10 sequential-random GE90",
        "analysis_start": args.start_date,
        "analysis_end_exclusive": args.end_date_exclusive,
        "training_design": "10 whole-basin virtual MODIS footprints in 10 fixed 10 km UTM blocks",
        "canonical_support_reconstruction": True,
        "ge90_period_whitelist_used_for_extraction": True,
        "real_station_footprints_used_for_training": False,
        "virtual_support_ids": sorted(virtual_population["station_id"].astype(str).unique().tolist()),
        "virtual_spatial_blocks": sorted(virtual_population["spatial_block"].astype(str).unique().tolist()),
        "spatial_block_definition": "fixed EPSG:32618 10 km grid; floor(easting/10000)_floor(northing/10000)",
        "target_definition": "Kc_target = MODIS_ET / ETo",
        "final_predictor_count": 25,
        "final_model": "RandomForestRegressor trained separately",
        "google_drive_used": False,
        "interpretation_guardrail": "Training target is MODIS-derived Kc; this is not independent 20 m ET validation.",
    }
    (results_root / "training_population_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Virtual master rows:", len(virtual_master))
    print("RF-25 GE90 rows:", len(virtual_population))
    print("Virtual supports:", virtual_population["station_id"].nunique())
    print("Spatial blocks:", virtual_population["spatial_block"].nunique())
    print("Training population:", output_path)


if __name__ == "__main__":
    main()
