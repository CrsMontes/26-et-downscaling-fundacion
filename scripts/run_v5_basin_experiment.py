"""Explicit reproduction of the frozen V5 training design.

Reads frozen selection from training/selection, extracts training/raw, builds
training/master and writes evaluation/results. This command performs extraction
and training and is not called by the operational read-only V5 entry point.
Stable5 transfer diagnostics require an explicit --reference-workspace; V5
training itself is independent. Target, predictors, CV and AOA are unchanged.
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

from et_downscaling.virtual_station import (
    resolve_virtual_workspace, resolve_reference_workspace, resolve_reference_run,
)


DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE_EXCLUSIVE = "2025-01-01"
EXPERIMENT_NAME = "v5_basin_random_ge90_canonical"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the frozen operational Virtual Station V5 training design."
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
            "Directory containing virtual_modis_footprints.geojson. Defaults to "
            "../ET_fundacion_workspace_virtual_station/training/selection."
        ),
    )
    parser.add_argument(
        "--workspace-root",
        default=None,
        help=(
            "Virtual Station workspace root. Defaults to repository sibling "
            "../ET_fundacion_workspace_virtual_station."
        ),
    )
    parser.add_argument("--reference-workspace", default=None, help="Optional explicit Stable5 workspace for reciprocal comparison.")
    parser.add_argument(
        "--stable-run-dir",
        default=None,
        help=(
            "Frozen stable run directory used for the baseline master. "
            "Defaults to current/runs/20260907T162048Z_2020_2024 when present, "
            "inside --reference-workspace; no latest-run fallback."
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
        help="Rebuild completed experimental extraction checkpoints.",
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


def resolve_stable_run_dir(reference_root: Path, value: str | None) -> Path:
    return resolve_reference_run(reference_root, value)


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


def stable_master_from_population(
    stable_run_dir: Path,
    current_master_path: Path,
) -> pd.DataFrame:
    """Prefer the full stable master; fail rather than reconstruct from population."""
    if current_master_path.is_file():
        return pd.read_csv(
            current_master_path,
            dtype={"station_id": str},
        )

    raise FileNotFoundError(
        "Stable raw-derived master was not found. The experiment intentionally "
        "does not reconstruct a full master from the filtered training population.\n"
        f"Expected: {current_master_path}\n"
        f"Stable run: {stable_run_dir}"
    )



def load_frozen_selection(
    selection_dir: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load the frozen V5 support selection and exact GE90 period whitelist."""
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
            f"Frozen V5 selection contains {len(selected)} supports; expected 10."
        )
    if selected["virtual_id"].nunique() != 10:
        raise RuntimeError(
            "Frozen V5 selection contains duplicate virtual IDs."
        )
    if selected["modis_pixel_id"].nunique() != 10:
        raise RuntimeError(
            "Frozen V5 selection contains duplicate MODIS pixel IDs."
        )
    if selected["spatial_block_utm10km"].nunique() != 10:
        raise RuntimeError(
            "Frozen V5 selection does not contain 10 distinct fixed UTM blocks."
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
    from et_downscaling.modeling import (
        prepare_ridge25_population,
        train_and_validate_ridge25,
    )
    from et_downscaling.optical import (
        get_optical_collection,
    )
    from et_downscaling.schema import (
        get_satellite_export_selectors,
    )
    from et_downscaling.aoa_ridge25 import (
        build_unweighted_aoa,
    )

    root = project_root()
    workspace_root = resolve_workspace_root(args.workspace_root)
    current_root = workspace_root / "current"
    reference_root = resolve_reference_workspace(args.reference_workspace) if args.reference_workspace else None
    if args.stable_run_dir and reference_root is None:
        raise ValueError("--stable-run-dir requires --reference-workspace.")
    if reference_root == workspace_root:
        raise ValueError("Stable5 reference must be separate from the V5 workspace.")
    stable_run_dir = resolve_stable_run_dir(reference_root, args.stable_run_dir) if reference_root else None
    selection_dir = resolve_selection_dir(
        workspace_root,
        args.selection_dir,
    )

    experiment_root = (
        workspace_root
        / "training"
    )
    raw_root = experiment_root / "raw"
    satellite_root = raw_root / "satellite" / "S2"
    meteorology_root = raw_root / "meteorology"
    master_root = experiment_root / "master"
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
    print("ET FUNDACION - V5 CANONICAL WHOLE-BASIN RANDOM GE90 EXPERIMENT")
    print("=" * 88)
    print("Stable run:", stable_run_dir.name if stable_run_dir else "comparison not requested")
    print("Analysis period:", args.start_date, "to", args.end_date_exclusive)
    print("Experiment:", experiment_root)
    print("Frozen selection source: selected_supports.csv + availability_checks/")
    print("MODIS polygons reconstructed canonically from frozen centroid points: YES")
    print("Stable current/ modified: NO")
    print("Stable final/ modified: NO")
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

    virtual_population = prepare_ridge25_population(
        virtual_master
    )

    print("Virtual master rows:", len(virtual_master))
    print("Virtual GE90 Ridge25 rows:", len(virtual_population))
    print("Virtual supports with GE90 rows:", virtual_population["station_id"].nunique())

    # ------------------------------------------------------------------
    # 4. Virtual-only Ridge25 design and reciprocal transfer diagnostics
    # ------------------------------------------------------------------
    print()
    print("=== VIRTUAL-ONLY RIDGE25 DESIGN ===")

    from et_downscaling.modeling import (
        SPATIAL_BLOCK_SIZE_KM,
        TARGET_COLUMN,
        build_ridge25_model,
        calculate_metrics as model_metrics,
    )
    from et_downscaling.ridge25 import RIDGE25_MODEL_FEATURES
    from et_downscaling.aoa_ridge25 import score_unweighted_aoa

    stable_population = None
    if reference_root is not None:
        stable_master_path = (
            reference_root / "current"
            / "master"
            / "S2"
            / build_training_output_filename("S2")
        )
        stable_master = stable_master_from_population(
            stable_run_dir=stable_run_dir,
            current_master_path=stable_master_path,
        )
        stable_population = prepare_ridge25_population(
            stable_master
        ).copy()
    virtual_population = prepare_ridge25_population(
        virtual_master
    ).copy()

    # ------------------------------------------------------------------
    # Frozen spatial-block design for V5
    # ------------------------------------------------------------------
    # Stable-5 keeps its original frozen spatial blocks exactly as accepted.
    # V5 uses the fixed 10 km UTM grid frozen BEFORE selection:
    # floor(EPSG:32618 easting/10000)_floor(northing/10000).
    if len(selected_supports) != 10:
        raise RuntimeError(
            "V5 selected-support table does not contain exactly 10 supports."
        )
    if selected_supports["spatial_block_utm10km"].nunique() != 10:
        raise RuntimeError(
            "V5 selected supports are not in 10 distinct fixed UTM blocks."
        )

    block_map = dict(
        zip(
            selected_supports["virtual_id"].astype(str),
            selected_supports["spatial_block_utm10km"].astype(str),
            strict=True,
        )
    )

    if stable_population is not None:
        stable_population["spatial_block"] = (
            stable_population["spatial_block"].astype(str)
        )
    virtual_population["spatial_block"] = (
        virtual_population["station_id"]
        .astype(str)
        .map(block_map)
    )

    if virtual_population["spatial_block"].isna().any():
        missing_ids = sorted(
            virtual_population.loc[
                virtual_population["spatial_block"].isna(),
                "station_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise RuntimeError(
            "Missing frozen UTM block for virtual support(s): "
            + ", ".join(missing_ids)
        )

    if stable_population is not None and stable_population["station_id"].astype(str).str.startswith("VF").any():
        raise RuntimeError(
            "Stable population unexpectedly contains virtual IDs."
        )
    if not virtual_population["station_id"].astype(str).str.startswith("VF").all():
        raise RuntimeError(
            "Virtual population contains non-virtual support IDs."
        )
    if virtual_population["station_id"].nunique() != 10:
        raise RuntimeError(
            "V5 experiment does not contain 10 supports after GE90."
        )
    if virtual_population["spatial_block"].nunique() != 10:
        raise RuntimeError(
            "V5 GE90 population does not retain 10 distinct fixed UTM blocks."
        )

    stable_blocks = set(
        stable_population["spatial_block"].astype(str).unique() if stable_population is not None else []
    )
    virtual_blocks = set(
        virtual_population["spatial_block"].astype(str).unique()
    )

    def oof_by_group(
        data: pd.DataFrame,
        group_column: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        predictions = np.full(len(data), np.nan, dtype=float)
        fold_rows = []
        groups = data[group_column].astype(str)

        for fold_number, group in enumerate(
            sorted(groups.unique()),
            start=1,
        ):
            test_mask = groups.eq(group).to_numpy()
            train_mask = ~test_mask

            model = build_ridge25_model()
            model.fit(
                data.loc[train_mask, RIDGE25_MODEL_FEATURES],
                data.loc[train_mask, TARGET_COLUMN],
            )
            prediction = model.predict(
                data.loc[test_mask, RIDGE25_MODEL_FEATURES]
            )
            predictions[test_mask] = prediction

            fold_rows.append(
                {
                    "fold": fold_number,
                    "group": group,
                    **model_metrics(
                        data.loc[test_mask, TARGET_COLUMN],
                        prediction,
                    ),
                }
            )

        if np.isnan(predictions).any():
            raise RuntimeError(
                f"OOF predictions contain missing values for {group_column}."
            )

        output = data[
            [
                "station_id",
                "period_start",
                "spatial_block",
                "year",
                TARGET_COLUMN,
            ]
        ].copy()
        output["prediction"] = predictions
        output["error"] = (
            predictions
            - data[TARGET_COLUMN].to_numpy(float)
        )
        metrics = model_metrics(
            output[TARGET_COLUMN],
            output["prediction"],
        )
        return output, pd.DataFrame(fold_rows), metrics

    def persistence_metrics(
        population: pd.DataFrame,
        label: str,
    ) -> list[dict[str, object]]:
        ordered = population.copy()
        ordered["period_start"] = pd.to_datetime(
            ordered["period_start"],
            errors="raise",
        )
        ordered = ordered.sort_values(
            ["station_id", "period_start"]
        ).reset_index(drop=True)

        ordered["previous_target"] = (
            ordered.groupby("station_id")[TARGET_COLUMN]
            .shift(1)
        )
        ordered["previous_date"] = (
            ordered.groupby("station_id")["period_start"]
            .shift(1)
        )
        ordered["lag_days"] = (
            ordered["period_start"]
            - ordered["previous_date"]
        ).dt.days

        rows = []
        masks = {
            "previous_available": ordered["previous_target"].notna(),
            "previous_exact_8_days": (
                ordered["previous_target"].notna()
                & ordered["lag_days"].eq(8)
            ),
            "previous_within_16_days": (
                ordered["previous_target"].notna()
                & ordered["lag_days"].le(16)
            ),
        }

        for baseline_name, mask in masks.items():
            subset = ordered.loc[mask].copy()
            if subset.empty:
                continue
            rows.append(
                {
                    "population": label,
                    "baseline": baseline_name,
                    "n": int(len(subset)),
                    **model_metrics(
                        subset[TARGET_COLUMN],
                        subset["previous_target"],
                    ),
                }
            )
        return rows

    if stable_population is None:
        virtual_spatial_oof, virtual_spatial_folds, virtual_spatial_metrics = (
            oof_by_group(virtual_population, "spatial_block")
        )
        virtual_temporal_oof, virtual_temporal_folds, virtual_temporal_metrics = (
            oof_by_group(virtual_population, "year")
        )

        virtual_model = build_ridge25_model()
        virtual_model.fit(
            virtual_population[RIDGE25_MODEL_FEATURES],
            virtual_population[TARGET_COLUMN],
        )

        virtual_aoa = build_unweighted_aoa(
            virtual_population,
            group_column="spatial_block",
        )

        metric_table = pd.DataFrame([
            {"training_design": "virtual_10_supports", "evaluation": "spatial_block_CV_internal", **virtual_spatial_metrics},
            {"training_design": "virtual_10_supports", "evaluation": "leave_one_year_out_internal", **virtual_temporal_metrics},
        ])
        outputs = {
            "virtual10_metrics.csv": metric_table,
            "virtual10_spatial_oof.csv": virtual_spatial_oof,
            "virtual10_spatial_fold_metrics.csv": virtual_spatial_folds,
            "virtual10_temporal_oof.csv": virtual_temporal_oof,
            "virtual10_temporal_fold_metrics.csv": virtual_temporal_folds,
            "virtual10_training_population.csv": virtual_population,
            "persistence_baselines.csv": pd.DataFrame(persistence_metrics(virtual_population, "v5_basin10")),
        }
        for filename, frame in outputs.items():
            frame.to_csv(results_root / filename, index=False)
        metadata = {
            "experiment": EXPERIMENT_NAME,
            "analysis_start": args.start_date,
            "analysis_end_exclusive": args.end_date_exclusive,
            "canonical_support_reconstruction": True,
            "frozen_GE90_period_whitelist_used_for_extraction": True,
            "real_station_footprints_used_for_training": False,
            "virtual_support_ids": virtual_ids,
            "virtual_modis_pixel_ids": modis_ids,
            "virtual_spatial_blocks": sorted(virtual_blocks),
            "virtual_spatial_block_definition": "fixed EPSG:32618 10 km grid; floor(easting/10000)_floor(northing/10000)",
            "target_definition": "Kc_target = MODIS_ET / ETo",
            "ridge25_predictor_count": 25,
            "virtual10_AOA_threshold": float(virtual_aoa.threshold),
            "reference_workspace": None,
            "interpretation_guardrail": "Internal CV evaluates a MODIS-derived Kc target, not independent 20 m ET validation.",
        }
        (results_root / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(metric_table.to_string(index=False))
        print("V5 results:", results_root)
        return

    stable_spatial_oof, stable_spatial_folds, stable_spatial_metrics = (
        oof_by_group(stable_population, "spatial_block")
    )
    stable_temporal_oof, stable_temporal_folds, stable_temporal_metrics = (
        oof_by_group(stable_population, "year")
    )
    virtual_spatial_oof, virtual_spatial_folds, virtual_spatial_metrics = (
        oof_by_group(virtual_population, "spatial_block")
    )
    virtual_temporal_oof, virtual_temporal_folds, virtual_temporal_metrics = (
        oof_by_group(virtual_population, "year")
    )

    stable_model = build_ridge25_model()
    stable_model.fit(
        stable_population[RIDGE25_MODEL_FEATURES],
        stable_population[TARGET_COLUMN],
    )
    virtual_model = build_ridge25_model()
    virtual_model.fit(
        virtual_population[RIDGE25_MODEL_FEATURES],
        virtual_population[TARGET_COLUMN],
    )

    stable_to_virtual_prediction = stable_model.predict(
        virtual_population[RIDGE25_MODEL_FEATURES]
    )
    virtual_to_stable_prediction = virtual_model.predict(
        stable_population[RIDGE25_MODEL_FEATURES]
    )

    stable_to_virtual_metrics = model_metrics(
        virtual_population[TARGET_COLUMN],
        stable_to_virtual_prediction,
    )
    virtual_to_stable_metrics = model_metrics(
        stable_population[TARGET_COLUMN],
        virtual_to_stable_prediction,
    )

    stable_aoa = build_unweighted_aoa(
        stable_population,
        group_column="spatial_block",
    )
    virtual_aoa = build_unweighted_aoa(
        virtual_population,
        group_column="spatial_block",
    )

    virtual_di_in_stable, virtual_inside_stable = score_unweighted_aoa(
        virtual_population[RIDGE25_MODEL_FEATURES].to_numpy(float),
        stable_aoa,
    )
    stable_di_in_virtual, stable_inside_virtual = score_unweighted_aoa(
        stable_population[RIDGE25_MODEL_FEATURES].to_numpy(float),
        virtual_aoa,
    )

    virtual_scored = virtual_population[
        ["station_id", "period_start", "spatial_block", TARGET_COLUMN]
    ].copy()
    virtual_scored["stable_AOA_DI"] = virtual_di_in_stable
    virtual_scored["stable_AOA_inside"] = virtual_inside_stable

    stable_scored = stable_population[
        ["station_id", "period_start", "spatial_block", TARGET_COLUMN]
    ].copy()
    stable_scored["virtual_AOA_DI"] = stable_di_in_virtual
    stable_scored["virtual_AOA_inside"] = stable_inside_virtual

    def support_aoa_summary(
        scored: pd.DataFrame,
        di_column: str,
        inside_column: str,
        label: str,
    ) -> pd.DataFrame:
        rows = []
        for support_id, group in scored.groupby("station_id", sort=True):
            values = pd.to_numeric(group[di_column], errors="coerce").to_numpy(float)
            rows.append(
                {
                    "direction": label,
                    "support_id": str(support_id),
                    "spatial_block": str(group["spatial_block"].iloc[0]),
                    "n": len(group),
                    "AOA_inside_fraction": float(
                        pd.to_numeric(group[inside_column], errors="coerce").mean()
                    ),
                    "DI_median": float(np.nanmedian(values)),
                    "DI_p90": float(np.nanquantile(values, 0.90)),
                    "DI_max": float(np.nanmax(values)),
                }
            )
        return pd.DataFrame(rows)

    reciprocal_aoa = pd.concat(
        [
            support_aoa_summary(
                virtual_scored,
                "stable_AOA_DI",
                "stable_AOA_inside",
                "stable_training_to_virtual_test",
            ),
            support_aoa_summary(
                stable_scored,
                "virtual_AOA_DI",
                "virtual_AOA_inside",
                "virtual_training_to_stable_test",
            ),
        ],
        ignore_index=True,
    )

    metric_rows = []
    for design, evaluation, metrics in (
        (
            "stable_5_supports",
            "spatial_block_CV_internal",
            stable_spatial_metrics,
        ),
        (
            "stable_5_supports",
            "leave_one_year_out_internal",
            stable_temporal_metrics,
        ),
        (
            "virtual_10_supports",
            "spatial_block_CV_internal",
            virtual_spatial_metrics,
        ),
        (
            "virtual_10_supports",
            "leave_one_year_out_internal",
            virtual_temporal_metrics,
        ),
        (
            "stable_5_supports",
            "full_model_transfer_to_virtual10",
            stable_to_virtual_metrics,
        ),
        (
            "virtual_10_supports",
            "full_model_transfer_to_stable5",
            virtual_to_stable_metrics,
        ),
    ):
        metric_rows.append(
            {
                "training_design": design,
                "evaluation": evaluation,
                **metrics,
            }
        )

    metric_table = pd.DataFrame(metric_rows)

    target_distributions = pd.DataFrame(
        [
            {
                "population": label,
                "n": len(pop),
                "supports": int(pop["station_id"].nunique()),
                "spatial_blocks": int(pop["spatial_block"].nunique()),
                "Kc_mean": float(pop[TARGET_COLUMN].mean()),
                "Kc_sd": float(pop[TARGET_COLUMN].std(ddof=0)),
                "Kc_p05": float(pop[TARGET_COLUMN].quantile(0.05)),
                "Kc_median": float(pop[TARGET_COLUMN].median()),
                "Kc_p95": float(pop[TARGET_COLUMN].quantile(0.95)),
            }
            for label, pop in (
                ("stable", stable_population),
                ("virtual10", virtual_population),
            )
        ]
    )

    transfer_stable = stable_population[
        ["station_id", "period_start", "spatial_block", "year", TARGET_COLUMN]
    ].copy()
    transfer_stable["virtual10_prediction"] = virtual_to_stable_prediction
    transfer_stable["error"] = (
        transfer_stable["virtual10_prediction"]
        - transfer_stable[TARGET_COLUMN]
    )

    transfer_virtual = virtual_population[
        ["station_id", "period_start", "spatial_block", "year", TARGET_COLUMN]
    ].copy()
    transfer_virtual["stable5_prediction"] = stable_to_virtual_prediction
    transfer_virtual["error"] = (
        transfer_virtual["stable5_prediction"]
        - transfer_virtual[TARGET_COLUMN]
    )

    # ------------------------------------------------------------------
    # Persistence baselines (coarse temporal diagnostic only)
    # ------------------------------------------------------------------
    # These are reported separately from spatial CV/LOYO because persistence
    # uses the previous observed Kc_target from the same support and therefore
    # is not the same prediction task as held-out spatial-block CV.
    persistence_table = pd.DataFrame(
        persistence_metrics(
            stable_population,
            "stable5",
        )
        + persistence_metrics(
            virtual_population,
            "v5_basin10",
        )
    )

    outputs = {
        "virtual10_vs_stable_metrics.csv": metric_table,
        "virtual10_target_distribution.csv": target_distributions,
        "reciprocal_AOA_by_support.csv": reciprocal_aoa,
        "virtual_rows_scored_against_stable_AOA.csv": virtual_scored,
        "stable_rows_scored_against_virtual_AOA.csv": stable_scored,
        "virtual10_spatial_oof.csv": virtual_spatial_oof,
        "virtual10_spatial_fold_metrics.csv": virtual_spatial_folds,
        "virtual10_temporal_oof.csv": virtual_temporal_oof,
        "virtual10_temporal_fold_metrics.csv": virtual_temporal_folds,
        "stable_spatial_oof_reproduced.csv": stable_spatial_oof,
        "stable_temporal_oof_reproduced.csv": stable_temporal_oof,
        "virtual10_full_model_transfer_to_stable5.csv": transfer_stable,
        "stable5_full_model_transfer_to_virtual10.csv": transfer_virtual,
        "virtual10_training_population.csv": virtual_population,
        "persistence_baselines.csv": persistence_table,
    }
    for filename, frame in outputs.items():
        frame.to_csv(results_root / filename, index=False)

    experiment_metadata = {
        "experiment": EXPERIMENT_NAME,
        "status": "operational_virtual_station",
        "stable_run": stable_run_dir.name,
        "reference_workspace": str(reference_root),
        "analysis_start": args.start_date,
        "analysis_end_exclusive": args.end_date_exclusive,
        "training_design": "10 whole-basin virtual MODIS footprints selected by frozen sequential random GE90 rejection",
        "canonical_support_reconstruction": True,
        "frozen_GE90_period_whitelist_used_for_extraction": True,
        "real_station_footprints_used_for_training": False,
        "real_station_rows_reserved_for_transfer_test": True,
        "virtual_support_ids": virtual_ids,
        "virtual_modis_pixel_ids": modis_ids,
        "stable_reference_latitude_deg": None,
        "virtual_spatial_blocks": sorted(virtual_blocks),
        "virtual_spatial_block_definition": "fixed EPSG:32618 10 km grid; floor(easting/10000)_floor(northing/10000)",
        "stable_spatial_blocks": sorted(stable_blocks),
        "target_definition": "Kc_target = MODIS_ET / ETo",
        "ridge25_predictor_count": 25,
        "stable_AOA_threshold": float(stable_aoa.threshold),
        "virtual10_AOA_threshold": float(virtual_aoa.threshold),
        "stable_current_modified": False,
        "stable_final_modified": False,
        "google_drive_used": False,
        "production_model_replaced": False,
        "field_validation_changed": False,
        "interpretation_guardrail": (
            "Virtual-only CV and reciprocal transfer tests evaluate a MODIS-derived "
            "Kc target. They are not independent 20 m ET validation. The five real "
            "stations remain reserved for subsequent field-derived ET proxy comparison."
        ),
    }
    (results_root / "experiment_metadata.json").write_text(
        json.dumps(experiment_metadata, indent=2),
        encoding="utf-8",
    )

    print("Stable GE90 rows:", len(stable_population))
    print("Stable spatial blocks:", stable_population["spatial_block"].nunique())
    print("Virtual10 GE90 rows:", len(virtual_population))
    print("Virtual10 supports:", virtual_population["station_id"].nunique())
    print("Virtual10 spatial blocks:", virtual_population["spatial_block"].nunique())
    print()
    print("MODEL / TRANSFER METRICS")
    print(metric_table.to_string(index=False))
    print()
    print("RECIPROCAL AOA BY SUPPORT")
    print(
        reciprocal_aoa.to_string(
            index=False,
            formatters={
                "AOA_inside_fraction": "{:.3f}".format,
                "DI_median": "{:.3f}".format,
                "DI_p90": "{:.3f}".format,
                "DI_max": "{:.3f}".format,
            },
        )
    )
    print()
    print("TARGET DISTRIBUTIONS")
    print(target_distributions.to_string(index=False))
    print()
    print("PERSISTENCE BASELINES")
    print(persistence_table.to_string(index=False))
    print()
    print("Results:", results_root)
    print("Stable current/ modified: NO")
    print("Stable final/ modified: NO")
    print("Production model replaced: NO")
    print("Real station footprints used for training: NO")
    print("Field comparison changed: NO")


if __name__ == "__main__":
    main()
