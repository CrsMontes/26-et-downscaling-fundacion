"""Select 10 whole-basin canonical MODIS supports by sequential random GE90 rejection.

Final training-support design
--------------------------
Sampling frame
    Every native MODIS pixel whose centroid lies inside the canonical
    FundaciÃ³n basin.

Randomization
    One reproducible random ordering of the full MODIS sampling frame using
    seed=42 by default. The random order is generated ONCE. A rejected
    candidate does not trigger a new draw; the algorithm simply advances to
    the next candidate in the same frozen random order.

Constraints fixed before model evaluation
    - exactly 10 accepted MODIS supports;
    - maximum one accepted support per fixed 10 km UTM block;
    - fixed 10 km blocks are floor(EPSG:32618 easting/10000) and
      floor(northing/10000), so the block grid does not move with the sample;
    - blocks containing any of the five real stations are excluded so the
      field locations remain spatially external to training;
    - a candidate is availability-eligible only if it has at least
      `min_ge90_per_year` periods in EACH year 2020-2024 satisfying:
          MODIS good == 1
          AND S2 optical union coverage >= 90%
          AND period fully inside the analysis interval.
      The final rule requires at least 20 GE90 periods in each year 2020-2024.
    - no Kc_target, model output, residual, AOA, field ET, R2, RMSE, MAE,
      KGE, or downstream map coverage is used in selection.

Every evaluated candidate is logged, including rejected candidates and the
reason for rejection. Exact per-period availability checks are cached so a
rerun replays the same frozen sequence instead of resampling.

This script SELECTS supports only. It does not train a model.

Outputs
-------
outputs/training/
â””â”€â”€ selection/
    â”œâ”€â”€ selected_supports.csv
    â”œâ”€â”€ selection_log.csv
    â”œâ”€â”€ virtual_points.geojson
    â”œâ”€â”€ virtual_modis_footprints.geojson
    â”œâ”€â”€ selection_metadata.json
    â””â”€â”€ availability_checks/
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import math
import os
from pathlib import Path
import time

import ee
import numpy as np
import pandas as pd

from et_downscaling.virtual_station import resolve_virtual_workspace

from et_downscaling.config import ANALYSIS_CRS
from et_downscaling.dataset import build_availability_table
from et_downscaling.modis import (
    assign_station_footprints,
    build_modis_grid,
    build_modis_pixel_id,
    get_modis_collection,
    get_modis_projection,
    get_modis_scale,
)
from et_downscaling.optical import get_optical_collection
from et_downscaling.stations import get_station_collection


SELECTION_NAME = "virtual10_random_ge90_seed42"
START_DATE = "2020-01-01"
END_DATE_EXCLUSIVE = "2025-01-01"
YEARS = tuple(range(2020, 2025))
GE90_THRESHOLD_PCT = 90.0
BLOCK_SIZE_M = 10_000
DEFAULT_SEED = 42
DEFAULT_N_SUPPORTS = 10
DEFAULT_MIN_GE90_PER_YEAR = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential constrained-random MODIS support selection across "
            "the full FundaciÃ³n basin."
        )
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--n-supports",
        type=int,
        default=DEFAULT_N_SUPPORTS,
    )
    parser.add_argument(
        "--min-ge90-per-year",
        type=int,
        default=DEFAULT_MIN_GE90_PER_YEAR,
        help=(
            "Minimum MODIS-good + S2>=90%% periods required in EACH year "
            "2020-2024. Final default: 20."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=9123,
        help=(
            "Safety limit on sequential candidates evaluated. The algorithm "
            "stops earlier as soon as 10 supports are accepted."
        ),
    )
    parser.add_argument(
        "--force-selection",
        action="store_true",
        help=(
            "Delete selection outputs/checks and replay the SAME frozen "
            "random design. This does not change the seed."
        ),
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_workspace_root(value: str | None) -> Path:
    return resolve_virtual_workspace(value, project_root())



def load_basin(root: Path) -> ee.Geometry:
    path = (
        root
        / "data"
        / "boundaries"
        / "fundacion_basin.geojson"
    )
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    geometries = [
        feature["geometry"]
        for feature in payload.get("features", [])
        if feature.get("geometry") is not None
    ]
    if not geometries:
        raise RuntimeError(f"No geometry found in {path}")

    if len(geometries) == 1:
        return ee.Geometry(geometries[0])

    return ee.FeatureCollection(
        [ee.Feature(ee.Geometry(geometry)) for geometry in geometries]
    ).geometry()


def fixed_utm_block_from_geometry(
    geometry: ee.Geometry,
) -> ee.Dictionary:
    centroid = geometry.centroid(1)
    wgs84 = centroid.transform("EPSG:4326", 1)
    utm = centroid.transform(ANALYSIS_CRS, 1)

    lon_lat = wgs84.coordinates()
    xy = utm.coordinates()

    x = ee.Number(xy.get(0))
    y = ee.Number(xy.get(1))
    block_x = x.divide(BLOCK_SIZE_M).floor().toInt()
    block_y = y.divide(BLOCK_SIZE_M).floor().toInt()

    return ee.Dictionary(
        {
            "longitude": lon_lat.get(0),
            "latitude": lon_lat.get(1),
            "utm_x_m": x,
            "utm_y_m": y,
            "spatial_block_utm10km": (
                block_x.format()
                .cat("_")
                .cat(block_y.format())
            ),
        }
    )


def build_sampling_frame(
    *,
    basin: ee.Geometry,
    pixel_id: ee.Image,
    modis_projection: ee.Projection,
    modis_scale: ee.Number,
) -> ee.FeatureCollection:
    """Native MODIS polygons whose centroids are inside the basin."""
    grid = pixel_id.reduceToVectors(
        geometry=basin.buffer(modis_scale.multiply(1.5)),
        crs=modis_projection,
        scale=modis_scale,
        geometryType="polygon",
        eightConnected=False,
        labelProperty="modis_pixel_id",
        reducer=ee.Reducer.countEvery(),
        geometryInNativeProjection=True,
        maxPixels=1e9,
        tileScale=4,
    )

    def annotate(feature):
        feature = ee.Feature(feature)
        centroid = feature.geometry().centroid(1)
        centroid_wgs84 = centroid.transform("EPSG:4326", 1)
        block = fixed_utm_block_from_geometry(feature.geometry())

        return feature.set(
            block
        ).set(
            {
                "center_in_basin": ee.Number(
                    ee.Algorithms.If(
                        basin.contains(centroid_wgs84, 1),
                        1,
                        0,
                    )
                ),
                "footprint_area_m2": feature.geometry().area(1),
            }
        )

    return (
        ee.FeatureCollection(grid.map(annotate))
        .filter(ee.Filter.eq("center_in_basin", 1))
    )


def real_station_blocks() -> tuple[set[str], dict[str, str]]:
    stations = get_station_collection().getInfo()
    blocks: set[str] = set()
    station_map: dict[str, str] = {}

    for feature in stations.get("features", []):
        station_id = str(feature["properties"]["station_id"])
        lon, lat = feature["geometry"]["coordinates"]

        utm_point = (
            ee.Geometry.Point([float(lon), float(lat)])
            .transform(ANALYSIS_CRS, 1)
            .coordinates()
            .getInfo()
        )
        x = float(utm_point[0])
        y = float(utm_point[1])
        block = f"{math.floor(x / BLOCK_SIZE_M)}_{math.floor(y / BLOCK_SIZE_M)}"
        blocks.add(block)
        station_map[station_id] = block

    return blocks, station_map


def availability_check_path(
    check_root: Path,
    candidate_order: int,
    modis_pixel_id: int,
) -> Path:
    return (
        check_root
        / f"{candidate_order:05d}_{modis_pixel_id}.csv"
    )



def reconstruct_canonical_candidate_support(
    *,
    candidate: ee.Feature,
    pixel_id: ee.Image,
    modis_projection: ee.Projection,
    modis_scale: ee.Number,
) -> ee.Feature:
    """Reconstruct the exact native MODIS polygon for a random candidate.

    The random sampling frame provides the candidate MODIS pixel ID and
    centroid. Availability is NEVER evaluated directly on the sampling-frame
    polygon. Instead, the centroid point is round-tripped through the exact
    repository production functions `build_modis_grid()` and
    `assign_station_footprints()`.

    This makes the selection support identical to the support that will later
    be used for extraction/training. The selected MODIS pixel ID is verified
    before any GE90 decision is made.
    """
    expected_id = int(
        ee.Number(
            candidate.get("modis_pixel_id")
        ).getInfo()
    )

    longitude = float(
        ee.Number(
            candidate.get("longitude")
        ).getInfo()
    )
    latitude = float(
        ee.Number(
            candidate.get("latitude")
        ).getInfo()
    )

    temporary_id = f"CANDIDATE_{expected_id}"

    point = ee.Feature(
        ee.Geometry.Point(
            [longitude, latitude]
        ),
        candidate.toDictionary()
        .set("station_id", temporary_id)
        .set("station", temporary_id),
    )
    points = ee.FeatureCollection([point])

    grid = build_modis_grid(
        points,
        pixel_id,
        modis_projection,
        modis_scale,
    )
    assigned = ee.Feature(
        assign_station_footprints(
            points,
            grid,
            pixel_id,
            modis_projection,
            modis_scale,
        ).first()
    )

    actual_id = int(
        ee.Number(
            assigned.get("modis_pixel_id")
        ).getInfo()
    )

    if actual_id != expected_id:
        raise RuntimeError(
            "Canonical candidate reconstruction changed the MODIS pixel ID: "
            f"expected={expected_id}, assigned={actual_id}."
        )

    # Preserve the frozen sampling-frame properties while keeping ONLY the
    # reconstructed native MODIS polygon as geometry.
    return ee.Feature(
        assigned.geometry(),
        candidate.toDictionary()
        .set("station_id", temporary_id)
        .set("station", temporary_id)
        .set("modis_pixel_id", actual_id)
        .set(
            "footprint_area_m2",
            assigned.get("footprint_area_m2"),
        ),
    )


def evaluate_exact_ge90_availability(
    *,
    candidate: ee.Feature,
    modis_collection: ee.ImageCollection,
    modis_projection: ee.Projection,
    modis_scale: ee.Number,
    check_path: Path,
    max_attempts: int = 5,
) -> pd.DataFrame:
    """Exact repository availability for one canonical native MODIS support."""
    if check_path.is_file():
        return pd.read_csv(check_path)

    station_id = (
        "CANDIDATE_"
        + str(
            int(
                ee.Number(
                    candidate.get("modis_pixel_id")
                ).getInfo()
            )
        )
    )

    props = candidate.toDictionary()
    support = ee.Feature(
        candidate.geometry(),
        props.set("station_id", station_id).set("station", station_id),
    )
    supports = ee.FeatureCollection([support])

    optical_collection = get_optical_collection(
        supports,
        "S2",
    )

    availability = build_availability_table(
        modis_inputs={
            "collection": modis_collection,
            "projection": modis_projection,
            "scale": modis_scale,
            "station_footprints": supports,
        },
        optical_collection=optical_collection,
        s1_collection=None,
        optical_source="S2",
    )

    def annotate(feature):
        feature = ee.Feature(feature)
        coverage = ee.Number(
            feature.get("optical_union_coverage_pct")
        )
        ge90 = (
            ee.Number(feature.get("modis_good"))
            .eq(1)
            .And(
                ee.Number(
                    feature.get("period_within_analysis")
                ).eq(1)
            )
            .And(coverage.gte(GE90_THRESHOLD_PCT))
        )
        year = ee.Date.parse(
            "YYYY-MM-dd",
            ee.String(feature.get("period_start")),
        ).get("year")

        return feature.set(
            {
                "ge90": ee.Number(
                    ee.Algorithms.If(ge90, 1, 0)
                ),
                "year": year,
            }
        )

    compact = ee.FeatureCollection(
        availability.map(annotate)
    ).select(
        [
            "period_start",
            "modis_good",
            "period_within_analysis",
            "optical_union_coverage_pct",
            "ge90",
            "year",
        ]
    )

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            payload = compact.getInfo()
            rows = [
                item["properties"]
                for item in payload.get("features", [])
            ]
            frame = pd.DataFrame(rows)
            if len(frame) == 0:
                raise RuntimeError(
                    "Availability check returned zero periods."
                )
            check_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(check_path, index=False)
            return frame
        except Exception as error:
            last_error = error
            if attempt == max_attempts:
                break
            print(
                "Availability check failed; retrying "
                f"({attempt}/{max_attempts}): {error}"
            )
            time.sleep(10)

    raise RuntimeError(
        f"Availability check failed after {max_attempts} attempts: {last_error}"
    )


def summarize_availability(
    frame: pd.DataFrame,
    min_ge90_per_year: int,
) -> tuple[bool, dict[str, int], str]:
    frame = frame.copy()
    frame["year"] = pd.to_numeric(
        frame["year"],
        errors="raise",
    ).astype(int)
    frame["ge90"] = pd.to_numeric(
        frame["ge90"],
        errors="raise",
    ).astype(int)

    counts = {
        year: int(
            frame.loc[
                frame["year"].eq(year),
                "ge90",
            ].sum()
        )
        for year in YEARS
    }
    total = int(sum(counts.values()))

    missing_years = [
        year
        for year, count in counts.items()
        if count < min_ge90_per_year
    ]

    if missing_years:
        reason = (
            "insufficient_GE90_in_years:"
            + ",".join(
                f"{year}={counts[year]}"
                for year in missing_years
            )
        )
        return False, counts, reason

    return True, counts, "accepted_GE90_temporal_rule"


def feature_to_wgs84_geojson(
    feature: ee.Feature,
) -> dict:
    converted = ee.Feature(
        feature.geometry().transform("EPSG:4326", 1),
        feature.toDictionary(),
    )
    return converted.getInfo()


def write_geojson(
    path: Path,
    features: list[dict],
) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if args.n_supports != 10:
        raise ValueError(
            "The Virtual10 design is frozen to exactly 10 supports."
        )
    if args.min_ge90_per_year < 1:
        raise ValueError("--min-ge90-per-year must be >= 1.")
    if args.seed < 0:
        raise ValueError("--seed must be >= 0.")

    os.environ["ET_START_DATE"] = START_DATE
    os.environ["ET_END_DATE_EXCLUSIVE"] = END_DATE_EXCLUSIVE

    root = project_root()
    workspace_root = resolve_workspace_root(
        args.workspace_root
    )
    training_root = (
        workspace_root
        / "training"
    )
    selection_root = training_root / "selection"
    check_root = selection_root / "availability_checks"

    for path in (
        selection_root,
        check_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    for protected in (
        (workspace_root / "current").resolve(),
        (workspace_root / "final").resolve(),
    ):
        resolved = selection_root.resolve()
        if (
            resolved == protected
            or protected in resolved.parents
        ):
            raise RuntimeError(
                "Experimental output resolves inside current/ or final/."
            )

    metadata_path = (
        selection_root
        / "selection_metadata.json"
    )

    if args.force_selection:
        for path in selection_root.glob("*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                import shutil
                shutil.rmtree(path)
        check_root.mkdir(parents=True, exist_ok=True)

    if metadata_path.is_file():
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        if metadata.get("selection_complete", False):
            expected = {
                "seed": args.seed,
                "n_supports": args.n_supports,
                "min_ge90_per_year": args.min_ge90_per_year,
            }
            actual = {
                key: metadata.get(key)
                for key in expected
            }
            if actual != expected:
                raise RuntimeError(
                    "A completed Virtual10 selection already exists with different "
                    f"frozen parameters. Existing={actual}; requested={expected}."
                )

            print("=" * 100)
            print("VIRTUAL10 SELECTION ALREADY COMPLETE - REUSING FROZEN DESIGN")
            print("=" * 100)
            print(
                pd.read_csv(
                    selection_root / "selected_supports.csv"
                ).to_string(index=False)
            )
            return

    ee.Initialize(project=args.project)
    ee.Number(1).getInfo()

    basin = load_basin(root)
    modis_collection = get_modis_collection()
    modis_projection = get_modis_projection(
        modis_collection
    )
    modis_scale = get_modis_scale(
        modis_projection
    )
    pixel_id = build_modis_pixel_id(
        modis_projection
    )

    frame = build_sampling_frame(
        basin=basin,
        pixel_id=pixel_id,
        modis_projection=modis_projection,
        modis_scale=modis_scale,
    )

    candidate_count = int(
        frame.size().getInfo()
    )
    station_blocks, station_block_map = (
        real_station_blocks()
    )

    # Exclude real-station blocks BEFORE random ordering.
    station_block_filters = [
        ee.Filter.neq(
            "spatial_block_utm10km",
            block,
        )
        for block in sorted(station_blocks)
    ]
    eligible_frame = (
        frame.filter(
            ee.Filter.And(
                *station_block_filters
            )
        )
        if station_block_filters
        else frame
    )

    eligible_count = int(
        eligible_frame.size().getInfo()
    )

    # ------------------------------------------------------------------
    # Freeze the candidate order LOCALLY, once.
    # ------------------------------------------------------------------
    # Do not rely on repeated evaluation of a lazy Earth Engine randomColumn
    # collection. The previous implementation demonstrated that separate
    # server evaluations of the same lazy candidate expression could return a
    # different feature at the same list position.
    #
    # Instead:
    #   1. materialize the eligible candidate metadata once;
    #   2. sort deterministically by MODIS pixel ID;
    #   3. generate one NumPy permutation with the frozen seed;
    #   4. iterate that immutable local table sequentially.
    #
    # A rejected candidate NEVER triggers a new permutation.
    candidate_properties = [
        "modis_pixel_id",
        "spatial_block_utm10km",
        "longitude",
        "latitude",
        "utm_x_m",
        "utm_y_m",
        "footprint_area_m2",
    ]

    # Earth Engine aborts client-side FeatureCollection materialization after
    # >5000 returned features. Materialize the deterministic, NON-RANDOM
    # eligible frame in small batches, stripping geometry to minimize payload.
    # The random permutation is still generated only once, locally, AFTER all
    # candidate metadata have been assembled and verified.
    metadata_frame = eligible_frame.map(
        lambda feature: ee.Feature(
            None,
            ee.Feature(feature).toDictionary(candidate_properties),
        )
    )

    candidate_rows: list[dict] = []
    batch_size = 2000
    last_modis_id: int | None = None

    while len(candidate_rows) < eligible_count:
        remaining = eligible_count - len(candidate_rows)
        take = min(batch_size, remaining)

        batch_frame = metadata_frame
        if last_modis_id is not None:
            batch_frame = batch_frame.filter(
                ee.Filter.gt("modis_pixel_id", last_modis_id)
            )

        batch_payload = (
            batch_frame
            .sort("modis_pixel_id")
            .limit(take)
            .getInfo()
        )
        batch_rows = [
            feature["properties"]
            for feature in batch_payload.get("features", [])
        ]

        if not batch_rows:
            raise RuntimeError(
                "Earth Engine returned an empty candidate-metadata batch "
                f"after materializing {len(candidate_rows)} of "
                f"{eligible_count} eligible candidates."
            )

        batch_rows = sorted(
            batch_rows,
            key=lambda row: int(row["modis_pixel_id"]),
        )

        first_id = int(batch_rows[0]["modis_pixel_id"])
        new_last_id = int(batch_rows[-1]["modis_pixel_id"])
        if last_modis_id is not None and first_id <= last_modis_id:
            raise RuntimeError(
                "Candidate metadata batches are not strictly increasing by "
                "MODIS pixel ID."
            )
        if new_last_id <= (last_modis_id if last_modis_id is not None else -1):
            raise RuntimeError(
                "Candidate metadata batching did not advance."
            )

        candidate_rows.extend(batch_rows)
        last_modis_id = new_last_id
        print(
            "Candidate metadata materialized: "
            f"{len(candidate_rows)}/{eligible_count}"
        )

    if len(candidate_rows) != eligible_count:
        raise RuntimeError(
            "Materialized candidate count differs from the eligible sampling "
            f"frame: {len(candidate_rows)} vs {eligible_count}."
        )

    candidate_table = pd.DataFrame(candidate_rows)
    candidate_table["modis_pixel_id"] = pd.to_numeric(
        candidate_table["modis_pixel_id"],
        errors="raise",
    ).astype("int64")

    if candidate_table["modis_pixel_id"].duplicated().any():
        raise RuntimeError(
            "Eligible sampling frame contains duplicated MODIS pixel IDs."
        )

    candidate_table = (
        candidate_table
        .sort_values("modis_pixel_id")
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(candidate_table))
    randomized_table = (
        candidate_table
        .iloc[permutation]
        .reset_index(drop=True)
    )
    randomized_table["candidate_order"] = (
        np.arange(len(randomized_table), dtype=int) + 1
    )

    frozen_order_path = (
        selection_root
        / "frozen_candidate_order.csv"
    )
    randomized_table.to_csv(
        frozen_order_path,
        index=False,
    )

    print("=" * 100)
    print("VIRTUAL10 WHOLE-BASIN SEQUENTIAL RANDOM GE90 SELECTION")
    print("=" * 100)
    print("Sampling frame MODIS pixels:", candidate_count)
    print(
        "Eligible after excluding real-station 10 km blocks:",
        eligible_count,
    )
    print("Random seed:", args.seed)
    print(
        "Frozen ordering mechanism: NumPy default_rng(seed) permutation of "
        "the MODIS-ID-sorted eligible frame"
    )
    print("Target supports:", args.n_supports)
    print(
        "Availability rule:",
        f">={args.min_ge90_per_year} MODIS-good + S2>=90% period(s) "
        "in EACH year 2020-2024",
    )
    print(
        "One accepted support per fixed UTM 10 km block: YES"
    )
    print(
        "Rejected candidate triggers new random draw: NO "
        "(advance to next candidate in same frozen order)"
    )
    print("Kc/target/AOA/model/field used in selection: NO")
    print("Stable current/ modified: NO")
    print("Stable final/ modified: NO")
    print()

    selected: list[dict] = []
    selected_features: list[ee.Feature] = []
    selected_blocks: set[str] = set()
    log_rows: list[dict] = []

    max_candidates = min(
        args.max_candidates,
        eligible_count,
    )

    for zero_index in range(max_candidates):
        if len(selected) >= args.n_supports:
            break

        props = randomized_table.iloc[zero_index].to_dict()
        candidate_order = int(props["candidate_order"])
        modis_id = int(props["modis_pixel_id"])
        block = str(props["spatial_block_utm10km"])
        longitude = float(props["longitude"])
        latitude = float(props["latitude"])

        # Rebuild a lightweight candidate feature from the immutable local
        # metadata. No subsequent server-side re-evaluation can change which
        # MODIS pixel belongs to this candidate order.
        candidate = ee.Feature(
            ee.Geometry.Point([longitude, latitude]),
            {
                "modis_pixel_id": modis_id,
                "spatial_block_utm10km": block,
                "longitude": longitude,
                "latitude": latitude,
                "utm_x_m": float(props["utm_x_m"]),
                "utm_y_m": float(props["utm_y_m"]),
                "footprint_area_m2": float(props["footprint_area_m2"]),
            },
        )

        base_log = {
            "candidate_order": candidate_order,
            "modis_pixel_id": modis_id,
            "spatial_block_utm10km": block,
            "longitude": longitude,
            "latitude": latitude,
        }

        if block in selected_blocks:
            row = {
                **base_log,
                "availability_evaluated": False,
                "accepted": False,
                "rejection_reason": "block_already_selected",
            }
            log_rows.append(row)
            pd.DataFrame(log_rows).to_csv(
                selection_root / "selection_log.csv",
                index=False,
            )
            print(
                f"[{candidate_order:04d}] MODIS={modis_id} block={block} "
                "-> REJECT block already selected"
            )
            continue

        check_path = availability_check_path(
            check_root,
            candidate_order,
            modis_id,
        )

        print(
            f"[{candidate_order:04d}] MODIS={modis_id} block={block} "
            "-> exact GE90 availability..."
        )

        canonical_candidate = reconstruct_canonical_candidate_support(
            candidate=candidate,
            pixel_id=pixel_id,
            modis_projection=modis_projection,
            modis_scale=modis_scale,
        )

        availability = evaluate_exact_ge90_availability(
            candidate=canonical_candidate,
            modis_collection=modis_collection,
            modis_projection=modis_projection,
            modis_scale=modis_scale,
            check_path=check_path,
        )

        accepted, counts, reason = (
            summarize_availability(
                availability,
                args.min_ge90_per_year,
            )
        )

        total_ge90 = int(sum(counts.values()))

        row = {
            **base_log,
            "availability_evaluated": True,
            "ge90_total": total_ge90,
            **{
                f"ge90_{year}": counts[year]
                for year in YEARS
            },
            "accepted": bool(accepted),
            "rejection_reason": (
                "" if accepted else reason
            ),
        }
        log_rows.append(row)

        if accepted:
            virtual_id = (
                f"VF{len(selected) + 1:02d}"
            )
            selected_blocks.add(block)

            selected_row = {
                "virtual_id": virtual_id,
                "candidate_order": candidate_order,
                "modis_pixel_id": modis_id,
                "spatial_block_utm10km": block,
                "longitude": longitude,
                "latitude": latitude,
                "utm_x_m": float(props["utm_x_m"]),
                "utm_y_m": float(props["utm_y_m"]),
                "footprint_area_m2": float(
                    props["footprint_area_m2"]
                ),
                "ge90_total": total_ge90,
                **{
                    f"ge90_{year}": counts[year]
                    for year in YEARS
                },
            }
            selected.append(selected_row)

            selected_feature = ee.Feature(
                canonical_candidate.geometry(),
                canonical_candidate.toDictionary()
                .set("station_id", virtual_id)
                .set("station", virtual_id)
                .set("virtual_site", 1)
                .set(
                    "spatial_block",
                    block,
                )
                .set(
                    "spatial_block_utm10km",
                    block,
                ),
            )
            selected_features.append(
                selected_feature
            )

            print(
                "    ACCEPT "
                f"{virtual_id} | GE90 total={total_ge90} | "
                + " ".join(
                    f"{year}={counts[year]}"
                    for year in YEARS
                )
            )
        else:
            print(
                "    REJECT | "
                f"GE90 total={total_ge90} | {reason}"
            )

        pd.DataFrame(log_rows).to_csv(
            selection_root / "selection_log.csv",
            index=False,
        )
        if selected:
            pd.DataFrame(selected).to_csv(
                selection_root / "selected_supports_partial.csv",
                index=False,
            )

    if len(selected) != args.n_supports:
        raise RuntimeError(
            f"Sequential screen stopped with {len(selected)} accepted "
            f"supports after {len(log_rows)} candidates. "
            "Do not change the seed after seeing this result. Increase "
            "--max-candidates and rerun the same command if needed."
        )

    selected_frame = pd.DataFrame(
        selected
    )
    if (
        selected_frame[
            "spatial_block_utm10km"
        ].nunique()
        != args.n_supports
    ):
        raise RuntimeError(
            "Accepted supports are not in 10 distinct fixed UTM blocks."
        )

    points_features = []
    footprint_features = []

    for row, feature in zip(
        selected,
        selected_features,
        strict=True,
    ):
        point_properties = dict(row)
        point_properties.update(
            {
                "station_id": row["virtual_id"],
                "station": row["virtual_id"],
                "station_slug": str(row["virtual_id"]).lower(),
                "virtual_site": 1,
                "spatial_block": row[
                    "spatial_block_utm10km"
                ],
            }
        )
        points_features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        row["longitude"],
                        row["latitude"],
                    ],
                },
                "properties": point_properties,
            }
        )

        footprint_features.append(
            feature_to_wgs84_geojson(
                feature
            )
        )

    selected_frame.to_csv(
        selection_root / "selected_supports.csv",
        index=False,
    )
    write_geojson(
        selection_root / "virtual_points.geojson",
        points_features,
    )
    write_geojson(
        selection_root / "virtual_modis_footprints.geojson",
        footprint_features,
    )

    metadata = {
        "experiment": SELECTION_NAME,
        "selection_complete": True,
        "selection_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "analysis_start": START_DATE,
        "analysis_end_exclusive": END_DATE_EXCLUSIVE,
        "sampling_domain": (
            "All native MODIS pixels whose centroids lie inside the "
            "canonical FundaciÃ³n basin."
        ),
        "sampling_algorithm": (
            "Eligible candidate metadata are materialized once, sorted by "
            "MODIS pixel ID, and permuted locally with NumPy default_rng(seed). "
            "Candidates are then evaluated sequentially; each candidate is "
            "reconstructed to its canonical native MODIS polygon BEFORE "
            "availability screening; rejected candidates do not cause a new "
            "permutation or restart."
        ),
        "random_order_implementation": (
            "numpy.default_rng(seed).permutation over MODIS-ID-sorted eligible frame"
        ),
        "candidate_metadata_materialization": (
            "geometry-free deterministic batches of <=2000 features, strictly "
            "increasing by MODIS pixel ID, to stay below Earth Engine's 5000-"
            "element client query limit"
        ),
        "frozen_candidate_order_file": str(
            selection_root / "frozen_candidate_order.csv"
        ),
        "canonical_footprint_reconstruction_before_GE90_screen": True,
        "seed": args.seed,
        "n_supports": args.n_supports,
        "candidate_count_full_basin": candidate_count,
        "candidate_count_after_station_block_exclusion": eligible_count,
        "candidates_examined_until_completion": len(log_rows),
        "min_ge90_per_year": args.min_ge90_per_year,
        "ge90_definition": (
            "modis_good=1 AND period_within_analysis=1 AND "
            "optical_union_coverage_pct>=90"
        ),
        "fixed_spatial_block_definition": (
            "EPSG:32618; floor(easting/10000)_floor(northing/10000)"
        ),
        "one_accepted_support_per_spatial_block": True,
        "real_station_blocks_excluded_before_randomization": True,
        "real_station_block_map": station_block_map,
        "selected_modis_pixel_ids": (
            selected_frame["modis_pixel_id"]
            .astype(int)
            .tolist()
        ),
        "selected_spatial_blocks": (
            selected_frame[
                "spatial_block_utm10km"
            ]
            .astype(str)
            .tolist()
        ),
        "selection_uses_kc_target": False,
        "selection_uses_model_output": False,
        "selection_uses_residuals": False,
        "selection_uses_aoa": False,
        "selection_uses_field_et": False,
        "selection_uses_validation_metrics": False,
        "stable_current_modified": False,
        "stable_final_modified": False,
        "google_drive_used": False,
    }
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    partial = (
        selection_root
        / "selected_supports_partial.csv"
    )
    partial.unlink(missing_ok=True)

    print()
    print("=" * 100)
    print("VIRTUAL10 SELECTION COMPLETE")
    print("=" * 100)
    print(
        selected_frame.to_string(
            index=False
        )
    )
    print()
    print(
        "Candidates examined:",
        len(log_rows),
    )
    print(
        "Rejected candidates:",
        int(
            (~pd.DataFrame(log_rows)["accepted"].astype(bool)).sum()
        ),
    )
    print("Selection:", selection_root)
    print("Stable current/ modified: NO")
    print("Stable final/ modified: NO")


if __name__ == "__main__":
    main()
