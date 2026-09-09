"""Compare existing frozen V5 and Stable5 rasters on the same 20 m grid.

Reads V5 from current/ and requires an explicit --reference-workspace for
Stable5. Writes comparison tables/maps under evaluation/basin_coverage_comparison.
Missing rasters fail; this command never fits a model or downloads products.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from et_downscaling.virtual_station import (
    resolve_virtual_workspace, resolve_reference_workspace, find_v5_raster, validate_frozen_v5,
)
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from et_downscaling.ridge25_overlap_production import (
    OUTPUT_BANDS,
)


EXPERIMENT_NAME = "v5_basin_random_ge90_canonical"
DEFAULT_DATES = (
    "2020-03-13",
    "2021-11-25",
    "2022-03-30",
)
TRANSITION_DESCRIPTIONS = {
    0: "outside_domain",
    1: "neither",
    2: "stable5_only",
    3: "virtual10_only",
    4: "both",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Stable-5 and Virtual-10 basin applicability/publication "
            "coverage on the same 20 m grid."
        )
    )
    parser.add_argument("--reference-workspace", required=True)
    parser.add_argument(
        "--workspace-root",
        default=None,
        help=(
            "Virtual Station workspace root. Defaults to the repository sibling."
        ),
    )
    parser.add_argument(
        "--date",
        dest="dates",
        action="append",
        default=None,
        help="MODIS-period start date. Repeat for multiple dates.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_workspace_root(value: str | None) -> Path:
    return resolve_virtual_workspace(value, project_root())



def load_union_geojson(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    geometries = [
        shape(feature["geometry"])
        for feature in payload.get("features", [])
        if feature.get("geometry")
    ]
    if not geometries:
        raise ValueError(f"No geometries found in {path}")
    return unary_union(geometries)


def find_stable_scientific_raster(
    workspace_root: Path,
    date_text: str,
) -> Path:
    directory = (
        workspace_root
        / "current"
        / "rasters"
        / date_text
    )
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Stable raster directory not found: {directory}"
        )

    candidates = []
    for path in directory.glob("ET_*_20m.tif"):
        try:
            with rasterio.open(path) as dataset:
                descriptions = tuple(dataset.descriptions)
                if (
                    dataset.count == len(OUTPUT_BANDS)
                    and descriptions == tuple(OUTPUT_BANDS)
                ):
                    priority = (
                        1
                        if "exact_overlap" in path.name.lower()
                        else 0
                    )
                    candidates.append(
                        (
                            priority,
                            path.stat().st_mtime,
                            path,
                        )
                    )
        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(
            "No top-level Stable-5 scientific 9-band raster with the accepted "
            f"band contract was found in {directory}."
        )

    candidates.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    return candidates[0][2]


def ensure_same_grid(
    stable: rasterio.io.DatasetReader,
    virtual: rasterio.io.DatasetReader,
) -> None:
    problems = []

    if stable.crs != virtual.crs:
        problems.append(
            f"CRS: stable={stable.crs}, virtual={virtual.crs}"
        )
    if stable.width != virtual.width or stable.height != virtual.height:
        problems.append(
            "shape: "
            f"stable={stable.width}x{stable.height}, "
            f"virtual={virtual.width}x{virtual.height}"
        )
    if not stable.transform.almost_equals(virtual.transform):
        problems.append(
            f"transform: stable={stable.transform}, virtual={virtual.transform}"
        )

    if problems:
        raise RuntimeError(
            "Stable-5 and Virtual-10 rasters are not on the same grid:\n"
            + "\n".join(problems)
        )


def basin_masks(
    basin_wgs84,
    dataset: rasterio.io.DatasetReader,
) -> tuple[dict[str, np.ndarray], float]:
    if dataset.crs is None:
        raise RuntimeError("Raster has no CRS.")

    basin_projected = shape(
        transform_geom(
            "EPSG:4326",
            dataset.crs.to_string(),
            mapping(basin_wgs84),
            precision=-1,
        )
    )

    centroid_x = float(
        basin_projected.centroid.x
    )
    minx, miny, maxx, maxy = basin_projected.bounds

    east_half = basin_projected.intersection(
        box(
            centroid_x,
            miny - 1000.0,
            maxx + 1000.0,
            maxy + 1000.0,
        )
    )

    masks = {}
    for name, geometry in (
        ("full_basin", basin_projected),
        (
            "eastern_upstream_diagnostic_half",
            east_half,
        ),
    ):
        masks[name] = rasterize(
            [(mapping(geometry), 1)],
            out_shape=(dataset.height, dataset.width),
            transform=dataset.transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        ).astype(bool)

    return masks, centroid_x


def band_index(
    dataset: rasterio.io.DatasetReader,
    name: str,
) -> int:
    descriptions = list(dataset.descriptions)
    if name not in descriptions:
        raise RuntimeError(
            f"Band {name!r} is absent from {dataset.name}."
        )
    return descriptions.index(name) + 1


def read_boolean_band(
    dataset: rasterio.io.DatasetReader,
    name: str,
) -> np.ndarray:
    index = band_index(dataset, name)
    array = dataset.read(index)
    nodata = dataset.nodata
    valid = np.isfinite(array)

    if nodata is not None:
        valid &= array != nodata

    return valid & (array > 0.5)


def read_published_et(
    dataset: rasterio.io.DatasetReader,
) -> np.ndarray:
    index = band_index(dataset, "ET_mm_period")
    array = dataset.read(index)
    nodata = dataset.nodata
    valid = np.isfinite(array)

    if nodata is not None:
        valid &= array != nodata

    return valid


def area_per_pixel_km2(transform: Affine) -> float:
    return (
        abs(
            float(transform.a)
            * float(transform.e)
        )
        / 1_000_000.0
    )


def summarize_model(
    *,
    date_text: str,
    region: str,
    model_name: str,
    mask: np.ndarray,
    pixel_area_km2: float,
    stack_valid: np.ndarray,
    aoa_inside: np.ndarray,
    usable: np.ndarray,
    coarse_eligible: np.ndarray,
    published: np.ndarray,
) -> dict[str, object]:
    basin_pixels = int(mask.sum())
    basin_area = basin_pixels * pixel_area_km2

    stack = mask & stack_valid
    aoa = stack & aoa_inside
    usable_mask = mask & usable
    coarse = mask & coarse_eligible
    published_mask = mask & published

    stack_n = int(stack.sum())
    aoa_n = int(aoa.sum())
    usable_n = int(usable_mask.sum())
    coarse_n = int(coarse.sum())
    published_n = int(published_mask.sum())

    return {
        "date": date_text,
        "region": region,
        "model": model_name,
        "basin_pixels": basin_pixels,
        "basin_area_km2": basin_area,
        "stack_valid_pixels": stack_n,
        "stack_valid_area_km2": stack_n * pixel_area_km2,
        "stack_valid_pct_basin": (
            100.0 * stack_n / basin_pixels
            if basin_pixels
            else np.nan
        ),
        "AOA_pixels": aoa_n,
        "AOA_area_km2": aoa_n * pixel_area_km2,
        "AOA_pct_basin": (
            100.0 * aoa_n / basin_pixels
            if basin_pixels
            else np.nan
        ),
        "AOA_pct_stack_valid": (
            100.0 * aoa_n / stack_n
            if stack_n
            else np.nan
        ),
        "usable_pixels": usable_n,
        "usable_area_km2": usable_n * pixel_area_km2,
        "usable_pct_basin": (
            100.0 * usable_n / basin_pixels
            if basin_pixels
            else np.nan
        ),
        "coarse_eligible_pixels": coarse_n,
        "coarse_eligible_area_km2": coarse_n * pixel_area_km2,
        "coarse_eligible_pct_basin": (
            100.0 * coarse_n / basin_pixels
            if basin_pixels
            else np.nan
        ),
        "published_pixels": published_n,
        "published_area_km2": published_n * pixel_area_km2,
        "published_pct_basin": (
            100.0 * published_n / basin_pixels
            if basin_pixels
            else np.nan
        ),
    }


def transition_counts(
    *,
    date_text: str,
    region: str,
    mask: np.ndarray,
    stable: np.ndarray,
    virtual: np.ndarray,
    pixel_area_km2: float,
    transition_type: str,
) -> dict[str, object]:
    both = mask & stable & virtual
    stable_only = mask & stable & ~virtual
    virtual_only = mask & ~stable & virtual
    neither = mask & ~stable & ~virtual

    basin_n = int(mask.sum())

    def record(name: str, value: np.ndarray):
        n = int(value.sum())
        return {
            f"{name}_pixels": n,
            f"{name}_area_km2": n * pixel_area_km2,
            f"{name}_pct_basin": (
                100.0 * n / basin_n
                if basin_n
                else np.nan
            ),
        }

    output = {
        "date": date_text,
        "region": region,
        "transition_type": transition_type,
        "basin_pixels": basin_n,
        "basin_area_km2": basin_n * pixel_area_km2,
    }
    for name, value in (
        ("both", both),
        ("stable_only", stable_only),
        ("virtual_only", virtual_only),
        ("neither", neither),
    ):
        output.update(record(name, value))

    output["net_virtual_minus_stable_area_km2"] = (
        int(virtual_only.sum())
        - int(stable_only.sum())
    ) * pixel_area_km2

    output["net_virtual_minus_stable_pct_points"] = (
        100.0
        * (
            int(virtual_only.sum())
            - int(stable_only.sum())
        )
        / basin_n
        if basin_n
        else np.nan
    )
    return output


def write_transition_raster(
    path: Path,
    template: rasterio.io.DatasetReader,
    domain: np.ndarray,
    stable: np.ndarray,
    virtual: np.ndarray,
    band_name: str,
) -> None:
    data = np.zeros(
        domain.shape,
        dtype=np.uint8,
    )
    data[domain & ~stable & ~virtual] = 1
    data[domain & stable & ~virtual] = 2
    data[domain & ~stable & virtual] = 3
    data[domain & stable & virtual] = 4

    profile = template.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with rasterio.open(
        path,
        "w",
        **profile,
    ) as destination:
        destination.write(
            data,
            1,
        )
        destination.set_band_description(
            1,
            band_name,
        )
        destination.update_tags(
            transition_codes=json.dumps(
                TRANSITION_DESCRIPTIONS
            )
        )


def main() -> None:
    args = parse_args()
    root = project_root()
    workspace_root = resolve_workspace_root(
        args.workspace_root
    )
    experiment_root = (
        workspace_root
        / "evaluation"
    )
    output_root = (
        experiment_root
        / "basin_coverage_comparison"
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_root = (
        workspace_root
        / "current"
    ).resolve()
    final_root = (
        workspace_root
        / "final"
    ).resolve()
    resolved_output = output_root.resolve()
    if (
        resolved_output == current_root
        or current_root in resolved_output.parents
        or resolved_output == final_root
        or final_root in resolved_output.parents
    ):
        raise RuntimeError(
            "Experimental output resolved inside current/ or final/."
        )

    dates = tuple(
        args.dates
        if args.dates
        else DEFAULT_DATES
    )

    frozen = validate_frozen_v5(workspace_root, check_rasters=False)
    reference_root = resolve_reference_workspace(args.reference_workspace)
    if reference_root == workspace_root:
        raise ValueError("Stable5 reference must be separate from the V5 workspace.")
    virtual_population = pd.read_csv(
        experiment_root / "results" / "virtual10_training_population.csv",
        dtype={"station_id": str},
    )
    frozen_threshold = frozen["AOA_threshold"]
    stable_rasters = {
        date_text: find_stable_scientific_raster(reference_root, date_text)
        for date_text in dates
    }
    virtual_rasters = {
        date_text: find_v5_raster(workspace_root, date_text)
        for date_text in dates
    }

    basin_path = (
        root
        / "data"
        / "boundaries"
        / "fundacion_basin.geojson"
    )
    basin = load_union_geojson(
        basin_path
    )

    model_rows = []
    comparison_rows = []
    publication_transition_rows = []
    aoa_transition_rows = []

    transition_raster_root = (
        output_root
        / "rasters"
    )
    transition_raster_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for date_text in dates:
        stable_path = stable_rasters[
            date_text
        ]
        virtual_path = virtual_rasters[
            date_text
        ]

        print()
        print("=" * 104)
        print(f"SUMMARIZING: {date_text}")
        print("=" * 104)
        print("Stable:", stable_path)
        print("Virtual:", virtual_path)

        with rasterio.open(
            stable_path
        ) as stable_ds, rasterio.open(
            virtual_path
        ) as virtual_ds:
            ensure_same_grid(
                stable_ds,
                virtual_ds,
            )
            if tuple(
                stable_ds.descriptions
            ) != tuple(OUTPUT_BANDS):
                raise RuntimeError(
                    "Stable band contract differs from accepted exact-overlap output."
                )
            if tuple(
                virtual_ds.descriptions
            ) != tuple(OUTPUT_BANDS):
                raise RuntimeError(
                    "Virtual band contract differs from accepted exact-overlap output."
                )

            masks, centroid_x = basin_masks(
                basin,
                stable_ds,
            )
            pixel_area = area_per_pixel_km2(
                stable_ds.transform
            )

            fields = {}
            for model_name, dataset in (
                ("Stable5", stable_ds),
                ("Virtual10", virtual_ds),
            ):
                fields[model_name] = {
                    "stack_valid": read_boolean_band(
                        dataset,
                        "stack_valid",
                    ),
                    "aoa_inside": read_boolean_band(
                        dataset,
                        "AOA_inside",
                    ),
                    "usable": read_boolean_band(
                        dataset,
                        "usable",
                    ),
                    "coarse_eligible": read_boolean_band(
                        dataset,
                        "coarse_eligible",
                    ),
                    "published": read_published_et(
                        dataset
                    ),
                }

            for region, mask in masks.items():
                for model_name in (
                    "Stable5",
                    "Virtual10",
                ):
                    model_rows.append(
                        summarize_model(
                            date_text=date_text,
                            region=region,
                            model_name=model_name,
                            mask=mask,
                            pixel_area_km2=pixel_area,
                            **fields[model_name],
                        )
                    )

                stable_aoa = (
                    fields["Stable5"][
                        "stack_valid"
                    ]
                    & fields["Stable5"][
                        "aoa_inside"
                    ]
                )
                virtual_aoa_mask = (
                    fields["Virtual10"][
                        "stack_valid"
                    ]
                    & fields["Virtual10"][
                        "aoa_inside"
                    ]
                )

                publication_transition_rows.append(
                    transition_counts(
                        date_text=date_text,
                        region=region,
                        mask=mask,
                        stable=fields["Stable5"][
                            "published"
                        ],
                        virtual=fields["Virtual10"][
                            "published"
                        ],
                        pixel_area_km2=pixel_area,
                        transition_type="published_ET",
                    )
                )
                aoa_transition_rows.append(
                    transition_counts(
                        date_text=date_text,
                        region=region,
                        mask=mask,
                        stable=stable_aoa,
                        virtual=virtual_aoa_mask,
                        pixel_area_km2=pixel_area,
                        transition_type="AOA_with_stack_valid",
                    )
                )

            full_mask = masks[
                "full_basin"
            ]
            write_transition_raster(
                transition_raster_root
                / (
                    "publication_transition_"
                    + date_text
                    + ".tif"
                ),
                stable_ds,
                full_mask,
                fields["Stable5"][
                    "published"
                ],
                fields["Virtual10"][
                    "published"
                ],
                "publication_transition",
            )

            stable_aoa = (
                fields["Stable5"][
                    "stack_valid"
                ]
                & fields["Stable5"][
                    "aoa_inside"
                ]
            )
            virtual_aoa_mask = (
                fields["Virtual10"][
                    "stack_valid"
                ]
                & fields["Virtual10"][
                    "aoa_inside"
                ]
            )
            write_transition_raster(
                transition_raster_root
                / (
                    "aoa_transition_"
                    + date_text
                    + ".tif"
                ),
                stable_ds,
                full_mask,
                stable_aoa,
                virtual_aoa_mask,
                "AOA_transition",
            )

            comparison_rows.append(
                {
                    "date": date_text,
                    "basin_centroid_x_m": centroid_x,
                    "stable_raster": str(
                        stable_path
                    ),
                    "virtual_raster": str(
                        virtual_path
                    ),
                }
            )

    coverage = pd.DataFrame(
        model_rows
    )
    publication_transitions = pd.DataFrame(
        publication_transition_rows
    )
    aoa_transitions = pd.DataFrame(
        aoa_transition_rows
    )

    stable = coverage.loc[
        coverage["model"].eq("Stable5")
    ].copy()
    virtual = coverage.loc[
        coverage["model"].eq("Virtual10")
    ].copy()

    key = ["date", "region"]
    pair = stable.merge(
        virtual,
        on=key,
        suffixes=("_Stable5", "_Virtual10"),
        validate="one_to_one",
    )

    comparison = pair[key].copy()
    comparison[
        "AOA_pct_stack_valid_Stable5"
    ] = pair[
        "AOA_pct_stack_valid_Stable5"
    ]
    comparison[
        "AOA_pct_stack_valid_Virtual10"
    ] = pair[
        "AOA_pct_stack_valid_Virtual10"
    ]
    comparison[
        "AOA_delta_percentage_points"
    ] = (
        pair[
            "AOA_pct_stack_valid_Virtual10"
        ]
        - pair[
            "AOA_pct_stack_valid_Stable5"
        ]
    )
    comparison[
        "published_area_km2_Stable5"
    ] = pair[
        "published_area_km2_Stable5"
    ]
    comparison[
        "published_area_km2_Virtual10"
    ] = pair[
        "published_area_km2_Virtual10"
    ]
    comparison[
        "published_area_delta_km2"
    ] = (
        pair[
            "published_area_km2_Virtual10"
        ]
        - pair[
            "published_area_km2_Stable5"
        ]
    )
    comparison[
        "published_pct_basin_Stable5"
    ] = pair[
        "published_pct_basin_Stable5"
    ]
    comparison[
        "published_pct_basin_Virtual10"
    ] = pair[
        "published_pct_basin_Virtual10"
    ]
    comparison[
        "published_delta_percentage_points"
    ] = (
        pair[
            "published_pct_basin_Virtual10"
        ]
        - pair[
            "published_pct_basin_Stable5"
        ]
    )

    comparison = comparison.merge(
        publication_transitions[
            [
                "date",
                "region",
                "both_area_km2",
                "stable_only_area_km2",
                "virtual_only_area_km2",
                "neither_area_km2",
                "net_virtual_minus_stable_area_km2",
            ]
        ],
        on=key,
        how="left",
        validate="one_to_one",
    )

    coverage_path = (
        output_root
        / "coverage_by_model.csv"
    )
    comparison_path = (
        output_root
        / "coverage_comparison.csv"
    )
    publication_transition_path = (
        output_root
        / "publication_transitions.csv"
    )
    aoa_transition_path = (
        output_root
        / "aoa_transitions.csv"
    )
    metadata_path = (
        output_root
        / "metadata.json"
    )

    coverage.to_csv(
        coverage_path,
        index=False,
    )
    comparison.to_csv(
        comparison_path,
        index=False,
    )
    publication_transitions.to_csv(
        publication_transition_path,
        index=False,
    )
    aoa_transitions.to_csv(
        aoa_transition_path,
        index=False,
    )

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "dates": list(dates),
        "virtual10_training_rows": int(
            len(virtual_population)
        ),
        "virtual10_supports": int(
            virtual_population[
                "station_id"
            ].nunique()
        ),
        "virtual10_spatial_blocks": int(
            virtual_population[
                "spatial_block"
            ].nunique()
        ),
        "virtual10_AOA_threshold": float(
            frozen_threshold
        ),
        "primary_domain": "full_basin",
        "secondary_domain": (
            "eastern_upstream_diagnostic_half: "
            "basin x >= projected basin centroid x; diagnostic only, "
            "not a hydrologically delineated upper sub-basin"
        ),
        "stable_current_modified": False,
        "stable_final_modified": False,
        "google_drive_used": False,
        "transition_codes": TRANSITION_DESCRIPTIONS,
    }
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print("COVERAGE COMPARISON")
    print("=" * 104)
    display_columns = [
        "date",
        "region",
        "AOA_pct_stack_valid_Stable5",
        "AOA_pct_stack_valid_Virtual10",
        "AOA_delta_percentage_points",
        "published_area_km2_Stable5",
        "published_area_km2_Virtual10",
        "published_area_delta_km2",
        "virtual_only_area_km2",
        "stable_only_area_km2",
    ]
    print(
        comparison[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print()
    print("Saved:")
    for path in (
        coverage_path,
        comparison_path,
        publication_transition_path,
        aoa_transition_path,
        metadata_path,
    ):
        print(" -", path)
    print(
        " -",
        transition_raster_root,
    )
    print()
    print("Stable current/ modified: NO")
    print("Stable final/ modified: NO")
    print("Production model replaced: NO")


if __name__ == "__main__":
    main()
