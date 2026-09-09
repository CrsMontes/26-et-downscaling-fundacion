"""Summarize preserved V5 results and existing explicit Stable5 comparisons.

This command reads local results only; it does not retrain or query Earth Engine.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from et_downscaling.virtual_station import (
    resolve_virtual_workspace, resolve_reference_workspace, resolve_reference_run,
)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=None)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_workspace_root(value: str | None) -> Path:
    return resolve_virtual_workspace(value, project_root())



def extract_internal_metrics(path: Path, design_label: str) -> list[dict]:
    if not path.is_file():
        return []

    data = pd.read_csv(path)
    rows = []

    # Stable rows are included in each experiment's comparison file.
    stable_spatial = data.loc[
        (data["training_design"] == "stable_5_supports")
        & (data["evaluation"] == "spatial_block_CV_internal")
    ]
    stable_temporal = data.loc[
        (data["training_design"] == "stable_5_supports")
        & (data["evaluation"] == "leave_one_year_out_internal")
    ]

    if not stable_spatial.empty and not stable_temporal.empty:
        rows.append(
            {
                "design": "Stable5",
                "status": "frozen_reference",
                "spatial_R2": float(stable_spatial.iloc[0]["R2"]),
                "spatial_RMSE": float(stable_spatial.iloc[0]["RMSE"]),
                "spatial_MAE": float(stable_spatial.iloc[0]["MAE"]),
                "spatial_KGE": float(stable_spatial.iloc[0]["KGE"]),
                "temporal_R2": float(stable_temporal.iloc[0]["R2"]),
                "temporal_RMSE": float(stable_temporal.iloc[0]["RMSE"]),
                "temporal_MAE": float(stable_temporal.iloc[0]["MAE"]),
                "temporal_KGE": float(stable_temporal.iloc[0]["KGE"]),
            }
        )

    virtual_spatial = data.loc[
        (data["training_design"] == "virtual_10_supports")
        & (data["evaluation"] == "spatial_block_CV_internal")
    ]
    virtual_temporal = data.loc[
        (data["training_design"] == "virtual_10_supports")
        & (data["evaluation"] == "leave_one_year_out_internal")
    ]

    if not virtual_spatial.empty and not virtual_temporal.empty:
        rows.append(
            {
                "design": design_label,
                "status": "frozen_v5",
                "spatial_R2": float(virtual_spatial.iloc[0]["R2"]),
                "spatial_RMSE": float(virtual_spatial.iloc[0]["RMSE"]),
                "spatial_MAE": float(virtual_spatial.iloc[0]["MAE"]),
                "spatial_KGE": float(virtual_spatial.iloc[0]["KGE"]),
                "temporal_R2": float(virtual_temporal.iloc[0]["R2"]),
                "temporal_RMSE": float(virtual_temporal.iloc[0]["RMSE"]),
                "temporal_MAE": float(virtual_temporal.iloc[0]["MAE"]),
                "temporal_KGE": float(virtual_temporal.iloc[0]["KGE"]),
            }
        )

    return rows


def field_primary_rows(
    path: Path,
    design_label: str,
) -> list[dict]:
    if not path.is_file():
        return []

    data = pd.read_csv(path)
    rows = []

    scenarios = {
        "all_with_AOA": (
            "primary_ge5_of_8__matched_all_with_AOA"
        ),
        "all_without_AOA": (
            "primary_ge5_of_8__matched_all_without_AOA"
        ),
        "fixed_Kc_with_AOA": (
            "primary_ge5_of_8__matched_fixed_Kc_with_AOA"
        ),
        "fixed_Kc_without_AOA": (
            "primary_ge5_of_8__matched_fixed_Kc_without_AOA"
        ),
    }

    for short_name, comparison in scenarios.items():
        subset = data.loc[
            data["comparison"].eq(comparison)
        ]
        if subset.empty:
            continue

        for model in (
            "MODIS_parent",
            "Stable5_Ridge25",
            "Virtual10_Ridge25",
        ):
            row = subset.loc[
                subset["model"].eq(model)
            ]
            if row.empty:
                continue

            label = model
            if model == "Virtual10_Ridge25":
                label = design_label

            item = row.iloc[0]
            rows.append(
                {
                    "source_experiment": design_label,
                    "scenario": short_name,
                    "model": label,
                    "n": int(item["n"]),
                    "R2": float(item["R2"]),
                    "RMSE": float(item["RMSE"]),
                    "MAE": float(item["MAE"]),
                    "BIAS": float(item["BIAS"]),
                    "r": float(item["r"]),
                    "KGE": float(item["KGE"]),
                }
            )

    return rows


def main() -> None:
    args = parse_args()
    workspace_root = resolve_workspace_root(
        args.workspace_root
    )

    evaluation_root = (
        workspace_root
        / "evaluation"
    )
    output_root = evaluation_root / "summary"
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    v5_metrics = (
        evaluation_root
        / "results"
        / "virtual10_vs_stable_metrics.csv"
    )
    if not v5_metrics.is_file():
        v5_metrics = evaluation_root / "results" / "virtual10_metrics.csv"

    comparison_rows = []
    comparison_rows.extend(
        extract_internal_metrics(
            v5_metrics,
            "V5_Basin10",
        )
    )

    internal = pd.DataFrame(
        comparison_rows
    )
    if not internal.empty:
        internal = (
            internal
            .drop_duplicates(
                subset=["design"],
                keep="first",
            )
            .reset_index(drop=True)
        )
        internal.to_csv(
            output_root
            / "internal_model_comparison.csv",
            index=False,
        )

    field_rows = []
    field_rows.extend(
        field_primary_rows(
            evaluation_root
            / "field_comparison"
            / "virtual10_vs_stable_field_metrics.csv",
            "V5_Basin10",
        )
    )
    field = pd.DataFrame(
        field_rows
    )
    if not field.empty:
        field.to_csv(
            output_root
            / "field_model_comparison.csv",
            index=False,
        )

    persistence_path = (
        evaluation_root
        / "results"
        / "persistence_baselines.csv"
    )
    persistence = (
        pd.read_csv(persistence_path)
        if persistence_path.is_file()
        else pd.DataFrame()
    )
    if not persistence.empty:
        # Old labels in frozen CSVs are preserved on disk and normalized for display.
        persistence["population"] = persistence["population"].replace({"v4_basin10": "v5_basin10"})
        persistence.to_csv(
            output_root
            / "persistence_comparison.csv",
            index=False,
        )

    coverage_path = (
        evaluation_root
        / "basin_coverage_comparison"
        / "coverage_comparison.csv"
    )
    coverage = (
        pd.read_csv(coverage_path)
        if coverage_path.is_file()
        else pd.DataFrame()
    )
    if not coverage.empty:
        coverage.to_csv(
            output_root
            / "basin_coverage_comparison.csv",
            index=False,
        )

    selection_metadata_path = (
        workspace_root / "training"
        / "selection"
        / "selection_metadata.json"
    )
    selection_metadata = (
        json.loads(
            selection_metadata_path.read_text(
                encoding="utf-8"
            )
        )
        if selection_metadata_path.is_file()
        else {}
    )

    lines = [
        "=" * 100,
        "ET FUNDACION - V5 SCIENTIFIC SUMMARY",
        "=" * 100,
        "",
        "Design status",
        "-------------",
        "Stable5: frozen comparison reference, when recorded in the input results.",
        (
            "V5_Basin10: frozen whole-basin sequential-random GE90 "
            "design; operational Virtual Station."
        ),
        "",
        "V5 selection",
        "------------",
        (
            f"Seed: {selection_metadata.get('seed', 'NA')} | "
            f"supports: {selection_metadata.get('n_supports', 'NA')} | "
            f"candidates examined: "
            f"{selection_metadata.get('candidates_examined_until_completion', 'NA')}"
        ),
        (
            "GE90 rule: "
            + str(
                selection_metadata.get(
                    "ge90_definition",
                    "NA",
                )
            )
        ),
        (
            "Minimum GE90 per year: "
            + str(
                selection_metadata.get(
                    "min_ge90_per_year",
                    "NA",
                )
            )
        ),
        "",
    ]

    if not internal.empty:
        lines.extend(
            [
                "Internal model / transfer context",
                "---------------------------------",
                internal.to_string(index=False),
                "",
            ]
        )

    if not persistence.empty:
        lines.extend(
            [
                "Persistence baselines",
                "---------------------",
                persistence.to_string(index=False),
                "",
            ]
        )

    if not field.empty:
        lines.extend(
            [
                "Primary field-proxy comparisons (>=5/8 valid days)",
                "-------------------------------------------------",
                field.to_string(index=False),
                "",
            ]
        )

    if not coverage.empty:
        lines.extend(
            [
                "Basin coverage",
                "--------------",
                coverage.to_string(index=False),
                "",
            ]
        )

    lines.extend(
        [
            "Interpretation guardrails",
            "-------------------------",
            (
                "- Internal CV and reciprocal transfer use the MODIS-derived "
                "Kc target; they are not independent 20 m ET validation."
            ),
            (
                "- Field comparison remains a field-derived ET proxy comparison, "
                "not direct independent 20 m ET validation."
            ),
            (
                "- Production Stable5 is not replaced automatically by this "
                "summary command."
            ),
        ]
    )

    report_path = (
        output_root
        / "summary_report.txt"
    )
    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print(
        report_path.read_text(
            encoding="utf-8"
        )
    )
    print()
    print("Summary directory:", output_root)


if __name__ == "__main__":
    main()
