"""Audit consistency between frozen V5 GE90 selection and extracted training rows.

This is a local-only QA script. It does not call Earth Engine. It is
read-only by default and writes QA tables only when --write-report is
explicitly supplied.

It verifies, support by support:
1. selected MODIS pixel ID;
2. GE90 period count frozen during sequential selection;
3. period count actually extracted into the V5 satellite master;
4. MODIS pixel ID(s) present in the extracted satellite rows, when available;
5. exact period-start set agreement between the cached selection availability
   check and the extracted satellite master.

The frozen selection is scientifically expected to define the same GE90
support-period set used for training. Any discrepancy must be resolved before
V5 model metrics are treated as final.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from et_downscaling.virtual_station import resolve_virtual_workspace




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace-root",
        default=None,
    )
    parser.add_argument("--write-report", action="store_true", help="Explicitly write QA tables to evaluation/qa.")
    return parser.parse_args()


def find_modis_column(frame: pd.DataFrame) -> str | None:
    candidates = [
        "modis_pixel_id",
        "MODIS_pixel_id",
        "modis_id",
    ]
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def main() -> None:
    args = parse_args()
    workspace_root = resolve_virtual_workspace(args.workspace_root)
    root = (
        workspace_root
        / "training"
    )
    selection_root = root / "selection"

    selected_path = (
        selection_root
        / "selected_supports.csv"
    )
    satellite_path = (
        root
        / "raw"
        / "satellite"
        / "S2"
        / "ET_S2_S1_SATELLITE_FOOTPRINT_2020_2024.csv"
    )

    for path in (selected_path, satellite_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    selected = pd.read_csv(
        selected_path,
        dtype={"virtual_id": str},
    )
    satellite = pd.read_csv(
        satellite_path,
        dtype={"station_id": str},
    )

    if "period_start" not in satellite.columns:
        raise RuntimeError(
            "Satellite master has no period_start column."
        )

    satellite["period_start"] = pd.to_datetime(
        satellite["period_start"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    modis_column = find_modis_column(
        satellite
    )

    output_rows = []
    period_disagreement_rows = []

    for row in selected.itertuples(index=False):
        virtual_id = str(row.virtual_id)
        candidate_order = int(row.candidate_order)
        selected_modis_id = int(row.modis_pixel_id)
        frozen_ge90_total = int(row.ge90_total)

        subset = satellite.loc[
            satellite["station_id"].astype(str).eq(
                virtual_id
            )
        ].copy()

        extracted_periods = set(
            subset["period_start"].astype(str)
        )

        check_candidates = sorted(
            (
                selection_root
                / "availability_checks"
            ).glob(
                f"{candidate_order:05d}_{selected_modis_id}.csv"
            )
        )
        if len(check_candidates) != 1:
            raise RuntimeError(
                f"{virtual_id}: expected one availability check for "
                f"candidate {candidate_order}, MODIS {selected_modis_id}; "
                f"found {len(check_candidates)}."
            )

        check = pd.read_csv(
            check_candidates[0]
        )
        check["period_start"] = pd.to_datetime(
            check["period_start"],
            errors="raise",
        ).dt.strftime("%Y-%m-%d")

        ge90 = pd.to_numeric(
            check["ge90"],
            errors="raise",
        ).astype(int)

        frozen_periods = set(
            check.loc[
                ge90.eq(1),
                "period_start",
            ].astype(str)
        )

        only_frozen = sorted(
            frozen_periods - extracted_periods
        )
        only_extracted = sorted(
            extracted_periods - frozen_periods
        )

        extracted_modis_ids = ""
        modis_match = None
        if modis_column is not None:
            actual_ids = sorted(
                pd.to_numeric(
                    subset[modis_column],
                    errors="coerce",
                )
                .dropna()
                .astype("int64")
                .unique()
                .tolist()
            )
            extracted_modis_ids = ";".join(
                str(value)
                for value in actual_ids
            )
            modis_match = (
                len(actual_ids) == 1
                and actual_ids[0] == selected_modis_id
            )

        output_rows.append(
            {
                "virtual_id": virtual_id,
                "candidate_order": candidate_order,
                "selected_modis_pixel_id": selected_modis_id,
                "frozen_ge90_total_csv": frozen_ge90_total,
                "frozen_ge90_total_check": len(
                    frozen_periods
                ),
                "extracted_training_rows": int(
                    len(subset)
                ),
                "count_difference_extracted_minus_frozen": (
                    int(len(subset))
                    - int(len(frozen_periods))
                ),
                "period_sets_identical": (
                    frozen_periods == extracted_periods
                ),
                "frozen_only_periods": len(
                    only_frozen
                ),
                "extracted_only_periods": len(
                    only_extracted
                ),
                "satellite_modis_pixel_ids": extracted_modis_ids,
                "satellite_modis_matches_selection": modis_match,
            }
        )

        for period in only_frozen:
            period_disagreement_rows.append(
                {
                    "virtual_id": virtual_id,
                    "period_start": period,
                    "status": (
                        "frozen_GE90_but_not_extracted"
                    ),
                }
            )

        for period in only_extracted:
            period_disagreement_rows.append(
                {
                    "virtual_id": virtual_id,
                    "period_start": period,
                    "status": (
                        "extracted_but_not_frozen_GE90"
                    ),
                }
            )

    audit = pd.DataFrame(
        output_rows
    )
    disagreements = pd.DataFrame(
        period_disagreement_rows,
        columns=[
            "virtual_id",
            "period_start",
            "status",
        ],
    )

    output_root = (
        workspace_root / "evaluation"
        / "qa"
    )
    if args.write_report:
        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        audit_path = (
            output_root
            / "selection_vs_training_GE90_audit.csv"
        )
        disagreement_path = (
            output_root
            / "selection_vs_training_period_disagreements.csv"
        )

        audit.to_csv(
            audit_path,
            index=False,
        )
        disagreements.to_csv(
            disagreement_path,
            index=False,
        )

    print("=" * 110)
    print("V5 FROZEN-SELECTION VS TRAINING GE90 AUDIT")
    print("=" * 110)
    print(
        audit.to_string(
            index=False
        )
    )
    print()
    print(
        "Frozen GE90 periods total:",
        int(
            audit[
                "frozen_ge90_total_check"
            ].sum()
        ),
    )
    print(
        "Extracted training rows total:",
        int(
            audit[
                "extracted_training_rows"
            ].sum()
        ),
    )
    print(
        "Supports with identical period sets:",
        int(
            audit[
                "period_sets_identical"
            ].sum()
        ),
        "/",
        len(audit),
    )
    print()
    if args.write_report:
        print("Saved:")
        print(" -", audit_path)
        print(" -", disagreement_path)

    if not audit[
        "period_sets_identical"
    ].all():
        print()
        print(
            "QA RESULT: FAIL. Do not treat V5 model/field metrics as final "
            "until this discrepancy is resolved."
        )
        raise SystemExit(2)

    print()
    print("QA RESULT: PASS.")


if __name__ == "__main__":
    main()
