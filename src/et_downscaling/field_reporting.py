"""Pure reporting helpers for the final ETgage comparison.

This module contains no Earth Engine calls and does not alter field inclusion
rules. It only summarizes already-built field-comparison tables.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


ScenarioSpec = tuple[str, str, str, str]


def build_temporal_completeness_sensitivity(
    final_table: pd.DataFrame,
    scenarios: list[ScenarioSpec],
    calculate_metrics: Callable,
) -> pd.DataFrame:
    """Evaluate frozen field scenarios under primary 5/8 and strict 8/8 support."""
    rows = []
    completeness_rules = [
        ("primary_5of8_or_more", 5),
        ("complete_8of8_only", 8),
    ]

    valid_days = pd.to_numeric(
        final_table["n_valid_field_days"],
        errors="coerce",
    )

    for scenario, membership, ridge_column, _description in scenarios:
        for completeness_label, minimum_days in completeness_rules:
            mask = (
                final_table[membership].astype(bool)
                & valid_days.ge(minimum_days)
            )
            subset = final_table.loc[mask].dropna(
                subset=[
                    "ET_field_proxy_mm_period",
                    "ET_MODIS_parent_mm_period",
                    ridge_column,
                ]
            )
            for model_name, prediction in [
                ("MODIS_parent", "ET_MODIS_parent_mm_period"),
                ("Ridge25_exact_overlap_spatial_OOF", ridge_column),
            ]:
                rows.append(
                    {
                        "scenario": scenario,
                        "field_completeness": completeness_label,
                        "minimum_valid_days": minimum_days,
                        "model": model_name,
                        **calculate_metrics(
                            subset["ET_field_proxy_mm_period"],
                            subset[prediction],
                        ),
                    }
                )
    return pd.DataFrame(rows)
