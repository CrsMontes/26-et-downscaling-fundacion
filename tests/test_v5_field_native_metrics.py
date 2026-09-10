from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "evaluate_v5_field_proxy.py"

    spec = importlib.util.spec_from_file_location(
        "evaluate_v5_field_proxy_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeFieldModule:
    @staticmethod
    def calculate_metrics(observed, predicted):
        return {
            "n": int(len(observed)),
            "R2": 0.0,
            "RMSE": 0.0,
            "MAE": 0.0,
            "BIAS": 0.0,
            "r": 0.0,
            "KGE": 0.0,
        }


def test_virtual_native_does_not_depend_on_stable5_availability():
    module = load_module()

    table = pd.DataFrame(
        {
            "ET_field_proxy_mm_period": [10.0, 11.0, 12.0],
            "ET_MODIS_parent_mm_period": [10.5, 11.5, 12.5],
            "ET_virtual10_with_AOA_mm_period": [10.2, 11.2, 12.2],
            "ET_Ridge_with_AOA_mm_period": [10.1, np.nan, 12.1],
        }
    )

    rows = module.metric_rows_for_subset(
        FakeFieldModule,
        table,
        comparison_label="virtual_native",
        virtual_column="ET_virtual10_with_AOA_mm_period",
        stable_column="ET_Ridge_with_AOA_mm_period",
        include_stable=False,
    )

    assert [row["model"] for row in rows] == [
        "MODIS_parent",
        "Virtual10_Ridge25",
    ]

    assert all(row["n"] == 3 for row in rows)


def test_matched_comparison_requires_common_stable5_rows():
    module = load_module()

    table = pd.DataFrame(
        {
            "ET_field_proxy_mm_period": [10.0, 11.0, 12.0],
            "ET_MODIS_parent_mm_period": [10.5, 11.5, 12.5],
            "ET_virtual10_with_AOA_mm_period": [10.2, 11.2, 12.2],
            "ET_Ridge_with_AOA_mm_period": [10.1, np.nan, 12.1],
        }
    )

    rows = module.metric_rows_for_subset(
        FakeFieldModule,
        table,
        comparison_label="matched",
        virtual_column="ET_virtual10_with_AOA_mm_period",
        stable_column="ET_Ridge_with_AOA_mm_period",
        include_stable=True,
    )

    assert [row["model"] for row in rows] == [
        "MODIS_parent",
        "Stable5_Ridge25",
        "Virtual10_Ridge25",
    ]

    assert all(row["n"] == 2 for row in rows)
