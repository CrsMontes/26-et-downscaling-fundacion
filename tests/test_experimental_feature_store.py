"""Guards for the non-destructive experimental feature store."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_experimental_feature_store",
    PROJECT_ROOT / "reproducibility" / "scripts" / "build_experimental_feature_store.py",
)
FEATURE_STORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FEATURE_STORE)


class ExperimentalFeatureStoreTests(unittest.TestCase):
    def test_declared_materialized_predictor_count(self):
        expected = 18 + 11 + 6 + 12 + 4 + 1
        metadata = FEATURE_STORE.predictor_metadata()
        materialized = [
            row for row in metadata
            if row["current_local_file"] and row["training_support"] != "Not materialized"
        ]
        self.assertEqual(len(materialized), expected)

    def test_nonmaterialized_candidates_are_explicit(self):
        metadata = {row["feature_name"]: row for row in FEATURE_STORE.predictor_metadata()}
        for feature in (
            "hls_Albedo_mean",
            "hls_FVC_mean",
            "landsat_l8_only_LST_K",
            "landsat_l8_l9_combined_LST_K",
            "terrain_slope",
            "terrain_aspect",
        ):
            self.assertEqual(metadata[feature]["requires_new_Earth_Engine_processing"], True)

    def test_left_join_preserves_rows_keys_and_missing_values(self):
        store = pd.DataFrame(
            {"station_id": ["A", "B"], "period_start": ["2020-01-01", "2020-01-01"]}
        )
        partial = pd.DataFrame(
            {"station_id": ["A"], "period_start": ["2020-01-01"], "candidate": [1.0]}
        )
        result = FEATURE_STORE.left_join(store, partial, "test join")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[FEATURE_STORE.KEYS].to_dict("records"), store.to_dict("records"))
        self.assertTrue(np.isnan(result.loc[1, "candidate"]))

    def test_harmonics_do_not_drop_rows(self):
        store = pd.DataFrame(
            {"station_id": ["A", "B"], "period_start": ["2020-01-01", "2020-07-01"]}
        )
        result = FEATURE_STORE.add_harmonics(store)
        self.assertEqual(len(result), len(store))
        self.assertTrue(set(FEATURE_STORE.HARMONICS).issubset(result.columns))


if __name__ == "__main__":
    unittest.main()
