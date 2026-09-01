"""Preregistration guards for incremental feature-family screening."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "screen_feature_families",
    PROJECT_ROOT / "scripts" / "screen_feature_families.py",
)
SCREENING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREENING)


class FeatureFamilyScreeningTests(unittest.TestCase):
    def test_exact_feature_counts(self):
        for source, configurations in SCREENING.SOURCE_CONFIGURATIONS.items():
            for configuration, features in configurations.items():
                with self.subTest(source=source, configuration=configuration):
                    self.assertEqual(
                        len(features), SCREENING.EXPECTED_FEATURE_COUNTS[configuration]
                    )
                    self.assertEqual(len(features), len(set(features)))

    def test_exact_algorithms_by_source(self):
        self.assertEqual(
            SCREENING.SOURCE_ALGORITHMS,
            {
                "S2": ["random_forest", "extra_trees", "ridge"],
                "HLS": ["random_forest", "extra_trees"],
            },
        )

    def test_rich_features_are_s2_only(self):
        hls_features = {
            feature
            for features in SCREENING.SOURCE_CONFIGURATIONS["HLS"].values()
            for feature in features
        }
        self.assertTrue(set(SCREENING.S2_RICH).isdisjoint(hls_features))

    def test_excluded_families_do_not_enter_matrices(self):
        all_features = {
            feature
            for configurations in SCREENING.SOURCE_CONFIGURATIONS.values()
            for features in configurations.values()
            for feature in features
        }
        forbidden_tokens = (
            "VV", "VH", "LST", "elevation", "slope", "aspect",
            "ETo", "ETr", "Kc_previous",
        )
        for token in forbidden_tokens:
            self.assertFalse(any(token in feature for feature in all_features))

    def test_base_configurations_match_previous_screening(self):
        self.assertEqual(
            SCREENING.SOURCE_CONFIGURATIONS["S2"]["s2_base"],
            SCREENING.S2_COMMON + SCREENING.ETO_DRIVERS,
        )
        self.assertEqual(
            SCREENING.SOURCE_CONFIGURATIONS["HLS"]["hls_base"],
            SCREENING.HLS_COMMON + SCREENING.ETO_DRIVERS,
        )


if __name__ == "__main__":
    unittest.main()
