"""Preregistration guards for coverage-threshold sensitivity."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "evaluate_coverage_threshold_sensitivity",
    PROJECT_ROOT / "scripts" / "evaluate_coverage_threshold_sensitivity.py",
)
SENSITIVITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SENSITIVITY)


class CoverageThresholdSensitivityTests(unittest.TestCase):
    def test_exact_thresholds(self):
        self.assertEqual(SENSITIVITY.THRESHOLDS, (80, 90, 99))

    def test_exact_candidate_count_and_ridge_scope(self):
        self.assertEqual(len(SENSITIVITY.CANDIDATES), 11)
        ridge_candidates = [
            candidate for candidate in SENSITIVITY.CANDIDATES
            if candidate["algorithm"] == "ridge"
        ]
        self.assertEqual(len(ridge_candidates), 2)
        self.assertTrue(all(candidate["source"] == "S2" for candidate in ridge_candidates))

    def test_exact_feature_counts(self):
        self.assertEqual(len(SENSITIVITY.FEATURES["S2"]["base"]), 16)
        self.assertEqual(len(SENSITIVITY.FEATURES["S2"]["seasonality"]), 20)
        self.assertEqual(len(SENSITIVITY.FEATURES["S2"]["rich7"]), 23)
        self.assertEqual(len(SENSITIVITY.FEATURES["HLS"]["base"]), 16)
        self.assertEqual(len(SENSITIVITY.FEATURES["HLS"]["seasonality"]), 20)

    def test_forbidden_features_are_excluded(self):
        all_features = {
            feature
            for configurations in SENSITIVITY.FEATURES.values()
            for features in configurations.values()
            for feature in features
        }
        for token in (
            "Precip", "VV", "VH", "LST", "elevation", "slope", "aspect",
            "ETo", "ETr", "Kc_previous",
        ):
            self.assertFalse(any(token in feature for feature in all_features))

    def test_comparisons_are_preregistered(self):
        comparisons = SENSITIVITY.comparison_definitions()
        seasonality = [row for row in comparisons if row["comparison"] == "seasonality_vs_base"]
        rich = [row for row in comparisons if row["comparison"] == "rich7_vs_rf_base"]
        self.assertEqual(len(seasonality), 5)
        self.assertEqual(len(rich), 1)
        self.assertEqual(rich[0]["algorithm"], "random_forest")


if __name__ == "__main__":
    unittest.main()
