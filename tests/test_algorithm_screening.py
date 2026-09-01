"""Configuration guards for the exploratory algorithm screening."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screen_optical_algorithms",
    PROJECT_ROOT / "scripts" / "screen_optical_algorithms.py",
)
SCREENING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREENING)


class AlgorithmScreeningTests(unittest.TestCase):
    def test_exact_sources_and_feature_count(self):
        self.assertEqual(set(SCREENING.SOURCE_FEATURES), {"S2", "HLS"})
        for features in SCREENING.SOURCE_FEATURES.values():
            self.assertEqual(len(features), 16)
            self.assertEqual(features[-5:], SCREENING.ETO_DRIVERS)

    def test_exact_algorithm_order(self):
        self.assertEqual(
            SCREENING.ALGORITHM_ORDER,
            [
                "dummy_mean",
                "ridge",
                "random_forest",
                "extra_trees",
                "hist_gradient_boosting",
            ],
        )

    def test_ridge_scaling_is_inside_pipeline(self):
        ridge = SCREENING.build_algorithms()["ridge"]
        self.assertIsInstance(ridge, Pipeline)
        self.assertEqual(list(ridge.named_steps), ["scaler", "regressor"])
        self.assertEqual(ridge.named_steps["regressor"].alpha, 1.0)

    def test_tree_configurations_are_preregistered(self):
        algorithms = SCREENING.build_algorithms()
        extra_trees = algorithms["extra_trees"]
        boosting = algorithms["hist_gradient_boosting"]
        self.assertIsInstance(extra_trees, ExtraTreesRegressor)
        self.assertIsInstance(boosting, HistGradientBoostingRegressor)
        for key, value in SCREENING.EXTRA_TREES_PARAMETERS.items():
            self.assertEqual(extra_trees.get_params()[key], value)
        for key, value in SCREENING.HIST_GRADIENT_BOOSTING_PARAMETERS.items():
            self.assertEqual(boosting.get_params()[key], value)

    def test_fold_definitions_are_fixed(self):
        self.assertEqual(
            SCREENING.EXPECTED_SPATIAL_FOLDS[2]["stations"],
            ["ST02", "ST03"],
        )
        self.assertEqual(
            [row["group"] for row in SCREENING.EXPECTED_TEMPORAL_FOLDS.values()],
            ["2020", "2021", "2022", "2023", "2024"],
        )


if __name__ == "__main__":
    unittest.main()
