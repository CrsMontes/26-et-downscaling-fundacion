import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/map_s2_ridge_rf_gate.py"
    spec = importlib.util.spec_from_file_location("cartographic_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = load_script()


class CartographicGateTests(unittest.TestCase):
    def test_frozen_periods_and_features(self):
        self.assertEqual(GATE.PERIODS["extreme_dry"], "2020-03-13")
        self.assertEqual(GATE.PERIODS["extreme_wet"], "2021-11-25")
        self.assertEqual(len(GATE.FEATURES), 20)
        self.assertEqual(len(set(GATE.FEATURES)), 20)

    def test_aoa_weights_are_normalized_for_both_models(self):
        training, _ = GATE.load_training()
        models = GATE.fit_models(training)
        for name, model in models.items():
            weights = GATE.model_weights(name, model)
            self.assertAlmostEqual(float(weights.sum()), 1.0)
            self.assertTrue(np.all(weights >= 0))

    def test_same_di_rule_accepts_training_like_point(self):
        training, _ = GATE.load_training()
        models = GATE.fit_models(training)
        for name, model in models.items():
            spec = GATE.build_aoa_spec(training, name, model)
            di, inside = GATE.calculate_di(training.iloc[[0]], spec)
            self.assertAlmostEqual(float(di[0]), 0.0, places=12)
            self.assertTrue(bool(inside[0]))


if __name__ == "__main__":
    unittest.main()
