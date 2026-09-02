import importlib.util
import sys
import unittest
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_predictor_availability_ladder.py"
SPEC = importlib.util.spec_from_file_location("predictor_availability_ladder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PredictorAvailabilityLadderTests(unittest.TestCase):
    def test_fixed_candidate_count_and_reference(self):
        self.assertEqual(len(MODULE.BASE20), 20)
        self.assertEqual(len(MODULE.CANDIDATES), 37)
        self.assertEqual(MODULE.EXPECTED_GE90_ROWS, 799)

    def test_fixed_models(self):
        models = MODULE.build_models()
        self.assertIsInstance(models["ridge"], Pipeline)
        self.assertIsInstance(models["ridge"].named_steps["scaler"], StandardScaler)
        self.assertEqual(models["ridge"].named_steps["regressor"].alpha, 1.0)
        forest = models["random_forest"]
        self.assertEqual(forest.n_estimators, 300)
        self.assertEqual(forest.max_features, 0.33)
        self.assertEqual(forest.min_samples_leaf, 3)
        self.assertEqual(forest.random_state, 42)


if __name__ == "__main__":
    unittest.main()
