import importlib.util
import unittest
from pathlib import Path

import pandas as pd


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/compare_s2_ridge_rf_gate.py"
    spec = importlib.util.spec_from_file_location("ridge_rf_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = load_script()


class S2RidgeRFGateTests(unittest.TestCase):
    def test_fixed_scope(self):
        self.assertEqual(GATE.ALGORITHMS, ("ridge", "random_forest"))
        self.assertEqual(GATE.CONFIGURATION, "s2_base_plus_seasonality")
        self.assertEqual(GATE.MANGROVE_STATION, "ST04")

    def test_paired_error_sign(self):
        rows = []
        for algorithm, prediction in (("ridge", 0.9), ("random_forest", 0.7)):
            rows.append({"algorithm": algorithm, "split_type": "spatial", "fold": 1,
                         "station_id": "ST01", "period_start": "2020-01-01",
                         "year": 2020, "spatial_block": "x", "Kc_target": 1.0,
                         "prediction": prediction})
        result = GATE.paired_errors(pd.DataFrame(rows)).iloc[0]
        self.assertAlmostEqual(result.delta_absolute_error_ridge_minus_rf, -0.2)


if __name__ == "__main__":
    unittest.main()
