import importlib.util
import unittest
from pathlib import Path


def load_script():
    path = Path(__file__).resolve().parents[1] / "reproducibility/scripts/export_landsat_lst_predictor.py"
    spec = importlib.util.spec_from_file_location("lst_predictor_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORT = load_script()


class LandsatLSTPredictorExportTests(unittest.TestCase):
    def test_approved_scope_and_support(self):
        manifest = EXPORT.method_manifest()
        self.assertEqual(manifest["view"], "L8_L9_COMBINED")
        self.assertEqual(manifest["predictor"], "LST_parent_mean_K")
        self.assertEqual(manifest["native_thermal_support_m_approx"], 100)
        self.assertEqual(manifest["distributed_grid_m"], 30)
        self.assertEqual(manifest["fine_grid_m"], 20)
        self.assertEqual(manifest["resampling_method"], "bilinear")
        self.assertEqual(
            manifest["status"], "APPROVED_PREDICTOR_LADDER_EXTRACTION"
        )
        self.assertFalse(manifest["historical_dn_ge_293_filter_used"])

    def test_dry_plan_does_not_execute(self):
        self.assertEqual(EXPORT.main([]), 0)
        with self.assertRaises(ValueError):
            EXPORT.main(["--execute"])


if __name__ == "__main__":
    unittest.main()
