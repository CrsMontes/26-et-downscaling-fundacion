import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import et_downscaling.thermal_availability as thermal


ROOT = Path(__file__).resolve().parents[1]


def load_script(filename):
    path = ROOT / "reproducibility" / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


export_script = load_script("export_thermal_availability.py")


class ThermalConfigurationTests(unittest.TestCase):
    def test_period_and_namespace_are_isolated(self):
        path = export_script.validate_context(*export_script.EXPECTED_CONTEXT)
        self.assertEqual(path.name, "thermal_availability")
        self.assertEqual(path.parent.name, "2020_2024")
        with self.assertRaises(ValueError):
            export_script.validate_context("2021-01-01", "2024-01-01", "2021_2023")

    def test_expected_rows_per_view(self):
        self.assertEqual(export_script.expected_rows(), 1150)

    def test_l8_only_columns_retain_zero_auditable_l9_slot(self):
        columns = set(thermal.EXPORT_SELECTORS)
        self.assertIn("l8_only_l8_products", columns)
        self.assertIn("l8_only_l9_products", columns)
        source = inspect.getsource(thermal.build_thermal_availability)
        self.assertIn('filter(ee.Filter.eq("sensor", "L8"))', source)

    def test_combined_retains_sensor_provenance(self):
        columns = set(thermal.EXPORT_SELECTORS)
        for column in (
            "l8_l9_combined_l8_products", "l8_l9_combined_l9_products",
            "l8_l9_combined_sensors_present", "l8_l9_combined_selected_l8_area_m2",
            "l8_l9_combined_selected_l9_area_m2",
        ):
            self.assertIn(column, columns)

    def test_no_20m_reprojection(self):
        source = inspect.getsource(thermal)
        self.assertNotIn(".reproject(", source)
        config = thermal.configuration_manifest()
        self.assertFalse(config["reprojected_to_20m"])
        self.assertEqual(config["distributed_grid_m"], 30)
        self.assertEqual(config["native_thermal_support_m_approx"], 100)

    def test_acquisition_and_valid_lst_are_distinct(self):
        columns = set(thermal.EXPORT_SELECTORS)
        self.assertIn("l8_only_acquisition_present", columns)
        self.assertIn("l8_only_any_valid_lst", columns)
        self.assertNotEqual("l8_only_acquisition_present", "l8_only_any_valid_lst")

    def test_continuous_coverage_and_flags_exist(self):
        columns = set(thermal.EXPORT_SELECTORS)
        for prefix in thermal.VIEWS:
            self.assertIn(f"{prefix}_valid_coverage_pct", columns)
            for threshold in (80, 90, 99):
                self.assertIn(f"{prefix}_ge_{threshold}", columns)

    def test_historical_dn_is_sensitivity_not_primary_truth(self):
        config = thermal.configuration_manifest()
        self.assertIsNone(config["primary_dn_minimum"])
        self.assertEqual(config["historical_dn_sensitivity_minimum"], 293)
        self.assertFalse(config["historical_dn_minimum_is_final_methodology"])

    def test_no_training(self):
        self.assertFalse(thermal.configuration_manifest()["training_performed"])
        args = export_script.parse_arguments([
            "--start-date", "2020-01-01", "--end-date-exclusive", "2025-01-01",
            "--period-label", "2020_2024",
        ])
        self.assertFalse(args.execute)



if __name__ == "__main__":
    unittest.main()
