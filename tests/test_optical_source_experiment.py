import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import pandas as pd

import et_downscaling.optical_source_experiment as experiment


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "reproducibility" / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


export_script = load_script("export_optical_source_experiment.py")
build_script = load_script("build_optical_source_populations.py")


class OpticalSourceExperimentTests(unittest.TestCase):
    def test_context_and_namespace_are_isolated(self):
        experiment.validate_context("2020-01-01", "2025-01-01", "2020_2024")
        self.assertEqual(export_script.output_root("2020_2024").name, "optical_source_experiment")
        with self.assertRaises(ValueError):
            experiment.validate_context("2021-01-01", "2024-01-01", "2021_2023")

    def test_exact_common_predictors(self):
        self.assertEqual(experiment.COMMON_PREDICTORS, (
            "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
            "NDVI", "EVI", "SAVI", "NDWI", "NDMI",
        ))

    def test_hls_is_not_scaled_again(self):
        source = inspect.getsource(experiment)
        self.assertNotIn("multiply(0.0001)", source)
        config = experiment.experiment_configuration()
        self.assertTrue(config["hls_values_already_scaled_by_earth_engine"])
        self.assertFalse(config["additional_hls_reflectance_scaling_applied"])

    def test_no_training_or_aoa(self):
        config = experiment.experiment_configuration()
        self.assertFalse(config["training_performed"])
        self.assertFalse(config["aoa_di_performed"])
        source = inspect.getsource(experiment)
        self.assertNotIn("RandomForest", source)

    def test_expected_rows_and_requests(self):
        self.assertEqual(experiment.expected_rows(), 1150)
        self.assertEqual(len(export_script.annual_era5_windows()), 5)

    def test_fold_assignments_are_grouped_and_paired(self):
        rows = []
        for station, block in (("A", "x"), ("B", "y")):
            for year in (2020, 2021):
                rows.append({
                    "station_id": station, "period_start": pd.Timestamp(f"{year}-01-01"),
                    "spatial_block": block, "year": year,
                })
        population = pd.DataFrame(rows)
        assignments, definitions = build_script.build_fold_tables({90: population})
        self.assertEqual(len(assignments), 8)
        self.assertEqual(set(assignments.split_type), {"spatial", "temporal"})
        self.assertEqual(len(definitions), 4)


if __name__ == "__main__":
    unittest.main()
