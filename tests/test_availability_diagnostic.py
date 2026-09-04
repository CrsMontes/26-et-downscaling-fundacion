import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

from et_downscaling.availability_diagnostic import (
    annual_partitions,
    scientific_configuration,
    split_partition,
    validate_period,
)


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "reproducibility" / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


export_script = load_script("export_availability_diagnostic.py")
summary_script = load_script("summarize_availability_diagnostic.py")


class PeriodTests(unittest.TestCase):
    def test_approved_period_has_five_annual_partitions(self):
        self.assertEqual(
            annual_partitions("2020-01-01", "2025-01-01"),
            [(f"{year}-01-01", f"{year + 1}-01-01") for year in range(2020, 2025)],
        )

    def test_invalid_period_rejected(self):
        with self.assertRaises(ValueError):
            validate_period("2025-01-01", "2025-01-01")

    def test_adaptive_split_is_contiguous(self):
        left, right = split_partition("2020-01-01", "2021-01-01")
        self.assertEqual(left[0], "2020-01-01")
        self.assertEqual(left[1], right[0])
        self.assertEqual(right[1], "2021-01-01")


class GuardTests(unittest.TestCase):
    def test_only_approved_namespace_is_allowed(self):
        with self.assertRaises(ValueError):
            export_script.validate_approved_context(
                "2021-01-01", "2024-01-01", "2021_2023"
            )
        path = export_script.validate_approved_context(*export_script.EXPECTED_PERIOD)
        self.assertEqual(path.name, "availability")
        self.assertEqual(path.parent.name, "2020_2024")

    def test_default_invocation_does_not_execute_ee(self):
        args = export_script.parse_arguments([
            "--start-date", "2020-01-01", "--end-date-exclusive", "2025-01-01",
            "--period-label", "2020_2024",
        ])
        self.assertFalse(args.execute)

    def test_configuration_has_no_s1_geometry_filter_or_training(self):
        config = scientific_configuration()
        self.assertIsNone(config["sentinel1"]["pass_filter"])
        self.assertIsNone(config["sentinel1"]["relative_orbit_filter"])
        self.assertFalse(config["training_performed"])


class LocalSummaryTests(unittest.TestCase):
    def test_optical_long_preserves_all_four_sources(self):
        s2 = pd.DataFrame({
            "station_id": ["a"], "period_start": ["2020-01-01"],
            "continuous_valid_coverage_pct": [90.0],
        })
        hls = pd.DataFrame({
            "station_id": ["a"], "period_start": ["2020-01-01"],
            "s30_continuous_valid_coverage_pct": [80.0],
            "l30_continuous_valid_coverage_pct": [70.0],
            "combined_continuous_valid_coverage_pct": [99.0],
        })
        result = summary_script.optical_long(s2, hls)
        self.assertEqual(
            set(result["source"]), {"S2", "HLS_S30", "HLS_L30", "HLS_COMBINED"}
        )

    def test_intersections_are_geometry_specific(self):
        modis = pd.DataFrame({
            "station_id": ["a"], "period_start": ["2020-01-01"], "modis_good": [1]
        })
        optical = pd.DataFrame([
            {"station_id": "a", "period_start": "2020-01-01", "source": source,
             "continuous_valid_coverage_pct": 100.0}
            for source in ("S2", "HLS_S30", "HLS_L30", "HLS_COMBINED")
        ])
        s1 = pd.DataFrame({
            "station_id": ["a", "a"], "period_start": ["2020-01-01"] * 2,
            "pass": ["ASCENDING", "DESCENDING"], "relative_orbit": [77, 142],
            "has_valid_vv_vh_coverage": [1, 0],
        })
        result = summary_script.intersection_rows(modis, optical, s1)
        base = result[result["intersection"].eq("target ∩ S1")]
        counts = dict(zip(base["relative_orbit"], base["count"]))
        self.assertEqual(counts, {77: 1, 142: 0})


if __name__ == "__main__":
    unittest.main()
