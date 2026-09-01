"""Guards for the single-pass recalibrated FVC exporter."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("recalibrated", ROOT / "scripts/export_recalibrated_fvc_predictors.py")
EXPORT = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(EXPORT)


class RecalibratedFvcPredictorExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = EXPORT.freeze_calibrations()

    def test_approved_global_endmembers_are_recalculated(self):
        for source, (low, high, n) in EXPORT.EXPECTED_GLOBAL.items():
            actual = self.manifest["sources"][source]["global_2020_2024"]
            self.assertAlmostEqual(actual["low"], low, places=14)
            self.assertAlmostEqual(actual["high"], high, places=14)
            self.assertEqual(actual["n"], n)

    def test_fold_specific_calibrations_are_explicit(self):
        for source in EXPORT.SOURCES:
            record = self.manifest["sources"][source]
            self.assertEqual(set(record["spatial_training_only"]), {"1", "2", "3", "4"})
            self.assertEqual(set(record["temporal_training_only"]), {str(y) for y in EXPORT.YEARS})
            self.assertEqual(len(EXPORT.calibration_variants(source, self.manifest)), 11)

    def test_hls_albedo_is_in_same_export_schema(self):
        self.assertIn("hls_Albedo_mean", EXPORT.output_columns("HLS", self.manifest))
        self.assertNotIn("hls_Albedo_mean", EXPORT.output_columns("S2", self.manifest))

    def test_dry_run_does_not_initialize_earth_engine(self):
        self.assertEqual(EXPORT.main(["--freeze-only"]), 0)
