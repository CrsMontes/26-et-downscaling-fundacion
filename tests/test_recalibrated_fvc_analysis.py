"""Local QA tests for recalibrated FVC outputs."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("analysis", ROOT / "scripts/analyze_recalibrated_fvc_predictors.py")
ANALYSIS = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ANALYSIS)


class RecalibratedFvcAnalysisTests(unittest.TestCase):
    def test_completed_sources_have_exact_universe(self):
        calibration = ANALYSIS.exporter.freeze_calibrations()
        for source in ANALYSIS.exporter.SOURCES:
            table = ANALYSIS.load_source(source, calibration)
            self.assertEqual(len(table), 1150)
            self.assertFalse(table.duplicated(["station_id", "period_start"]).any())

    def test_comparison_metrics_are_paired(self):
        calibration = ANALYSIS.exporter.freeze_calibrations()
        table = ANALYSIS.load_source("S2", calibration)
        metrics = ANALYSIS.comparison_metrics(table)
        self.assertEqual(metrics["n_paired"], table.fvc_historical_mean.notna().sum())
        self.assertGreaterEqual(metrics["MAE"], 0)
