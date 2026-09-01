"""Guards for the focused HLS Albedo/FVC export."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_hls_albedo_fvc", ROOT / "scripts" / "export_hls_albedo_fvc.py"
)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class HlsAlbedoFvcExportTests(unittest.TestCase):
    def test_context_is_fixed(self):
        self.assertEqual(EXPORT.EXPECTED_CONTEXT, ("2020-01-01", "2025-01-01", "2020_2024"))

    def test_only_requested_predictors_are_exported(self):
        self.assertEqual(EXPORT.PREDICTORS, ("Albedo", "FVC"))
        predictor_columns = [name for name in EXPORT.EXPORT_SELECTORS if name.endswith("_mean")]
        self.assertEqual(predictor_columns, ["hls_Albedo_mean", "hls_FVC_mean"])

    def test_dry_run_does_not_initialize_earth_engine(self):
        result = EXPORT.main([
            "--start-date", "2020-01-01", "--end-date-exclusive", "2025-01-01",
            "--period-label", "2020_2024",
        ])
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
