"""Local-only tests for the FVC recalibration preflight."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import subprocess
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "preflight_fvc_recalibration", ROOT / "scripts" / "preflight_fvc_recalibration.py"
)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def historical_candidates():
    result = subprocess.run(
        ["git", "show", "8722b8b:outputs/diagnostics/_fvc_endmember_calibration_checkpoint_v2.csv"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return pd.read_csv(io.StringIO(result.stdout), dtype={"station_id": str})


class FvcRecalibrationPreflightTests(unittest.TestCase):
    def test_historical_s2_reproduction(self):
        table = historical_candidates().loc[lambda frame: frame.source.eq("S2")]
        result, eligible = PREFLIGHT.calculate_endmembers(table)
        self.assertEqual(len(eligible), 498)
        self.assertAlmostEqual(result["ndvi_low_endmember"], 0.30906052790151156, places=15)
        self.assertAlmostEqual(result["ndvi_high_endmember"], 0.9240448371180946, places=15)

    def test_historical_uncorrected_hls_is_not_accepted_as_corrected(self):
        table = historical_candidates().loc[lambda frame: frame.source.eq("HLS")]
        result, eligible = PREFLIGHT.calculate_endmembers(table)
        self.assertEqual(len(eligible), 434)
        self.assertAlmostEqual(result["ndvi_low_endmember"], 0.4163919518084051, places=15)
        self.assertNotAlmostEqual(result["ndvi_low_endmember"], 0.411908487478892, places=12)

    def test_quantile_logic_is_float64_linear_and_filters_eligibility(self):
        table = pd.DataFrame({
            "station_id": ["1", "2", "3", "4"],
            "coverage_pct": [80, 100, 79.999, 100],
            "nonwater_pixel_count": [1, 2, 5, 0],
            "NDVI_p05": np.array([0.1, 0.3, 0.9, 0.8], dtype=np.float32),
            "NDVI_p95": np.array([0.7, 0.9, 0.95, 0.99], dtype=np.float32),
        })
        result, eligible = PREFLIGHT.calculate_endmembers(table)
        self.assertEqual(len(eligible), 2)
        self.assertAlmostEqual(result["ndvi_low_endmember"], 0.11)
        self.assertAlmostEqual(result["ndvi_high_endmember"], 0.89)

    def test_date_interval_and_exact_universe(self):
        periods = pd.date_range("2020-01-01", periods=230, freq="8D")
        # MODIS periods restart at each year; use the real local period keys instead.
        local = pd.read_csv(
            ROOT / "outputs/diagnostics/2020_2024/optical_source_experiment/raw/paired_optical_common.csv",
            dtype={"station_id": str},
        )[["station_id", "period_start"]]
        validated = PREFLIGHT.validate_universe(local)
        self.assertEqual(len(validated), 1150)
        self.assertEqual(validated.period_start.nunique(), 230)
        self.assertEqual(len(periods), 230)

    def test_duplicate_key_rejection(self):
        local = pd.read_csv(
            ROOT / "outputs/diagnostics/2020_2024/optical_source_experiment/raw/paired_optical_common.csv",
            dtype={"station_id": str},
        )[["station_id", "period_start"]]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            PREFLIGHT.validate_universe(pd.concat([local, local.iloc[[0]]]))

    def test_chunk_resume_and_manifest_compatibility(self):
        keys = pd.DataFrame({"station_id": ["1"], "period_start": ["2020-01-01"]})
        expected = PREFLIGHT.chunk_manifest(
            "S2", "2020-01-01", "2021-01-01", PREFLIGHT.key_digest(keys), 1
        )
        table = keys.assign(
            number_days=8, source="S2", optical_coverage_pct=100,
            nonwater_pixel_count=10, ndvi_p05_nonwater=0.2,
            ndvi_p95_nonwater=0.9, valid_for_fvc_calibration=1,
            optical_products=2, optical_unique_dates=2,
        )
        self.assertTrue(PREFLIGHT.validate_completed_chunk(table, expected, expected, keys))
        incompatible = dict(expected, algorithm_version="changed")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            PREFLIGHT.validate_completed_chunk(table, incompatible, expected, keys)

    def test_hls_mgrs_and_source_separation(self):
        keys = pd.DataFrame({"station_id": ["1"], "period_start": ["2020-01-01"]})
        expected = PREFLIGHT.chunk_manifest(
            "HLS", "2020-01-01", "2021-01-01", PREFLIGHT.key_digest(keys), 1
        )
        self.assertEqual(expected["hls_mgrs_rule"], "verified_local_mgrs_before_composite")
        self.assertNotEqual(PREFLIGHT.MEDOID_DEFINITIONS["S2"], PREFLIGHT.MEDOID_DEFINITIONS["HLS"])
        table = keys.assign(
            number_days=8, source="HLS", optical_coverage_pct=100,
            nonwater_pixel_count=10, ndvi_p05_nonwater=0.2,
            ndvi_p95_nonwater=0.9, valid_for_fvc_calibration=1,
            hls_s30_products=1, hls_l30_products=1, hls_s30_unique_dates=1,
            hls_l30_unique_dates=1, local_mgrs_tiles=np.nan,
        )
        with self.assertRaisesRegex(ValueError, "MGRS"):
            PREFLIGHT.validate_completed_chunk(table, expected, expected, keys)

    def test_historical_config_overwrite_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "never be overwritten"):
            PREFLIGHT.safe_experimental_output(ROOT / "config/fvc_endmembers.json")


if __name__ == "__main__":
    unittest.main()
