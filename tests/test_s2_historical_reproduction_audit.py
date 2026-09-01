"""Tests for the local S2 historical-reproduction audit."""

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("s2_audit", ROOT / "scripts/audit_s2_historical_reproduction.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class HistoricalReproductionAuditTests(unittest.TestCase):
    def test_station_mapping_and_fold_mapping_are_complete(self):
        self.assertEqual(set(AUDIT.STATION_MAP.values()), {"ST01", "ST02", "ST03", "ST04", "ST05"})
        self.assertEqual(set(AUDIT.SPATIAL_FOLD), set(AUDIT.STATION_MAP.values()))

    def test_pairwise_detects_same_eligibility_and_key_mismatch(self):
        historical = pd.DataFrame({"source": ["S2"], "station": ["x"], "station_id": ["ST01"],
            "period_start": ["2021-01-01"], "period_end": ["2021-01-09"], "source_scale_m": [20],
            "products": [2], "coverage_pct": [100], "NDVI_p05": [.2], "NDVI_p95": [.8],
            "nonwater_pixel_count": [4], "historical_station_id": ["0"]})
        current = pd.DataFrame({"station_id": ["ST01"], "period_start": ["2021-01-01"],
            "number_days": [8], "source": ["S2"], "optical_coverage_pct": [100],
            "nonwater_pixel_count": [4], "ndvi_p05_nonwater": [.2], "ndvi_p95_nonwater": [.8],
            "valid_for_fvc_calibration": [1], "optical_products": [2], "optical_unique_dates": [2]})
        result = AUDIT.build_pairwise(historical, current)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]._merge, "both")
        self.assertTrue(result.iloc[0].historical_eligible and result.iloc[0].current_eligible)

    def test_quantile_rank_uses_linear_interpolation_positions(self):
        frame = pd.DataFrame({"station_id": [f"ST{i:02d}" for i in range(1, 6)] * 100,
            "period_start": pd.date_range("2020-01-01", periods=500).astype(str),
            "historical_eligible": True, "current_eligible": True,
            "historical_p05": range(500), "current_p05": range(500),
            "historical_p95": range(500), "current_p95": range(500),
            "abs_delta_p05": 0.0, "abs_delta_p95": 0.0})
        controls = AUDIT.quantile_controls(frame)
        p05 = controls.query("version == 'historical' and candidate == 'p05'")
        self.assertEqual(p05.zero_based_rank.tolist(), [24, 25])
        self.assertAlmostEqual(p05.upper_interpolation_weight.iloc[0], 0.95, places=14)

    def test_method_drift_is_scientifically_relevant(self):
        row = AUDIT.method_diff().query("component == 'medoid score bands'").iloc[0]
        self.assertFalse(row.same)
        self.assertTrue(row.scientifically_relevant and row.could_explain_difference)
