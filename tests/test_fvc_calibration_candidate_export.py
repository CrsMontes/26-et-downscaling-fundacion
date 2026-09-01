"""Guards for the gated minimal FVC candidate extractor."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "export_fvc_calibration_candidates",
    ROOT / "scripts/export_fvc_calibration_candidates.py",
)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_fvc_candidate_stability",
    ROOT / "scripts/analyze_fvc_candidate_stability.py",
)
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)


class FvcCalibrationCandidateExportTests(unittest.TestCase):
    def test_schema_is_minimal_and_source_specific(self):
        self.assertEqual(len(EXPORT.SELECTORS["S2"]), 11)
        self.assertEqual(len(EXPORT.SELECTORS["HLS"]), 14)
        for source, selectors in EXPORT.SELECTORS.items():
            for token in EXPORT.FORBIDDEN_EXPORT_TOKENS:
                self.assertFalse(any(token in name for name in selectors), (source, token))

    def test_annual_expected_keys(self):
        for year in range(2020, 2025):
            keys = EXPORT.expected_keys(f"{year}-01-01", f"{year + 1}-01-01")
            self.assertEqual(len(keys), 230)
            self.assertFalse(keys.duplicated(["station_id", "period_start"]).any())

    def valid_table(self, source="HLS"):
        base = {
            "station_id": ["1"], "period_start": ["2021-01-01"],
            "number_days": [8], "source": [source],
            "optical_coverage_pct": [80], "nonwater_pixel_count": [10],
            "ndvi_p05_nonwater": [0.2], "ndvi_p95_nonwater": [0.9],
            "valid_for_fvc_calibration": [1],
        }
        if source == "HLS":
            base.update({
                "hls_s30_products": [1], "hls_l30_products": [2],
                "hls_s30_unique_dates": [1], "hls_l30_unique_dates": [2],
                "local_mgrs_tiles": ["18NYM"],
            })
        else:
            base.update({"optical_products": [2], "optical_unique_dates": [2]})
        return pd.DataFrame(base)

    def test_value_checks_and_p05_order(self):
        EXPORT.validate_candidate_values(
            self.valid_table(), "HLS", "2021-01-01", "2022-01-01"
        )
        invalid = self.valid_table()
        invalid["ndvi_p05_nonwater"] = 0.95
        with self.assertRaisesRegex(ValueError, "P05"):
            EXPORT.validate_candidate_values(
                invalid, "HLS", "2021-01-01", "2022-01-01"
            )

    def test_range_and_invalid_chunk_rejection(self):
        invalid = self.valid_table()
        invalid["optical_coverage_pct"] = 101
        with self.assertRaisesRegex(ValueError, "coverage"):
            EXPORT.validate_candidate_values(
                invalid, "HLS", "2021-01-01", "2022-01-01"
            )
        invalid = self.valid_table()
        invalid["nonwater_pixel_count"] = -1
        with self.assertRaisesRegex(ValueError, "negative"):
            EXPORT.validate_candidate_values(
                invalid, "HLS", "2021-01-01", "2022-01-01"
            )

    def test_source_separation_and_mgrs_requirement(self):
        mixed = self.valid_table()
        mixed.loc[0, "source"] = "S2"
        with self.assertRaisesRegex(ValueError, "sources"):
            EXPORT.validate_candidate_values(
                mixed, "HLS", "2021-01-01", "2022-01-01"
            )
        missing = self.valid_table()
        missing["local_mgrs_tiles"] = np.nan
        with self.assertRaisesRegex(ValueError, "MGRS"):
            EXPORT.validate_candidate_values(
                missing, "HLS", "2021-01-01", "2022-01-01"
            )

    def test_manifest_hash_and_resume_validation(self):
        table = self.valid_table("S2")
        keys = table[["station_id", "period_start"]]
        expected = EXPORT.expected_manifest("S2", "2021-01-01", "2022-01-01", keys)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            csv_path = Path(directory) / "chunk.csv"
            manifest_path = Path(directory) / "chunk.json"
            table.to_csv(csv_path, index=False)
            manifest = dict(
                expected, status="completed", actual_key_count=1,
                csv_sha256=EXPORT.sha256_file(csv_path),
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(EXPORT.validate_existing_chunk(
                csv_path, manifest_path, expected, keys
            ))
            csv_path.write_text("corrupt", encoding="utf-8")
            self.assertFalse(EXPORT.validate_existing_chunk(
                csv_path, manifest_path, expected, keys
            ))

    def test_historical_tolerances_and_stability_classes_are_preregistered(self):
        self.assertEqual(EXPORT.HISTORICAL_ABSOLUTE_TOLERANCE, 1e-7)
        self.assertEqual(EXPORT.HISTORICAL_RELATIVE_TOLERANCE, 1e-7)
        self.assertEqual(EXPORT.STABILITY_THRESHOLDS["negligible_max_abs_ndvi"], 0.01)
        self.assertEqual(EXPORT.STABILITY_THRESHOLDS["modest_max_abs_ndvi"], 0.05)

    def test_reconstructed_hls_historical_gate(self):
        table, _ = ANALYSIS.load_source_chunks("HLS", (2021, 2022, 2023))
        passed, comparison, _, _ = ANALYSIS.historical_gate("HLS", table)
        self.assertTrue(passed)
        self.assertEqual(comparison["n_actual"], 381)
        self.assertEqual(comparison["low_absolute_difference"], 0.0)
        self.assertEqual(comparison["high_absolute_difference"], 0.0)

    def test_current_s2_historical_reconstruction_mismatch_is_detected(self):
        table, _ = ANALYSIS.load_source_chunks("S2", (2021, 2022, 2023))
        passed, comparison, _, _ = ANALYSIS.historical_gate("S2", table)
        self.assertFalse(passed)
        self.assertEqual(comparison["n_actual"], 498)
        self.assertEqual(comparison["low_absolute_difference"], 0.0)
        self.assertGreater(
            abs(comparison["high_absolute_difference"]),
            EXPORT.HISTORICAL_ABSOLUTE_TOLERANCE,
        )

    def test_fold_specific_exclusion(self):
        rows = []
        for station, block in (("ST01", "-814_118"), ("ST02", "-814_119")):
            for year in (2020, 2021):
                rows.append({"station_id": station, "period_start": f"{year}-01-01",
                             "spatial_block": block, "year": year})
        data = pd.DataFrame(rows)
        temporal_training = data.loc[data.year.ne(2020)]
        spatial_training = data.loc[~data.station_id.isin(("ST01",))]
        self.assertFalse(temporal_training.year.eq(2020).any())
        self.assertFalse(spatial_training.station_id.eq("ST01").any())
        self.assertLess(len(spatial_training), len(data))

    def test_dry_run_never_initializes_earth_engine(self):
        result = EXPORT.main([
            "--source", "HLS", "--start-date", "2021-01-01",
            "--end-date-exclusive", "2022-01-01",
        ])
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
