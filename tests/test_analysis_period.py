import json
from datetime import date
from pathlib import Path
import unittest
import sys
from unittest.mock import mock_open, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from et_downscaling.period import (
    AnalysisPeriod,
    expected_observation_count,
    period_directory,
    require_matching_period_metadata,
)


class AnalysisPeriodTest(unittest.TestCase):
    def test_baseline_period_label(self):
        period = AnalysisPeriod.from_strings("2021-01-01", "2024-01-01")
        self.assertEqual(period.label, "2021_2023")

    def test_five_year_period_label(self):
        period = AnalysisPeriod.from_strings("2020-01-01", "2025-01-01")
        self.assertEqual(period.label, "2020_2024")

    def test_expected_count_is_derived(self):
        self.assertEqual(expected_observation_count(4, 2), 8)
        self.assertEqual(expected_observation_count(6, 3), 18)

    def test_period_directories_are_separate(self):
        baseline = AnalysisPeriod(date(2021, 1, 1), date(2024, 1, 1))
        experiment = AnalysisPeriod(date(2020, 1, 1), date(2025, 1, 1))
        self.assertNotEqual(
            period_directory(Path("outputs"), baseline),
            period_directory(Path("outputs"), experiment),
        )

    def test_mismatched_artifact_metadata_is_rejected(self):
        baseline = AnalysisPeriod.from_strings("2021-01-01", "2024-01-01")
        requested = AnalysisPeriod.from_strings("2020-01-01", "2025-01-01")
        with patch.object(Path, "is_file", return_value=True), patch.object(
            Path, "open", mock_open(read_data=json.dumps(baseline.metadata()))
        ):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                require_matching_period_metadata(Path("metadata.json"), requested)


if __name__ == "__main__":
    unittest.main()
