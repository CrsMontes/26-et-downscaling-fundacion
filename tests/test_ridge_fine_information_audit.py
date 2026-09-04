"""Preregistration guards for the Ridge fine-information audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "reproducibility" / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "audit_ridge_fine_information",
    PROJECT_ROOT / "reproducibility" / "scripts" / "audit_ridge_fine_information.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class RidgeFineInformationAuditTests(unittest.TestCase):
    def test_exact_thresholds(self):
        self.assertEqual(AUDIT.THRESHOLDS, (80, 90, 99))

    def test_preregistered_feature_sets(self):
        self.assertEqual(len(AUDIT.FEATURES["ridge_optical_only"]), 11)
        self.assertEqual(len(AUDIT.FEATURES["ridge_coarse_only"]), 9)
        self.assertEqual(len(AUDIT.FEATURES["ridge_eto_drivers_only"]), 5)
        self.assertEqual(len(AUDIT.FEATURES["ridge_optical_plus_eto"]), 16)
        self.assertEqual(len(AUDIT.FEATURES["ridge_full_candidate"]), 20)

    def test_critical_comparisons(self):
        comparisons = {row["comparison"]: row for row in AUDIT.COMPARISONS}
        critical = comparisons["full_candidate_vs_coarse_only"]
        self.assertEqual(critical["variant"], "ridge_full_candidate")
        self.assertEqual(critical["reference"], "ridge_coarse_only")
        secondary = comparisons["optical_plus_eto_vs_eto_drivers_only"]
        self.assertEqual(secondary["reference"], "ridge_eto_drivers_only")

    def test_feature_families_are_disjoint_and_complete(self):
        sets = [set(AUDIT.OPTICAL), set(AUDIT.ETO_DRIVERS), set(AUDIT.SEASONALITY)]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual(set().union(*sets), set(AUDIT.FEATURE_FAMILY))

    def test_no_forbidden_features(self):
        all_features = set().union(*(set(features) for features in AUDIT.FEATURES.values()))
        for token in ("Precip", "VV", "VH", "LST", "elevation", "slope", "aspect",
                      "ETo", "ETr", "Kc_previous"):
            self.assertFalse(any(token in feature for feature in all_features))


if __name__ == "__main__":
    unittest.main()
