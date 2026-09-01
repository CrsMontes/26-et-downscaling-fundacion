"""Tests for the local FVC coverage-population audit."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "coverage_audit", ROOT / "scripts/audit_fvc_coverage_populations.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class FvcCoveragePopulationAuditTests(unittest.TestCase):
    def test_full_key_join_for_both_sources(self):
        for source in ("S2", "HLS"):
            table = AUDIT.load_tables(source)
            self.assertEqual(len(table), 1150)
            self.assertFalse(table.duplicated(AUDIT.KEYS).any())

    def test_definitions_distinguish_union_from_medoid(self):
        table = AUDIT.definitions()
        temporal = table.loc[table.component.eq("temporal support")]
        self.assertTrue((~temporal.same).all())
        water = table.loc[table.component.eq("water mask")]
        self.assertTrue(water.same.all())
