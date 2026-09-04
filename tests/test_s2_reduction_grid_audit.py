"""Guards for the local S2 reduction-grid audit."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "grid_audit", ROOT / "reproducibility/scripts/audit_s2_reduction_grid.py"
)
GRID = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRID)


class S2ReductionGridAuditTests(unittest.TestCase):
    def test_first_band_and_nir_contract(self):
        result = GRID.audit()
        self.assertEqual(result["first_band"], "Blue")
        self.assertEqual(result["nir_source_band"], "B8A")
        self.assertEqual(result["classification"], "EXACTLY_EQUIVALENT")
        self.assertFalse(result["metadata_only_gee_query_used"])
