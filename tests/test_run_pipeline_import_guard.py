"""Local tests for the run_pipeline editable-package fail-fast guard."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_pipeline


class ImportedPackageRootGuardTest(unittest.TestCase):
    def test_accepts_package_from_current_repository(self):
        imported_file = run_pipeline.validate_imported_package_root(PROJECT_ROOT)

        expected_root = (PROJECT_ROOT / "src" / "et_downscaling").resolve()
        self.assertTrue(imported_file.is_relative_to(expected_root))

    def test_rejects_package_from_different_repository(self):
        wrong_file = (
            PROJECT_ROOT.parent
            / "other_repository"
            / "src"
            / "et_downscaling"
            / "__init__.py"
        ).resolve()

        with patch.object(run_pipeline.et_downscaling, "__file__", str(wrong_file)):
            with self.assertRaises(RuntimeError) as context:
                run_pipeline.validate_imported_package_root(PROJECT_ROOT)

        message = str(context.exception)
        self.assertIn("different repository", message)
        self.assertIn(str((PROJECT_ROOT / "src" / "et_downscaling").resolve()), message)
        self.assertIn(str(wrong_file), message)
        self.assertIn("No pipeline work was started", message)


if __name__ == "__main__":
    unittest.main()
