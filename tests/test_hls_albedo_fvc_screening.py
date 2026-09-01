"""Guards for fold-aware HLS Albedo/FVC screening."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("screening", ROOT / "scripts/screen_hls_albedo_fvc.py")
SCREEN = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(SCREEN)


class HlsAlbedoFvcScreeningTests(unittest.TestCase):
    def test_exact_paired_population_and_features(self):
        data, folds, _, audit = SCREEN.load_data()
        self.assertEqual(len(data), 550); self.assertEqual(audit["both_valid"], 550)
        self.assertEqual(len(SCREEN.BASE), 20); self.assertEqual(len(folds), 1100)

    def test_fold_columns_are_explicit(self):
        self.assertEqual(SCREEN.fold_fvc_column("spatial", 2), "fvc_spatial_train_excl_fold2_mean")
        self.assertEqual(SCREEN.fold_fvc_column("temporal", 2), "fvc_temporal_train_excl_2021_mean")
        self.assertEqual(SCREEN.fold_fvc_column("spatial", 2, True), "fvc_global_2020_2024_mean")

    def test_only_preregistered_algorithms_and_configurations(self):
        self.assertEqual(SCREEN.ALGORITHMS, ("random_forest", "extra_trees"))
        self.assertEqual(set(SCREEN.CONFIGURATIONS), {"HLS_BASE", "HLS_BASE_ALBEDO", "HLS_BASE_FVC", "HLS_BASE_ALBEDO_FVC"})
