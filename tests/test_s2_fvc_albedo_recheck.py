from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


def load_diagnostic_module():
    project_root = Path(__file__).resolve().parents[1]
    path = (
        project_root
        / "reproducibility"
        / "scripts"
        / "recheck_s2_fvc_albedo.py"
    )
    spec = spec_from_file_location("recheck_s2_fvc_albedo", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fvc_endmember_recalibration_uses_only_eligible_candidates():
    diagnostic = load_diagnostic_module()
    table = pd.DataFrame(
        {
            "station_id": ["ST01", "ST02", "ST03", "ST04", "ST05", "ST01"],
            "optical_coverage_pct": [95, 95, 95, 95, 95, 70],
            "nonwater_pixel_count": [10, 10, 10, 10, 10, 10],
            "ndvi_p05_nonwater": [0.10, 0.20, 0.30, 0.40, 0.50, -0.90],
            "ndvi_p95_nonwater": [0.60, 0.70, 0.80, 0.90, 1.00, 0.10],
        }
    )

    result, eligible = diagnostic.calculate_endmembers(table)

    assert len(eligible) == 5
    assert np.isclose(result["low"], 0.12)
    assert np.isclose(result["high"], 0.98)
    assert result["n_stations"] == 5


def test_s2_daily_mosaic_has_explicit_deterministic_order():
    project_root = Path(__file__).resolve().parents[1]
    source = (
        project_root / "src" / "et_downscaling" / "sentinel2.py"
    ).read_text(encoding="utf-8")

    function = source.split("def build_s2_daily_collection", 1)[1].split(
        "def build_s2_medoid", 1
    )[0]

    assert "S2_DAILY_MOSAIC_SORT_PROPERTY" in function
    assert function.index(".sort(") < function.index(".mosaic()")


def test_fvc_albedo_diagnostic_uses_production_analysis_crs():
    project_root = Path(__file__).resolve().parents[1]
    source = (
        project_root
        / "reproducibility"
        / "scripts"
        / "recheck_s2_fvc_albedo.py"
    ).read_text(encoding="utf-8")

    assert "ANALYSIS_CRS" in source
    assert source.count("crs=ANALYSIS_CRS") >= 3
