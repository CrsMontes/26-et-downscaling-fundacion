from __future__ import annotations

import pandas as pd

from et_downscaling.field_reporting import build_temporal_completeness_sensitivity


def test_temporal_completeness_keeps_primary_and_strict_complete_case_counts():
    table = pd.DataFrame(
        {
            "included": [True, True, True, True],
            "n_valid_field_days": [5, 7, 8, 8],
            "ET_field_proxy_mm_period": [10.0, 11.0, 12.0, 13.0],
            "ET_MODIS_parent_mm_period": [10.5, 11.5, 12.5, 13.5],
            "ET_Ridge_with_AOA_mm_period": [10.2, 11.2, 12.2, 13.2],
        }
    )
    scenarios = [
        (
            "demo",
            "included",
            "ET_Ridge_with_AOA_mm_period",
            "demo scenario",
        )
    ]

    def metrics(observed, predicted):
        return {
            "n": int(len(observed)),
            "R2": 0.0,
            "RMSE": 0.0,
            "MAE": 0.0,
            "BIAS": 0.0,
            "r": 0.0,
            "KGE": 0.0,
        }

    output = build_temporal_completeness_sensitivity(
        table,
        scenarios,
        metrics,
    )

    counts = output.groupby("field_completeness")["n"].first().to_dict()
    assert counts["primary_5of8_or_more"] == 4
    assert counts["complete_8of8_only"] == 2
