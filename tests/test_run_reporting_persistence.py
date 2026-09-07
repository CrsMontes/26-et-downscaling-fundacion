from __future__ import annotations

import pandas as pd

from et_downscaling.run_reporting import _persistence_baseline_metrics


def test_persistence_baseline_matches_ridge_on_identical_rows():
    population = pd.DataFrame(
        {
            "station_id": ["A", "A", "A", "B", "B"],
            "period_start": [
                "2020-01-01",
                "2020-01-09",
                "2020-01-25",
                "2020-01-01",
                "2020-01-09",
            ],
            "Kc_target": [1.0, 1.2, 1.4, 0.8, 0.9],
        }
    )
    oof = pd.DataFrame(
        {
            "station_id": population["station_id"],
            "period_start": population["period_start"],
            "prediction": [1.1, 1.1, 1.3, 0.85, 0.85],
        }
    )

    metrics = _persistence_baseline_metrics(population, oof)
    strict = metrics[metrics["definition"].eq("strict_previous_8day_composite")]

    assert set(strict["model"]) == {"persistence", "Ridge25_spatial_OOF"}
    assert set(strict["n"]) == {2}

    within = metrics[metrics["definition"].eq("previous_observation_within_16days")]
    assert set(within["n"]) == {3}
