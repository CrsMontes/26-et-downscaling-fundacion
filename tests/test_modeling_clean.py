import numpy as np
import pandas as pd

from et_downscaling.modeling import (
    MASTER_MODEL_FEATURES,
    OPTICAL_COVERAGE_THRESHOLD_PCT,
    prepare_ridge25_population,
    train_and_validate_ridge25,
)
from et_downscaling.ridge25 import RIDGE25_MODEL_FEATURES


def build_synthetic_master():
    rows = []
    spatial_blocks = ["A", "B", "C", "D"]
    years = [2020, 2021, 2022, 2023]

    row_number = 0
    for block_index, block in enumerate(spatial_blocks):
        for year_index, year in enumerate(years):
            for replicate in range(2):
                row_number += 1
                values = {
                    feature: (
                        0.1 * row_number
                        + 0.01 * feature_index
                        + 0.02 * block_index
                        + 0.03 * year_index
                    )
                    for feature_index, feature in enumerate(
                        MASTER_MODEL_FEATURES
                    )
                }
                target = (
                    0.6
                    + 0.015 * row_number
                    + 0.02 * block_index
                    - 0.01 * year_index
                )
                rows.append(
                    {
                        "station_id": f"ST{block_index + 1:02d}",
                        "period_start": f"{year}-01-{1 + replicate:02d}",
                        "year": year,
                        "spatial_block": block,
                        "modis_good": 1,
                        "target_complete": 1,
                        "s2_coverage_pct": 95.0,
                        "Kc_target": target,
                        **values,
                    }
                )

    return pd.DataFrame(rows)


def test_ge90_population_is_rebuilt_from_master():
    master = build_synthetic_master()

    master.loc[0, "s2_coverage_pct"] = (
        OPTICAL_COVERAGE_THRESHOLD_PCT - 0.01
    )
    master.loc[1, "modis_good"] = 0
    master.loc[2, MASTER_MODEL_FEATURES[0]] = np.nan

    population = prepare_ridge25_population(master)

    assert len(population) == len(master) - 3
    assert population["s2_coverage_pct"].ge(
        OPTICAL_COVERAGE_THRESHOLD_PCT
    ).all()


def test_ridge_is_fit_in_memory_with_spatial_and_loyo_oof():
    master = build_synthetic_master()

    result = train_and_validate_ridge25(master)

    assert len(result.population) == len(master)
    assert len(result.spatial_oof) == len(master)
    assert len(result.temporal_oof) == len(master)
    assert len(result.spatial_fold_metrics) == 4
    assert len(result.temporal_fold_metrics) == 4
    assert result.model.named_steps["regressor"].alpha == 1.0

    prediction = result.model.predict(
        result.population[RIDGE25_MODEL_FEATURES]
    )
    assert len(prediction) == len(master)


def test_training_module_contains_no_reconciliation_step():
    import et_downscaling.modeling as modeling

    assert not hasattr(modeling, "build_modis_constrained_et")
