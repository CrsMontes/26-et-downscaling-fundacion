import numpy as np
import pandas as pd

from et_downscaling.aoa_ridge25 import (
    build_unweighted_aoa,
    score_unweighted_aoa,
)
from et_downscaling.ridge25 import (
    RIDGE25_MODEL_FEATURES,
)


def _population():
    rows = []
    for group, offset in [
        ("a", 0.0),
        ("b", 1.0),
        ("c", 2.0),
        ("d", 3.0),
    ]:
        for replicate in range(3):
            row = {
                feature: (
                    offset
                    + replicate * 0.05
                    + index * 0.01
                )
                for index, feature
                in enumerate(
                    RIDGE25_MODEL_FEATURES
                )
            }
            row["spatial_block"] = group
            rows.append(row)

    return pd.DataFrame(rows)


def test_unweighted_aoa_uses_all_features():
    parameters = build_unweighted_aoa(
        _population()
    )

    assert parameters.feature_names == tuple(
        RIDGE25_MODEL_FEATURES
    )


def test_threshold_matches_cast_rule():
    parameters = build_unweighted_aoa(
        _population()
    )

    q1 = np.quantile(
        parameters.training_di,
        0.25,
    )
    q3 = np.quantile(
        parameters.training_di,
        0.75,
    )

    expected = min(
        np.max(
            parameters.training_di
        ),
        q3 + 1.5 * (q3 - q1),
    )

    assert np.isclose(
        parameters.threshold,
        expected,
    )


def test_identical_training_point_has_zero_di():
    population = _population()

    parameters = build_unweighted_aoa(
        population
    )

    matrix = population[
        RIDGE25_MODEL_FEATURES
    ].iloc[[0]].to_numpy()

    di, inside = score_unweighted_aoa(
        matrix,
        parameters,
    )

    assert np.isclose(di[0], 0.0)
    assert inside[0]

