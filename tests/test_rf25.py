import numpy as np
import pandas as pd

from et_downscaling.aoa_rf25 import build_rf_weighted_aoa, score_rf_weighted_aoa
from et_downscaling.rf25 import (
    RF25_MODEL_FEATURES,
    RF25_PARAMETERS,
    build_rf25_model,
    validate_rf25_model,
)


def synthetic_population(n=80):
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(n, len(RF25_MODEL_FEATURES)))
    data = pd.DataFrame(matrix, columns=RF25_MODEL_FEATURES)
    data["Kc_target"] = (
        1.0
        + 0.25 * data[RF25_MODEL_FEATURES[0]]
        - 0.15 * data[RF25_MODEL_FEATURES[1]]
        + rng.normal(0, 0.04, n)
    )
    data["spatial_block"] = [f"B{i % 5}" for i in range(n)]
    return data


def test_rf25_fixed_configuration_and_schema():
    data = synthetic_population()
    model = build_rf25_model()
    assert model.get_params(deep=False)["n_estimators"] == RF25_PARAMETERS["n_estimators"]
    model.fit(data[RF25_MODEL_FEATURES], data["Kc_target"])
    validate_rf25_model(model)
    assert model.n_features_in_ == 25


def test_rf_weighted_aoa_uses_cv_threshold_and_scores_training_points():
    data = synthetic_population()
    model = build_rf25_model()
    model.fit(data[RF25_MODEL_FEATURES], data["Kc_target"])
    aoa = build_rf_weighted_aoa(
        data,
        model,
        importance_repeats=3,
        random_state=42,
    )

    assert aoa.weights.shape == (25,)
    assert np.isfinite(aoa.weights).all()
    assert np.isclose(aoa.weights.max(), 1.0)
    assert np.isfinite(aoa.training_di).all()
    assert np.isfinite(aoa.threshold)
    assert aoa.threshold > 0
    assert np.isclose(
        aoa.threshold,
        min(
            float(np.max(aoa.training_di)),
            float(np.quantile(aoa.training_di, 0.75))
            + 1.5
            * (
                float(np.quantile(aoa.training_di, 0.75))
                - float(np.quantile(aoa.training_di, 0.25))
            ),
        ),
    )

    di, inside, lpd = score_rf_weighted_aoa(
        data[RF25_MODEL_FEATURES].to_numpy(float),
        aoa,
    )
    # Scoring the exact final-training points against the full final-training
    # reference gives zero nearest-neighbour distance by construction.
    np.testing.assert_allclose(di, 0.0, atol=1e-7)
    assert inside.all()
    assert (lpd >= 1).all()
