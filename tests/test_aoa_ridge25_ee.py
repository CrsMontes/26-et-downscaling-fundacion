from dataclasses import replace

import numpy as np
import pytest

import et_downscaling.aoa_ridge25 as aoa_ridge25
from et_downscaling.aoa_ridge25 import (
    build_ee_unweighted_aoa,
    build_unweighted_aoa,
    score_unweighted_aoa,
)
from et_downscaling.ridge25 import RIDGE25_MODEL_FEATURES
from test_aoa import _population


class _ArrayConstant:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)


class _Image:
    def __init__(self, values, bands=None, is_array=False):
        self.values = np.asarray(values)
        self.bands = None if bands is None else list(bands)
        self.is_array = is_array

    def select(self, bands):
        indexes = [self.bands.index(band) for band in bands]
        return _Image(
            self.values[indexes],
            bands=bands,
        )

    def rename(self, bands):
        if isinstance(bands, str):
            bands = [bands]
        return _Image(
            self.values,
            bands=bands,
            is_array=self.is_array,
        )

    def toDouble(self):
        return _Image(
            self.values.astype(float),
            bands=self.bands,
            is_array=self.is_array,
        )

    def toByte(self):
        return _Image(
            self.values.astype(np.uint8),
            bands=self.bands,
            is_array=self.is_array,
        )

    def subtract(self, other):
        return self._binary(other, np.subtract)

    def divide(self, other):
        return self._binary(other, np.divide)

    def multiply(self, other):
        return self._binary(other, np.multiply)

    def add(self, other):
        return self._binary(other, np.add)

    def max(self, other):
        return self._binary(other, np.maximum)

    def pow(self, exponent):
        return _Image(
            np.power(self.values, exponent),
            bands=self.bands,
            is_array=self.is_array,
        )

    def sqrt(self):
        return _Image(
            np.sqrt(self.values),
            bands=self.bands,
            is_array=self.is_array,
        )

    def lte(self, other):
        return self._binary(other, np.less_equal)

    def reduce(self, reducer):
        assert reducer == "sum"
        return _Image(np.sum(self.values))

    def toArray(self, axis=None):
        if axis is None:
            values = (
                self.values
                if self.is_array
                else np.atleast_1d(self.values)
            )
        else:
            values = np.expand_dims(self.values, axis=axis)
        return _Image(values, is_array=True)

    def arrayRepeat(self, axis, copies):
        return _Image(
            np.repeat(self.values, copies, axis=axis),
            is_array=True,
        )

    def matrixMultiply(self, other):
        return _Image(
            np.matmul(self.values, other.values),
            is_array=True,
        )

    def arrayReduce(self, reducer, axes):
        assert reducer == "min"
        values = self.values
        for axis in sorted(axes, reverse=True):
            values = np.min(values, axis=axis, keepdims=True)
        return _Image(values, is_array=True)

    def arrayGet(self, indexes):
        index = tuple(
            np.asarray(indexes.values, dtype=int).tolist()
        )
        return _Image(self.values[index])

    def _binary(self, other, operation):
        other_values = (
            other.values
            if isinstance(other, _Image)
            else other
        )
        return _Image(
            operation(self.values, other_values),
            bands=self.bands,
            is_array=self.is_array,
        )


class _ImageFactory:
    def __call__(self, value):
        assert isinstance(value, _Image)
        return value

    @staticmethod
    def constant(value):
        if isinstance(value, _ArrayConstant):
            return _Image(value.values, is_array=True)
        return _Image(value)


class _Reducer:
    @staticmethod
    def sum():
        return "sum"

    @staticmethod
    def min():
        return "min"


class _LocalEarthEngine:
    Image = _ImageFactory()
    Reducer = _Reducer()

    @staticmethod
    def Array(values):
        return _ArrayConstant(values)


def _server_side_scores(monkeypatch, predictors, parameters):
    monkeypatch.setattr(
        aoa_ridge25,
        "ee",
        _LocalEarthEngine(),
    )

    reversed_features = list(
        reversed(RIDGE25_MODEL_FEATURES)
    )
    outputs = []
    for row in predictors:
        values_by_feature = dict(
            zip(
                RIDGE25_MODEL_FEATURES,
                row,
                strict=True,
            )
        )
        stack = _Image(
            [values_by_feature[name] for name in reversed_features],
            bands=reversed_features,
        )
        result = build_ee_unweighted_aoa(
            stack,
            parameters,
        )
        outputs.append(
            (
                float(result["nearest_training_distance"].values),
                float(result["di"].values),
                bool(result["aoa"].values),
            )
        )

    return outputs


def test_ee_unweighted_aoa_matches_local_score(monkeypatch):
    population = _population()
    parameters = build_unweighted_aoa(population)
    training = population[
        RIDGE25_MODEL_FEATURES
    ].to_numpy(dtype=float)
    predictors = np.vstack(
        [
            training[0],
            (training[4] + training[7]) / 2,
            training[-1] + parameters.scales * 0.25,
            training[-1] + parameters.scales * 8.0,
        ]
    )

    local_di, local_inside = score_unweighted_aoa(
        predictors,
        parameters,
    )
    server = _server_side_scores(
        monkeypatch,
        predictors,
        parameters,
    )

    server_distance = np.asarray([row[0] for row in server])
    server_di = np.asarray([row[1] for row in server])
    server_inside = np.asarray([row[2] for row in server])

    np.testing.assert_allclose(
        server_distance,
        local_di * parameters.mean_training_distance,
        rtol=1e-10,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        server_di,
        local_di,
        rtol=1e-10,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        server_inside,
        local_inside,
    )
    assert server_inside.any()
    assert (~server_inside).any()


def test_ee_unweighted_aoa_rejects_non_ridge25_schema():
    parameters = build_unweighted_aoa(_population())
    inconsistent = replace(
        parameters,
        feature_names=tuple(
            reversed(parameters.feature_names)
        ),
    )

    with pytest.raises(
        ValueError,
        match="Ridge-25 model specification",
    ):
        build_ee_unweighted_aoa(
            object(),
            inconsistent,
        )
