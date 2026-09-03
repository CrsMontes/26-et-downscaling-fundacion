import numpy as np
from rasterio.transform import Affine

from et_downscaling.aoa_ridge25 import AOAParameters
from et_downscaling.local_reconciliation import (
    RIDGE25_RECONCILIATION_MAX_ITERATIONS,
    RIDGE25_RECONCILIATION_TOLERANCE_MM,
    RIDGE25_USABLE_SUPPORT_FRACTION,
    reconcile_local_ridge25,
    score_local_ridge25,
)
from et_downscaling.ridge25 import RIDGE25_MODEL_FEATURES


class _TestModel:
    def predict(self, frame):
        values = frame[
            RIDGE25_MODEL_FEATURES[0]
        ].to_numpy(dtype=float)

        return np.where(
            values < 0,
            -1.0,
            1.0,
        )


def _aoa_parameters():
    feature_count = len(
        RIDGE25_MODEL_FEATURES
    )

    return AOAParameters(
        feature_names=tuple(
            RIDGE25_MODEL_FEATURES
        ),
        means=np.zeros(
            feature_count,
            dtype=float,
        ),
        scales=np.ones(
            feature_count,
            dtype=float,
        ),
        training_scaled=np.zeros(
            (
                1,
                feature_count,
            ),
            dtype=float,
        ),
        mean_training_distance=1.0,
        threshold=0.5,
        training_di=np.array(
            [0.0],
            dtype=float,
        ),
    )


def test_final_constants_match_accepted_method():
    assert RIDGE25_USABLE_SUPPORT_FRACTION == 0.90
    assert RIDGE25_RECONCILIATION_TOLERANCE_MM == 0.01
    assert RIDGE25_RECONCILIATION_MAX_ITERATIONS == 30


def test_local_scoring_requires_stack_aoa_and_nonnegative_kc():
    feature_count = len(
        RIDGE25_MODEL_FEATURES
    )

    cube = np.zeros(
        (
            1,
            3,
            feature_count,
        ),
        dtype=float,
    )

    # Inside AOA and positive Kc.
    cube[
        0,
        0,
        0,
    ] = 0.0

    # Far outside the AOA, but positive Kc.
    cube[
        0,
        1,
        0,
    ] = 10.0

    # Inside AOA, but physically invalid negative Kc.
    cube[
        0,
        2,
        0,
    ] = -0.1

    result = score_local_ridge25(
        predictor_cube=cube,
        model=_TestModel(),
        aoa_parameters=_aoa_parameters(),
    )

    assert result.stack_valid.tolist() == [
        [
            True,
            True,
            True,
        ]
    ]

    assert result.aoa_inside.tolist() == [
        [
            True,
            False,
            True,
        ]
    ]

    assert result.usable.tolist() == [
        [
            True,
            False,
            False,
        ]
    ]

    assert result.kc_raw[0, 0] == 1.0
    assert result.kc_raw[0, 2] == -1.0


def test_aligned_reconciliation_conserves_modis_mean():
    kc_raw = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=float,
    )

    usable = np.ones(
        kc_raw.shape,
        dtype=bool,
    )

    modis_et = np.array(
        [
            [10.0],
        ],
        dtype=float,
    )

    fine_transform = Affine(
        1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        2.0,
    )

    modis_transform = Affine(
        2.0,
        0.0,
        0.0,
        0.0,
        -2.0,
        2.0,
    )

    result = reconcile_local_ridge25(
        kc_raw=kc_raw,
        usable=usable,
        modis_et=modis_et,
        fine_transform=fine_transform,
        fine_crs="EPSG:3857",
        modis_transform=modis_transform,
        modis_crs="EPSG:3857",
    )

    assert result.converged
    assert result.iterations_used == 0

    np.testing.assert_allclose(
        result.et_reaggregated,
        modis_et,
        rtol=0,
        atol=1e-10,
    )

    assert (
        result.max_abs_conservation_error
        <= RIDGE25_RECONCILIATION_TOLERANCE_MM
    )

    assert np.isfinite(
        result.et_published
    ).all()

    assert (
        result.et_published
        >= 0
    ).all()


def test_support_below_90_percent_is_not_published():
    kc_raw = np.ones(
        (
            2,
            2,
        ),
        dtype=float,
    )

    usable = np.array(
        [
            [True, True],
            [True, False],
        ],
        dtype=bool,
    )

    modis_et = np.array(
        [
            [10.0],
        ],
        dtype=float,
    )

    fine_transform = Affine(
        1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        2.0,
    )

    modis_transform = Affine(
        2.0,
        0.0,
        0.0,
        0.0,
        -2.0,
        2.0,
    )

    result = reconcile_local_ridge25(
        kc_raw=kc_raw,
        usable=usable,
        modis_et=modis_et,
        fine_transform=fine_transform,
        fine_crs="EPSG:3857",
        modis_transform=modis_transform,
        modis_crs="EPSG:3857",
    )

    np.testing.assert_allclose(
        result.usable_fraction,
        np.array(
            [[0.75]]
        ),
        rtol=0,
        atol=1e-12,
    )

    assert not result.eligible_coarse.any()
    assert not np.isfinite(
        result.et_published
    ).any()


def test_internal_fill_is_not_published():
    kc_raw = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    usable = np.array(
        [
            [True, True],
            [True, False],
        ],
        dtype=bool,
    )

    modis_et = np.array(
        [
            [8.0],
        ],
        dtype=float,
    )

    fine_transform = Affine(
        1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        2.0,
    )

    modis_transform = Affine(
        2.0,
        0.0,
        0.0,
        0.0,
        -2.0,
        2.0,
    )

    result = reconcile_local_ridge25(
        kc_raw=kc_raw,
        usable=usable,
        modis_et=modis_et,
        fine_transform=fine_transform,
        fine_crs="EPSG:3857",
        modis_transform=modis_transform,
        modis_crs="EPSG:3857",
        usable_support_fraction=0.75,
    )

    assert result.converged

    # Internal fill exists for conservation...
    assert np.isfinite(
        result.et_full_support[
            1,
            1,
        ]
    )

    # ...but that cell is never part of the published product.
    assert np.isnan(
        result.et_published[
            1,
            1,
        ]
    )
