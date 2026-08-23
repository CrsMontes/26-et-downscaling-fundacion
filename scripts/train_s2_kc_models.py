"""Compare candidate Sentinel-2 Kc models using spatial validation.

The script performs the minimum model-selection experiment required before
fine-scale evapotranspiration prediction:

1. Random Forest using the common scale-transferable feature set.
2. Random Forest using the full Sentinel-2 feature set.
3. Global-mean baseline.
4. Previous-MODIS-period persistence baseline.

Both Random Forest models use exactly the same Sentinel-2 observations and
exactly the same spatial validation folds.

Training unit:
    one station x one MODIS footprint x one MODIS period

Target:
    Kc_target = ET_MODIS / ETo

Primary validation:
    spatial GroupKFold using 10 km spatial blocks

This script makes no Earth Engine calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

SPATIAL_BLOCK_SIZE_KM = 10.0

TARGET_COLUMN = "Kc_target"

CANDIDATE_COLUMN = (
    "training_candidate_source_ge_90"
)

MAX_VALID_ET_QC = 254


# ============================================================
# Predictor sets
# ============================================================

COMMON_SATELLITE_FEATURES = [
    # Optical reflectance
    "Blue_mean",
    "Green_mean",
    "Red_mean",
    "NIR_mean",
    "SWIR1_mean",
    "SWIR2_mean",

    # Common spectral indices
    "NDVI_mean",
    "EVI_mean",
    "SAVI_mean",
    "NDWI_mean",
    "NDMI_mean",

    # Sentinel-1
    "VV_dB_mean",
    "VH_dB_mean",
    "VV_minus_VH_dB_mean",
]


S2_ADDITIONAL_FEATURES = [
    "RedEdge1_mean",
    "RedEdge2_mean",
    "RedEdge3_mean",
    "NDRE_mean",
    "Albedo_mean",
    "FVC_mean",
]


METEOROLOGICAL_FEATURES = [
    "Tair_mean_C",
    "Tair_max_C",
    "VPD_mean_kPa",
    "SolarRad_MJ_m2_day",
    "Wind_mean_ms",
    "Precip_period_mm",
    "Precip_prev30d_mm",
]


HARMONIC_FEATURES = [
    "doy_sin1",
    "doy_cos1",
    "doy_sin2",
    "doy_cos2",
]


COMMON_MODEL_FEATURES = (
    COMMON_SATELLITE_FEATURES
    + METEOROLOGICAL_FEATURES
    + HARMONIC_FEATURES
)


FULL_S2_MODEL_FEATURES = (
    COMMON_SATELLITE_FEATURES
    + S2_ADDITIONAL_FEATURES
    + METEOROLOGICAL_FEATURES
    + HARMONIC_FEATURES
)


# ============================================================
# Temporal harmonics
# ============================================================

def add_doy_harmonics(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add two annual harmonic pairs from MODIS period start."""

    result = data.copy()

    dates = pd.to_datetime(
        result["period_start"],
        errors="raise",
    )

    doy = (
        dates
        .dt
        .dayofyear
        .to_numpy(
            dtype=float
        )
    )

    harmonic_data = {}

    for harmonic in (1, 2):

        angle = (
            2.0
            * np.pi
            * harmonic
            * doy
            / 365.25
        )

        harmonic_data[
            f"doy_sin{harmonic}"
        ] = np.sin(
            angle
        )

        harmonic_data[
            f"doy_cos{harmonic}"
        ] = np.cos(
            angle
        )

    harmonics = pd.DataFrame(
        harmonic_data,
        index=result.index,
    )

    return pd.concat(
        [
            result,
            harmonics,
        ],
        axis=1,
    )


# ============================================================
# MODIS persistence baseline
# ============================================================

def add_previous_modis_kc(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Add previous valid MODIS-period Kc within each station.

    The lag is derived from the complete MODIS time series rather
    than only from optical training candidates.

    ET_QC values above the accepted MOD16 valid range are masked
    before calculating the lag, so an invalid target cannot be used
    as a persistence predictor.
    """

    result = master.copy()

    result["period_start"] = pd.to_datetime(
        result["period_start"],
        errors="raise",
    )

    et_qc = pd.to_numeric(
        result["ET_QC"],
        errors="coerce",
    )

    valid_previous_kc = pd.to_numeric(
        result[
            TARGET_COLUMN
        ],
        errors="coerce",
    ).copy()

    valid_previous_kc.loc[
        et_qc > MAX_VALID_ET_QC
    ] = np.nan

    result[
        "_Kc_for_persistence"
    ] = valid_previous_kc

    result = result.sort_values(
        [
            "station_id",
            "period_start",
        ]
    )

    result[
        "Kc_previous_modis"
    ] = (
        result
        .groupby(
            "station_id"
        )[
            "_Kc_for_persistence"
        ]
        .shift(1)
    )

    result = result.drop(
        columns=[
            "_Kc_for_persistence"
        ]
    )

    return result.sort_index()


# ============================================================
# Spatial blocks
# ============================================================

def resolve_coordinate_columns(
    data: pd.DataFrame,
) -> tuple[str, str]:
    """Find longitude and latitude columns available in the master."""

    candidates = [
        (
            "station_longitude",
            "station_latitude",
        ),
        (
            "longitude",
            "latitude",
        ),
        (
            "footprint_centroid_longitude",
            "footprint_centroid_latitude",
        ),
    ]

    for (
        longitude_column,
        latitude_column,
    ) in candidates:

        if (
            longitude_column
            in data.columns
            and latitude_column
            in data.columns
        ):
            return (
                longitude_column,
                latitude_column,
            )

    raise ValueError(
        "No usable longitude/latitude columns were found."
    )


def add_spatial_blocks(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Assign observations to approximately 10 km spatial blocks."""

    result = data.copy()

    (
        longitude_column,
        latitude_column,
    ) = resolve_coordinate_columns(
        result
    )

    longitude = pd.to_numeric(
        result[
            longitude_column
        ],
        errors="raise",
    )

    latitude = pd.to_numeric(
        result[
            latitude_column
        ],
        errors="raise",
    )

    km_per_degree_latitude = 111.32

    km_per_degree_longitude = (
        111.32
        * np.cos(
            np.radians(
                latitude.mean()
            )
        )
    )

    block_x = np.floor(
        longitude
        * km_per_degree_longitude
        / SPATIAL_BLOCK_SIZE_KM
    ).astype(
        int
    )

    block_y = np.floor(
        latitude
        * km_per_degree_latitude
        / SPATIAL_BLOCK_SIZE_KM
    ).astype(
        int
    )

    result[
        "spatial_block"
    ] = (
        block_x.astype(
            str
        )
        + "_"
        + block_y.astype(
            str
        )
    )

    return result


# ============================================================
# Random Forest
# ============================================================

def build_random_forest(
) -> RandomForestRegressor:
    """Return the pre-specified Random Forest configuration."""

    return RandomForestRegressor(
        n_estimators=500,
        max_features=0.33,
        min_samples_leaf=3,
        max_depth=None,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_kge(
    observed,
    predicted,
) -> float:
    """Calculate Kling-Gupta Efficiency."""

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    if len(
        observed
    ) < 2:
        return np.nan

    observed_sd = observed.std(
        ddof=0
    )

    predicted_sd = predicted.std(
        ddof=0
    )

    if (
        observed_sd == 0
        or predicted_sd == 0
    ):
        return np.nan

    correlation = float(
        np.corrcoef(
            observed,
            predicted,
        )[0, 1]
    )

    alpha = (
        predicted_sd
        / observed_sd
    )

    observed_mean = (
        observed.mean()
    )

    if observed_mean == 0:
        return np.nan

    beta = (
        predicted.mean()
        / observed_mean
    )

    return float(
        1.0
        - np.sqrt(
            (
                correlation
                - 1.0
            )
            ** 2
            + (
                alpha
                - 1.0
            )
            ** 2
            + (
                beta
                - 1.0
            )
            ** 2
        )
    )


def calculate_metrics(
    observed,
    predicted,
) -> dict[str, float]:
    """Calculate model-performance metrics."""

    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    error = (
        predicted
        - observed
    )

    return {
        "n": int(
            len(
                observed
            )
        ),
        "R2": float(
            r2_score(
                observed,
                predicted,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    observed,
                    predicted,
                )
            )
        ),
        "MAE": float(
            mean_absolute_error(
                observed,
                predicted,
            )
        ),
        "BIAS": float(
            error.mean()
        ),
        "KGE": calculate_kge(
            observed,
            predicted,
        ),
    }


# ============================================================
# Spatial validation
# ============================================================

def run_spatial_validation(
    candidate: pd.DataFrame,
):
    """Run identical spatial folds for models and baselines."""

    number_blocks = int(
        candidate[
            "spatial_block"
        ]
        .nunique()
    )

    if number_blocks < 3:
        raise RuntimeError(
            "Spatial validation requires at least "
            "three spatial blocks."
        )

    splitter = GroupKFold(
        n_splits=number_blocks
    )

    groups = (
        candidate[
            "spatial_block"
        ]
        .to_numpy()
    )

    y = (
        candidate[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    number_rows = len(
        candidate
    )

    common_predictions = np.full(
        number_rows,
        np.nan,
        dtype=float,
    )

    full_predictions = np.full(
        number_rows,
        np.nan,
        dtype=float,
    )

    mean_predictions = np.full(
        number_rows,
        np.nan,
        dtype=float,
    )

    persistence_predictions = np.full(
        number_rows,
        np.nan,
        dtype=float,
    )

    fold_values = np.full(
        number_rows,
        -1,
        dtype=int,
    )

    fold_rows = []

    split_input = np.zeros(
        (
            number_rows,
            1,
        )
    )

    for (
        fold,
        (
            train_index,
            test_index,
        ),
    ) in enumerate(
        splitter.split(
            split_input,
            y,
            groups=groups,
        ),
        start=1,
    ):

        train = candidate.iloc[
            train_index
        ]

        test = candidate.iloc[
            test_index
        ]

        # ----------------------------------------------------
        # Common predictor Random Forest
        # ----------------------------------------------------

        common_model = (
            build_random_forest()
        )

        common_model.fit(
            train[
                COMMON_MODEL_FEATURES
            ],
            train[
                TARGET_COLUMN
            ],
        )

        common_predictions[
            test_index
        ] = common_model.predict(
            test[
                COMMON_MODEL_FEATURES
            ]
        )

        # ----------------------------------------------------
        # Full Sentinel-2 Random Forest
        # ----------------------------------------------------

        full_model = (
            build_random_forest()
        )

        full_model.fit(
            train[
                FULL_S2_MODEL_FEATURES
            ],
            train[
                TARGET_COLUMN
            ],
        )

        full_predictions[
            test_index
        ] = full_model.predict(
            test[
                FULL_S2_MODEL_FEATURES
            ]
        )

        # ----------------------------------------------------
        # Training-fold mean baseline
        # ----------------------------------------------------

        training_mean = float(
            train[
                TARGET_COLUMN
            ].mean()
        )

        mean_predictions[
            test_index
        ] = training_mean

        # ----------------------------------------------------
        # MODIS previous-period persistence baseline
        #
        # Missing previous Kc values are replaced by the
        # training-fold mean.
        # ----------------------------------------------------

        persistence = (
            pd.to_numeric(
                test[
                    "Kc_previous_modis"
                ],
                errors="coerce",
            )
            .fillna(
                training_mean
            )
            .to_numpy(
                dtype=float
            )
        )

        persistence_predictions[
            test_index
        ] = persistence

        fold_values[
            test_index
        ] = fold

        stations = (
            test[
                "station"
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        blocks = (
            test[
                "spatial_block"
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        common_fold_metrics = (
            calculate_metrics(
                y[
                    test_index
                ],
                common_predictions[
                    test_index
                ],
            )
        )

        full_fold_metrics = (
            calculate_metrics(
                y[
                    test_index
                ],
                full_predictions[
                    test_index
                ],
            )
        )

        persistence_fold_metrics = (
            calculate_metrics(
                y[
                    test_index
                ],
                persistence_predictions[
                    test_index
                ],
            )
        )

        fold_rows.append(
            {
                "fold": fold,
                "test_rows": int(
                    len(
                        test_index
                    )
                ),
                "stations": ";".join(
                    stations
                ),
                "blocks": ";".join(
                    blocks
                ),
                "common_R2": (
                    common_fold_metrics[
                        "R2"
                    ]
                ),
                "common_RMSE": (
                    common_fold_metrics[
                        "RMSE"
                    ]
                ),
                "full_R2": (
                    full_fold_metrics[
                        "R2"
                    ]
                ),
                "full_RMSE": (
                    full_fold_metrics[
                        "RMSE"
                    ]
                ),
                "persistence_R2": (
                    persistence_fold_metrics[
                        "R2"
                    ]
                ),
                "persistence_RMSE": (
                    persistence_fold_metrics[
                        "RMSE"
                    ]
                ),
            }
        )

    predictions = {
        "rf_common":
            common_predictions,
        "rf_s2_full":
            full_predictions,
        "global_mean":
            mean_predictions,
        "modis_persistence":
            persistence_predictions,
    }

    metrics_rows = []

    for (
        model_name,
        predicted_values,
    ) in predictions.items():

        metrics_rows.append(
            {
                "model": model_name,
                **calculate_metrics(
                    y,
                    predicted_values,
                ),
            }
        )

    metrics = pd.DataFrame(
        metrics_rows
    )

    folds = pd.DataFrame(
        fold_rows
    )

    oof_columns = [
        "station",
        "station_id",
        "period_start",
        "spatial_block",
        TARGET_COLUMN,
        "ET_mm_period",
        "ETo_mm_period",
        "ET_QC",
        "optical_union_coverage_pct",
        "s1_union_coverage_pct",
        "Kc_previous_modis",
    ]

    oof = candidate[
        oof_columns
    ].copy()

    oof[
        "fold"
    ] = fold_values

    for (
        model_name,
        predicted_values,
    ) in predictions.items():

        oof[
            f"Kc_predicted_{model_name}"
        ] = predicted_values

        oof[
            f"Kc_residual_{model_name}"
        ] = (
            predicted_values
            - y
        )

    return (
        metrics,
        folds,
        oof,
    )


# ============================================================
# Main workflow
# ============================================================

def main():
    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    input_path = (
        project_root
        / "outputs"
        / "processed"
        / "training"
        / "S2"
        / "ET_S2_S1_METEO_KC_FOOTPRINT_2021_2023.csv"
    )

    output_directory = (
        project_root
        / "outputs"
        / "processed"
        / "models"
        / "S2"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: "
            f"{input_path}"
        )

    print(
        "Loading:",
        input_path,
    )

    master = pd.read_csv(
        input_path,
        dtype={
            "station_id": "string",
        },
    ).copy()

    print(
        "Master rows:",
        len(
            master
        ),
    )

    if len(
        master
    ) != 690:
        raise RuntimeError(
            "Expected 690 rows in the "
            "Sentinel-2 training master."
        )

    if master.duplicated(
        [
            "station_id",
            "period_start",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate station-period rows found "
            "in training master."
        )

    # ========================================================
    # Add persistence predictor before optical filtering
    # ========================================================

    master = add_previous_modis_kc(
        master
    )

    # ========================================================
    # Add temporal harmonics
    # ========================================================

    master = add_doy_harmonics(
        master
    )

    # ========================================================
    # Select >=90% Sentinel-2 candidate population
    # ========================================================

    if (
        CANDIDATE_COLUMN
        not in master.columns
    ):
        raise ValueError(
            "Missing candidate column: "
            f"{CANDIDATE_COLUMN}"
        )

    candidate = master.loc[
        master[
            CANDIDATE_COLUMN
        ]
        == 1
    ].copy()

    print(
        "S2 >=90% candidate rows:",
        len(
            candidate
        ),
    )

    # ========================================================
    # Remove invalid MOD16 ET_QC values
    #
    # This does not apply a stricter MODIS quality subset.
    # It only removes values outside the accepted QC range.
    # ========================================================

    if (
        "ET_QC"
        not in candidate.columns
    ):
        raise ValueError(
            "ET_QC column is required."
        )

    candidate_et_qc = pd.to_numeric(
        candidate[
            "ET_QC"
        ],
        errors="coerce",
    )

    invalid_et_qc_mask = (
        candidate_et_qc
        > MAX_VALID_ET_QC
    )

    invalid_et_qc_count = int(
        invalid_et_qc_mask.sum()
    )

    print(
        "Invalid ET_QC rows removed:",
        invalid_et_qc_count,
    )

    candidate = candidate.loc[
        ~invalid_et_qc_mask
    ].copy()

    print(
        "Final candidate rows:",
        len(
            candidate
        ),
    )

    # ========================================================
    # Report S1 support
    #
    # Do not introduce a new S1 coverage threshold silently.
    # ========================================================

    s1_below_90 = int(
        (
            candidate[
                "s1_union_coverage_pct"
            ]
            < 90.0
        ).sum()
    )

    print(
        "Candidates with S1 coverage <90%:",
        s1_below_90,
    )

    # ========================================================
    # Validate predictor definitions
    # ========================================================

    print(
        "Common model features:",
        len(
            COMMON_MODEL_FEATURES
        ),
    )

    print(
        "Full S2 model features:",
        len(
            FULL_S2_MODEL_FEATURES
        ),
    )

    if (
        len(
            COMMON_MODEL_FEATURES
        )
        != 25
    ):
        raise RuntimeError(
            "Expected 25 common model features."
        )

    if (
        len(
            FULL_S2_MODEL_FEATURES
        )
        != 31
    ):
        raise RuntimeError(
            "Expected 31 full Sentinel-2 "
            "model features."
        )

    required_columns = list(
        dict.fromkeys(
            FULL_S2_MODEL_FEATURES
            + [
                TARGET_COLUMN,
                "station",
                "station_id",
                "period_start",
                "ET_QC",
                "ET_mm_period",
                "ETo_mm_period",
                "optical_union_coverage_pct",
                "s1_union_coverage_pct",
                "Kc_previous_modis",
            ]
        )
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in candidate.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required model columns: "
            f"{sorted(missing_columns)}"
        )

    incomplete_model_rows = (
        candidate[
            FULL_S2_MODEL_FEATURES
            + [
                TARGET_COLUMN
            ]
        ]
        .isna()
        .any(
            axis=1
        )
    )

    if incomplete_model_rows.any():
        raise RuntimeError(
            "Final candidate population contains "
            f"{int(incomplete_model_rows.sum())} "
            "rows with incomplete model values."
        )

    # ========================================================
    # Spatial blocks
    # ========================================================

    candidate = add_spatial_blocks(
        candidate
    )

    station_blocks = (
        candidate[
            [
                "station",
                "spatial_block",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "spatial_block",
                "station",
            ]
        )
    )

    print()
    print(
        "=== SPATIAL BLOCKS ==="
    )

    print(
        station_blocks.to_string(
            index=False
        )
    )

    number_spatial_groups = int(
        candidate[
            "spatial_block"
        ]
        .nunique()
    )

    print(
        "Spatial groups:",
        number_spatial_groups,
    )

    # ========================================================
    # Spatial cross-validation
    # ========================================================

    (
        metrics,
        fold_metrics,
        oof,
    ) = run_spatial_validation(
        candidate
    )

    print()
    print(
        "=== SPATIAL VALIDATION ==="
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print()
    print(
        "=== PERFORMANCE BY SPATIAL FOLD ==="
    )

    print(
        fold_metrics.to_string(
            index=False
        )
    )

    # ========================================================
    # Train final candidate models
    #
    # Both are saved for traceability. Model selection is based
    # on the spatial-validation results rather than training fit.
    # ========================================================

    common_final_model = (
        build_random_forest()
    )

    common_final_model.fit(
        candidate[
            COMMON_MODEL_FEATURES
        ],
        candidate[
            TARGET_COLUMN
        ],
    )

    full_final_model = (
        build_random_forest()
    )

    full_final_model.fit(
        candidate[
            FULL_S2_MODEL_FEATURES
        ],
        candidate[
            TARGET_COLUMN
        ],
    )

    # ========================================================
    # Output paths
    # ========================================================

    common_model_path = (
        output_directory
        / "rf_kc_s2_common_ge90.joblib"
    )

    full_model_path = (
        output_directory
        / "rf_kc_s2_full_ge90.joblib"
    )

    metrics_path = (
        output_directory
        / "kc_model_comparison_ge90.csv"
    )

    folds_path = (
        output_directory
        / "kc_model_spatial_folds_ge90.csv"
    )

    oof_path = (
        output_directory
        / "kc_model_oof_predictions_ge90.csv"
    )

    training_population_path = (
        output_directory
        / "kc_model_training_population_ge90.csv"
    )

    metadata_path = (
        output_directory
        / "kc_model_comparison_ge90.json"
    )

    # ========================================================
    # Save models
    # ========================================================

    joblib.dump(
        common_final_model,
        common_model_path,
    )

    joblib.dump(
        full_final_model,
        full_model_path,
    )

    # ========================================================
    # Save diagnostics
    # ========================================================

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    fold_metrics.to_csv(
        folds_path,
        index=False,
    )

    oof.to_csv(
        oof_path,
        index=False,
    )

    training_population_columns = list(
        dict.fromkeys(
            [
                "station",
                "station_id",
                "period_start",
                "spatial_block",
                TARGET_COLUMN,
                "ET_QC",
                "optical_union_coverage_pct",
                "s1_union_coverage_pct",
            ]
            + FULL_S2_MODEL_FEATURES
        )
    )

    candidate[
        training_population_columns
    ].to_csv(
        training_population_path,
        index=False,
    )

    # ========================================================
    # Save metadata
    # ========================================================

    metadata = {
        "optical_source": "S2",
        "target": TARGET_COLUMN,
        "training_support": (
            "MODIS footprint x MODIS period"
        ),
        "prediction_grid_m": 20,
        "candidate_column": (
            CANDIDATE_COLUMN
        ),
        "optical_coverage_threshold_pct": 90,
        "maximum_valid_et_qc": (
            MAX_VALID_ET_QC
        ),
        "invalid_et_qc_rows_removed": (
            invalid_et_qc_count
        ),
        "training_rows": int(
            len(
                candidate
            )
        ),
        "spatial_block_size_km": (
            SPATIAL_BLOCK_SIZE_KM
        ),
        "spatial_groups": (
            number_spatial_groups
        ),
        "random_forest": {
            "n_estimators": 500,
            "max_features": 0.33,
            "min_samples_leaf": 3,
            "max_depth": None,
            "bootstrap": True,
            "random_state": RANDOM_STATE,
        },
        "common_features": (
            COMMON_MODEL_FEATURES
        ),
        "full_s2_features": (
            FULL_S2_MODEL_FEATURES
        ),
        "n_common_features": int(
            len(
                COMMON_MODEL_FEATURES
            )
        ),
        "n_full_s2_features": int(
            len(
                FULL_S2_MODEL_FEATURES
            )
        ),
        "baseline_definitions": {
            "global_mean": (
                "Mean Kc of the training folds."
            ),
            "modis_persistence": (
                "Previous valid MODIS-period Kc at the "
                "same station. The lag is derived from "
                "the complete 690-row MODIS master and "
                "is independent of optical availability."
            ),
        },
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print(
        "Saved common RF:",
        common_model_path,
    )

    print(
        "Saved full S2 RF:",
        full_model_path,
    )

    print(
        "Saved comparison:",
        metrics_path,
    )

    print(
        "Saved folds:",
        folds_path,
    )

    print(
        "Saved OOF predictions:",
        oof_path,
    )

    print(
        "Saved training population:",
        training_population_path,
    )

    print(
        "Saved metadata:",
        metadata_path,
    )


if __name__ == "__main__":
    main()
    