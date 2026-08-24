"""Validate 20 m downscaling at ETgage locations without exporting rasters.

This script performs the field test that is directly comparable to the
diagnostic repository's "downscaled vs in situ" analysis, but with the current
Sentinel-2/Sentinel-1 Random Forest and spatial out-of-fold (OOF) models.

Important design choices
------------------------
- No GeoTIFF is created.
- Each field station is predicted with the RF fold that excluded its complete
  10 km spatial block from training.
- Fine predictors are built only over the parent MODIS footprint.
- The seven meteorological predictors are taken from the accepted training
  master for that station-period and used as coarse constant context. They are
  not downscaled to 20 m.
- The fine Kc pattern is reconciled to the observed parent MODIS ET by a
  multiplicative factor:
      ET_20m(point) = Kc_20m(point) * ET_MODIS / mean(Kc_20m within footprint)
- A field comparison is accepted only when the station pixel itself has a valid
  fine model prediction and the rebuilt footprint support satisfies the same
  optical/S1 completeness logic used for model training.

This test evaluates whether the fine spatial redistribution improves local
agreement with ETgage-derived ET relative to the parent MODIS value. It does
not constitute independent validation of every 20 m pixel across the basin.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from et_downscaling.config import (
    ANALYSIS_CRS,
    S1_FULL_COVERAGE,
)
from et_downscaling.model_spec import (
    CANDIDATE_COLUMN,
    COMMON_MODEL_FEATURES,
    HARMONIC_FEATURES,
    METEOROLOGICAL_FEATURES,
    SPATIAL_BLOCK_SIZE_KM,
    TARGET_COLUMN,
    add_doy_harmonics,
    build_random_forest,
)
from et_downscaling.model_transfer import build_ee_regressor
from et_downscaling.stations import load_station_dataframe
from et_downscaling.modis import build_modis_inputs
from et_downscaling.optical import (
    build_optical_predictors,
    filter_optical_period,
)
from et_downscaling.sentinel1 import (
    build_s1_median,
    get_sentinel1_collection,
)
from et_downscaling.sentinel2 import get_sentinel2_collection


PREDICTION_SCALE_M = 20
OPTICAL_MIN_COVERAGE_FRACTION = 0.90
S1_MIN_COVERAGE_FRACTION = S1_FULL_COVERAGE
RANDOM_STATE = 42
BOOTSTRAP_REPLICATES = 5000

OPTICAL_SOURCE_BANDS = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
    "NDVI",
    "EVI",
    "SAVI",
    "NDWI",
    "NDMI",
]

OPTICAL_MODEL_BANDS = [
    "Blue_mean",
    "Green_mean",
    "Red_mean",
    "NIR_mean",
    "SWIR1_mean",
    "SWIR2_mean",
    "NDVI_mean",
    "EVI_mean",
    "SAVI_mean",
    "NDWI_mean",
    "NDMI_mean",
]

S1_SOURCE_BANDS = [
    "VV_dB",
    "VH_dB",
    "VV_minus_VH_dB",
]

S1_MODEL_BANDS = [
    "VV_dB_mean",
    "VH_dB_mean",
    "VV_minus_VH_dB_mean",
]


# ============================================================
# Arguments and paths
# ============================================================


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Validate spatial OOF ET downscaling at ETgage locations "
            "without exporting basin rasters."
        )
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Google Cloud Project ID with Earth Engine access.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard the fine-validation checkpoint and recompute all rows.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Optional maximum number of field station-period rows to process. "
            "Useful for a one-row smoke test."
        ),
    )
    return parser.parse_args()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_paths(project_root: Path) -> dict[str, Path]:
    base = (
        project_root
        / "outputs"
        / "processed"
        / "field_validation"
    )
    tables = base / "tables"
    figures = base / "figures"

    return {
        "field_pairs": (
            tables
            / "field_modis_period_pairs_diagnostic_reproduction.csv"
        ),
        "oof": (
            project_root
            / "outputs"
            / "processed"
            / "models"
            / "S2"
            / "kc_model_oof_predictions_ge90.csv"
        ),
        "training_master": (
            project_root
            / "outputs"
            / "processed"
            / "training"
            / "S2"
            / "ET_S2_S1_METEO_KC_FOOTPRINT_2021_2023.csv"
        ),
        "checkpoint": (
            tables
            / "field_oof_downscaling_checkpoint.csv"
        ),
        "pairs_output": (
            tables
            / "field_oof_downscaled_20m_pairs.csv"
        ),
        "metrics_output": (
            tables
            / "field_oof_downscaled_20m_metrics.csv"
        ),
        "station_output": (
            tables
            / "field_oof_downscaled_20m_by_station.csv"
        ),
        "figure_main": (
            figures
            / "FD08_oof_downscaling_vs_field.png"
        ),
        "figure_series": (
            figures
            / "FD09_oof_downscaled_series.png"
        ),
    }


def require_inputs(paths: dict[str, Path]) -> None:
    required = [
        "field_pairs",
        "oof",
        "training_master",
    ]

    missing = [
        str(paths[key])
        for key in required
        if not paths[key].is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n"
            + "\n".join(missing)
            + "\nRun scripts/analyze_field_diagnostics.py and "
            "scripts/train_s2_kc_models.py first."
        )


# ============================================================
# Metrics
# ============================================================


def calculate_kge(
    observed,
    predicted,
) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(observed) < 2:
        return np.nan

    observed_sd = observed.std(ddof=0)
    predicted_sd = predicted.std(ddof=0)

    if observed_sd == 0 or predicted_sd == 0:
        return np.nan

    correlation = float(
        np.corrcoef(observed, predicted)[0, 1]
    )

    alpha = predicted_sd / observed_sd

    observed_mean = observed.mean()
    if observed_mean == 0:
        return np.nan

    beta = predicted.mean() / observed_mean

    return float(
        1.0
        - np.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    )


def calculate_metrics(
    observed,
    predicted,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "observed": observed,
            "predicted": predicted,
        }
    ).dropna()

    if len(frame) < 2:
        return {
            "n": int(len(frame)),
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "BIAS": np.nan,
            "r": np.nan,
            "KGE": np.nan,
        }

    y = frame["observed"].to_numpy(float)
    p = frame["predicted"].to_numpy(float)

    correlation = (
        float(np.corrcoef(y, p)[0, 1])
        if np.std(y) > 0 and np.std(p) > 0
        else np.nan
    )

    return {
        "n": int(len(frame)),
        "R2": float(r2_score(y, p)),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(y, p)
            )
        ),
        "MAE": float(
            mean_absolute_error(y, p)
        ),
        "BIAS": float(
            np.mean(p - y)
        ),
        "r": correlation,
        "KGE": calculate_kge(y, p),
    }


def paired_bootstrap_delta_rmse(
    observed,
    coarse,
    fine,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = RANDOM_STATE,
) -> dict[str, float | bool]:
    """Bootstrap paired RMSE improvement.

    delta_RMSE = RMSE_MODIS - RMSE_downscaled
    Positive values favour downscaling.
    """
    frame = pd.DataFrame(
        {
            "observed": observed,
            "coarse": coarse,
            "fine": fine,
        }
    ).dropna()

    if len(frame) < 2:
        return {
            "delta_RMSE": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "significant": False,
        }

    y = frame["observed"].to_numpy(float)
    a = frame["coarse"].to_numpy(float)
    b = frame["fine"].to_numpy(float)

    def rmse(x, z):
        return float(
            np.sqrt(
                np.mean(
                    (x - z) ** 2
                )
            )
        )

    observed_delta = rmse(y, a) - rmse(y, b)

    rng = np.random.default_rng(seed)
    n = len(y)

    deltas = np.empty(n_boot, dtype=float)

    for index in range(n_boot):
        sampled = rng.integers(
            0,
            n,
            size=n,
        )

        deltas[index] = (
            rmse(
                y[sampled],
                a[sampled],
            )
            - rmse(
                y[sampled],
                b[sampled],
            )
        )

    low, high = np.percentile(
        deltas,
        [2.5, 97.5],
    )

    return {
        "delta_RMSE": float(observed_delta),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "significant": bool(
            low > 0 or high < 0
        ),
    }


# ============================================================
# Local model reconstruction
# ============================================================


def resolve_coordinate_columns(data: pd.DataFrame) -> tuple[str, str]:
    """Resolve the coordinate columns exactly as in model training."""
    candidates = [
        ("station_longitude", "station_latitude"),
        ("longitude", "latitude"),
        ("footprint_centroid_longitude", "footprint_centroid_latitude"),
    ]

    for longitude_column, latitude_column in candidates:
        if longitude_column in data.columns and latitude_column in data.columns:
            return longitude_column, latitude_column

    raise ValueError("No usable longitude/latitude columns were found.")


def add_spatial_blocks(data: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the same approximately 10 km spatial blocks used in training."""
    result = data.copy()

    longitude_column, latitude_column = resolve_coordinate_columns(result)

    longitude = pd.to_numeric(
        result[longitude_column],
        errors="raise",
    )

    latitude = pd.to_numeric(
        result[latitude_column],
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
    ).astype(int)

    block_y = np.floor(
        latitude
        * km_per_degree_latitude
        / SPATIAL_BLOCK_SIZE_KM
    ).astype(int)

    result["spatial_block"] = (
        block_x.astype(str)
        + "_"
        + block_y.astype(str)
    )

    return result


def rebuild_training_candidate(
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild the exact in-memory candidate population used by training.

    The saved training-population CSV is a reporting artifact. Re-fitting an RF
    from that round-tripped artifact can change a small number of tree splits.
    For exact OOF reconstruction, start again from the canonical training master
    and apply the same harmonic, candidate and spatial-block transformations as
    scripts/train_s2_kc_models.py.
    """
    result = master.copy()

    result["period_start"] = pd.to_datetime(
        result["period_start"],
        errors="raise",
    )

    result = add_doy_harmonics(
        result
    )

    if CANDIDATE_COLUMN not in result.columns:
        raise ValueError(
            f"Missing candidate column: {CANDIDATE_COLUMN}"
        )

    result = result.loc[
        result[CANDIDATE_COLUMN] == 1
    ].copy()

    result = add_spatial_blocks(
        result
    )

    if result[
        COMMON_MODEL_FEATURES
        + [TARGET_COLUMN]
    ].isna().any(axis=1).any():
        raise RuntimeError(
            "Rebuilt training candidate contains incomplete model values."
        )

    return result


def prepare_tables(paths: dict[str, Path]):
    pairs = pd.read_csv(
        paths["field_pairs"],
        dtype={
            "station_id": "string",
        },
    )

    metadata = load_station_dataframe()

    oof = pd.read_csv(
        paths["oof"],
        dtype={
            "station_id": "string",
        },
    )

    training_master = pd.read_csv(
        paths["training_master"],
        dtype={
            "station_id": "string",
        },
    )

    for table in [
        pairs,
        oof,
        training_master,
    ]:
        table["period_start"] = pd.to_datetime(
            table["period_start"],
            errors="raise",
        )

    training = rebuild_training_candidate(
        training_master
    )

    if len(training) != len(oof):
        raise RuntimeError(
            "Rebuilt training population and saved OOF table have different "
            f"row counts: training={len(training)}, oof={len(oof)}."
        )

    for boolean_column in [
        "inside_basin",
        "installation_conforms_manual",
    ]:
        metadata[boolean_column] = (
            metadata[boolean_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "si", "sí"})
        )

    validation = pairs.merge(
        oof[
            [
                "station_id",
                "period_start",
                "fold",
                "spatial_block",
                "Kc_predicted_rf_common",
                "ET_mm_period",
                "ETo_mm_period",
            ]
        ],
        on=[
            "station_id",
            "period_start",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_oof"),
    )

    validation = validation.merge(
        metadata[
            [
                "station_id",
                "longitude",
                "latitude",
                "inside_basin",
                "installation_conforms_manual",
            ]
        ],
        on="station_id",
        how="left",
        validate="many_to_one",
    )

    meteorology = (
        training[
            [
                "station_id",
                "period_start",
                *METEOROLOGICAL_FEATURES,
            ]
        ]
        .drop_duplicates(
            [
                "station_id",
                "period_start",
            ]
        )
    )

    validation = validation.merge(
        meteorology,
        on=[
            "station_id",
            "period_start",
        ],
        how="left",
        validate="one_to_one",
    )

    validation = validation.dropna(
        subset=[
            "field_actual_et_diagnostic_mm_period",
            "ET_MODIS_mm_period",
            "fold",
            *METEOROLOGICAL_FEATURES,
        ]
    ).copy()

    validation["fold"] = (
        validation["fold"]
        .astype(int)
    )

    validation = validation.sort_values(
        [
            "fold",
            "station",
            "period_start",
        ]
    ).reset_index(drop=True)

    if validation.empty:
        raise RuntimeError(
            "No field rows overlap the current spatial OOF population."
        )

    if validation[
        [
            *METEOROLOGICAL_FEATURES,
            "longitude",
            "latitude",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Fine field-validation population contains missing "
            "meteorology or station coordinates."
        )

    return (
        validation,
        oof,
        training,
    )


def train_fold_model(
    fold: int,
    oof: pd.DataFrame,
    training: pd.DataFrame,
):
    """Rebuild exactly the spatial OOF RF for one held-out block."""
    fold_rows = oof.loc[
        oof["fold"] == fold
    ].copy()

    test_blocks = sorted(
        fold_rows[
            "spatial_block"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not test_blocks:
        raise RuntimeError(
            f"No spatial block found for fold {fold}."
        )

    train = training.loc[
        ~training[
            "spatial_block"
        ]
        .astype(str)
        .isin(test_blocks)
    ].copy()

    test = training.loc[
        training[
            "spatial_block"
        ]
        .astype(str)
        .isin(test_blocks)
    ].copy()

    model = build_random_forest()

    model.fit(
        train[
            COMMON_MODEL_FEATURES
        ],
        train[
            TARGET_COLUMN
        ],
    )

    check = test[
        [
            "station_id",
            "period_start",
            *COMMON_MODEL_FEATURES,
        ]
    ].merge(
        fold_rows[
            [
                "station_id",
                "period_start",
                "Kc_predicted_rf_common",
            ]
        ],
        on=[
            "station_id",
            "period_start",
        ],
        how="inner",
        validate="one_to_one",
    )

    local_prediction = model.predict(
        check[
            COMMON_MODEL_FEATURES
        ]
    )

    difference = (
        local_prediction
        - check[
            "Kc_predicted_rf_common"
        ].to_numpy(float)
    )

    max_abs_difference = float(
        np.max(
            np.abs(
                difference
            )
        )
    )

    if max_abs_difference > 1e-10:
        raise RuntimeError(
            "Rebuilt fold model does not reproduce saved OOF predictions. "
            f"Fold {fold}, max absolute difference = "
            f"{max_abs_difference:.3e}."
        )

    print(
        f"Fold {fold}: local OOF reproduction PASS "
        f"(test blocks={';'.join(test_blocks)}, "
        f"train={len(train)}, test={len(test)}, "
        f"max diff={max_abs_difference:.3e})"
    )

    return (
        model,
        test_blocks,
    )


# ============================================================
# Earth Engine fine prediction
# ============================================================


def initialize_earth_engine(
    project_id: str,
) -> None:
    ee.Initialize(
        project=project_id
    )

    ee.Number(1).getInfo()

    print(
        "Earth Engine initialized with project:",
        project_id,
    )


def build_station_resources():
    modis_inputs = build_modis_inputs()

    footprints = ee.FeatureCollection(
        modis_inputs[
            "station_footprints"
        ]
    )

    return footprints


def get_station_footprint(
    footprints: ee.FeatureCollection,
    station_id: str,
) -> ee.Feature:
    footprint = ee.Feature(
        footprints
        .filter(
            ee.Filter.eq(
                "station_id",
                station_id,
            )
        )
        .first()
    )

    return footprint


def build_harmonic_image(
    period_start_text: str,
) -> ee.Image:
    day_of_year = (
        date
        .fromisoformat(
            period_start_text
        )
        .timetuple()
        .tm_yday
    )

    values = []

    for harmonic in (
        1,
        2,
    ):
        angle = (
            2.0
            * np.pi
            * harmonic
            * day_of_year
            / 365.25
        )

        values.extend(
            [
                float(
                    np.sin(
                        angle
                    )
                ),
                float(
                    np.cos(
                        angle
                    )
                ),
            ]
        )

    return (
        ee.Image.constant(
            values
        )
        .rename(
            HARMONIC_FEATURES
        )
        .toFloat()
    )


def build_fine_stack(
    row,
    footprint: ee.Feature,
):
    """Build current 25-feature model stack over one MODIS parent footprint."""
    geometry = footprint.geometry()

    study_features = ee.FeatureCollection(
        [
            footprint,
        ]
    )

    period_start_text = (
        pd.Timestamp(
            row.period_start
        )
        .date()
        .isoformat()
    )

    period_start = ee.Date(
        period_start_text
    )

    period_end = period_start.advance(
        int(
            row.number_days
        ),
        "day",
    )

    fine_projection = (
        ee.Projection(
            ANALYSIS_CRS
        )
        .atScale(
            PREDICTION_SCALE_M
        )
    )

    # --------------------------------------------------------
    # Sentinel-2: same temporal medoid and common predictors
    # used in training, now retained at the 20 m prediction
    # support instead of reduced to the whole MODIS footprint.
    # --------------------------------------------------------

    s2_collection = (
        get_sentinel2_collection(
            study_features
        )
    )

    s2_period = filter_optical_period(
        collection=s2_collection,
        period_start=period_start,
        period_end=period_end,
        geometry=geometry,
        source="S2",
    )

    optical_raw = build_optical_predictors(
        period_collection=s2_period,
        geometry=geometry,
        source="S2",
    )

    optical = (
        optical_raw
        .select(
            OPTICAL_SOURCE_BANDS,
            OPTICAL_MODEL_BANDS,
        )
        .reproject(
            fine_projection
        )
        .toFloat()
    )

    # --------------------------------------------------------
    # Sentinel-1: R077 ascending, temporal median at 10 m,
    # then area aggregation to the 20 m Sentinel-2 grid.
    # --------------------------------------------------------

    s1_collection = (
        get_sentinel1_collection(
            study_features
        )
    )

    s1_period = (
        s1_collection
        .filterDate(
            period_start,
            period_end,
        )
        .filterBounds(
            geometry
        )
    )

    s1_median = build_s1_median(
        s1_period,
        geometry,
    )

    s1 = (
        s1_median
        .select(
            S1_SOURCE_BANDS
        )
        .reproject(
            crs=ANALYSIS_CRS,
            scale=10,
        )
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=4,
        )
        .reproject(
            fine_projection
        )
        .rename(
            S1_MODEL_BANDS
        )
        .toFloat()
    )

    # --------------------------------------------------------
    # Coarse atmospheric context.
    #
    # These are the exact accepted station-period model values
    # from the training population. They are spatially constant
    # within this footprint and are NOT treated as 20 m
    # meteorological observations.
    # --------------------------------------------------------

    meteorological_values = [
        float(
            getattr(
                row,
                feature,
            )
        )
        for feature in METEOROLOGICAL_FEATURES
    ]

    meteorology = (
        ee.Image.constant(
            meteorological_values
        )
        .rename(
            METEOROLOGICAL_FEATURES
        )
        .toFloat()
    )

    harmonics = build_harmonic_image(
        period_start_text
    )

    stack = (
        optical
        .addBands(
            s1
        )
        .addBands(
            meteorology
        )
        .addBands(
            harmonics
        )
        .select(
            COMMON_MODEL_FEATURES
        )
        .reproject(
            fine_projection
        )
        .toFloat()
    )

    return {
        "stack": stack,
        "optical": optical,
        "s1": s1,
        "s2_period": s2_period,
        "s1_period": s1_period,
        "fine_projection": fine_projection,
        "geometry": geometry,
    }


def valid_fraction(
    image: ee.Image,
    geometry: ee.Geometry,
) -> ee.Number:
    valid = (
        ee.Image(
            image
        )
        .mask()
        .reduce(
            ee.Reducer.min()
        )
        .unmask(0)
        .rename(
            "valid"
        )
        .toFloat()
    )

    raw = (
        valid
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            crs=ANALYSIS_CRS,
            scale=PREDICTION_SCALE_M,
            maxPixels=1e6,
            tileScale=4,
        )
        .get(
            "valid"
        )
    )

    return ee.Number(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(
                raw,
                None,
            ),
            0.0,
            raw,
        )
    )


def calculate_fine_station_prediction(
    row,
    footprint: ee.Feature,
    classifier: ee.Classifier,
) -> dict[str, object]:
    context = build_fine_stack(
        row,
        footprint,
    )

    stack = context[
        "stack"
    ]

    geometry = context[
        "geometry"
    ]

    kc_raw = (
        stack
        .classify(
            classifier,
            "Kc_raw",
        )
        .rename(
            "Kc_raw"
        )
        .toFloat()
    )

    kc_stats = (
        kc_raw
        .reduceRegion(
            reducer=(
                ee.Reducer.mean()
                .combine(
                    reducer2=ee.Reducer.stdDev(),
                    sharedInputs=True,
                )
            ),
            geometry=geometry,
            crs=ANALYSIS_CRS,
            scale=PREDICTION_SCALE_M,
            maxPixels=1e6,
            tileScale=4,
        )
    )

    field_point = ee.Geometry.Point(
        [
            float(
                row.longitude
            ),
            float(
                row.latitude
            ),
        ]
    )

    kc_station_raw = (
        kc_raw
        .reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=field_point,
            crs=ANALYSIS_CRS,
            scale=PREDICTION_SCALE_M,
            maxPixels=100,
        )
        .get(
            "Kc_raw"
        )
    )

    station_stack_valid_raw = (
        stack
        .mask()
        .reduce(
            ee.Reducer.min()
        )
        .rename(
            "station_stack_valid"
        )
        .unmask(0)
        .reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=field_point,
            crs=ANALYSIS_CRS,
            scale=PREDICTION_SCALE_M,
            maxPixels=100,
        )
        .get(
            "station_stack_valid"
        )
    )

    summary = ee.Dictionary(
        {
            "kc_station":
                kc_station_raw,

            "kc_mean":
                kc_stats.get(
                    "Kc_raw_mean"
                ),

            "kc_sd":
                kc_stats.get(
                    "Kc_raw_stdDev"
                ),

            "optical_valid_fraction":
                valid_fraction(
                    context[
                        "optical"
                    ],
                    geometry,
                ),

            "s1_valid_fraction":
                valid_fraction(
                    context[
                        "s1"
                    ],
                    geometry,
                ),

            "stack_valid_fraction":
                valid_fraction(
                    stack,
                    geometry,
                ),

            "station_stack_valid":
                station_stack_valid_raw,

            "s2_products":
                context[
                    "s2_period"
                ].size(),

            "s1_products":
                context[
                    "s1_period"
                ].size(),
        }
    ).getInfo()

    def finite_or_nan(
        value,
    ) -> float:
        if value is None:
            return np.nan

        value = float(
            value
        )

        return (
            value
            if np.isfinite(
                value
            )
            else np.nan
        )

    kc_station = finite_or_nan(
        summary.get(
            "kc_station"
        )
    )

    kc_mean = finite_or_nan(
        summary.get(
            "kc_mean"
        )
    )

    kc_sd = finite_or_nan(
        summary.get(
            "kc_sd"
        )
    )

    optical_fraction = float(
        summary.get(
            "optical_valid_fraction",
            0.0,
        )
        or 0.0
    )

    s1_fraction = float(
        summary.get(
            "s1_valid_fraction",
            0.0,
        )
        or 0.0
    )

    stack_fraction = float(
        summary.get(
            "stack_valid_fraction",
            0.0,
        )
        or 0.0
    )

    station_stack_valid = int(
        float(
            summary.get(
                "station_stack_valid",
                0,
            )
            or 0
        )
        > 0
    )

    accepted = bool(
        np.isfinite(
            kc_station
        )
        and np.isfinite(
            kc_mean
        )
        and abs(
            kc_mean
        )
        > 1e-9
        and optical_fraction
        >= OPTICAL_MIN_COVERAGE_FRACTION
        and s1_fraction
        >= S1_MIN_COVERAGE_FRACTION
        and stack_fraction
        >= OPTICAL_MIN_COVERAGE_FRACTION
        and station_stack_valid
        == 1
    )

    et_modis = float(
        row.ET_MODIS_mm_period
    )

    if accepted:
        mass_scale = (
            et_modis
            / kc_mean
        )

        et_downscaled = (
            kc_station
            * mass_scale
        )

        subpixel_sd_et = (
            kc_sd
            * abs(
                mass_scale
            )
            if np.isfinite(
                kc_sd
            )
            else np.nan
        )

    else:
        mass_scale = np.nan
        et_downscaled = np.nan
        subpixel_sd_et = np.nan

    return {
        "status":
            (
                "accepted"
                if accepted
                else "rejected_support"
            ),

        "station":
            row.station,

        "station_id":
            str(
                row.station_id
            ),

        "period_start":
            pd.Timestamp(
                row.period_start
            )
            .date()
            .isoformat(),

        "fold":
            int(
                row.fold
            ),

        "spatial_block":
            str(
                row.spatial_block
            ),

        "number_days":
            int(
                row.number_days
            ),

        "field_actual_et_diagnostic_mm_period":
            float(
                row.field_actual_et_diagnostic_mm_period
            ),

        "Kc_source":
            str(
                row.Kc_source
            ),

        "ET_MODIS_mm_period":
            et_modis,

        "Kc_coarse_oof":
            float(
                row.Kc_predicted_rf_common
            ),

        "ETo_mm_period":
            float(
                row.ETo_mm_period
            ),

        "ET_coarse_oof_mm_period":
            float(
                row.Kc_predicted_rf_common
                * row.ETo_mm_period
            ),

        "Kc_20m_station_oof":
            kc_station,

        "Kc_20m_footprint_mean_oof":
            kc_mean,

        "Kc_20m_footprint_sd_oof":
            kc_sd,

        "mass_scale_mm_per_kc":
            mass_scale,

        "ET_downscaled_20m_oof_mm_period":
            et_downscaled,

        "subpixel_sd_et_mm_period":
            subpixel_sd_et,

        "optical_valid_fraction_rebuilt":
            optical_fraction,

        "s1_valid_fraction_rebuilt":
            s1_fraction,

        "stack_valid_fraction_rebuilt":
            stack_fraction,

        "station_stack_valid":
            station_stack_valid,

        "s2_products":
            int(
                summary.get(
                    "s2_products",
                    0,
                )
                or 0
            ),

        "s1_products":
            int(
                summary.get(
                    "s1_products",
                    0,
                )
                or 0
            ),

        "training_optical_coverage_pct":
            float(
                row.optical_union_coverage_pct
            ),

        "training_s1_coverage_pct":
            float(
                row.s1_union_coverage_pct
            ),

        "longitude":
            float(
                row.longitude
            ),

        "latitude":
            float(
                row.latitude
            ),

        "inside_basin":
            bool(
                row.inside_basin
            ),

        "installation_conforms_manual":
            bool(
                row.installation_conforms_manual
            ),
    }


# ============================================================
# Checkpoint
# ============================================================


def load_checkpoint(
    path: Path,
) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()

    result = pd.read_csv(
        path,
        dtype={
            "station_id": "string",
        },
    )

    return result


def save_checkpoint(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "fold",
                "station",
                "period_start",
            ]
        )
        .to_csv(
            path,
            index=False,
        )
    )


# ============================================================
# Evaluation tables
# ============================================================


def build_metrics_table(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    accepted = pairs.loc[
        pairs[
            "status"
        ]
        == "accepted"
    ].copy()

    subsets = {
        "all_diagnostic_reproduction":
            accepted,

        "external_fixed_kc_only":
            accepted.loc[
                accepted[
                    "Kc_source"
                ]
                == "FAO-56 fixed"
            ],

        "clean_pasture_only_installation_conforming":
            accepted.loc[
                accepted[
                    "station"
                ]
                == "Clean pasture"
            ],
    }

    observed_column = (
        "field_actual_et_diagnostic_mm_period"
    )

    models = {
        "MODIS":
            "ET_MODIS_mm_period",

        "rf_coarse_oof":
            "ET_coarse_oof_mm_period",

        "rf_downscaled_20m_oof":
            "ET_downscaled_20m_oof_mm_period",
    }

    rows = []

    for (
        subset_name,
        subset,
    ) in subsets.items():

        for (
            model_name,
            model_column,
        ) in models.items():

            rows.append(
                {
                    "subset":
                        subset_name,

                    "model":
                        model_name,

                    **calculate_metrics(
                        subset[
                            observed_column
                        ],
                        subset[
                            model_column
                        ],
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_station_analysis(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    accepted = pairs.loc[
        pairs[
            "status"
        ]
        == "accepted"
    ].copy()

    rows = []

    for (
        station,
        group,
    ) in accepted.groupby(
        "station"
    ):

        if len(
            group
        ) < 2:
            continue

        observed = (
            group[
                "field_actual_et_diagnostic_mm_period"
            ]
        )

        coarse = (
            group[
                "ET_MODIS_mm_period"
            ]
        )

        fine = (
            group[
                "ET_downscaled_20m_oof_mm_period"
            ]
        )

        coarse_metrics = (
            calculate_metrics(
                observed,
                coarse,
            )
        )

        fine_metrics = (
            calculate_metrics(
                observed,
                fine,
            )
        )

        bootstrap = (
            paired_bootstrap_delta_rmse(
                observed,
                coarse,
                fine,
            )
        )

        coarse_rmse = (
            coarse_metrics[
                "RMSE"
            ]
        )

        fine_rmse = (
            fine_metrics[
                "RMSE"
            ]
        )

        improvement_pct = (
            100.0
            * (
                1.0
                - fine_rmse
                / coarse_rmse
            )
            if (
                np.isfinite(
                    coarse_rmse
                )
                and coarse_rmse
                != 0
            )
            else np.nan
        )

        rows.append(
            {
                "station":
                    station,

                "n":
                    int(
                        len(
                            group
                        )
                    ),

                "Kc_source":
                    "; ".join(
                        sorted(
                            group[
                                "Kc_source"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                    ),

                "installation_conforms_manual":
                    bool(
                        group[
                            "installation_conforms_manual"
                        ]
                        .iloc[0]
                    ),

                "inside_basin":
                    bool(
                        group[
                            "inside_basin"
                        ]
                        .iloc[0]
                    ),

                "RMSE_MODIS":
                    coarse_rmse,

                "RMSE_downscaled_20m_oof":
                    fine_rmse,

                "MAE_MODIS":
                    coarse_metrics[
                        "MAE"
                    ],

                "MAE_downscaled_20m_oof":
                    fine_metrics[
                        "MAE"
                    ],

                "BIAS_MODIS":
                    coarse_metrics[
                        "BIAS"
                    ],

                "BIAS_downscaled_20m_oof":
                    fine_metrics[
                        "BIAS"
                    ],

                "r_MODIS":
                    coarse_metrics[
                        "r"
                    ],

                "r_downscaled_20m_oof":
                    fine_metrics[
                        "r"
                    ],

                "KGE_MODIS":
                    coarse_metrics[
                        "KGE"
                    ],

                "KGE_downscaled_20m_oof":
                    fine_metrics[
                        "KGE"
                    ],

                "delta_RMSE":
                    bootstrap[
                        "delta_RMSE"
                    ],

                "ci95_low":
                    bootstrap[
                        "ci95_low"
                    ],

                "ci95_high":
                    bootstrap[
                        "ci95_high"
                    ],

                "significant":
                    bootstrap[
                        "significant"
                    ],

                "improvement_pct":
                    improvement_pct,

                "mean_subpixel_sd_et_mm":
                    float(
                        group[
                            "subpixel_sd_et_mm_period"
                        ]
                        .mean()
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    return (
        result
        .sort_values(
            "delta_RMSE",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Figures
# ============================================================


def save_figure(
    fig: plt.Figure,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def make_main_figure(
    pairs: pd.DataFrame,
    station_analysis: pd.DataFrame,
    path: Path,
) -> None:
    accepted = pairs.loc[
        pairs[
            "status"
        ]
        == "accepted"
    ].copy()

    if accepted.empty:
        return

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            14,
            10,
        ),
    )

    axes = axes.ravel()

    station_names = sorted(
        accepted[
            "station"
        ]
        .unique()
        .tolist()
    )

    colors = {
        station:
            plt.get_cmap(
                "tab10"
            )(
                index
                % 10
            )
        for (
            index,
            station,
        ) in enumerate(
            station_names
        )
    }

    # --------------------------------------------------------
    # Panel A: point-to-point agreement
    # --------------------------------------------------------

    columns = [
        "field_actual_et_diagnostic_mm_period",
        "ET_MODIS_mm_period",
        "ET_downscaled_20m_oof_mm_period",
    ]

    minimum = float(
        accepted[
            columns
        ]
        .min()
        .min()
    )

    maximum = float(
        accepted[
            columns
        ]
        .max()
        .max()
    )

    margin = 0.05 * (
        maximum
        - minimum
        if maximum
        > minimum
        else 1.0
    )

    limits = [
        minimum
        - margin,
        maximum
        + margin,
    ]

    axes[
        0
    ].plot(
        limits,
        limits,
        "k--",
        linewidth=1,
    )

    for (
        station,
        group,
    ) in accepted.groupby(
        "station"
    ):

        color = colors[
            station
        ]

        for row in group.itertuples(
            index=False
        ):
            axes[
                0
            ].plot(
                [
                    row.field_actual_et_diagnostic_mm_period,
                    row.field_actual_et_diagnostic_mm_period,
                ],
                [
                    row.ET_MODIS_mm_period,
                    row.ET_downscaled_20m_oof_mm_period,
                ],
                linewidth=0.8,
                alpha=0.35,
                color=color,
            )

        axes[
            0
        ].scatter(
            group[
                "field_actual_et_diagnostic_mm_period"
            ],
            group[
                "ET_MODIS_mm_period"
            ],
            marker="o",
            facecolors="none",
            edgecolors=[
                color
            ],
            s=55,
        )

        axes[
            0
        ].scatter(
            group[
                "field_actual_et_diagnostic_mm_period"
            ],
            group[
                "ET_downscaled_20m_oof_mm_period"
            ],
            marker="s",
            s=45,
            label=station,
            color=[
                color
            ],
        )

    axes[
        0
    ].set_xlim(
        limits
    )

    axes[
        0
    ].set_ylim(
        limits
    )

    axes[
        0
    ].set_xlabel(
        "Field-derived ET (mm period$^{-1}$)"
    )

    axes[
        0
    ].set_ylabel(
        "Satellite ET (mm period$^{-1}$)"
    )

    axes[
        0
    ].set_title(
        "(a) Field agreement: MODIS vs OOF downscaled 20 m",
        loc="left",
    )

    axes[
        0
    ].legend(
        frameon=False,
        fontsize=8,
    )

    # --------------------------------------------------------
    # Panel B: paired RMSE improvement
    # --------------------------------------------------------

    if not station_analysis.empty:
        table = station_analysis.sort_values(
            "delta_RMSE"
        )

        y = np.arange(
            len(
                table
            )
        )

        axes[
            1
        ].barh(
            y,
            table[
                "delta_RMSE"
            ],
        )

        lower_error = (
            table[
                "delta_RMSE"
            ]
            - table[
                "ci95_low"
            ]
        )

        upper_error = (
            table[
                "ci95_high"
            ]
            - table[
                "delta_RMSE"
            ]
        )

        axes[
            1
        ].errorbar(
            table[
                "delta_RMSE"
            ],
            y,
            xerr=[
                lower_error,
                upper_error,
            ],
            fmt="none",
            capsize=4,
        )

        axes[
            1
        ].axvline(
            0,
            linewidth=1,
        )

        labels = [
            f"{row.station} (n={int(row.n)})"
            for row in table.itertuples(
                index=False
            )
        ]

        axes[
            1
        ].set_yticks(
            y
        )

        axes[
            1
        ].set_yticklabels(
            labels
        )

        axes[
            1
        ].set_xlabel(
            "ΔRMSE = MODIS − downscaled (mm period$^{-1}$)\n"
            "positive values favour downscaling"
        )

        axes[
            1
        ].set_title(
            "(b) Paired improvement with bootstrap 95% CI",
            loc="left",
        )

    # --------------------------------------------------------
    # Panel C: subpixel signal vs improvement
    # --------------------------------------------------------

    if len(
        station_analysis
    ) >= 2:
        axes[
            2
        ].axhline(
            0,
            linewidth=1,
        )

        for row in station_analysis.itertuples(
            index=False
        ):
            axes[
                2
            ].scatter(
                row.mean_subpixel_sd_et_mm,
                row.improvement_pct,
                s=55,
                label=row.station,
            )

            axes[
                2
            ].annotate(
                row.station,
                (
                    row.mean_subpixel_sd_et_mm,
                    row.improvement_pct,
                ),
                xytext=(
                    4,
                    4,
                ),
                textcoords="offset points",
                fontsize=8,
            )

        axes[
            2
        ].set_xlabel(
            "Mean within-footprint ET SD from OOF downscaling\n"
            "(mm period$^{-1}$)"
        )

        axes[
            2
        ].set_ylabel(
            "RMSE improvement (%)"
        )

        axes[
            2
        ].set_title(
            "(c) Does stronger subpixel signal yield more improvement?",
            loc="left",
        )

    # --------------------------------------------------------
    # Panel D: station RMSE comparison
    # --------------------------------------------------------

    if not station_analysis.empty:
        table = station_analysis.sort_values(
            "station"
        )

        x = np.arange(
            len(
                table
            )
        )

        width = 0.36

        axes[
            3
        ].bar(
            x
            - width
            / 2,
            table[
                "RMSE_MODIS"
            ],
            width=width,
            label="MODIS parent pixel",
        )

        axes[
            3
        ].bar(
            x
            + width
            / 2,
            table[
                "RMSE_downscaled_20m_oof"
            ],
            width=width,
            label="OOF downscaled 20 m",
        )

        axes[
            3
        ].set_xticks(
            x
        )

        axes[
            3
        ].set_xticklabels(
            table[
                "station"
            ],
            rotation=25,
            ha="right",
        )

        axes[
            3
        ].set_ylabel(
            "RMSE (mm period$^{-1}$)"
        )

        axes[
            3
        ].set_title(
            "(d) Error by station",
            loc="left",
        )

        axes[
            3
        ].legend(
            frameon=False
        )

    fig.suptitle(
        "Spatial OOF field test of 20 m ET downscaling",
        fontsize=14,
    )

    fig.tight_layout()

    save_figure(
        fig,
        path,
    )


def make_series_figure(
    pairs: pd.DataFrame,
    path: Path,
) -> None:
    accepted = pairs.loc[
        pairs[
            "status"
        ]
        == "accepted"
    ].copy()

    stations = sorted(
        accepted[
            "station"
        ]
        .unique()
        .tolist()
    )

    if not stations:
        return

    fig, axes = plt.subplots(
        len(
            stations
        ),
        1,
        figsize=(
            12,
            max(
                3,
                2.5
                * len(
                    stations
                ),
            ),
        ),
        sharex=True,
    )

    if len(
        stations
    ) == 1:
        axes = [
            axes,
        ]

    for (
        axis,
        station,
    ) in zip(
        axes,
        stations,
    ):
        group = (
            accepted.loc[
                accepted[
                    "station"
                ]
                == station
            ]
            .sort_values(
                "period_start"
            )
        )

        dates = pd.to_datetime(
            group[
                "period_start"
            ]
        )

        axis.plot(
            dates,
            group[
                "field_actual_et_diagnostic_mm_period"
            ],
            marker="o",
            label="Field-derived ET",
        )

        axis.plot(
            dates,
            group[
                "ET_MODIS_mm_period"
            ],
            marker="o",
            label="MODIS parent pixel",
        )

        axis.plot(
            dates,
            group[
                "ET_downscaled_20m_oof_mm_period"
            ],
            marker="s",
            label="OOF downscaled 20 m",
        )

        axis.set_ylabel(
            "ET\n(mm period$^{-1}$)"
        )

        axis.set_title(
            station,
            loc="left",
        )

        axis.grid(
            alpha=0.2
        )

    axes[
        0
    ].legend(
        frameon=False,
        ncol=3,
    )

    axes[
        -1
    ].set_xlabel(
        "MODIS period start"
    )

    fig.suptitle(
        "Field, MODIS and spatial OOF 20 m downscaled ET",
        fontsize=14,
    )

    fig.tight_layout()

    save_figure(
        fig,
        path,
    )


# ============================================================
# Main
# ============================================================


def main():
    args = parse_arguments()

    project_root = get_project_root()
    paths = get_paths(
        project_root
    )

    require_inputs(
        paths
    )

    (
        validation,
        oof,
        training,
    ) = prepare_tables(
        paths
    )

    print(
        "Field rows overlapping spatial OOF population:",
        len(
            validation
        ),
    )

    print(
        "Rows by station:"
    )

    print(
        validation[
            "station"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if (
        args.max_rows
        is not None
    ):
        if (
            args.max_rows
            < 1
        ):
            raise ValueError(
                "--max-rows must be >= 1."
            )

        validation = (
            validation
            .head(
                args.max_rows
            )
            .copy()
        )

        print(
            "Rows limited by --max-rows:",
            len(
                validation
            ),
        )

    paths[
        "checkpoint"
    ].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths[
        "figure_main"
    ].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        args.restart
        and paths[
            "checkpoint"
        ].exists()
    ):
        paths[
            "checkpoint"
        ].unlink()

    checkpoint = load_checkpoint(
        paths[
            "checkpoint"
        ]
    )

    completed_keys = set()

    if not checkpoint.empty:
        completed = checkpoint.loc[
            checkpoint[
                "status"
            ]
            .isin(
                [
                    "accepted",
                    "rejected_support",
                ]
            )
        ].copy()

        completed_keys = set(
            zip(
                completed[
                    "station_id"
                ].astype(str),
                completed[
                    "period_start"
                ].astype(str),
            )
        )

        print(
            "Existing completed checkpoint rows:",
            len(
                completed_keys
            ),
        )

    initialize_earth_engine(
        args.project
    )

    footprints = build_station_resources()

    rows = (
        checkpoint.to_dict(
            orient="records"
        )
        if not checkpoint.empty
        else []
    )

    folds_to_process = sorted(
        validation[
            "fold"
        ]
        .unique()
        .tolist()
    )

    for fold in folds_to_process:
        fold_validation = validation.loc[
            validation[
                "fold"
            ]
            == fold
        ].copy()

        pending = []

        for row in fold_validation.itertuples(
            index=False
        ):
            key = (
                str(
                    row.station_id
                ),
                pd.Timestamp(
                    row.period_start
                )
                .date()
                .isoformat(),
            )

            if key not in completed_keys:
                pending.append(
                    row
                )

        if not pending:
            print(
                f"Fold {fold}: all requested field rows already checkpointed."
            )
            continue

        model, test_blocks = train_fold_model(
            fold=fold,
            oof=oof,
            training=training,
        )

        classifier, trees = build_ee_regressor(
            model,
            COMMON_MODEL_FEATURES,
        )

        print(
            f"Fold {fold}: transferred {len(trees)} trees to Earth Engine."
        )

        for (
            row_index,
            row,
        ) in enumerate(
            pending,
            start=1,
        ):
            period_text = (
                pd.Timestamp(
                    row.period_start
                )
                .date()
                .isoformat()
            )

            print(
                f"Fold {fold} | "
                f"{row_index}/{len(pending)} | "
                f"{row.station} | "
                f"{period_text}"
            )

            footprint = get_station_footprint(
                footprints,
                str(
                    row.station_id
                ),
            )

            try:
                result = calculate_fine_station_prediction(
                    row=row,
                    footprint=footprint,
                    classifier=classifier,
                )

            except Exception as error:
                failed = {
                    "status":
                        "error",

                    "station":
                        row.station,

                    "station_id":
                        str(
                            row.station_id
                        ),

                    "period_start":
                        period_text,

                    "fold":
                        int(
                            fold
                        ),

                    "spatial_block":
                        str(
                            row.spatial_block
                        ),

                    "error":
                        repr(
                            error
                        ),
                }

                rows = [
                    existing
                    for existing in rows
                    if not (
                        str(
                            existing.get(
                                "station_id"
                            )
                        )
                        == str(
                            row.station_id
                        )
                        and str(
                            existing.get(
                                "period_start"
                            )
                        )
                        == period_text
                    )
                ]

                rows.append(
                    failed
                )

                save_checkpoint(
                    rows,
                    paths[
                        "checkpoint"
                    ],
                )

                raise

            rows = [
                existing
                for existing in rows
                if not (
                    str(
                        existing.get(
                            "station_id"
                        )
                    )
                    == str(
                        row.station_id
                    )
                    and str(
                        existing.get(
                            "period_start"
                        )
                    )
                    == period_text
                )
            ]

            rows.append(
                result
            )

            save_checkpoint(
                rows,
                paths[
                    "checkpoint"
                ],
            )

            print(
                "  status:",
                result[
                    "status"
                ],
                "| ET field:",
                round(
                    result[
                        "field_actual_et_diagnostic_mm_period"
                    ],
                    3,
                ),
                "| MODIS:",
                round(
                    result[
                        "ET_MODIS_mm_period"
                    ],
                    3,
                ),
                "| OOF 20 m:",
                (
                    round(
                        result[
                            "ET_downscaled_20m_oof_mm_period"
                        ],
                        3,
                    )
                    if np.isfinite(
                        result[
                            "ET_downscaled_20m_oof_mm_period"
                        ]
                    )
                    else "NA"
                ),
            )

    final_pairs = load_checkpoint(
        paths[
            "checkpoint"
        ]
    )

    requested_keys = set(
        (
            str(
                row.station_id
            ),
            pd.Timestamp(
                row.period_start
            )
            .date()
            .isoformat(),
        )
        for row in validation.itertuples(
            index=False
        )
    )

    final_pairs = final_pairs.loc[
        [
            (
                str(
                    station_id
                ),
                str(
                    period_start
                ),
            )
            in requested_keys
            for (
                station_id,
                period_start,
            ) in zip(
                final_pairs[
                    "station_id"
                ],
                final_pairs[
                    "period_start"
                ],
            )
        ]
    ].copy()

    final_pairs = final_pairs.sort_values(
        [
            "station",
            "period_start",
        ]
    ).reset_index(
        drop=True
    )

    final_pairs.to_csv(
        paths[
            "pairs_output"
        ],
        index=False,
    )

    accepted_count = int(
        (
            final_pairs[
                "status"
            ]
            == "accepted"
        ).sum()
    )

    rejected_count = int(
        (
            final_pairs[
                "status"
            ]
            == "rejected_support"
        ).sum()
    )

    print()
    print(
        "=== FINE OOF FIELD VALIDATION SUPPORT ==="
    )

    print(
        "Requested rows:",
        len(
            validation
        ),
    )

    print(
        "Accepted fine rows:",
        accepted_count,
    )

    print(
        "Rejected fine-support rows:",
        rejected_count,
    )

    if accepted_count == 0:
        print(
            "No fine rows were accepted. "
            "Metrics and figures were not generated."
        )
        return

    metrics = build_metrics_table(
        final_pairs
    )

    station_analysis = build_station_analysis(
        final_pairs
    )

    metrics.to_csv(
        paths[
            "metrics_output"
        ],
        index=False,
    )

    station_analysis.to_csv(
        paths[
            "station_output"
        ],
        index=False,
    )

    make_main_figure(
        final_pairs,
        station_analysis,
        paths[
            "figure_main"
        ],
    )

    make_series_figure(
        final_pairs,
        paths[
            "figure_series"
        ],
    )

    print()
    print(
        "=== FIELD COMPARISON: MODIS VS OOF 20 m ==="
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print()
    print(
        "=== BY STATION ==="
    )

    if station_analysis.empty:
        print(
            "Not enough accepted rows for station-level metrics."
        )
    else:
        columns = [
            "station",
            "n",
            "RMSE_MODIS",
            "RMSE_downscaled_20m_oof",
            "delta_RMSE",
            "ci95_low",
            "ci95_high",
            "improvement_pct",
            "r_MODIS",
            "r_downscaled_20m_oof",
        ]

        print(
            station_analysis[
                columns
            ]
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Interpretation safeguards:"
    )

    print(
        "- all_diagnostic_reproduction includes NDVI-derived field Kc "
        "for Mangrove and Dry forest."
    )

    print(
        "- external_fixed_kc_only is the more independent ET-conversion subset."
    )

    print(
        "- Clean pasture is the only station flagged as conforming to "
        "the ETgage installation guidance."
    )

    print(
        "- The RF is spatial OOF: the held-out station block is excluded "
        "from training."
    )

    print(
        "- The 20 m result is reconciled to its parent MODIS ET, so this test "
        "evaluates local spatial redistribution rather than independent "
        "prediction of the parent-pixel ET amount."
    )

    print()
    print(
        "Saved pairs:",
        paths[
            "pairs_output"
        ],
    )

    print(
        "Saved metrics:",
        paths[
            "metrics_output"
        ],
    )

    print(
        "Saved station analysis:",
        paths[
            "station_output"
        ],
    )

    print(
        "Saved figure:",
        paths[
            "figure_main"
        ],
    )

    print(
        "Saved series figure:",
        paths[
            "figure_series"
        ],
    )


if __name__ == "__main__":
    main()
