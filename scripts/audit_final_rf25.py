"""Final local scientific audit for the frozen RF-25 workflow.

The audit performs no Earth Engine access and no data download. It compares
RF-25 with Ridge-25 and a fold-specific training-mean baseline on exactly the
same final population, verifies exact OOF population identity, evaluates fold
stability and random-state sensitivity, quantifies spatial uncertainty with a
cluster bootstrap, summarizes monthly RF25 population retention and the
RF-weighted AOA, and audits the three selected production rasters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PROJ_NETWORK", "OFF")

import joblib
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from et_downscaling.metrics import TARGET_COLUMN, calculate_metrics
from et_downscaling.rf25 import (
    RF25_MODEL_FEATURES,
    build_rf25_model,
)


ROOT = Path.cwd().resolve()
OUTPUTS = ROOT / "outputs"
RESULTS = OUTPUTS / "evaluation" / "results"
AUDIT = OUTPUTS / "evaluation" / "audit" / "final_rf25_closure"
AUDIT.mkdir(parents=True, exist_ok=True)

POPULATION = RESULTS / "virtual10_training_population.csv"
RF_SPATIAL = RESULTS / "rf25_spatial_oof.csv"
RF_TEMPORAL = RESULTS / "rf25_temporal_oof.csv"
AOA_TABLE = RESULTS / "rf25_training_aoa.csv"
WEIGHTS_TABLE = RESULTS / "rf25_aoa_weights.csv"
AOA_MODEL = (
    OUTPUTS / "current" / "models" / "rf25_weighted_aoa.joblib"
)
MASTER = (
    OUTPUTS
    / "training"
    / "master"
    / "virtual10_all_predictors_2020_2024.parquet"
)

EXPECTED_POPULATION_ROWS = 1526

RANDOM_STATE_SEEDS = [
    1,
    7,
    21,
    42,
    77,
    101,
    202,
    303,
    404,
    505,
]

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 42


required_paths = [
    POPULATION,
    RF_SPATIAL,
    RF_TEMPORAL,
    AOA_TABLE,
    WEIGHTS_TABLE,
    AOA_MODEL,
    MASTER,
]

missing = [str(path) for path in required_paths if not path.is_file()]

if missing:
    raise FileNotFoundError(
        "Missing required audit inputs:\n" + "\n".join(missing)
    )


population = pd.read_csv(
    POPULATION,
    dtype={"station_id": str},
)

population["period_start"] = pd.to_datetime(
    population["period_start"],
    errors="raise",
)

if "year" not in population.columns:
    population["year"] = population["period_start"].dt.year.astype(int)

if population["station_id"].nunique() != 10:
    raise RuntimeError("Expected exactly 10 Virtual10 supports.")

if population["spatial_block"].nunique() != 10:
    raise RuntimeError("Expected exactly 10 spatial blocks.")

required_columns = set(
    RF25_MODEL_FEATURES
    + [
        TARGET_COLUMN,
        "station_id",
        "spatial_block",
        "year",
    ]
)

missing_columns = sorted(
    required_columns - set(population.columns)
)

if missing_columns:
    raise RuntimeError(
        "Population missing columns: "
        + ", ".join(missing_columns)
    )


if len(population) != EXPECTED_POPULATION_ROWS:
    raise RuntimeError(
        "Expected exactly "
        f"{EXPECTED_POPULATION_ROWS} RF25 population rows; "
        f"found {len(population)}."
    )

population_keys = [
    "station_id",
    "period_start",
]

if population.duplicated(population_keys).any():
    raise RuntimeError(
        "RF25 population contains duplicate "
        "station_id x period_start keys."
    )

training_years = sorted(
    population["year"]
    .astype(int)
    .unique()
    .tolist()
)

if training_years != [
    2020,
    2021,
    2022,
    2023,
    2024,
]:
    raise RuntimeError(
        "Unexpected RF25 training years: "
        f"{training_years}"
    )

feature_matrix = population[
    RF25_MODEL_FEATURES
].to_numpy(dtype=float)

if not np.isfinite(feature_matrix).all():
    raise RuntimeError(
        "RF25 training population contains "
        "missing or non-finite predictor values."
    )

target_values = population[
    TARGET_COLUMN
].to_numpy(dtype=float)

if not np.isfinite(target_values).all():
    raise RuntimeError(
        "RF25 training population contains "
        "missing or non-finite target values."
    )


def build_ridge25():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                Ridge(
                    alpha=1.0,
                    fit_intercept=True,
                ),
            ),
        ]
    )


def grouped_prediction(
    data,
    group_column,
    model_type,
):
    y = data[TARGET_COLUMN].to_numpy(dtype=float)
    groups = data[group_column].astype(str)

    prediction = np.full(
        len(data),
        np.nan,
        dtype=float,
    )

    for group in sorted(groups.unique()):
        test = groups.eq(group).to_numpy()
        train = ~test

        if model_type == "ridge":
            model = build_ridge25()

            model.fit(
                data.loc[
                    train,
                    RF25_MODEL_FEATURES,
                ],
                y[train],
            )

            pred = model.predict(
                data.loc[
                    test,
                    RF25_MODEL_FEATURES,
                ]
            )

        elif model_type == "training_mean":
            pred = np.full(
                int(test.sum()),
                float(y[train].mean()),
            )

        else:
            raise ValueError(model_type)

        prediction[test] = pred

    output = data[
        [
            "station_id",
            "period_start",
            "spatial_block",
            "year",
            TARGET_COLUMN,
        ]
    ].copy()

    output["prediction"] = prediction

    return output


def load_rf(path):
    table = pd.read_csv(
        path,
        dtype={"station_id": str},
    )

    table["period_start"] = pd.to_datetime(
        table["period_start"],
        errors="raise",
    )

    return table


def validate_rf_oof(
    table,
    evaluation_name,
):
    required = {
        "station_id",
        "period_start",
        "spatial_block",
        "year",
        TARGET_COLUMN,
        "prediction",
    }

    missing = sorted(
        required - set(table.columns)
    )

    if missing:
        raise RuntimeError(
            f"{evaluation_name}: missing OOF columns: "
            + ", ".join(missing)
        )

    if len(table) != EXPECTED_POPULATION_ROWS:
        raise RuntimeError(
            f"{evaluation_name}: expected "
            f"{EXPECTED_POPULATION_ROWS} rows; "
            f"found {len(table)}."
        )

    keys = [
        "station_id",
        "period_start",
    ]

    if table.duplicated(keys).any():
        raise RuntimeError(
            f"{evaluation_name}: duplicate "
            "station_id x period_start keys."
        )

    predictions = table[
        "prediction"
    ].to_numpy(dtype=float)

    if not np.isfinite(predictions).all():
        raise RuntimeError(
            f"{evaluation_name}: predictions contain "
            "missing or non-finite values."
        )

    reference = population[
        [
            "station_id",
            "period_start",
            "spatial_block",
            "year",
            TARGET_COLUMN,
        ]
    ].copy()

    candidate = table[
        [
            "station_id",
            "period_start",
            "spatial_block",
            "year",
            TARGET_COLUMN,
            "prediction",
        ]
    ].copy()

    merged = reference.merge(
        candidate,
        on=keys,
        how="outer",
        suffixes=(
            "_population",
            "_oof",
        ),
        indicator=True,
        validate="one_to_one",
    )

    if not merged["_merge"].eq(
        "both"
    ).all():
        raise RuntimeError(
            f"{evaluation_name}: OOF keys differ "
            "from the RF25 population."
        )

    if not (
        merged[
            "spatial_block_population"
        ].astype(str)
        == merged[
            "spatial_block_oof"
        ].astype(str)
    ).all():
        raise RuntimeError(
            f"{evaluation_name}: spatial block "
            "mismatch between OOF and population."
        )

    if not (
        merged[
            "year_population"
        ].astype(int)
        == merged[
            "year_oof"
        ].astype(int)
    ).all():
        raise RuntimeError(
            f"{evaluation_name}: year mismatch "
            "between OOF and population."
        )

    np.testing.assert_allclose(
        merged[
            f"{TARGET_COLUMN}_population"
        ].to_numpy(dtype=float),
        merged[
            f"{TARGET_COLUMN}_oof"
        ].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
        err_msg=(
            f"{evaluation_name}: target differs "
            "between OOF and population."
        ),
    )

    print(
        f"{evaluation_name}: exact OOF identity PASS | "
        f"rows={len(table)}"
    )


def rf_grouped_prediction(
    data,
    group_column,
    random_state,
):
    y = data[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    groups = data[
        group_column
    ].astype(str)

    prediction = np.full(
        len(data),
        np.nan,
        dtype=float,
    )

    for group in sorted(
        groups.unique()
    ):
        test = groups.eq(
            group
        ).to_numpy()

        train = ~test

        model = build_rf25_model()

        model.set_params(
            random_state=int(
                random_state
            )
        )

        model.fit(
            data.loc[
                train,
                RF25_MODEL_FEATURES,
            ],
            y[train],
        )

        prediction[test] = (
            model.predict(
                data.loc[
                    test,
                    RF25_MODEL_FEATURES,
                ]
            )
        )

    output = data[
        [
            "station_id",
            "period_start",
            "spatial_block",
            "year",
            TARGET_COLUMN,
        ]
    ].copy()

    output["prediction"] = prediction

    return output


def assert_predictions_match_oof(
    generated,
    stored,
    evaluation_name,
):
    keys = [
        "station_id",
        "period_start",
    ]

    paired = generated[
        keys + ["prediction"]
    ].merge(
        stored[
            keys + ["prediction"]
        ],
        on=keys,
        how="inner",
        suffixes=(
            "_generated",
            "_stored",
        ),
        validate="one_to_one",
    )

    if len(paired) != EXPECTED_POPULATION_ROWS:
        raise RuntimeError(
            f"{evaluation_name}: seed-42 comparison "
            "did not recover all population rows."
        )

    np.testing.assert_allclose(
        paired[
            "prediction_generated"
        ].to_numpy(dtype=float),
        paired[
            "prediction_stored"
        ].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
        err_msg=(
            f"{evaluation_name}: regenerated seed-42 "
            "RF25 predictions differ from stored OOF."
        ),
    )


print("=" * 100)
print("FINAL RF-25 LOCAL SCIENTIFIC AUDIT")
print("=" * 100)

print(
    f"Population: {len(population)} rows | "
    f"supports={population['station_id'].nunique()} | "
    f"blocks={population['spatial_block'].nunique()}"
)


# ------------------------------------------------------------------
# 1. Same-population RF vs Ridge vs simple baseline
# ------------------------------------------------------------------

print("\n[1/7] Same-population model comparison...")

rf_spatial = load_rf(RF_SPATIAL)
rf_temporal = load_rf(RF_TEMPORAL)

validate_rf_oof(
    rf_spatial,
    "RF25 spatial OOF",
)

validate_rf_oof(
    rf_temporal,
    "RF25 temporal OOF",
)

ridge_spatial = grouped_prediction(
    population,
    "spatial_block",
    "ridge",
)

ridge_temporal = grouped_prediction(
    population,
    "year",
    "ridge",
)

mean_spatial = grouped_prediction(
    population,
    "spatial_block",
    "training_mean",
)

mean_temporal = grouped_prediction(
    population,
    "year",
    "training_mean",
)

comparison_rows = []

for model_name, evaluation, table in [
    (
        "RF25",
        "spatial_block_OOF",
        rf_spatial,
    ),
    (
        "Ridge25",
        "spatial_block_OOF",
        ridge_spatial,
    ),
    (
        "TrainingMean",
        "spatial_block_OOF",
        mean_spatial,
    ),
    (
        "RF25",
        "leave_one_year_out",
        rf_temporal,
    ),
    (
        "Ridge25",
        "leave_one_year_out",
        ridge_temporal,
    ),
    (
        "TrainingMean",
        "leave_one_year_out",
        mean_temporal,
    ),
]:
    comparison_rows.append(
        {
            "model": model_name,
            "evaluation": evaluation,
            **calculate_metrics(
                table[TARGET_COLUMN],
                table["prediction"],
            ),
        }
    )

comparison = pd.DataFrame(comparison_rows)

comparison.to_csv(
    AUDIT / "model_comparison_same_1526.csv",
    index=False,
)

print(comparison.to_string(index=False))


# ------------------------------------------------------------------
# 2. Fold stability
# ------------------------------------------------------------------

print("\n[2/7] Fold stability...")


def fold_metrics(
    table,
    group_column,
    model_name,
    evaluation,
):
    rows = []

    groups = table[group_column].astype(str)

    for group in sorted(groups.unique()):
        subset = table.loc[groups.eq(group)]

        rows.append(
            {
                "model": model_name,
                "evaluation": evaluation,
                "group": group,
                **calculate_metrics(
                    subset[TARGET_COLUMN],
                    subset["prediction"],
                ),
            }
        )

    return pd.DataFrame(rows)


fold_table = pd.concat(
    [
        fold_metrics(
            rf_spatial,
            "spatial_block",
            "RF25",
            "spatial_block_OOF",
        ),
        fold_metrics(
            ridge_spatial,
            "spatial_block",
            "Ridge25",
            "spatial_block_OOF",
        ),
        fold_metrics(
            mean_spatial,
            "spatial_block",
            "TrainingMean",
            "spatial_block_OOF",
        ),
        fold_metrics(
            rf_temporal,
            "year",
            "RF25",
            "leave_one_year_out",
        ),
        fold_metrics(
            ridge_temporal,
            "year",
            "Ridge25",
            "leave_one_year_out",
        ),
        fold_metrics(
            mean_temporal,
            "year",
            "TrainingMean",
            "leave_one_year_out",
        ),
    ],
    ignore_index=True,
)

fold_table.to_csv(
    AUDIT / "fold_metrics.csv",
    index=False,
)

summary_rows = []

for evaluation in [
    "spatial_block_OOF",
    "leave_one_year_out",
]:
    for model_name in [
        "RF25",
        "Ridge25",
        "TrainingMean",
    ]:
        subset = fold_table.loc[
            fold_table["evaluation"].eq(evaluation)
            & fold_table["model"].eq(model_name)
        ]

        summary_rows.append(
            {
                "evaluation": evaluation,
                "model": model_name,
                "folds": len(subset),
                "negative_R2_folds": int(
                    (subset["R2"] < 0).sum()
                ),
                "median_R2": subset["R2"].median(),
                "median_RMSE": subset["RMSE"].median(),
                "median_MAE": subset["MAE"].median(),
                "max_abs_BIAS": subset["BIAS"].abs().max(),
            }
        )

fold_summary = pd.DataFrame(summary_rows)

fold_summary.to_csv(
    AUDIT / "fold_stability_summary.csv",
    index=False,
)

print(fold_summary.to_string(index=False))


# Paired RF-Ridge absolute errors

paired_rows = []

for evaluation, rf_table, ridge_table in [
    (
        "spatial_block_OOF",
        rf_spatial,
        ridge_spatial,
    ),
    (
        "leave_one_year_out",
        rf_temporal,
        ridge_temporal,
    ),
]:
    rf_pair = rf_table[
        [
            "station_id",
            "period_start",
            TARGET_COLUMN,
            "prediction",
        ]
    ].rename(
        columns={
            "prediction": "rf_prediction",
        }
    )

    ridge_pair = ridge_table[
        [
            "station_id",
            "period_start",
            "prediction",
        ]
    ].rename(
        columns={
            "prediction": "ridge_prediction",
        }
    )

    paired = rf_pair.merge(
        ridge_pair,
        on=[
            "station_id",
            "period_start",
        ],
        validate="one_to_one",
    )

    rf_error = (
        paired["rf_prediction"]
        - paired[TARGET_COLUMN]
    ).abs()

    ridge_error = (
        paired["ridge_prediction"]
        - paired[TARGET_COLUMN]
    ).abs()

    paired_rows.append(
        {
            "evaluation": evaluation,
            "n": len(paired),
            "RF_lower_abs_error": int(
                (rf_error < ridge_error).sum()
            ),
            "Ridge_lower_abs_error": int(
                (ridge_error < rf_error).sum()
            ),
            "RF_lower_abs_error_pct": (
                (rf_error < ridge_error).mean()
                * 100
            ),
        }
    )

paired_summary = pd.DataFrame(paired_rows)

paired_summary.to_csv(
    AUDIT / "paired_rf_vs_ridge.csv",
    index=False,
)

print("\nPaired RF vs Ridge:")
print(paired_summary.to_string(index=False))


# ------------------------------------------------------------------
# 3. RF25 random-state sensitivity
# ------------------------------------------------------------------

print(
    "\n[3/7] RF25 random-state sensitivity..."
)

seed_rows = []

for seed in RANDOM_STATE_SEEDS:

    seed_spatial = rf_grouped_prediction(
        population,
        "spatial_block",
        seed,
    )

    seed_temporal = rf_grouped_prediction(
        population,
        "year",
        seed,
    )

    if seed == 42:
        assert_predictions_match_oof(
            seed_spatial,
            rf_spatial,
            "spatial_block_OOF",
        )

        assert_predictions_match_oof(
            seed_temporal,
            rf_temporal,
            "leave_one_year_out",
        )

        print(
            "Seed 42 regenerated predictions "
            "match stored RF25 OOF."
        )

    for evaluation, table in [
        (
            "spatial_block_OOF",
            seed_spatial,
        ),
        (
            "leave_one_year_out",
            seed_temporal,
        ),
    ]:
        seed_rows.append(
            {
                "random_state": seed,
                "evaluation": evaluation,
                **calculate_metrics(
                    table[
                        TARGET_COLUMN
                    ],
                    table[
                        "prediction"
                    ],
                ),
            }
        )

seed_results = pd.DataFrame(
    seed_rows
)

seed_results.to_csv(
    AUDIT
    / "rf25_random_state_sensitivity.csv",
    index=False,
)

seed_summary_rows = []

metric_names = [
    "R2",
    "RMSE",
    "MAE",
    "BIAS",
    "KGE",
]

for evaluation in [
    "spatial_block_OOF",
    "leave_one_year_out",
]:

    subset = seed_results.loc[
        seed_results[
            "evaluation"
        ].eq(evaluation)
    ]

    row = {
        "evaluation": evaluation,
        "seeds": len(subset),
    }

    for metric in metric_names:
        values = subset[
            metric
        ].to_numpy(dtype=float)

        row[
            f"{metric}_mean"
        ] = float(
            np.mean(values)
        )

        row[
            f"{metric}_sd"
        ] = float(
            np.std(
                values,
                ddof=1,
            )
        )

        row[
            f"{metric}_min"
        ] = float(
            np.min(values)
        )

        row[
            f"{metric}_max"
        ] = float(
            np.max(values)
        )

    seed_summary_rows.append(
        row
    )

seed_summary = pd.DataFrame(
    seed_summary_rows
)

seed_summary.to_csv(
    AUDIT
    / "rf25_random_state_sensitivity_summary.csv",
    index=False,
)

print(seed_summary.to_string(index=False))


# ------------------------------------------------------------------
# 4. Spatial cluster bootstrap
# ------------------------------------------------------------------

print(
    "\n[4/7] Spatial cluster bootstrap..."
)

bootstrap_rng = (
    np.random.default_rng(
        BOOTSTRAP_SEED
    )
)

bootstrap_groups = (
    rf_spatial[
        "spatial_block"
    ]
    .astype(str)
    .to_numpy()
)

block_labels = np.array(
    sorted(
        np.unique(
            bootstrap_groups
        )
    ),
    dtype=object,
)

if len(block_labels) != 10:
    raise RuntimeError(
        "Cluster bootstrap expected exactly "
        f"10 spatial blocks; found {len(block_labels)}."
    )

block_indices = {
    block: np.flatnonzero(
        bootstrap_groups == block
    )
    for block in block_labels
}

observed = rf_spatial[
    TARGET_COLUMN
].to_numpy(dtype=float)

predicted = rf_spatial[
    "prediction"
].to_numpy(dtype=float)

point_metrics = calculate_metrics(
    observed,
    predicted,
)

bootstrap_rows = []

for replicate in range(
    BOOTSTRAP_REPLICATES
):
    sampled_blocks = (
        bootstrap_rng.choice(
            block_labels,
            size=len(block_labels),
            replace=True,
        )
    )

    sampled_indices = np.concatenate(
        [
            block_indices[block]
            for block
            in sampled_blocks
        ]
    )

    metrics = calculate_metrics(
        observed[
            sampled_indices
        ],
        predicted[
            sampled_indices
        ],
    )

    bootstrap_rows.append(
        {
            "replicate": (
                replicate + 1
            ),
            "unique_blocks": int(
                len(
                    np.unique(
                        sampled_blocks
                    )
                )
            ),
            **metrics,
        }
    )

bootstrap_table = pd.DataFrame(
    bootstrap_rows
)

bootstrap_table.to_csv(
    AUDIT
    / "rf25_spatial_cluster_bootstrap_replicates.csv",
    index=False,
)

bootstrap_summary_rows = []

for metric in metric_names:

    values = bootstrap_table[
        metric
    ].to_numpy(dtype=float)

    finite = values[
        np.isfinite(values)
    ]

    bootstrap_summary_rows.append(
        {
            "metric": metric,
            "point_estimate": float(
                point_metrics[
                    metric
                ]
            ),
            "bootstrap_median": float(
                np.median(finite)
            ),
            "bootstrap_q025": float(
                np.quantile(
                    finite,
                    0.025,
                )
            ),
            "bootstrap_q975": float(
                np.quantile(
                    finite,
                    0.975,
                )
            ),
            "valid_replicates": int(
                len(finite)
            ),
            "requested_replicates": (
                BOOTSTRAP_REPLICATES
            ),
            "bootstrap_seed": (
                BOOTSTRAP_SEED
            ),
        }
    )

bootstrap_summary = pd.DataFrame(
    bootstrap_summary_rows
)

bootstrap_summary.to_csv(
    AUDIT
    / "rf25_spatial_cluster_bootstrap_summary.csv",
    index=False,
)

print(
    bootstrap_summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------------
# 5. RF25 population retention by month
# ------------------------------------------------------------------

print(
    "\n[5/7] RF25 population retention by month..."
)

master = pd.read_parquet(
    MASTER,
    columns=[
        "station_id",
        "period_start",
    ],
)

master[
    "station_id"
] = master[
    "station_id"
].astype(str)

master["period_start"] = pd.to_datetime(
    master["period_start"],
    errors="raise",
)

population_supports = set(
    population[
        "station_id"
    ].astype(str)
)

master = master.loc[
    master[
        "station_id"
    ].isin(
        population_supports
    )
].copy()

master_supports = set(
    master[
        "station_id"
    ].astype(str)
)

if master_supports != population_supports:
    raise RuntimeError(
        "Candidate master and RF25 population "
        "do not contain the same Virtual10 supports."
    )

if master.duplicated(
    [
        "station_id",
        "period_start",
    ]
).any():
    raise RuntimeError(
        "Candidate master contains duplicate "
        "station_id x period_start rows."
    )

master["month"] = (
    master[
        "period_start"
    ].dt.month
)

selected = population[
    [
        "station_id",
        "period_start",
    ]
].copy()

selected["month"] = (
    selected[
        "period_start"
    ].dt.month
)

master_rows = (
    master.groupby(
        "month"
    )
    .size()
    .rename(
        "candidate_master_rows"
    )
)

rf25_rows = (
    selected.groupby(
        "month"
    )
    .size()
    .rename(
        "rf25_population_rows"
    )
)

monthly = pd.concat(
    [
        master_rows,
        rf25_rows,
    ],
    axis=1,
).fillna(0).reset_index()

monthly[
    "retention_pct"
] = (
    monthly[
        "rf25_population_rows"
    ]
    / monthly[
        "candidate_master_rows"
    ]
    * 100
)

monthly.to_csv(
    AUDIT
    / "rf25_population_retention_by_month.csv",
    index=False,
)

print(monthly.to_string(index=False))

print(
    "\nMonthly RF25 population retention range: "
    f"{monthly['retention_pct'].min():.2f}% - "
    f"{monthly['retention_pct'].max():.2f}%"
)

print(
    "Interpretation: this is monthly retention "
    "of the final RF25 population relative to the "
    "Virtual10 candidate master. It is not an "
    "independent climatology of Sentinel-2 GE90 "
    "availability."
)


# ------------------------------------------------------------------
# 6. Training AOA
# ------------------------------------------------------------------

print("\n[6/7] RF-weighted AOA audit...")

aoa = joblib.load(AOA_MODEL)

aoa_training = pd.read_csv(AOA_TABLE)
weights = pd.read_csv(WEIGHTS_TABLE)

training_di = pd.to_numeric(
    aoa_training["training_DI"],
    errors="raise",
).to_numpy(dtype=float)

training_lpd = pd.to_numeric(
    aoa_training["training_LPD"],
    errors="raise",
).to_numpy(dtype=float)

aoa_summary = {
    "threshold": float(aoa.threshold),
    "mean_training_distance": float(
        aoa.mean_training_distance
    ),
    "training_DI_min": float(
        np.min(training_di)
    ),
    "training_DI_q25": float(
        np.quantile(training_di, 0.25)
    ),
    "training_DI_median": float(
        np.median(training_di)
    ),
    "training_DI_q75": float(
        np.quantile(training_di, 0.75)
    ),
    "training_DI_max": float(
        np.max(training_di)
    ),
    "training_rows_outside_threshold": int(
        (training_di > aoa.threshold).sum()
    ),
    "training_LPD_min": float(
        np.min(training_lpd)
    ),
    "training_LPD_median": float(
        np.median(training_lpd)
    ),
    "training_LPD_mean": float(
        np.mean(training_lpd)
    ),
    "training_LPD_max": float(
        np.max(training_lpd)
    ),
}

(
    AUDIT / "aoa_training_summary.json"
).write_text(
    json.dumps(
        aoa_summary,
        indent=2,
    ),
    encoding="utf-8",
)

for key, value in aoa_summary.items():
    print(f"{key}: {value}")

print("\nTop 10 AOA weights:")
print(
    weights.head(10).to_string(index=False)
)


# ------------------------------------------------------------------
# 7. Existing raster audit
# ------------------------------------------------------------------

print("\n[7/7] Three-date raster QC...")

dates = [
    "2020-03-13",
    "2022-10-24",
    "2022-03-30",
]

raster_rows = []

for date in dates:

    folder = (
        OUTPUTS
        / "current"
        / "rasters"
        / date
    )

    metadata_files = sorted(
        folder.glob(
            "production_metadata_rf25_*.json"
        )
    )

    if len(metadata_files) != 1:
        raise RuntimeError(
            f"{date}: expected exactly one "
            "RF25 production metadata file; "
            f"found {len(metadata_files)}."
        )

    metadata = json.loads(
        metadata_files[0].read_text(
            encoding="utf-8"
        )
    )

    raster_path = Path(
        metadata["raster"]
    )

    if not raster_path.is_file():
        raster_path = (
            folder / raster_path.name
        )

    if not raster_path.is_file():
        raise FileNotFoundError(
            raster_path
        )

    with rasterio.open(
        raster_path
    ) as src:

        bands = {
            name: index + 1
            for index, name
            in enumerate(src.descriptions)
            if name
        }

        required = [
            "ET_mm_period",
            "Kc_raw",
            "dissimilarity_index",
            "local_point_density",
            "stack_valid",
            "AOA_inside",
            "usable",
        ]

        missing_bands = [
            band
            for band in required
            if band not in bands
        ]

        if missing_bands:
            raise RuntimeError(
                f"{date}: missing raster bands "
                f"{missing_bands}"
            )

        basin_cells = 0
        stack_cells = 0
        aoa_cells = 0
        usable_cells = 0
        published_cells = 0

        negative_published = 0
        negative_kc_stack = 0
        negative_kc_aoa = 0

        for _, window in src.block_windows(
            bands["ET_mm_period"]
        ):

            et = src.read(
                bands["ET_mm_period"],
                window=window,
                masked=True,
            )

            kc = src.read(
                bands["Kc_raw"],
                window=window,
                masked=True,
            )

            stack = src.read(
                bands["stack_valid"],
                window=window,
                masked=True,
            )

            inside = src.read(
                bands["AOA_inside"],
                window=window,
                masked=True,
            )

            usable = src.read(
                bands["usable"],
                window=window,
                masked=True,
            )

            basin = (
                ~np.ma.getmaskarray(stack)
            )

            stack_bool = (
                basin
                & (stack.data > 0.5)
            )

            inside_bool = (
                basin
                & (inside.data > 0.5)
            )

            usable_bool = (
                basin
                & (usable.data > 0.5)
            )

            et_valid = (
                ~np.ma.getmaskarray(et)
            )

            kc_valid = (
                ~np.ma.getmaskarray(kc)
            )

            basin_cells += int(
                basin.sum()
            )

            stack_cells += int(
                stack_bool.sum()
            )

            aoa_cells += int(
                (
                    stack_bool
                    & inside_bool
                ).sum()
            )

            usable_cells += int(
                usable_bool.sum()
            )

            published_cells += int(
                et_valid.sum()
            )

            negative_published += int(
                (
                    et_valid
                    & (et.data < 0)
                ).sum()
            )

            negative_kc_stack += int(
                (
                    stack_bool
                    & kc_valid
                    & (kc.data < 0)
                ).sum()
            )

            negative_kc_aoa += int(
                (
                    stack_bool
                    & inside_bool
                    & kc_valid
                    & (kc.data < 0)
                ).sum()
            )

    tolerance = float(
        metadata[
            "conservation_tolerance_mm"
        ]
    )

    error_after = float(
        metadata[
            "max_abs_conservation_error_after_floor_mm"
        ]
    )

    raster_rows.append(
        {
            "date": date,
            "basin_cells": basin_cells,
            "stack_valid_pct_basin": (
                stack_cells
                / basin_cells
                * 100
            ),
            "aoa_inside_pct_of_stack": (
                aoa_cells
                / stack_cells
                * 100
                if stack_cells
                else np.nan
            ),
            "usable_pct_basin": (
                usable_cells
                / basin_cells
                * 100
            ),
            "published_pct_basin": (
                published_cells
                / basin_cells
                * 100
            ),
            "negative_published_et": (
                negative_published
            ),
            "negative_kc_stack": (
                negative_kc_stack
            ),
            "negative_kc_inside_aoa": (
                negative_kc_aoa
            ),
            "eligible_modis_parents": int(
                metadata[
                    "eligible_modis_parents"
                ]
            ),
            "max_conservation_error_mm": (
                error_after
            ),
            "tolerance_mm": tolerance,
            "conservation_pass": (
                error_after
                <= tolerance + 1e-12
            ),
            "mae_adjustment_mm": float(
                metadata[
                    "mae_adjustment_mm"
                ]
            ),
            "rmse_adjustment_mm": float(
                metadata[
                    "rmse_adjustment_mm"
                ]
            ),
            "pearson_final_vs_initial": float(
                metadata[
                    "pearson_final_vs_initial"
                ]
            ),
        }
    )

raster_qc = pd.DataFrame(
    raster_rows
)

raster_qc.to_csv(
    AUDIT / "raster_qc_three_dates.csv",
    index=False,
)

print(raster_qc.to_string(index=False))


print("\n" + "=" * 100)
print("FINAL MACHINE CHECKS")
print("=" * 100)

print(
    "RF population rows:",
    len(population),
)

print(
    "All raster conservation tests pass:",
    bool(
        raster_qc[
            "conservation_pass"
        ].all()
    ),
)

print(
    "Negative ET in published rasters:",
    int(
        raster_qc[
            "negative_published_et"
        ].sum()
    ),
)

print(
    "Audit outputs:",
    AUDIT,
)

print(
    "\nFINAL RF-25 LOCAL AUDIT COMPLETED SUCCESSFULLY"
)
