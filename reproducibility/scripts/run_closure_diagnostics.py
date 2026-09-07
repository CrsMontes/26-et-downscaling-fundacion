"""Run the final offline closure diagnostics for the Ridge-25 workflow.

This script does not contact Earth Engine and does not modify production
outputs. It consolidates the remaining manuscript/closure checks that can be
computed from the latest saved run and final field-comparison tables:

1. Reproduce the current equal-weight DI/AOA from the 833-row population.
2. Build a model-weighted sensitivity in which standardized predictors are
   multiplied by |standardized Ridge coefficient|. This is a project-specific
   sensitivity, not a claim of exact CAST model-importance weighting.
3. Compare both DI definitions against spatial OOF errors without using those
   errors to tune the AOA.
4. Recalculate persistence baselines with explicit temporal definitions and
   compare Ridge on exactly the same rows.
5. Quantify the field-comparison sensitivity to requiring 8/8 valid field days.

Outputs are written to workspace/diagnostics/closure_tests/offline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import pairwise_distances


WORKSPACE_ENV_VAR = "ET_FUNDACION_WORKSPACE"
FIELD_SCENARIO_DIRECTORY = "field_ridge25_final_scenarios"
FIELD_COMPARISON_FILENAME = "field_comparison_scenarios.csv"


@dataclass(frozen=True)
class DIResult:
    name: str
    training_di: np.ndarray
    threshold: float
    mean_training_distance: float
    scaled: np.ndarray


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final offline ET Fundación closure diagnostics."
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help=(
            "External workspace root. Defaults to ET_FUNDACION_WORKSPACE or "
            "the repository sibling ET_fundacion_workspace/current."
        ),
    )
    parser.add_argument(
        "--run-directory",
        default=None,
        help="Specific saved run directory. Defaults to the newest run.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_workspace(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    override = os.environ.get(WORKSPACE_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return project_root().parent / "ET_fundacion_workspace" / "current"


def resolve_run_directory(workspace: Path, value: str | None) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Run directory not found: {path}")
        return path

    runs = workspace / "runs"
    candidates = sorted(
        path for path in runs.iterdir()
        if path.is_dir() and (path / "tables" / "ridge25_training_population.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No completed Ridge-25 runs found under {runs}")
    return candidates[-1]


def build_training_di(
    matrix: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray | None,
    name: str,
) -> DIResult:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=1)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Predictors contain zero or invalid sample standard deviations.")

    scaled = (matrix - means) / scales
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (matrix.shape[1],):
            raise ValueError("AOA weight count differs from feature count.")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("AOA weights must be finite and non-negative.")
        if not np.any(weights > 0):
            raise ValueError("At least one AOA weight must be positive.")
        scaled = scaled * weights

    distances = pairwise_distances(scaled, metric="euclidean")
    np.fill_diagonal(distances, np.nan)
    mean_training_distance = float(np.nanmean(np.nanmean(distances, axis=1)))

    training_di = np.full(matrix.shape[0], np.nan, dtype=float)
    for index in range(matrix.shape[0]):
        candidate = groups != groups[index]
        if not candidate.any():
            raise ValueError("Each AOA row requires observations from another spatial block.")
        training_di[index] = (
            float(np.nanmin(distances[index, candidate])) / mean_training_distance
        )

    q1 = float(np.quantile(training_di, 0.25))
    q3 = float(np.quantile(training_di, 0.75))
    threshold = min(float(np.max(training_di)), q3 + 1.5 * (q3 - q1))

    return DIResult(
        name=name,
        training_di=training_di,
        threshold=threshold,
        mean_training_distance=mean_training_distance,
        scaled=scaled,
    )


def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
    mean_true = float(np.mean(y_true))
    std_true = float(np.std(y_true, ddof=0))
    if mean_true == 0 or std_true == 0:
        return float("nan")
    alpha = float(np.std(y_pred, ddof=0) / std_true)
    beta = float(np.mean(y_pred) / mean_true)
    return float(1 - math.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if len(y) == 0:
        return {
            "n": 0,
            "R2": float("nan"),
            "RMSE": float("nan"),
            "MAE": float("nan"),
            "BIAS": float("nan"),
            "r": float("nan"),
            "KGE": float("nan"),
        }
    correlation = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else float("nan")
    return {
        "n": int(len(y)),
        "R2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        "RMSE": float(mean_squared_error(y, p) ** 0.5),
        "MAE": float(mean_absolute_error(y, p)),
        "BIAS": float(np.mean(p - y)),
        "r": correlation,
        "KGE": kge(y, p),
    }


def field_sensitivity(workspace: Path) -> pd.DataFrame:
    path = (
        workspace
        / "diagnostics"
        / FIELD_SCENARIO_DIRECTORY
        / FIELD_COMPARISON_FILENAME
    )
    if not path.is_file():
        return pd.DataFrame()

    frame = pd.read_csv(path)
    scenarios = [
        (
            "all_with_AOA",
            "included_all_with_AOA",
            "ET_Ridge_with_AOA_mm_period",
        ),
        (
            "all_without_AOA_pure_extrapolations",
            "included_all_without_AOA_pure_extrapolations",
            "ET_Ridge_without_AOA_mm_period",
        ),
        (
            "fixed_Kc_with_AOA",
            "included_fixed_Kc_with_AOA",
            "ET_Ridge_with_AOA_mm_period",
        ),
        (
            "fixed_Kc_without_AOA_pure_extrapolations",
            "included_fixed_Kc_without_AOA_pure_extrapolations",
            "ET_Ridge_without_AOA_mm_period",
        ),
    ]

    rows: list[dict[str, object]] = []
    for scenario, include_column, ridge_column in scenarios:
        if include_column not in frame.columns:
            continue
        included = frame[include_column].fillna(False).astype(bool)
        for completeness, extra in (
            ("current_5of8_or_more", pd.Series(True, index=frame.index)),
            ("complete_8of8_only", frame["n_valid_field_days"].eq(8)),
        ):
            subset = frame.loc[included & extra].copy()
            for product, prediction_column in (
                ("Ridge25", ridge_column),
                ("MODIS", "ET_MODIS_parent_mm_period"),
            ):
                metrics = regression_metrics(
                    subset["ET_field_proxy_mm_period"],
                    subset[prediction_column],
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "field_completeness": completeness,
                        "product": product,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_arguments()
    workspace = resolve_workspace(args.workspace)
    run_directory = resolve_run_directory(workspace, args.run_directory)
    tables = run_directory / "tables"
    output_directory = workspace / "diagnostics" / "closure_tests" / "offline"
    output_directory.mkdir(parents=True, exist_ok=True)

    population = pd.read_csv(
        tables / "ridge25_training_population.csv",
        dtype={"station_id": str},
        parse_dates=["period_start"],
    )
    oof = pd.read_csv(
        tables / "ridge25_spatial_oof.csv",
        dtype={"station_id": str},
        parse_dates=["period_start"],
    )
    model_parameters = pd.read_csv(tables / "ridge25_model_parameters.csv")

    features = model_parameters["feature"].astype(str).tolist()
    missing = [name for name in features if name not in population.columns]
    if missing:
        raise ValueError(f"Training population is missing Ridge-25 features: {missing}")

    matrix = population[features].to_numpy(dtype=float)
    groups = population["spatial_block"].astype(str).to_numpy()
    weights = model_parameters["ridge_coefficient_standardized"].abs().to_numpy(dtype=float)

    equal = build_training_di(matrix, groups, weights=None, name="equal_weight")
    weighted = build_training_di(
        matrix,
        groups,
        weights=weights,
        name="abs_standardized_ridge_coefficient",
    )

    saved_meta_path = run_directory / "ridge25_aoa_metadata.json"
    saved_meta = json.loads(saved_meta_path.read_text(encoding="utf-8"))
    saved_threshold = float(saved_meta["threshold"])
    if not math.isclose(equal.threshold, saved_threshold, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            "Current equal-weight AOA was not reproduced exactly: "
            f"{equal.threshold} != {saved_threshold}"
        )

    oof_keyed = oof.set_index(["station_id", "period_start"])
    population_keyed = population.set_index(["station_id", "period_start"])
    aligned = population_keyed[["Kc_target"]].join(
        oof_keyed[["prediction", "error"]], how="left"
    ).reset_index()
    if aligned["prediction"].isna().any():
        raise RuntimeError("Spatial OOF predictions do not cover the full training population.")

    aoa_rows = []
    for result in (equal, weighted):
        inside = result.training_di <= result.threshold
        absolute_error = aligned["error"].abs().to_numpy(dtype=float)
        aoa_rows.append(
            {
                "method": result.name,
                "threshold": result.threshold,
                "mean_training_distance": result.mean_training_distance,
                "training_rows": int(len(inside)),
                "inside_rows": int(inside.sum()),
                "outside_rows": int((~inside).sum()),
                "outside_fraction": float((~inside).mean()),
                "spearman_DI_vs_abs_OOF_error": float(
                    pd.Series(result.training_di).corr(
                        pd.Series(absolute_error), method="spearman"
                    )
                ),
                "OOF_RMSE_inside": float(
                    mean_squared_error(
                        aligned.loc[inside, "Kc_target"],
                        aligned.loc[inside, "prediction"],
                    ) ** 0.5
                ),
                "OOF_RMSE_outside": float(
                    mean_squared_error(
                        aligned.loc[~inside, "Kc_target"],
                        aligned.loc[~inside, "prediction"],
                    ) ** 0.5
                ),
            }
        )
    aoa_summary = pd.DataFrame(aoa_rows)
    aoa_summary["classification_changes_vs_equal_weight"] = [
        0,
        int(
            np.sum(
                (equal.training_di <= equal.threshold)
                != (weighted.training_di <= weighted.threshold)
            )
        ),
    ]
    aoa_summary.to_csv(output_directory / "aoa_training_comparison.csv", index=False)

    feature_rows = pd.DataFrame(
        {
            "feature": features,
            "abs_standardized_ridge_coefficient": weights,
            "squared_distance_weight": weights ** 2,
        }
    )
    optical = set(features[:16])
    meteorology = set(features[16:21])
    feature_rows["family"] = feature_rows["feature"].map(
        lambda name: "optical" if name in optical else (
            "meteorology" if name in meteorology else "harmonics"
        )
    )
    total_weight = float(feature_rows["squared_distance_weight"].sum())
    feature_rows["squared_distance_contribution_fraction"] = (
        feature_rows["squared_distance_weight"] / total_weight
    )
    feature_rows.sort_values(
        "abs_standardized_ridge_coefficient", ascending=False
    ).to_csv(output_directory / "aoa_ridge_coefficient_weights.csv", index=False)

    family_summary = (
        feature_rows.groupby("family", as_index=False)["squared_distance_weight"]
        .sum()
        .rename(columns={"squared_distance_weight": "weight_squared_sum"})
    )
    family_summary["contribution_fraction"] = (
        family_summary["weight_squared_sum"] / family_summary["weight_squared_sum"].sum()
    )
    family_summary.to_csv(output_directory / "aoa_weight_contribution_by_family.csv", index=False)

    persistence = population[["station_id", "period_start", "Kc_target"]].copy()
    persistence = persistence.sort_values(["station_id", "period_start"])
    persistence["previous_Kc"] = persistence.groupby("station_id")["Kc_target"].shift(1)
    persistence["previous_period_start"] = (
        persistence.groupby("station_id")["period_start"].shift(1)
    )
    persistence["gap_days"] = (
        persistence["period_start"] - persistence["previous_period_start"]
    ).dt.days
    persistence = persistence.merge(
        oof[["station_id", "period_start", "prediction"]],
        on=["station_id", "period_start"],
        how="left",
        validate="one_to_one",
    )

    persistence_rows: list[dict[str, object]] = []
    masks = {
        "previous_available_observation": persistence["previous_Kc"].notna(),
        "strict_previous_8day_composite": (
            persistence["previous_Kc"].notna() & persistence["gap_days"].eq(8)
        ),
        "previous_observation_within_16days": (
            persistence["previous_Kc"].notna() & persistence["gap_days"].le(16)
        ),
    }
    for definition, mask in masks.items():
        subset = persistence.loc[mask]
        for model_name, prediction_column in (
            ("persistence", "previous_Kc"),
            ("Ridge25_spatial_OOF", "prediction"),
        ):
            persistence_rows.append(
                {
                    "definition": definition,
                    "model": model_name,
                    **regression_metrics(subset["Kc_target"], subset[prediction_column]),
                }
            )
    persistence_summary = pd.DataFrame(persistence_rows)
    persistence_summary.to_csv(output_directory / "persistence_baselines.csv", index=False)

    gap_counts = (
        persistence.loc[persistence["previous_Kc"].notna(), "gap_days"]
        .value_counts()
        .sort_index()
        .rename_axis("gap_days")
        .reset_index(name="n")
    )
    gap_counts.to_csv(output_directory / "persistence_gap_counts.csv", index=False)

    field_summary = field_sensitivity(workspace)
    if not field_summary.empty:
        field_summary.to_csv(
            output_directory / "field_valid_day_sensitivity.csv", index=False
        )

    summary = {
        "workspace": str(workspace),
        "run_directory": str(run_directory),
        "training_rows": int(len(population)),
        "predictor_count": int(len(features)),
        "equal_weight_AOA_threshold": equal.threshold,
        "coefficient_weighted_AOA_threshold": weighted.threshold,
        "training_classification_changes": int(
            np.sum(
                (equal.training_di <= equal.threshold)
                != (weighted.training_di <= weighted.threshold)
            )
        ),
        "spearman_equal_vs_weighted_DI": float(
            pd.Series(equal.training_di).corr(
                pd.Series(weighted.training_di), method="spearman"
            )
        ),
        "weighted_distance_family_contribution": {
            row["family"]: float(row["contribution_fraction"])
            for _, row in family_summary.iterrows()
        },
        "decision_rule": (
            "Do not choose the AOA by field R2, OOF error, or mapped area. "
            "Use the map sensitivity only to determine whether coefficient "
            "weighting creates a stable and scientifically interpretable "
            "support domain relative to the current equal-weight domain."
        ),
    }
    (output_directory / "closure_offline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 80)
    print("ET FUNDACION - OFFLINE CLOSURE DIAGNOSTICS")
    print("=" * 80)
    print("Run:", run_directory.name)
    print("Training rows:", len(population))
    print()
    print("AOA")
    print(aoa_summary.to_string(index=False))
    print()
    print("Weighted distance contribution by family")
    print(family_summary.to_string(index=False))
    print()
    print("Persistence / matched Ridge baselines")
    print(persistence_summary.to_string(index=False))
    if not field_summary.empty:
        print()
        print("Field valid-day sensitivity")
        print(field_summary.to_string(index=False))
    print()
    print("Outputs:", output_directory)


if __name__ == "__main__":
    main()
