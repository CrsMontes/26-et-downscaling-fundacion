"""Build the frozen RF-25 Virtual10 population from the complete candidate master.

The comprehensive candidate master is preserved unchanged.  Only the already
accepted 25 RF predictors are selected for model fitting, and only the frozen
GE90 support-period whitelist produced by ``select_virtual_stations.py`` is
eligible.  Unused candidate predictors never become additional eligibility
gates for the final RF-25 model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from et_downscaling.training import prepare_rf25_population


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
SELECTION_ROOT = OUTPUTS / "training" / "selection"
SOURCE_MASTER = OUTPUTS / "current" / "master" / "master_predictor_store.parquet"
MASTER_ROOT = OUTPUTS / "training" / "master"
RESULTS_ROOT = OUTPUTS / "evaluation" / "results"


def load_frozen_selection() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_path = SELECTION_ROOT / "selected_supports.csv"
    checks_root = SELECTION_ROOT / "availability_checks"
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    if not checks_root.is_dir():
        raise FileNotFoundError(checks_root)

    selected = pd.read_csv(selected_path, dtype={"virtual_id": str})
    if len(selected) != 10:
        raise RuntimeError(f"Expected 10 selected Virtual10 supports; found {len(selected)}.")
    if selected["virtual_id"].nunique() != 10:
        raise RuntimeError("Virtual10 selection contains duplicate virtual IDs.")
    if selected["modis_pixel_id"].nunique() != 10:
        raise RuntimeError("Virtual10 selection contains duplicate MODIS supports.")
    if selected["spatial_block_utm10km"].nunique() != 10:
        raise RuntimeError("Virtual10 selection does not occupy 10 distinct fixed UTM blocks.")

    rows: list[dict[str, object]] = []
    for support in selected.itertuples(index=False):
        check_path = checks_root / f"{int(support.candidate_order):05d}_{int(support.modis_pixel_id)}.csv"
        if not check_path.is_file():
            raise FileNotFoundError(check_path)
        check = pd.read_csv(check_path)
        check["period_start"] = pd.to_datetime(check["period_start"], errors="raise").dt.strftime("%Y-%m-%d")
        ge90 = pd.to_numeric(check["ge90"], errors="raise").astype(int)
        periods = check.loc[ge90.eq(1), "period_start"].drop_duplicates().sort_values().tolist()
        if len(periods) != int(support.ge90_total):
            raise RuntimeError(
                f"{support.virtual_id}: GE90 whitelist has {len(periods)} rows; "
                f"selection records {int(support.ge90_total)}."
            )
        for period_start in periods:
            rows.append(
                {
                    "station_id": str(support.virtual_id),
                    "period_start": period_start,
                    "modis_pixel_id": int(support.modis_pixel_id),
                    "spatial_block": str(support.spatial_block_utm10km),
                }
            )

    whitelist = pd.DataFrame(rows)
    if whitelist.duplicated(["station_id", "period_start"]).any():
        raise RuntimeError("Frozen GE90 whitelist contains duplicate support-period keys.")
    return selected, whitelist


def main() -> None:
    if not SOURCE_MASTER.is_file():
        raise FileNotFoundError(
            f"Complete candidate master not found: {SOURCE_MASTER}\n"
            "Run `python scripts/run_pipeline.py download-candidates --project <PROJECT>` first."
        )

    selected, whitelist = load_frozen_selection()
    master = pd.read_parquet(SOURCE_MASTER)
    required_keys = {"station_id", "period_start", "modis_pixel_id"}
    missing = sorted(required_keys - set(master.columns))
    if missing:
        raise RuntimeError(f"Candidate master is missing keys: {missing}")

    master["station_id"] = master["station_id"].astype(str)
    master["period_start"] = pd.to_datetime(master["period_start"], errors="raise").dt.strftime("%Y-%m-%d")
    master["modis_pixel_id"] = pd.to_numeric(master["modis_pixel_id"], errors="raise").astype("int64")
    if master.duplicated(["station_id", "period_start"]).any():
        raise RuntimeError("Complete candidate master contains duplicate support-period keys.")

    actual_supports = sorted(master["station_id"].unique().tolist())
    expected_supports = sorted(selected["virtual_id"].astype(str).tolist())
    if actual_supports != expected_supports:
        raise RuntimeError(
            "Candidate master is not the Virtual10 archive. "
            f"Expected {expected_supports}; found {actual_supports}."
        )

    MASTER_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Preserve the complete archive in a training-facing location as both
    # Parquet (typed, compact) and CSV (transparent/manual inspection).
    full_parquet = MASTER_ROOT / "virtual10_all_predictors_2020_2024.parquet"
    full_csv = MASTER_ROOT / "virtual10_all_predictors_2020_2024.csv"
    master.to_parquet(full_parquet, index=False)
    master.to_csv(full_csv, index=False)

    keys = ["station_id", "period_start"]
    expected_pairs = whitelist[keys].copy()
    selected_master = expected_pairs.merge(master, on=keys, how="left", validate="one_to_one", indicator=True)
    missing_pairs = selected_master.loc[selected_master["_merge"].ne("both"), keys]
    if not missing_pairs.empty:
        raise RuntimeError(
            "Complete candidate master is missing frozen GE90 support-periods. First rows:\n"
            + missing_pairs.head(20).to_string(index=False)
        )
    selected_master = selected_master.drop(columns="_merge")

    # Verify MODIS identity against the frozen selection and use the exact UTM
    # block labels rather than a coordinate-derived approximation.
    frozen = whitelist[keys + ["modis_pixel_id", "spatial_block"]].rename(
        columns={"modis_pixel_id": "frozen_modis_pixel_id", "spatial_block": "frozen_spatial_block"}
    )
    selected_master = selected_master.merge(frozen, on=keys, how="left", validate="one_to_one")
    actual_modis = pd.to_numeric(selected_master["modis_pixel_id"], errors="raise").astype("int64")
    expected_modis = pd.to_numeric(selected_master["frozen_modis_pixel_id"], errors="raise").astype("int64")
    if not actual_modis.equals(expected_modis):
        raise RuntimeError("Candidate master MODIS support IDs differ from the frozen Virtual10 selection.")
    selected_master["spatial_block"] = selected_master["frozen_spatial_block"].astype(str)
    selected_master = selected_master.drop(columns=["frozen_modis_pixel_id", "frozen_spatial_block"])

    population = prepare_rf25_population(selected_master)
    expected_key_set = set(map(tuple, expected_pairs.to_numpy()))
    actual_key_set = set(
        map(
            tuple,
            population.assign(
                period_start=pd.to_datetime(population["period_start"]).dt.strftime("%Y-%m-%d")
            )[keys].to_numpy(),
        )
    )
    if actual_key_set != expected_key_set:
        missing_rf = sorted(expected_key_set - actual_key_set)
        extra_rf = sorted(actual_key_set - expected_key_set)
        raise RuntimeError(
            "The RF-25 eligibility contract changed the frozen GE90 population. "
            f"Missing={len(missing_rf)}, extra={len(extra_rf)}; "
            f"first missing={missing_rf[:10]}, first extra={extra_rf[:10]}."
        )

    if population["station_id"].nunique() != 10 or population["spatial_block"].nunique() != 10:
        raise RuntimeError("RF-25 population must retain 10 supports and 10 fixed spatial blocks.")

    output_path = RESULTS_ROOT / "virtual10_training_population.csv"
    population.to_csv(output_path, index=False)

    metadata = {
        "design": "Virtual10 sequential-random GE90",
        "candidate_master_source": str(SOURCE_MASTER.relative_to(ROOT)),
        "complete_candidate_archive_rows": int(len(master)),
        "complete_candidate_archive_columns": int(len(master.columns)),
        "rf25_population_rows": int(len(population)),
        "rf25_supports": int(population["station_id"].nunique()),
        "rf25_spatial_blocks": int(population["spatial_block"].nunique()),
        "frozen_ge90_whitelist_used": True,
        "unused_candidate_predictors_are_eligibility_gates": False,
        "real_station_footprints_used_for_training": False,
        "target_definition": "Kc_target = MODIS_ET / ETo",
        "final_predictor_count": 25,
        "final_model": "RandomForestRegressor; fixed hyperparameters; no tuning",
        "interpretation_guardrail": "Training target is MODIS-derived Kc; this is not independent 20 m ET validation.",
    }
    (RESULTS_ROOT / "training_population_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("=" * 92)
    print("RF-25 TRAINING POPULATION FROM COMPLETE VIRTUAL10 CANDIDATE MASTER")
    print("=" * 92)
    print("Complete candidate rows:", len(master))
    print("Complete candidate columns:", len(master.columns))
    print("Frozen GE90 rows:", len(whitelist))
    print("RF-25 rows:", len(population))
    print("Virtual supports:", population["station_id"].nunique())
    print("Spatial blocks:", population["spatial_block"].nunique())
    print("Complete master CSV:", full_csv)
    print("Complete master Parquet:", full_parquet)
    print("RF-25 population:", output_path)


if __name__ == "__main__":
    main()
