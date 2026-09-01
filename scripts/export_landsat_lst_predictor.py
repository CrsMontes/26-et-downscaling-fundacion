"""Frozen experimental L8+L9 LST predictor export contract.

LST remains a NO-GO experiment and is not part of model training.  This file
is intentionally retained so the previously designed extraction can be
reproduced later without changing the approved thermal composite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from et_downscaling.thermal_availability import configuration_manifest


START_DATE = "2020-01-01"
END_DATE_EXCLUSIVE = "2025-01-01"
YEARS = tuple(range(2020, 2025))
SELECTORS = (
    "station_id", "period_start", "period_end_exclusive", "LST_mean_K",
    "LST_valid_count", "LST_valid_coverage_pct", "landsat_products",
    "landsat_unique_dates", "landsat_acquisition_dates",
    "landsat_dates_with_valid_lst", "l8_products", "l9_products",
    "l8_unique_dates", "l9_unique_dates", "sensors_present",
    "distributed_grid_m", "native_thermal_support_m_approx",
    "composite_method",
)


def method_manifest():
    method = configuration_manifest()
    return {
        **method,
        "period": {"start": START_DATE, "end_exclusive": END_DATE_EXCLUSIVE},
        "view": "L8_L9_COMBINED",
        "predictor": "LST_mean_K",
        "historical_dn_ge_293_filter_used": False,
        "status": "NO_GO_NOT_USED_FOR_TRAINING",
        "selectors": list(SELECTORS),
    }


def output_root():
    return Path(__file__).resolve().parents[1] / "outputs/diagnostics/2020_2024/landsat_lst_predictor"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(method_manifest(), indent=2))
    if args.execute:
        raise RuntimeError(
            "LST is frozen as NO-GO; extraction requires a separately approved reactivation."
        )
    print("Dry plan only: Earth Engine was not initialized; LST remains NO-GO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
