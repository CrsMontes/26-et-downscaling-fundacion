"""Regression tests for Virtual Station V5 provenance metadata."""

from __future__ import annotations

import ast
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_v5_basin_experiment.py"
)


def _experiment_metadata_literals() -> dict[str, object]:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        if not any(
            isinstance(target, ast.Name)
            and target.id == "experiment_metadata"
            for target in node.targets
        ):
            continue

        if not isinstance(node.value, ast.Dict):
            raise AssertionError("experiment_metadata must remain a dictionary.")

        values: dict[str, object] = {}
        for key_node, value_node in zip(
            node.value.keys,
            node.value.values,
        ):
            if (
                isinstance(key_node, ast.Constant)
                and isinstance(key_node.value, str)
            ):
                try:
                    values[key_node.value] = ast.literal_eval(value_node)
                except (ValueError, TypeError):
                    pass

        return values

    raise AssertionError("experiment_metadata assignment was not found.")


def test_v5_metadata_reports_operational_design() -> None:
    metadata = _experiment_metadata_literals()

    assert metadata["status"] == "operational_virtual_station"


def test_training_reproduction_remains_non_destructive() -> None:
    metadata = _experiment_metadata_literals()

    assert metadata["production_model_replaced"] is False
    assert metadata["stable_current_modified"] is False
    assert metadata["stable_final_modified"] is False
    assert metadata["google_drive_used"] is False
