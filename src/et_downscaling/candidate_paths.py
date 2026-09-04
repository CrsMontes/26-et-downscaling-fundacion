"""Canonical external-workspace paths for candidate predictors and sensitivity.

The complete 2020-2024 candidate universe is stored outside the Git repository.
The same candidate data support both the frozen final model and reproducibility
analyses; only derived reports and diagnostics are regenerated per run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .workspace import get_workspace_paths


CANDIDATE_PERIOD_LABEL = "2020_2024"


@dataclass(frozen=True)
class CandidateStudyPaths:
    workspace_root: Path
    raw_root: Path
    intermediate_root: Path
    master_store: Path
    sensitivity_root: Path
    station_support: Path

    @property
    def optical_root(self) -> Path:
        return self.raw_root / "optical_source_experiment"

    @property
    def availability_root(self) -> Path:
        return self.raw_root / "availability"

    @property
    def meteorology_root(self) -> Path:
        return self.raw_root / "meteorology_experiment"

    @property
    def s1_root(self) -> Path:
        return self.raw_root / "s1_geometry_experiment"

    @property
    def thermal_root(self) -> Path:
        return self.raw_root / "thermal_availability"

    @property
    def landsat_lst_root(self) -> Path:
        return self.raw_root / "predictor_availability_ladder" / "raw" / "landsat_lst"

    def ensure(self) -> "CandidateStudyPaths":
        for path in (
            self.raw_root,
            self.intermediate_root,
            self.master_store.parent,
            self.sensitivity_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


def get_candidate_study_paths(project_root: Path) -> CandidateStudyPaths:
    """Return canonical paths for the complete 2020-2024 candidate universe."""
    workspace = get_workspace_paths(project_root).ensure()
    return CandidateStudyPaths(
        workspace_root=workspace.root,
        raw_root=workspace.raw_cache / "candidates" / CANDIDATE_PERIOD_LABEL,
        intermediate_root=workspace.master / "candidates" / CANDIDATE_PERIOD_LABEL,
        master_store=workspace.master / "master_predictor_store.parquet",
        sensitivity_root=workspace.diagnostics / "sensitivity" / CANDIDATE_PERIOD_LABEL,
        station_support=workspace.raw_cache / "meteorology" / "station_support.csv",
    ).ensure()
