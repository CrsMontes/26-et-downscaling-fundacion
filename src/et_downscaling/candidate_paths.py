"""Canonical in-repository output paths for candidate predictor materialization.

Generated candidate data live under ``outputs/`` and are ignored by Git. They
are kept separate from the frozen RF-25 training population.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import build_satellite_output_filename
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
    operational_s2_table: Path

    @property
    def optical_root(self) -> Path:
        return self.raw_root / "optical"

    @property
    def availability_root(self) -> Path:
        return self.raw_root / "availability"

    @property
    def meteorology_root(self) -> Path:
        return self.raw_root / "meteorology"

    @property
    def s1_root(self) -> Path:
        return self.raw_root / "sentinel1"

    @property
    def thermal_root(self) -> Path:
        return self.raw_root / "landsat_thermal"

    @property
    def landsat_lst_root(self) -> Path:
        return self.raw_root / "landsat_lst"

    @property
    def hls_albedo_fvc_root(self) -> Path:
        return self.raw_root / "hls_albedo_fvc"

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
        sensitivity_root=workspace.diagnostics / "candidates" / CANDIDATE_PERIOD_LABEL,
        station_support=workspace.raw_cache / "meteorology" / "station_support.csv",
        operational_s2_table=(
            workspace.raw_cache
            / "satellite"
            / "S2"
            / build_satellite_output_filename("S2")
        ),
    ).ensure()
