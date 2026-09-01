"""Analysis-period configuration and period-scoped artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path


@dataclass(frozen=True)
class AnalysisPeriod:
    """Half-open analysis interval: start inclusive, end exclusive."""

    start_date: date
    end_date_exclusive: date

    def __post_init__(self) -> None:
        if self.end_date_exclusive <= self.start_date:
            raise ValueError("end_date_exclusive must be later than start_date")

    @classmethod
    def from_strings(cls, start_date: str, end_date_exclusive: str) -> "AnalysisPeriod":
        return cls(date.fromisoformat(start_date), date.fromisoformat(end_date_exclusive))

    @property
    def label(self) -> str:
        end_inclusive = self.end_date_exclusive - timedelta(days=1)
        if (
            self.start_date.month == 1
            and self.start_date.day == 1
            and self.end_date_exclusive.month == 1
            and self.end_date_exclusive.day == 1
        ):
            if self.start_date.year == end_inclusive.year:
                return str(self.start_date.year)
            return f"{self.start_date.year}_{end_inclusive.year}"
        return f"{self.start_date:%Y%m%d}_{end_inclusive:%Y%m%d}"

    def metadata(self) -> dict[str, str]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date_exclusive": self.end_date_exclusive.isoformat(),
            "period_label": self.label,
        }


def expected_observation_count(number_periods: int, number_supports: int) -> int:
    if number_periods < 0 or number_supports < 0:
        raise ValueError("Expected-count inputs cannot be negative")
    return number_periods * number_supports


def period_directory(base: Path, period: AnalysisPeriod) -> Path:
    return Path(base) / period.label


def require_matching_period_metadata(path: Path, period: AnalysisPeriod) -> dict:
    """Load JSON metadata and reject missing or mismatched period provenance."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact metadata not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    expected = period.metadata()
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Artifact period metadata does not match {period.label}: {mismatches}"
        )
    return metadata
