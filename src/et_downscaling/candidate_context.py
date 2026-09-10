"""Fixed context for the comprehensive Virtual10 candidate archive."""

from __future__ import annotations

import os


CANDIDATE_SUPPORT_COUNT_ENV = "ET_CANDIDATE_SUPPORT_COUNT"
CANDIDATE_PERIOD_COUNT_2020_2024 = 230


def candidate_support_count(default: int = 5) -> int:
    """Return the support count supplied by the orchestration layer."""
    raw = os.environ.get(CANDIDATE_SUPPORT_COUNT_ENV, "").strip()
    count = int(raw) if raw else int(default)
    if count <= 0:
        raise ValueError("Candidate support count must be positive.")
    return count


def candidate_expected_rows(number_periods: int = CANDIDATE_PERIOD_COUNT_2020_2024) -> int:
    """Expected support-period rows in the fixed 2020-2024 candidate universe."""
    return int(number_periods) * candidate_support_count()
