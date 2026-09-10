"""Performance metrics used by the final RF-25 workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

TARGET_COLUMN = "Kc_target"


def calculate_metrics(
    observed: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Return R2, RMSE, MAE, bias and KGE for paired finite values."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if observed.size == 0:
        raise ValueError("No finite paired values are available for metric calculation.")

    error = predicted - observed
    observed_sd = observed.std(ddof=0)
    predicted_sd = predicted.std(ddof=0)
    if (
        observed.size < 2
        or observed_sd == 0
        or predicted_sd == 0
        or observed.mean() == 0
    ):
        kge = np.nan
    else:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
        alpha = float(predicted_sd / observed_sd)
        beta = float(predicted.mean() / observed.mean())
        kge = 1.0 - np.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )

    return {
        "n": int(observed.size),
        "R2": float(r2_score(observed, predicted)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "BIAS": float(np.mean(error)),
        "KGE": float(kge),
    }
