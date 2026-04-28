"""Iterative two-sided Grubbs test for chain-final-state outlier detection."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.stats import t as _student_t


def grubbs_critical_value(n: int, alpha: float) -> float:
    """Two-sided Grubbs critical value for *n* observations at significance *alpha*.

    Uses the standard formula

    .. math::

        G = \\frac{n-1}{\\sqrt{n}} \\sqrt{\\frac{t^2}{n-2+t^2}}

    where ``t = t_{1 - alpha/(2n), n-2}`` is the upper-tail Student-t quantile.

    Parameters
    ----------
    n:
        Number of observations.  Must satisfy ``n >= 3``.
    alpha:
        Two-sided significance level.

    Returns
    -------
    float
    """
    if n < 3:
        raise ValueError(f"Grubbs test requires n >= 3; got n={n}")
    t_quantile = _student_t.ppf(1.0 - alpha / (2.0 * n), n - 2)
    return float(
        (n - 1) / np.sqrt(n) * np.sqrt(t_quantile**2 / (n - 2 + t_quantile**2))
    )


def iterative_grubbs(
    points: np.ndarray,
    *,
    alpha: float = 0.05,
    max_outliers: int = 10,
) -> Tuple[int, ...]:
    """Identify outlier rows of *points* via iterative two-sided Grubbs.

    At each iteration the row whose deviation ``|x_ij − mean_j| / sd_j`` (taken
    over any column ``j``) is largest is declared an outlier if that maximum
    deviation exceeds the Grubbs critical value for the current active set.
    Iteration stops when no row exceeds the critical value or after
    ``max_outliers`` removals.  Means and standard deviations are recomputed
    each iteration over the still-active rows, with sample-variance denominator
    ``n_active - 1``.

    Parameters
    ----------
    points:
        Array of shape ``(n_rows, n_features)``.  Each row is a candidate; each
        column is one feature on which the test is independently evaluated.
    alpha:
        Two-sided significance level.  Defaults to ``0.05``, matching the
        Galacticus Perl implementation.
    max_outliers:
        Maximum number of outliers to remove.

    Returns
    -------
    tuple of int
        Row indices (into the original *points*) flagged as outliers, in the
        order they were removed.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"points must be 2-D; got shape {pts.shape!r}")
    n_rows = pts.shape[0]
    if n_rows < 3:
        return ()

    active = np.ones(n_rows, dtype=bool)
    outliers: List[int] = []

    for _ in range(max_outliers):
        n_active = int(active.sum())
        if n_active < 3:
            break
        active_pts = pts[active]
        means = active_pts.mean(axis=0)
        stds = active_pts.std(axis=0, ddof=1)
        # Avoid division by zero when a column is constant across active rows.
        safe_stds = np.where(stds > 0, stds, np.inf)
        deviations = np.abs(active_pts - means) / safe_stds  # (n_active, n_features)
        per_row_max = deviations.max(axis=1)
        crit = grubbs_critical_value(n_active, alpha)
        if per_row_max.max() <= crit:
            break
        local_idx = int(np.argmax(per_row_max))
        global_idx = int(np.flatnonzero(active)[local_idx])
        outliers.append(global_idx)
        active[global_idx] = False

    return tuple(outliers)
