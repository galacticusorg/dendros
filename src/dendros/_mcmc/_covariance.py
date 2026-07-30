"""Utilities for working with estimated data covariance matrices.

An estimated covariance is not the true one, and the ways it goes wrong matter
for parameter errors: a matrix estimated from a finite number of realizations has
a biased inverse (:func:`hartlap_factor`) and injects extra parameter variance
(:func:`dodelson_schneider_factor`); one estimated at a different amplitude than
the measurement it will be applied to should usually contribute only its
*correlation* structure (:func:`rescale_correlation`); and one that is singular or
noise-dominated needs regularizing before it is inverted
(:func:`shrink_to_diagonal`).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def to_correlation(covariance: np.ndarray) -> np.ndarray:
    """Return the correlation matrix of *covariance*.

    Rows and columns with non-positive variance are returned as zero off the
    diagonal and one on it, rather than producing NaNs.
    """
    C = np.asarray(covariance, dtype=float)
    d = np.diag(C).copy()
    ok = d > 0.0
    s = np.zeros_like(d)
    s[ok] = 1.0 / np.sqrt(d[ok])
    R = C * s[:, None] * s[None, :]
    R[~ok, :] = 0.0
    R[:, ~ok] = 0.0
    np.fill_diagonal(R, np.where(ok, np.diag(R), 1.0))
    return 0.5 * (R + R.T)


def rescale_correlation(correlation: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Combine a correlation matrix with a separate set of standard deviations.

    ``C = diag(sigma) R diag(sigma)``.  Use this when the correlation structure
    and the variances come from different sources — for example a correlation
    matrix measured from a sub-volume or a subsample, applied at the variances the
    likelihood actually used.  It sidesteps any amplitude mismatch in the
    estimate, and keeps the resulting covariance's diagonal consistent with the
    fit being corrected, which is what makes the validity check in
    :mod:`dendros._mcmc._fisher` meaningful.

    Parameters
    ----------
    correlation:
        ``(n, n)`` correlation matrix.
    sigma:
        ``(n,)`` standard deviations.
    """
    R = np.asarray(correlation, dtype=float)
    s = np.asarray(sigma, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"correlation must be square; got shape {R.shape}")
    if s.shape != (R.shape[0],):
        raise ValueError(f"sigma must have shape ({R.shape[0]},); got {s.shape}")
    return 0.5 * ((R * s[:, None] * s[None, :]) + (R * s[:, None] * s[None, :]).T)


def shrink_to_diagonal(covariance: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate between the diagonal of *covariance* and the full matrix.

    ``alpha=0`` returns the diagonal (no correlations), ``alpha=1`` the input
    unchanged, and intermediate values scale the off-diagonal terms.  Values
    above 1 strengthen the correlations, which is useful for bracketing: if a
    parameter-error inflation is flat across a range of *alpha*, the precise
    correlation estimate does not matter and a better one is not worth
    obtaining.  Large *alpha* can leave the matrix indefinite, so check the
    eigenvalues.

    Parameters
    ----------
    covariance:
        ``(n, n)`` covariance or correlation matrix.
    alpha:
        Off-diagonal scaling.  Must be non-negative.
    """
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative; got {alpha!r}")
    C = np.asarray(covariance, dtype=float)
    d = np.diag(C)
    return alpha * C + (1.0 - alpha) * np.diag(d)


def nearest_positive_definite(
    covariance: np.ndarray, *, floor: float = 1e-10
) -> np.ndarray:
    """Clip eigenvalues of a symmetric matrix to make it positive definite.

    Bootstrap and jackknife covariance estimates are commonly rank-deficient
    (more bins than independent resamplings, or exactly degenerate bins), leaving
    zero or slightly negative eigenvalues that break a Cholesky factorization.

    Parameters
    ----------
    covariance:
        ``(n, n)`` symmetric matrix.
    floor:
        Minimum eigenvalue, as a fraction of the largest eigenvalue.
    """
    C = np.asarray(covariance, dtype=float)
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    if vals.max() <= 0.0:
        raise ValueError("covariance has no positive eigenvalues")
    vals = np.maximum(vals, floor * vals.max())
    return (vecs * vals) @ vecs.T


def hartlap_factor(n_realizations: int, n_data: int) -> float:
    """Return the Hartlap debiasing factor for an inverse sample covariance.

    The inverse of a covariance estimated from ``n_realizations`` independent
    realizations of an ``n_data``-dimensional vector is biased high; multiplying
    it by ``(n_realizations - n_data - 2) / (n_realizations - 1)`` removes the
    bias.  Raises when there are too few realizations for the inverse to exist.
    """
    if n_realizations <= n_data + 2:
        raise ValueError(
            f"need n_realizations > n_data + 2 for an unbiased inverse; got "
            f"n_realizations={n_realizations}, n_data={n_data}"
        )
    return (n_realizations - n_data - 2.0) / (n_realizations - 1.0)


def dodelson_schneider_factor(
    n_realizations: int, n_data: int, n_params: int
) -> float:
    """Return the variance inflation of parameter errors from a noisy covariance.

    Using a covariance estimated from a finite number of realizations inflates
    parameter variances by ``1 + (n_data - n_params) / (n_realizations - n_data - 2)``
    beyond the errors a perfectly known covariance would give.  Take the square
    root for the 1-sigma inflation.
    """
    if n_realizations <= n_data + 2:
        raise ValueError(
            f"need n_realizations > n_data + 2; got n_realizations="
            f"{n_realizations}, n_data={n_data}"
        )
    return 1.0 + (n_data - n_params) / (n_realizations - n_data - 2.0)
