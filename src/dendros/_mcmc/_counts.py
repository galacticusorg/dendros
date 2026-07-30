"""Likelihoods and Fisher weights for binned count data.

Galacticus' count-based likelihoods (``haloMassFunction`` with
``likelihoodPoisson=true``, for instance) treat the number of objects in each bin
as Poisson distributed, optionally over-dispersed into a negative binomial to
absorb model discrepancy.  This module reproduces those log-likelihoods and
supplies the corresponding Fisher weights, which are what lets Gaussian
error-propagation machinery be applied to a count-based fit.

For any distribution whose mean is the model prediction, the expected Fisher
information about that prediction is the inverse of the variance:

    Poisson,           variance mu:              w = 1 / mu
    negative binomial, variance mu + f mu^2:     w = 1 / (mu + f mu^2)

so a count likelihood behaves, to second order, like a Gaussian one with those
variances — see :func:`fisher_weights`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import gammaln

#: Value Galacticus uses for an effectively impossible log-likelihood
#: (``Models_Likelihoods_Constants``).
LOG_IMPROBABLE = -1.0e30


def negative_binomial_shape(variance_fractional: float) -> float:
    """Return the negative-binomial shape ``r`` for a fractional model variance.

    Galacticus parameterizes over-dispersion by the fractional variance ``f``
    added in quadrature to the Poisson term, giving variance ``mu + f mu^2``; the
    corresponding negative-binomial shape is ``r = 1 / f``.  Returns ``inf`` for
    ``f <= 0``, the Poisson limit.
    """
    return np.inf if variance_fractional <= 0.0 else 1.0 / variance_fractional


def log_likelihood_bins(
    count: np.ndarray,
    mu: np.ndarray,
    variance_fractional: float = 0.0,
) -> np.ndarray:
    """Per-bin log-likelihood of observing *count* given model mean *mu*.

    Poisson when ``variance_fractional <= 0``, else negative binomial with
    variance ``mu + variance_fractional * mu**2``.

    Bins with ``mu <= 0`` contribute ``0`` where the observed count is also zero,
    and :data:`LOG_IMPROBABLE` where it is not — matching Galacticus, which
    treats a positive count against a zero prediction as impossible.

    Parameters
    ----------
    count:
        Observed counts per bin.
    mu:
        Model-predicted mean counts per bin, broadcastable against *count*.
    variance_fractional:
        Fractional model-discrepancy variance ``f``.

    Returns
    -------
    numpy.ndarray
        Per-bin log-likelihood.
    """
    count = np.asarray(count, dtype=float)
    mu = np.asarray(mu, dtype=float)
    bad = mu <= 0.0
    safe = np.where(bad, 1.0, mu)

    if variance_fractional <= 0.0:
        out = count * np.log(safe) - safe - gammaln(count + 1.0)
    else:
        r = 1.0 / variance_fractional
        out = (
            count * np.log(safe)
            - gammaln(count + 1.0)
            + gammaln(r + count)
            - gammaln(r)
            - count * np.log(r + safe)
            - r * np.log(safe / r + 1.0)
        )
    return np.where(bad, np.where(count > 0.0, LOG_IMPROBABLE, 0.0), out)


def count_variance(mu: np.ndarray, variance_fractional: float = 0.0) -> np.ndarray:
    """Return the per-bin count variance ``mu + f mu**2``."""
    mu = np.asarray(mu, dtype=float)
    if variance_fractional <= 0.0:
        return mu.copy()
    return mu + variance_fractional * mu**2


def fisher_weights(
    mu: np.ndarray,
    variance_fractional: float = 0.0,
    *,
    count: Optional[np.ndarray] = None,
    observed: bool = False,
) -> np.ndarray:
    """Return per-bin Fisher weights for the model prediction.

    These are the diagonal of the weight matrix ``W`` in ``J^T W J``, letting the
    Gaussian propagation in :mod:`dendros._mcmc._fisher` be applied to a count
    likelihood.

    Parameters
    ----------
    mu:
        Model-predicted mean counts per bin.
    variance_fractional:
        Fractional model-discrepancy variance ``f``.
    count:
        Observed counts.  Required when ``observed=True``.
    observed:
        When ``False`` (the default) return the *expected* information
        ``1 / (mu + f mu^2)``.  When ``True`` return the *observed* information
        ``-d2 lnL / d mu2`` evaluated at the data, which tracks the local
        curvature of the realized likelihood more closely but is noisier and can
        go negative in poorly-fit bins.

    Returns
    -------
    numpy.ndarray
        Per-bin weights.  Zero where ``mu <= 0``, those bins carrying no
        information about the model.
    """
    mu = np.asarray(mu, dtype=float)
    bad = mu <= 0.0
    safe = np.where(bad, 1.0, mu)

    if not observed:
        w = 1.0 / count_variance(safe, variance_fractional)
    else:
        if count is None:
            raise ValueError("count is required when observed=True")
        c = np.asarray(count, dtype=float)
        if variance_fractional <= 0.0:
            # Poisson: -d2/dmu2 [ c ln mu - mu ] = c / mu^2
            w = c / safe**2
        else:
            r = 1.0 / variance_fractional
            # -d2/dmu2 of the negative-binomial log-pmf.
            w = c / safe**2 - (c + r) / (r + safe) ** 2
    return np.where(bad, 0.0, w)
