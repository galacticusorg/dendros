"""Autocorrelation, integrated autocorrelation time, and effective sample size."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ._chains import ChainSet


# ---------------------------------------------------------------------------
# Autocorrelation function
# ---------------------------------------------------------------------------


def _acf_1d(x: np.ndarray) -> np.ndarray:
    """FFT-based normalized autocorrelation of a 1-D array.

    Returns the biased estimator (1/N normalization) so the ACF is well
    defined out to long lags; ``rho[0] == 1`` by construction.

    Parameters
    ----------
    x:
        1-D input.

    Returns
    -------
    np.ndarray
        Length-N array of normalized autocorrelations at lags ``0, 1, ..., N-1``.
    """
    n = x.size
    if n < 2:
        return np.array([1.0])
    centered = x - x.mean()
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(centered, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n].real / n
    if acov[0] <= 0:
        # Constant chain: zero variance, so the ACF is undefined.
        out = np.zeros(n)
        out[0] = 1.0
        return out
    return acov / acov[0]


def autocorrelation_function(
    chains: ChainSet,
    *,
    post_burn: int = 0,
    max_lag: Optional[int] = None,
) -> np.ndarray:
    """Per-chain, per-parameter normalized autocorrelation function.

    Parameters
    ----------
    chains:
        :class:`ChainSet`.
    post_burn:
        Number of leading rows to skip in each chain before computing the ACF.
    max_lag:
        Truncate the returned ACF to this lag (inclusive).  ``None`` returns
        the full lag range out to the post-burn chain length.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_chains, max_lag + 1, n_params)``.  Chains are
        truncated to a common length (the shortest post-burn chain) so this
        is rectangular.
    """
    if post_burn < 0:
        raise ValueError(f"post_burn must be non-negative; got {post_burn!r}")
    n_chains = len(chains)
    n_params = chains.n_params
    if n_chains == 0:
        return np.empty((0, 0, n_params), dtype=float)

    n_min = min(c.n_steps - post_burn for c in chains)
    if n_min < 2:
        raise ValueError(
            "All chains must have at least two samples after post_burn; "
            f"got minimum length {n_min}."
        )
    if max_lag is None:
        max_lag = n_min - 1
    elif max_lag < 0 or max_lag >= n_min:
        raise ValueError(
            f"max_lag must satisfy 0 <= max_lag < {n_min}; got {max_lag}"
        )

    out = np.empty((n_chains, max_lag + 1, n_params), dtype=float)
    for i, c in enumerate(chains):
        seg = c.state[post_burn : post_burn + n_min]
        for j in range(n_params):
            out[i, :, j] = _acf_1d(seg[:, j])[: max_lag + 1]
    return out


# ---------------------------------------------------------------------------
# Integrated autocorrelation time
# ---------------------------------------------------------------------------


def _auto_window(taus: np.ndarray, c: float) -> int:
    """Sokal automatic window: smallest M with ``M >= c * taus[M]``.

    Falls back to the largest available index if no such M exists.
    """
    fails = np.arange(taus.size) >= c * taus
    if not np.any(fails):
        return taus.size - 1
    return int(np.argmax(fails))


def autocorrelation_time(
    chains: ChainSet,
    *,
    post_burn: Optional[int] = None,
    c: float = 5.0,
) -> np.ndarray:
    """Integrated autocorrelation time per parameter, in steps.

    Implements the standard Sokal automatic-windowing estimator over the
    chain-averaged autocovariance.  For each parameter, the per-chain
    autocovariances are averaged before integrating, which is more stable
    than averaging per-chain ``τ_int`` estimates.

    Parameters
    ----------
    chains:
        :class:`ChainSet`.  All chains are truncated to the shortest
        post-burn length.
    post_burn:
        Number of leading rows to skip from each chain.  ``None`` triggers
        automatic detection via :func:`gelman_rubin` /
        :func:`convergence_step`; if convergence is not reached a
        :class:`UserWarning` is emitted and ``0`` is used.
    c:
        Sokal window constant.  Defaults to ``5.0``.

    Returns
    -------
    np.ndarray
        ``(n_params,)`` array of integrated autocorrelation times in steps.
    """
    from ._convergence import _resolve_post_burn

    burn = _resolve_post_burn(chains, post_burn)
    n_chains = len(chains)
    n_params = chains.n_params
    n_min = min(c.n_steps - burn for c in chains)
    if n_min < 2:
        raise ValueError(
            "All chains must have at least two samples after post_burn; "
            f"got minimum length {n_min}."
        )

    # Averaged autocovariance across chains.
    avg_acov = np.zeros((n_min, n_params), dtype=float)
    var_sum = np.zeros(n_params, dtype=float)
    for chain in chains:
        seg = chain.state[burn : burn + n_min]
        for j in range(n_params):
            x = seg[:, j] - seg[:, j].mean()
            nfft = 1 << int(np.ceil(np.log2(2 * n_min)))
            f = np.fft.rfft(x, nfft)
            acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n_min].real / n_min
            avg_acov[:, j] += acov
            var_sum[j] += acov[0]
    # The chain-averaged ACF: divide by sum of zero-lag variances so rho(0) = 1
    # for each parameter.
    safe_var = np.where(var_sum > 0, var_sum, np.nan)
    avg_acf = avg_acov / safe_var

    taus_cum = 2.0 * np.cumsum(avg_acf, axis=0) - 1.0
    out = np.empty(n_params, dtype=float)
    for j in range(n_params):
        if not np.isfinite(taus_cum[:, j]).all():
            out[j] = np.nan
            continue
        m = _auto_window(taus_cum[:, j], c)
        out[j] = taus_cum[m, j]
    return out


def effective_sample_size(
    chains: ChainSet,
    *,
    post_burn: Optional[int] = None,
    c: float = 5.0,
) -> np.ndarray:
    """Effective sample size per parameter.

    Defined as ``N_total / τ_int`` where ``N_total`` is the total number of
    post-burn samples summed across all chains and ``τ_int`` is the chain-
    averaged integrated autocorrelation time from
    :func:`autocorrelation_time`.

    Parameters
    ----------
    chains:
        :class:`ChainSet`.
    post_burn:
        See :func:`autocorrelation_time`.
    c:
        Sokal window constant.

    Returns
    -------
    np.ndarray
        ``(n_params,)`` array of effective sample sizes.
    """
    from ._convergence import _resolve_post_burn

    burn = _resolve_post_burn(chains, post_burn)
    n_min = min(ch.n_steps - burn for ch in chains)
    n_total = n_min * len(chains)
    tau = autocorrelation_time(chains, post_burn=burn, c=c)
    safe_tau = np.where(tau > 0, tau, np.nan)
    return n_total / safe_tau
