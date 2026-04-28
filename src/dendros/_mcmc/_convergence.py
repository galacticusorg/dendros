"""MCMC convergence diagnostics: Gelman-Rubin, Geweke, outlier-chain detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from ._chains import ChainSet
from ._grubbs import iterative_grubbs


# ---------------------------------------------------------------------------
# Gelman-Rubin
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RhatResult:
    """Result of :func:`gelman_rubin`.

    Attributes
    ----------
    steps:
        ``(n_eval,)`` 1-D array of truncation step counts at which Rhat was
        computed (i.e. each entry ``s`` means "use the first ``s`` rows of
        every chain").  These are 1-based step counts so the smallest value is
        the chosen ``min_steps``.
    Rhat_c:
        ``(n_eval, n_params)`` array of Brooks-Gelman corrected potential-scale
        reduction factors.
    R_interval:
        ``(n_eval, n_params)`` array of non-parametric interval-length ratios
        (mixed-chain credible interval / mean per-chain credible interval) at
        the chosen ``alpha_interval``.
    parameter_names:
        Names of the parameters along ``axis=1``.
    alpha_interval:
        Significance level used to compute ``R_interval``.
    chains_used:
        ``chain_index`` values of the chains that contributed (after
        ``drop_chains`` was applied).

    Methods
    -------
    Rhat_c_max:
        Per-step max-over-parameters of :attr:`Rhat_c`, useful as the input to
        :func:`convergence_step`.
    """

    steps: np.ndarray
    Rhat_c: np.ndarray
    R_interval: np.ndarray
    parameter_names: Tuple[str, ...]
    alpha_interval: float
    chains_used: Tuple[int, ...]

    def Rhat_c_max(self) -> np.ndarray:
        """Return ``(n_eval,)`` max-over-parameters of :attr:`Rhat_c`."""
        return self.Rhat_c.max(axis=1)


def gelman_rubin(
    chains: ChainSet,
    *,
    drop_chains: Sequence[int] = (),
    step_grid: Optional[Sequence[int]] = None,
    n_grid: int = 200,
    min_steps: int = 10,
    alpha_interval: float = 0.15,
) -> RhatResult:
    """Brooks-Gelman corrected Rhat as a function of simulation step.

    For each chosen truncation point ``s`` the first ``s`` rows of every
    surviving chain are used to compute the standard between-chain (``B``)
    and within-chain (``W``) variances and the Brooks-Gelman corrected
    potential-scale reduction factor :math:`\\hat{R}_c`.  The non-parametric
    interval-length ratio :math:`R_{\\rm interval}` (Brooks & Gelman 1998
    section 1.3) is also computed at the same evaluation points.

    Parameters
    ----------
    chains:
        :class:`ChainSet` to evaluate.  Must contain at least two non-dropped
        chains and at least ``min_steps`` rows per chain.
    drop_chains:
        Iterable of ``chain_index`` values to exclude before computing.
        Use this with the indices returned by :func:`outlier_chains`.
    step_grid:
        Optional explicit 1-D iterable of truncation step counts (1-based).
        When given, ``n_grid`` and ``min_steps`` are ignored.
    n_grid:
        Number of evenly-spaced evaluation points to use when ``step_grid`` is
        ``None``.  Capped at the shortest surviving chain length minus
        ``min_steps`` + 1.
    min_steps:
        Smallest truncation step count to evaluate.  Must be ``>= 2``.
    alpha_interval:
        Two-sided significance level for ``R_interval`` (default 0.15, i.e.
        85 % credible intervals — matches the Galacticus Perl reference).

    Returns
    -------
    RhatResult

    Raises
    ------
    ValueError
        If fewer than two chains survive ``drop_chains`` or ``min_steps`` is
        too small.
    """
    if min_steps < 2:
        raise ValueError(f"min_steps must be >= 2; got {min_steps}")

    drop = set(int(i) for i in drop_chains)
    keep = [c for c in chains if c.chain_index not in drop]
    if len(keep) < 2:
        raise ValueError(
            f"gelman_rubin requires at least 2 chains; got {len(keep)} "
            f"after dropping {sorted(drop)!r}."
        )

    n_min = min(c.n_steps for c in keep)
    if n_min < min_steps:
        raise ValueError(
            f"Shortest surviving chain has {n_min} steps; min_steps={min_steps}."
        )

    if step_grid is None:
        n_eval = min(int(n_grid), n_min - min_steps + 1)
        steps = np.unique(
            np.linspace(min_steps, n_min, n_eval, dtype=int)
        )
    else:
        steps = np.asarray(list(step_grid), dtype=int)
        if (steps < 2).any():
            raise ValueError("step_grid entries must all be >= 2.")
        if (steps > n_min).any():
            raise ValueError(
                f"step_grid contains values exceeding the shortest chain "
                f"length ({n_min})."
            )

    n_params = chains.n_params
    # Stack to (n_chains, n_min_overall, n_params) so a single fancy index
    # gives us the truncated view at any step.
    stacked = np.stack([c.state[:n_min] for c in keep], axis=0)

    Rhat_c = np.empty((steps.size, n_params), dtype=float)
    R_interval = np.empty((steps.size, n_params), dtype=float)

    lo_q = alpha_interval / 2.0
    hi_q = 1.0 - alpha_interval / 2.0

    for i, s in enumerate(steps):
        sub = stacked[:, :s, :]                       # (m, s, n_params)
        Rhat_c[i] = _brooks_gelman_corrected(sub)
        R_interval[i] = _interval_ratio(sub, lo_q, hi_q)

    return RhatResult(
        steps=steps,
        Rhat_c=Rhat_c,
        R_interval=R_interval,
        parameter_names=chains.config.parameter_names,
        alpha_interval=alpha_interval,
        chains_used=tuple(c.chain_index for c in keep),
    )


def _brooks_gelman_corrected(sub: np.ndarray) -> np.ndarray:
    """Brooks-Gelman corrected Rhat for *sub* of shape ``(m, n, n_params)``.

    Returns a ``(n_params,)`` array.  Per-parameter computation follows
    Brooks & Gelman 1998, with the ``(d+3)/(d+1)`` correction applied.
    """
    m, n, _ = sub.shape
    chain_means = sub.mean(axis=1)              # (m, n_params)
    chain_vars = sub.var(axis=1, ddof=1)        # (m, n_params)  — s_j^2

    grand_mean = chain_means.mean(axis=0)       # (n_params,)

    # B = n * sample variance of chain means across chains, denom (m-1).
    B = n * chain_means.var(axis=0, ddof=1)     # (n_params,)
    # W = mean of within-chain variances.
    W = chain_vars.mean(axis=0)                 # (n_params,)

    # Posterior-variance estimate.
    Vhat = (n - 1) / n * W + (m + 1) / (m * n) * B

    # Variance of Vhat (Brooks-Gelman eq 4.5).
    var_s2 = chain_vars.var(axis=0, ddof=1)
    cov_s2_xbar = _sample_covariance(chain_vars, chain_means)
    cov_s2_xbar2 = _sample_covariance(chain_vars, chain_means ** 2)

    term_W = ((n - 1) / n) ** 2 * (1.0 / m) * var_s2
    term_B = ((m + 1) / (m * n)) ** 2 * (2.0 * B ** 2) / (m - 1)
    term_cov = (
        2.0 * (m + 1) * (n - 1) / (m ** 2 * n ** 2) * (n / m)
        * (cov_s2_xbar2 - 2.0 * grand_mean * cov_s2_xbar)
    )
    var_Vhat = term_W + term_B + term_cov

    # Degrees of freedom and corrected Rhat.  Guard against tiny / negative
    # var_Vhat (which can happen in pathological synthetic cases).
    safe_var = np.where(var_Vhat > 0, var_Vhat, np.inf)
    d = 2.0 * Vhat ** 2 / safe_var

    # Avoid div-by-zero when W is exactly 0 (constant chain).  Fall back to NaN
    # so the user can detect the degenerate case.
    safe_W = np.where(W > 0, W, np.nan)
    rhat_sq = (m + 1) / m * Vhat / safe_W - (n - 1) / (n * m)
    rhat_sq = np.where(rhat_sq > 0, rhat_sq, np.nan)
    rhat = np.sqrt(rhat_sq)

    return (d + 3) / (d + 1) * rhat


def _sample_covariance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column sample covariance of two ``(m, n_params)`` arrays."""
    m = a.shape[0]
    a_dev = a - a.mean(axis=0)
    b_dev = b - b.mean(axis=0)
    return (a_dev * b_dev).sum(axis=0) / (m - 1)


def _interval_ratio(sub: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    """Mixed-chain credible-interval length divided by the mean per-chain length.

    Parameters
    ----------
    sub:
        ``(m, n, n_params)`` array.
    lo_q, hi_q:
        Lower and upper quantile probabilities in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Per-parameter ratio.  ``NaN`` where any chain has zero spread.
    """
    m, n, _ = sub.shape
    mixed = sub.reshape(m * n, -1)
    mixed_lo = np.quantile(mixed, lo_q, axis=0)
    mixed_hi = np.quantile(mixed, hi_q, axis=0)
    mixed_len = mixed_hi - mixed_lo

    per_chain_lo = np.quantile(sub, lo_q, axis=1)   # (m, n_params)
    per_chain_hi = np.quantile(sub, hi_q, axis=1)
    per_chain_len = per_chain_hi - per_chain_lo
    mean_per_chain = per_chain_len.mean(axis=0)

    safe = np.where(mean_per_chain > 0, mean_per_chain, np.nan)
    return mixed_len / safe


# ---------------------------------------------------------------------------
# Convergence step
# ---------------------------------------------------------------------------


def convergence_step(
    rhat_max: np.ndarray,
    *,
    threshold: float = 1.1,
    sustained_for: int = 1,
) -> Optional[int]:
    """Index into the Rhat grid at which convergence is first declared.

    Searches for the smallest index ``i`` such that every entry of
    ``rhat_max[i : i + sustained_for]`` is at or below ``threshold``.

    Parameters
    ----------
    rhat_max:
        1-D array of (max-over-parameters) Rhat values, e.g.
        :meth:`RhatResult.Rhat_c_max`.
    threshold:
        Convergence threshold.  Defaults to ``1.1``.
    sustained_for:
        Number of consecutive grid points that must all be below the threshold
        before convergence is declared.  Defaults to ``1`` (strict first
        crossing).

    Returns
    -------
    int or None
        Grid index at which convergence is first sustained, or ``None`` if
        the threshold is never met.

    Notes
    -----
    Use :attr:`RhatResult.steps` to translate the returned grid index to a
    simulation-step count.
    """
    arr = np.asarray(rhat_max, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"rhat_max must be 1-D; got shape {arr.shape!r}")
    if sustained_for < 1:
        raise ValueError(f"sustained_for must be >= 1; got {sustained_for}")
    n = arr.size
    if n < sustained_for:
        return None
    below = arr <= threshold
    for i in range(n - sustained_for + 1):
        if below[i : i + sustained_for].all():
            return i
    return None


# ---------------------------------------------------------------------------
# Geweke
# ---------------------------------------------------------------------------


def geweke(
    chains: ChainSet,
    *,
    first: float = 0.1,
    last: float = 0.5,
) -> np.ndarray:
    """Per-chain Geweke z-scores comparing the means of two chain segments.

    For each chain and each parameter, returns

    .. math::

        z = \\frac{\\bar{x}_1 - \\bar{x}_2}{\\sqrt{s^2_1/n_1 + s^2_2/n_2}}

    where segment 1 covers the first ``first`` fraction of the chain and
    segment 2 covers the last ``last`` fraction.  Large ``|z|`` for any
    parameter suggests the chain has not yet reached a stationary
    distribution — useful when the chains were started from an
    under-dispersed state (which makes Gelman-Rubin uninformative).

    Parameters
    ----------
    chains:
        :class:`ChainSet`.
    first, last:
        Fractions in ``(0, 1)`` for the lengths of the two segments.  By
        default ``first=0.1`` and ``last=0.5`` (Geweke's original
        recommendation).

    Returns
    -------
    np.ndarray
        ``(n_chains, n_params)`` z-score array.  Chains shorter than 4 rows
        in either segment yield ``NaN``.

    Notes
    -----
    The variance estimator used here is the simple sample variance, which
    treats each draw as independent.  Autocorrelated chains will produce
    artificially-large ``|z|``; once a proper integrated-autocorrelation-time
    estimator lands (Phase 3) this can be inflated by the ACL to recover the
    classical spectral-density-at-zero variant.
    """
    if not (0.0 < first < 1.0):
        raise ValueError(f"first must be in (0, 1); got {first}")
    if not (0.0 < last < 1.0):
        raise ValueError(f"last must be in (0, 1); got {last}")
    if first + last > 1.0:
        raise ValueError(
            f"first + last must be <= 1; got first={first}, last={last}"
        )

    n_params = chains.n_params
    out = np.full((len(chains), n_params), np.nan)
    for i, c in enumerate(chains):
        n = c.n_steps
        n1 = int(n * first)
        n2 = int(n * last)
        if n1 < 2 or n2 < 2:
            continue
        seg1 = c.state[:n1]
        seg2 = c.state[-n2:]
        mu1 = seg1.mean(axis=0)
        mu2 = seg2.mean(axis=0)
        v1 = seg1.var(axis=0, ddof=1) / n1
        v2 = seg2.var(axis=0, ddof=1) / n2
        denom = np.sqrt(v1 + v2)
        z = np.where(denom > 0, (mu1 - mu2) / np.where(denom > 0, denom, 1.0), np.nan)
        out[i] = z
    return out


# ---------------------------------------------------------------------------
# Outlier chains
# ---------------------------------------------------------------------------


def outlier_chains(
    chains: ChainSet,
    *,
    alpha: float = 0.05,
    max_outliers: int = 10,
    parameters: Optional[Iterable[str]] = None,
) -> Tuple[int, ...]:
    """Iterative two-sided Grubbs test on each chain's final state.

    Each chain contributes its last row (the most recent state) as a single
    multivariate point.  The Grubbs test is applied iteratively over the
    active chains, dropping the chain whose maximum per-parameter deviation
    exceeds the critical value at each step, until none exceed it or
    ``max_outliers`` chains have been removed.

    Parameters
    ----------
    chains:
        :class:`ChainSet`.  Must contain at least three chains.
    alpha:
        Two-sided significance level.  Defaults to ``0.05`` to match the
        Galacticus Perl reference's hard-coded value.
    max_outliers:
        Maximum number of chains to declare as outliers.
    parameters:
        Optional iterable of parameter names to restrict the test to a
        subset.  Unknown names raise :class:`KeyError`.

    Returns
    -------
    tuple of int
        ``chain_index`` values of the chains flagged as outliers, in the
        order they were removed.
    """
    if len(chains) < 3:
        return ()

    # Restrict to selected parameter columns if requested.
    if parameters is None:
        cols = slice(None)
    else:
        wanted = list(parameters)
        index_by_name = {p.name: i for i, p in enumerate(chains.config.parameters)}
        try:
            cols = [index_by_name[name] for name in wanted]
        except KeyError as e:
            raise KeyError(
                f"Unknown parameter name {e.args[0]!r}; "
                f"available: {list(index_by_name)!r}"
            ) from None

    finals = np.stack([c.state[-1] for c in chains], axis=0)
    points = finals[:, cols] if isinstance(cols, list) else finals
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    flagged_rows = iterative_grubbs(points, alpha=alpha, max_outliers=max_outliers)
    return tuple(int(chains[i].chain_index) for i in flagged_rows)
