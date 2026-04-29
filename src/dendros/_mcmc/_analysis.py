"""Per-chain analyses that operate on post-burn samples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._chains import ChainSet


def acceptance_rate(
    chains: ChainSet,
    *,
    post_burn: Optional[int] = None,
) -> np.ndarray:
    """Per-chain post-burn acceptance rate.

    A step is "accepted" iff *any* parameter component differs from the
    previous step.  Galacticus emits the same row when a proposal is rejected,
    so this is the canonical acceptance count.

    Parameters
    ----------
    chains:
        :class:`ChainSet`.
    post_burn:
        Number of leading rows to skip in each chain.  ``None`` triggers
        automatic detection via :func:`gelman_rubin` /
        :func:`convergence_step`; if convergence is not reached a
        :class:`UserWarning` is emitted and ``0`` is used.

    Returns
    -------
    np.ndarray
        ``(n_chains,)`` array.  ``NaN`` for any chain with fewer than two
        post-burn rows.
    """
    from ._convergence import _resolve_post_burn

    burn = _resolve_post_burn(chains, post_burn)
    out = np.full(len(chains), np.nan, dtype=float)
    for i, c in enumerate(chains):
        seg = c.state[burn:]
        if seg.shape[0] < 2:
            continue
        diffs = np.any(seg[1:] != seg[:-1], axis=1)
        out[i] = diffs.mean()
    return out


def acceptance_rate_trace(
    chains: ChainSet,
    *,
    window: int = 30,
    post_burn: int = 0,
) -> List[np.ndarray]:
    """Sliding-window acceptance rate as a function of step.

    For each chain, returns a 1-D array whose ``i``-th entry is the fraction
    of the previous *window* transitions that were accepted (i.e. changed at
    least one parameter).  The first ``window`` entries are filled with
    :data:`numpy.nan` because the window is not yet full.

    Parameters
    ----------
    chains:
        :class:`ChainSet`.
    window:
        Sliding-window length in steps.  Defaults to ``30`` to match the
        Galacticus Perl reference.
    post_burn:
        Number of leading rows to skip in each chain before evaluation.

    Returns
    -------
    list of np.ndarray
        One 1-D array per chain, each of length ``n_steps_post_burn``.
        Returned as a list because chains may have different post-burn
        lengths.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window!r}")
    if post_burn < 0:
        raise ValueError(f"post_burn must be non-negative; got {post_burn!r}")

    out: List[np.ndarray] = []
    for c in chains:
        seg = c.state[post_burn:]
        n = seg.shape[0]
        trace = np.full(n, np.nan, dtype=float)
        if n > 1:
            diff = np.any(seg[1:] != seg[:-1], axis=1).astype(int)
            cumsum = np.concatenate(([0], np.cumsum(diff)))
            if n - 1 >= window:
                # trace[i] is the rate over transitions ending at row i.
                trace[window:] = (cumsum[window:] - cumsum[: n - window]) / window
        out.append(trace)
    return out


# ---------------------------------------------------------------------------
# Maximum posterior / likelihood
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxResult:
    """Result of :func:`maximum_posterior` or :func:`maximum_likelihood`.

    Attributes
    ----------
    state:
        ``(n_params,)`` parameter vector at the maximizing step.
    log_posterior:
        Log posterior at that step.
    log_likelihood:
        Log likelihood at that step.
    chain_index:
        ``chain_index`` of the chain holding the maximum.
    step:
        Simulation-step value of the maximizing row (1-based, matching
        :attr:`Chain.step`).
    parameter_names:
        Names along ``state``.
    """

    state: np.ndarray
    log_posterior: float
    log_likelihood: float
    chain_index: int
    step: int
    parameter_names: Tuple[str, ...]


def _argmax_across_chains(
    chains: ChainSet,
    *,
    drop_chains: Sequence[int],
    field: str,
) -> Tuple[int, int]:
    """Return ``(chain_position, row)`` of the per-chain maximum of *field*.

    *chain_position* is the index into ``chains`` (not the chain's own
    ``chain_index``).  Raises :class:`ValueError` if no chains remain.
    """
    drop = set(int(i) for i in drop_chains)
    best_chain_pos = -1
    best_row = -1
    best_val = -np.inf
    for pos, c in enumerate(chains):
        if c.chain_index in drop:
            continue
        arr = getattr(c, field)
        if arr.size == 0:
            continue
        row = int(np.argmax(arr))
        val = float(arr[row])
        if val > best_val:
            best_val = val
            best_chain_pos = pos
            best_row = row
    if best_chain_pos < 0:
        raise ValueError(
            "No chains contributed a maximum: every chain was either dropped "
            "or empty."
        )
    return best_chain_pos, best_row


def maximum_posterior(
    chains: ChainSet,
    *,
    drop_chains: Sequence[int] = (),
) -> MaxResult:
    """State vector at the maximum log posterior across all surviving chains.

    Parameters
    ----------
    chains:
        :class:`ChainSet`.
    drop_chains:
        ``chain_index`` values to exclude from the search (e.g. the result
        of :func:`outlier_chains`).

    Returns
    -------
    MaxResult
    """
    pos, row = _argmax_across_chains(
        chains, drop_chains=drop_chains, field="log_posterior"
    )
    c = chains[pos]
    return MaxResult(
        state=c.state[row].copy(),
        log_posterior=float(c.log_posterior[row]),
        log_likelihood=float(c.log_likelihood[row]),
        chain_index=int(c.chain_index),
        step=int(c.step[row]),
        parameter_names=chains.config.parameter_names,
    )


def maximum_likelihood(
    chains: ChainSet,
    *,
    drop_chains: Sequence[int] = (),
) -> MaxResult:
    """State vector at the maximum log likelihood across all surviving chains."""
    pos, row = _argmax_across_chains(
        chains, drop_chains=drop_chains, field="log_likelihood"
    )
    c = chains[pos]
    return MaxResult(
        state=c.state[row].copy(),
        log_posterior=float(c.log_posterior[row]),
        log_likelihood=float(c.log_likelihood[row]),
        chain_index=int(c.chain_index),
        step=int(c.step[row]),
        parameter_names=chains.config.parameter_names,
    )


# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PosteriorSamples:
    """Sampled rows from the post-burn concatenated chain.

    Attributes
    ----------
    state:
        ``(n_samples, n_params)`` parameter values.
    log_posterior:
        ``(n_samples,)`` log posterior at each sample.
    log_likelihood:
        ``(n_samples,)`` log likelihood at each sample.
    chain_index:
        ``(n_samples,)`` source chain indices.
    step:
        ``(n_samples,)`` source simulation-step values.
    parameter_names:
        Parameter names along ``state.shape[1]``.

    Notes
    -----
    Adjacent steps in an MCMC chain are correlated — *N* draws here represent
    significantly fewer independent samples from the posterior.  Use
    :func:`effective_sample_size` to estimate the equivalent count, and thin
    by the integrated autocorrelation time if independence is required.
    """

    state: np.ndarray
    log_posterior: np.ndarray
    log_likelihood: np.ndarray
    chain_index: np.ndarray
    step: np.ndarray
    parameter_names: Tuple[str, ...]


def posterior_samples(
    chains: ChainSet,
    n: int,
    *,
    post_burn: Optional[int] = None,
    drop_chains: Sequence[int] = (),
    rng: Optional[np.random.Generator] = None,
    replace: Optional[bool] = None,
) -> PosteriorSamples:
    """Draw *n* uniformly-random rows from the post-burn concatenated chain.

    Parameters
    ----------
    chains:
        :class:`ChainSet`.
    n:
        Number of samples to draw.  Must be positive.
    post_burn:
        Number of leading rows to skip in each chain.  ``None`` triggers
        automatic detection via :func:`gelman_rubin` /
        :func:`convergence_step`; if convergence is not reached a
        :class:`UserWarning` is emitted and ``0`` is used.
    drop_chains:
        Iterable of ``chain_index`` values to exclude.
    rng:
        :class:`numpy.random.Generator`.  Defaults to
        :func:`numpy.random.default_rng()`, which seeds from system entropy.
        Pass an explicit generator for reproducibility.
    replace:
        Whether to sample with replacement.  ``None`` (default) means "with
        replacement only when *n* exceeds the available pool", matching the
        common case where a small ``n`` is desired and identical rows would
        be misleading.

    Returns
    -------
    PosteriorSamples

    Raises
    ------
    ValueError
        If *n* is non-positive, or if all chains are dropped, or if
        ``replace=False`` and *n* exceeds the pool size.
    """
    from ._convergence import _resolve_post_burn

    if n <= 0:
        raise ValueError(f"n must be positive; got {n!r}")
    if rng is None:
        rng = np.random.default_rng()

    burn = _resolve_post_burn(chains, post_burn)
    drop = set(int(i) for i in drop_chains)

    state_parts: List[np.ndarray] = []
    logp_parts: List[np.ndarray] = []
    logl_parts: List[np.ndarray] = []
    chain_id_parts: List[np.ndarray] = []
    step_parts: List[np.ndarray] = []
    for c in chains:
        if c.chain_index in drop:
            continue
        if c.n_steps <= burn:
            continue
        state_parts.append(c.state[burn:])
        logp_parts.append(c.log_posterior[burn:])
        logl_parts.append(c.log_likelihood[burn:])
        n_post = c.n_steps - burn
        chain_id_parts.append(np.full(n_post, c.chain_index, dtype=np.int64))
        step_parts.append(c.step[burn:])

    if not state_parts:
        raise ValueError(
            "All chains dropped or shorter than post_burn; nothing to sample."
        )

    pool_state = np.concatenate(state_parts, axis=0)
    pool_logp = np.concatenate(logp_parts, axis=0)
    pool_logl = np.concatenate(logl_parts, axis=0)
    pool_chain_id = np.concatenate(chain_id_parts, axis=0)
    pool_step = np.concatenate(step_parts, axis=0)
    pool_size = pool_state.shape[0]

    if replace is None:
        replace = n > pool_size
    if not replace and n > pool_size:
        raise ValueError(
            f"Requested n={n} samples without replacement, but only "
            f"{pool_size} post-burn rows are available."
        )

    indices = rng.choice(pool_size, size=n, replace=replace)
    return PosteriorSamples(
        state=pool_state[indices].copy(),
        log_posterior=pool_logp[indices].copy(),
        log_likelihood=pool_logl[indices].copy(),
        chain_index=pool_chain_id[indices].copy(),
        step=pool_step[indices].copy(),
        parameter_names=chains.config.parameter_names,
    )
