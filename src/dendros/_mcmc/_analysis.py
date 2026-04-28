"""Per-chain analyses that operate on post-burn samples."""
from __future__ import annotations

from typing import List, Optional

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
