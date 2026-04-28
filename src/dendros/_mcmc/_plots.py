"""Corner-plot wrapper around the ``corner`` package."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from ._chains import ChainSet


def _import_corner():
    try:
        import corner  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "corner_plot requires the `corner` package.  Install it with: "
            "pip install 'dendros[mcmc]'  (which also installs matplotlib)."
        ) from exc
    return corner


def corner_plot(
    chains: ChainSet,
    *,
    parameters: Optional[Iterable[str]] = None,
    post_burn: Optional[int] = None,
    drop_chains: Sequence[int] = (),
    labels: Optional[Sequence[str]] = None,
    **corner_kwargs,
):
    """Render a corner plot of post-burn chain samples.

    Parameters
    ----------
    chains:
        :class:`ChainSet` whose post-burn samples will be plotted.
    parameters:
        Optional iterable of parameter names to restrict the plot to a
        subset (in the order given).  ``None`` plots every active
        parameter.
    post_burn:
        Number of leading rows to skip per chain.  ``None`` triggers
        automatic detection via :func:`gelman_rubin` /
        :func:`convergence_step`.
    drop_chains:
        Iterable of ``chain_index`` values to exclude.
    labels:
        Optional axis labels.  ``None`` uses each parameter's LaTeX
        :attr:`ModelParameter.display_label`, wrapped in ``$...$`` so
        :mod:`corner` renders them in math mode.
    **corner_kwargs:
        Additional keyword arguments forwarded to :func:`corner.corner`.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ImportError
        If the optional ``corner`` package is not installed.  Install via
        ``pip install 'dendros[mcmc]'``.
    KeyError
        If a name in *parameters* is not among the active parameters.
    ValueError
        If the post-burn pool is empty.
    """
    corner = _import_corner()
    from ._convergence import _resolve_post_burn

    burn = _resolve_post_burn(chains, post_burn)
    drop = set(int(i) for i in drop_chains)

    config = chains.config
    if parameters is None:
        cols = list(range(len(config.parameters)))
    else:
        index_by_name = {p.name: i for i, p in enumerate(config.parameters)}
        cols = []
        for name in parameters:
            if name not in index_by_name:
                raise KeyError(
                    f"Unknown parameter name {name!r}; "
                    f"available: {list(index_by_name)!r}"
                )
            cols.append(index_by_name[name])

    parts = [
        c.state[burn:]
        for c in chains
        if c.chain_index not in drop and c.n_steps > burn
    ]
    if not parts:
        raise ValueError(
            "No post-burn samples available — every chain was dropped or "
            "shorter than post_burn."
        )
    samples = np.concatenate(parts, axis=0)[:, cols]

    if labels is None:
        labels = [f"${config.parameters[i].display_label}$" for i in cols]
    else:
        labels = list(labels)
        if len(labels) != len(cols):
            raise ValueError(
                f"labels has length {len(labels)}; expected {len(cols)} "
                "(one per chosen parameter)."
            )

    return corner.corner(samples, labels=labels, **corner_kwargs)
