"""Projection pursuit: PCA on prior-variance-rescaled posterior samples.

The smallest eigenvalues of the rescaled-sample covariance correspond to the
linear combinations of parameters most tightened by the data relative to the
prior — those are the "best constrained" directions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._chains import ChainSet
from ._config import ModelParameter, PriorSpec


# ---------------------------------------------------------------------------
# Prior-variance helpers
# ---------------------------------------------------------------------------


def _prior_variance(prior: Optional[PriorSpec], parameter_name: str) -> float:
    """Return the variance of *prior*.

    Currently supports:

    * ``uniform`` with ``limitLower`` and ``limitUpper`` → ``(b - a)^2 / 12``.
    * ``normal`` with ``variance`` (and no truncation) → ``variance``.

    Other prior kinds, or normal priors carrying ``limitLower`` / ``limitUpper``
    (truncated), raise :class:`NotImplementedError`.
    """
    if prior is None:
        raise NotImplementedError(
            f"projection_pursuit requires a prior on every active parameter; "
            f"parameter {parameter_name!r} has none."
        )
    kind = prior.kind
    p = prior.params
    if kind == "uniform":
        lo = p.get("limitLower")
        hi = p.get("limitUpper")
        if lo is None or hi is None:
            raise NotImplementedError(
                f"Uniform prior on {parameter_name!r} is missing "
                "limitLower / limitUpper."
            )
        return (float(hi) - float(lo)) ** 2 / 12.0
    if kind == "normal":
        var = p.get("variance")
        if var is None:
            raise NotImplementedError(
                f"Normal prior on {parameter_name!r} is missing 'variance'."
            )
        if "limitLower" in p or "limitUpper" in p:
            raise NotImplementedError(
                f"Truncated-normal prior on {parameter_name!r} is not yet "
                "supported by projection_pursuit; please raise an issue if "
                "you need this."
            )
        return float(var)
    raise NotImplementedError(
        f"projection_pursuit does not yet support prior kind {kind!r} "
        f"(on parameter {parameter_name!r})."
    )


def _apply_mapper(values: np.ndarray, mapper: str, parameter_name: str) -> np.ndarray:
    """Apply *mapper* to *values* (sampler-space transform).

    Currently supports:

    * ``identity`` — pass through.

    Other mappers raise :class:`NotImplementedError`; please raise an issue if
    a particular mapper is needed.
    """
    if mapper == "identity":
        return values
    raise NotImplementedError(
        f"projection_pursuit does not yet support operatorUnaryMapper "
        f"value={mapper!r} (on parameter {parameter_name!r}).  Currently "
        "only 'identity' is implemented."
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionPursuitResult:
    """Result of :func:`projection_pursuit`.

    Attributes
    ----------
    eigenvalues:
        ``(n_params,)`` ascending eigenvalues of the rescaled-sample
        covariance matrix.  Smaller is "better constrained".
    eigenvectors:
        ``(n_params, n_params)`` matrix whose ``[:, k]`` column is the
        eigenvector for ``eigenvalues[k]``, expressed in *rescaled-mapped*
        space (i.e. in the same coordinates used for the eigendecomposition).
    parameter_names:
        Parameter names along axis 0.
    parameter_labels:
        :attr:`ModelParameter.display_label` strings parallel to
        :attr:`parameter_names`.
    prior_sigma:
        ``(n_params,)`` square root of the prior variance for each parameter,
        the rescaling that was applied before eigendecomposition.

    Methods
    -------
    direction:
        Components of one eigenvector that exceed a contribution threshold.
    latex_summary:
        LaTeX-rendered summary line for a chosen direction.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    parameter_names: Tuple[str, ...]
    parameter_labels: Tuple[str, ...]
    prior_sigma: np.ndarray

    def direction(
        self,
        index: int,
        *,
        contribution_minimum: float = 0.05,
    ) -> List[Tuple[str, float]]:
        """Significant components of eigenvector *index*.

        Parameters
        ----------
        index:
            Eigenvector index (0 = best constrained).
        contribution_minimum:
            Drop components whose squared loading is below this threshold.

        Returns
        -------
        list of (label, loading) pairs
            Sorted by descending absolute loading.
        """
        v = self.eigenvectors[:, index]
        out = [
            (self.parameter_labels[i], float(v[i]))
            for i in range(v.size)
            if v[i] ** 2 >= contribution_minimum
        ]
        out.sort(key=lambda t: abs(t[1]), reverse=True)
        return out

    def latex_summary(
        self,
        index: int,
        *,
        contribution_minimum: float = 0.05,
        precision: int = 3,
    ) -> str:
        """LaTeX-formatted summary of eigenvector *index*'s significant components."""
        parts = []
        for label, val in self.direction(index, contribution_minimum=contribution_minimum):
            sign = "+" if val >= 0 and parts else ("-" if val < 0 else "")
            parts.append(f"{sign} {abs(val):.{precision}f}\\,{label}".strip())
        return " ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def projection_pursuit(
    chains: ChainSet,
    *,
    post_burn: Optional[int] = None,
    drop_chains: Sequence[int] = (),
) -> ProjectionPursuitResult:
    """Find the linear combinations of parameters best constrained by the data.

    Each post-burn parameter column is mapped via its
    ``operatorUnaryMapper``, normalised by ``sqrt(prior variance)``, and
    mean-centred.  The covariance matrix of the resulting samples is
    eigendecomposed, and the eigenvalues/eigenvectors are returned sorted by
    ascending eigenvalue — so ``eigenvectors[:, 0]`` is the linear combination
    most tightly constrained relative to the prior.

    Parameters
    ----------
    chains:
        :class:`ChainSet`.
    post_burn:
        Number of leading rows to skip per chain.  ``None`` triggers
        automatic detection.
    drop_chains:
        ``chain_index`` values to exclude.

    Returns
    -------
    ProjectionPursuitResult

    Raises
    ------
    NotImplementedError
        If any active parameter uses a prior or mapper not yet supported by
        :func:`projection_pursuit` (currently uniform/normal priors and
        ``identity`` mapper only).
    ValueError
        If the post-burn pool is empty or contains fewer than two rows.
    """
    from ._convergence import _resolve_post_burn

    burn = _resolve_post_burn(chains, post_burn)
    drop = set(int(i) for i in drop_chains)

    parameters: Tuple[ModelParameter, ...] = chains.config.parameters
    n_params = len(parameters)

    # Pre-compute prior-sigma vector and mapper kinds.
    prior_sigma = np.empty(n_params, dtype=float)
    for i, p in enumerate(parameters):
        var = _prior_variance(p.prior, p.name)
        if var <= 0:
            raise ValueError(
                f"Prior variance on parameter {p.name!r} is non-positive."
            )
        prior_sigma[i] = np.sqrt(var)

    # Concatenate post-burn samples across chains (with mappers applied).
    parts: List[np.ndarray] = []
    for c in chains:
        if c.chain_index in drop:
            continue
        if c.n_steps <= burn:
            continue
        seg = c.state[burn:].copy()
        for i, p in enumerate(parameters):
            seg[:, i] = _apply_mapper(seg[:, i], p.mapper, p.name)
        parts.append(seg)
    if not parts:
        raise ValueError(
            "No post-burn samples available — every chain was dropped or "
            "shorter than post_burn."
        )
    samples = np.concatenate(parts, axis=0)
    if samples.shape[0] < 2:
        raise ValueError(
            f"Need at least two post-burn samples for projection_pursuit; "
            f"got {samples.shape[0]}."
        )

    # Rescale by prior sigma and centre.
    rescaled = samples / prior_sigma[np.newaxis, :]
    rescaled -= rescaled.mean(axis=0, keepdims=True)

    # Symmetric covariance → eigh.
    cov = rescaled.T @ rescaled / (rescaled.shape[0] - 1)
    cov = 0.5 * (cov + cov.T)  # symmetrise to suppress numerical asymmetry
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues already; double-check / be explicit.
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    return ProjectionPursuitResult(
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        parameter_names=chains.config.parameter_names,
        parameter_labels=tuple(p.display_label for p in parameters),
        prior_sigma=prior_sigma,
    )
