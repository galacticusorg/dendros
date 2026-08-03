"""Linearized error propagation for a completed posterior sample.

Given the model prediction vectors recorded alongside an MCMC (see
:mod:`dendros._mcmc._predictions`), the model can be linearized over the
posterior volume and parameter uncertainties re-derived under a *different* data
weighting than the one the chain actually used.  The motivating case is a
likelihood run with uncorrelated errors when the data are in fact correlated:
how much would the posterior have broadened had the covariance been included?

Write ``J = dm/dtheta`` for the Jacobian of the model over the posterior volume,
``W`` for the weight matrix a fit used (the inverse of its assumed data
covariance) and ``C`` for the true data covariance.  Then

- ``Sigma_assumed = (J^T W J + P)^-1`` reproduces the fit that was run.  It
  should match the chain's own covariance — check this before trusting anything
  downstream, via :func:`inflation_report`.
- ``Sigma_optimal = (J^T C^-1 J + P)^-1`` is what a correctly-weighted fit would
  have given.
- ``Sigma_sandwich = A C A^T`` with ``A = Sigma_assumed J^T W`` is the true
  uncertainty of the mis-weighted estimator that *was* used.  By the
  Gauss--Markov theorem it is never smaller than ``Sigma_optimal``.

``P`` is an optional prior precision.  Note that including correlations does not
necessarily inflate anything: a strongly correlated mode the model cannot
produce is effectively marginalized away, which can leave the remaining
information sharper.  The calculation decides; assumption does not.

Fisher matrices are additive over independent constraints, so a single
constraint's weighting can be corrected while the others are left alone, and
:func:`information_shares` can rank constraints by how much they actually
contribute before any of that work is done.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


@dataclass
class JacobianFit:
    """A linear model of the prediction vector over the posterior volume.

    Attributes
    ----------
    jacobian:
        ``(n_bins, n_params)`` matrix of ``dm/dtheta``.
    intercept:
        ``(n_bins,)`` prediction at :attr:`center`.
    center:
        ``(n_params,)`` parameter values the expansion is about.
    residual_rms:
        ``(n_bins,)`` root-mean-square residual of the linear fit per bin.
    prediction_rms:
        ``(n_bins,)`` root-mean-square variation of the prediction about its
        mean, per bin.  The ratio to :attr:`residual_rms` measures how well the
        model is described as linear over this volume.
    n_samples:
        Number of samples the fit used.
    degree:
        Polynomial degree fitted.  With ``degree=2`` the reported
        :attr:`jacobian` is the derivative at :attr:`center`, cleanly separated
        from curvature; with ``degree=1`` it is the best linear approximation
        over the whole sampled volume, which for a curved model is not the same
        thing.
    """

    jacobian: np.ndarray
    intercept: np.ndarray
    center: np.ndarray
    residual_rms: np.ndarray
    prediction_rms: np.ndarray
    n_samples: int
    degree: int = 1

    @property
    def nonlinearity(self) -> np.ndarray:
        """Per-bin residual RMS as a fraction of the prediction's own RMS.

        Small values mean the linearization captures the model's behaviour over
        the sampled volume.  Values approaching 1 mean it does not, and the
        Fisher results below should be treated as indicative only.
        """
        denom = np.where(self.prediction_rms > 0.0, self.prediction_rms, np.nan)
        return self.residual_rms / denom


def jacobian_from_samples(
    state: np.ndarray,
    prediction: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    active: Optional[Sequence[int]] = None,
    degree: int = 1,
    center: Optional[np.ndarray] = None,
    rcond: Optional[float] = None,
) -> JacobianFit:
    """Fit ``m(theta)`` as a polynomial in ``theta`` by least squares over samples.

    Regressing the recorded predictions on the recorded states gives the model's
    response over the posterior volume without any new model evaluations.

    Prefer ``degree=2`` unless the model is known to be near-linear.  A model
    that curves appreciably over the posterior volume will have its linear-fit
    slope biased, because the fit trades curvature against the slopes of
    correlated parameters; a quadratic fit separates the two and its linear
    coefficients are the derivative at :attr:`JacobianFit.center`.  Compare
    :attr:`JacobianFit.nonlinearity` between degrees to see whether it matters.

    Parameters
    ----------
    state:
        ``(n_samples, n_params)`` parameter states.
    prediction:
        ``(n_samples, n_bins)`` model predictions, row-matched to *state*.
    weights:
        ``(n_samples,)`` non-negative sample weights, e.g. the posterior
        multiplicity of each accepted state.  Uniform when omitted.
    active:
        Column indices of *state* to fit against.  Columns outside this set get
        a zero Jacobian column.  Use it for parameters a constraint cannot see
        (a Galacticus ``parameterMap`` subset), which are otherwise fitted to
        pure noise.
    degree:
        Polynomial degree, 1 or 2.  Degree 2 adds all cross- and square terms in
        the active columns, costing ``n_active (n_active + 3) / 2`` coefficients.
    center:
        ``(n_params,)`` point about which to expand.  Defaults to the
        weight-weighted mean of *state*.  With ``degree=2`` the reported Jacobian
        is the derivative *at this point*, so set it explicitly when the samples
        are not centred where the derivative is wanted — fitting over proposals
        (accepted and rejected alike) while expanding about the posterior mean,
        for instance.
    rcond:
        Cutoff passed to :func:`numpy.linalg.lstsq`.

    Returns
    -------
    JacobianFit
    """
    state = np.asarray(state, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if state.ndim != 2 or prediction.ndim != 2:
        raise ValueError("state and prediction must both be 2-dimensional")
    if state.shape[0] != prediction.shape[0]:
        raise ValueError(
            f"state has {state.shape[0]} samples but prediction has "
            f"{prediction.shape[0]}"
        )
    n_samples, n_params = state.shape
    n_bins = prediction.shape[1]

    cols = (
        np.arange(n_params)
        if active is None
        else np.asarray(sorted(set(int(i) for i in active)), dtype=int)
    )
    if cols.size and (cols.min() < 0 or cols.max() >= n_params):
        raise ValueError("active contains out-of-range parameter indices")

    if weights is None:
        w = np.ones(n_samples)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_samples,):
            raise ValueError(f"weights must have shape ({n_samples},)")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
    if w.sum() <= 0.0:
        raise ValueError("weights sum to zero")

    if center is None:
        center = (w @ state) / w.sum()
    else:
        center = np.asarray(center, dtype=float)
        if center.shape != (n_params,):
            raise ValueError(f"center must have shape ({n_params},); got {center.shape}")
    mean_pred = (w @ prediction) / w.sum()

    if degree not in (1, 2):
        raise ValueError(f"degree must be 1 or 2; got {degree!r}")

    x = state[:, cols] - center[cols]
    blocks = [np.ones((n_samples, 1)), x]
    if degree == 2 and cols.size:
        # Squares and cross terms; only the linear block is read back out, so
        # their ordering does not matter.
        iu, ju = np.triu_indices(cols.size)
        blocks.append(x[:, iu] * x[:, ju])
    design = np.hstack(blocks)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], prediction * sw[:, None], rcond=rcond)

    intercept = coef[0]
    jacobian = np.zeros((n_bins, n_params))
    if cols.size:
        jacobian[:, cols] = coef[1 : 1 + cols.size].T

    resid = prediction - design @ coef
    wsum = w.sum()
    residual_rms = np.sqrt((w @ resid**2) / wsum)
    prediction_rms = np.sqrt((w @ (prediction - mean_pred) ** 2) / wsum)

    return JacobianFit(
        jacobian=jacobian,
        intercept=intercept,
        center=center,
        residual_rms=residual_rms,
        prediction_rms=prediction_rms,
        n_samples=n_samples,
        degree=degree,
    )


# ---------------------------------------------------------------------------
# Covariance propagation
# ---------------------------------------------------------------------------


@dataclass
class InflationResult:
    """Parameter covariances under the assumed and true data covariances.

    Attributes
    ----------
    assumed:
        ``(n_params, n_params)`` covariance implied by the weighting the fit
        used.  Compare against the chain's own covariance to validate.
    optimal:
        Covariance a correctly-weighted fit would have given.
    sandwich:
        True covariance of the mis-weighted estimator that was used.
    parameter_names:
        Names in row/column order, when supplied.
    """

    assumed: np.ndarray
    optimal: np.ndarray
    sandwich: np.ndarray
    parameter_names: Optional[Tuple[str, ...]] = None

    def sigma(self, which: str = "assumed") -> np.ndarray:
        """Return per-parameter 1-sigma uncertainties for one covariance."""
        return np.sqrt(np.diag(getattr(self, which)))

    def factors(self, which: str = "optimal") -> np.ndarray:
        """Per-parameter 1-sigma inflation of *which* relative to :attr:`assumed`."""
        return self.sigma(which) / self.sigma("assumed")

    def volume_factor(self, which: str = "optimal") -> float:
        """Geometric-mean inflation, ``(det ratio)^(1/2p)``.

        Insensitive to which parameter basis is used, so it summarizes the
        change in overall constraining power better than any single parameter.
        """
        other = getattr(self, which)
        p = other.shape[0]
        sign_o, logdet_o = np.linalg.slogdet(other)
        sign_a, logdet_a = np.linalg.slogdet(self.assumed)
        if sign_o <= 0 or sign_a <= 0:
            return float("nan")
        return float(np.exp((logdet_o - logdet_a) / (2.0 * p)))


def parameter_covariances(
    jacobian: np.ndarray,
    weights_assumed: np.ndarray,
    covariance_true: np.ndarray,
    *,
    prior_precision: Optional[np.ndarray] = None,
    parameter_names: Optional[Sequence[str]] = None,
    rcond: float = 1e-12,
) -> InflationResult:
    """Propagate a data covariance through a linearized model.

    Parameters
    ----------
    jacobian:
        ``(n_data, n_params)`` model Jacobian.
    weights_assumed:
        The weight matrix the fit used: ``(n_data,)`` for a diagonal weighting
        (e.g. :func:`dendros._mcmc._counts.fisher_weights`) or
        ``(n_data, n_data)`` for a general one.
    covariance_true:
        ``(n_data, n_data)`` true data covariance.
    prior_precision:
        ``(n_params, n_params)`` prior precision to add to each Fisher matrix.
        Omit for flat priors.
    parameter_names:
        Optional names, carried through to the result.
    rcond:
        Relative eigenvalue cutoff for the pseudo-inverses.  Fisher matrices are
        singular whenever a parameter is unconstrained by the data — which is
        normal for a single constraint of a larger fit — so inverses here are
        Moore--Penrose throughout.

    Returns
    -------
    InflationResult
    """
    J = np.asarray(jacobian, dtype=float)
    C = np.asarray(covariance_true, dtype=float)
    n_data, n_params = J.shape
    if C.shape != (n_data, n_data):
        raise ValueError(
            f"covariance_true has shape {C.shape}; expected ({n_data}, {n_data})"
        )

    W = np.asarray(weights_assumed, dtype=float)
    if W.ndim == 1:
        if W.shape != (n_data,):
            raise ValueError(f"weights_assumed has shape {W.shape}; expected ({n_data},)")
        W = np.diag(W)
    elif W.shape != (n_data, n_data):
        raise ValueError(
            f"weights_assumed has shape {W.shape}; expected ({n_data},) or "
            f"({n_data}, {n_data})"
        )

    P = (
        np.zeros((n_params, n_params))
        if prior_precision is None
        else np.asarray(prior_precision, dtype=float)
    )

    C_inv = np.linalg.pinv(C, rcond=rcond, hermitian=True)

    fisher_assumed = J.T @ W @ J + P
    sigma_assumed = np.linalg.pinv(fisher_assumed, rcond=rcond, hermitian=True)
    sigma_optimal = np.linalg.pinv(J.T @ C_inv @ J + P, rcond=rcond, hermitian=True)

    A = sigma_assumed @ J.T @ W
    sigma_sandwich = A @ C @ A.T

    return InflationResult(
        assumed=_symmetrize(sigma_assumed),
        optimal=_symmetrize(sigma_optimal),
        sandwich=_symmetrize(sigma_sandwich),
        parameter_names=None if parameter_names is None else tuple(parameter_names),
    )


def _symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


# ---------------------------------------------------------------------------
# Per-constraint information budget
# ---------------------------------------------------------------------------


def fisher_matrix(
    jacobian: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Return ``J^T W J`` for diagonal (1-D) or full (2-D) *weights*."""
    J = np.asarray(jacobian, dtype=float)
    W = np.asarray(weights, dtype=float)
    if W.ndim == 1:
        return J.T @ (W[:, None] * J)
    return J.T @ W @ J


def information_shares(
    fishers: Mapping[str, np.ndarray],
    *,
    prior_precision: Optional[np.ndarray] = None,
    parameters: Optional[Sequence[int]] = None,
    rcond: float = 1e-12,
) -> Mapping[str, float]:
    """Rank constraints by how much they tighten the posterior.

    For each constraint, the fraction by which dropping it would inflate the
    parameter volume — ``(det Sigma_without / det Sigma_all)^(1/2p) - 1``.  A
    constraint scoring near zero cannot move the result no matter how its data
    covariance is treated, which is what makes this the cheap triage step before
    any covariance work: it is computed from the diagonal weights already in
    hand.

    Parameters
    ----------
    fishers:
        ``{label: (n_params, n_params) Fisher matrix}``, additive over
        independent constraints.
    prior_precision:
        Added to the total and to each leave-one-out total.  Strongly
        recommended: without a prior, dropping a constraint can leave a singular
        Fisher matrix and an undefined determinant.
    parameters:
        Restrict the comparison to this subset of parameter indices, to ask about
        the constraints on particular parameters rather than all of them.
    rcond:
        Relative eigenvalue cutoff for the pseudo-inverses.

    Returns
    -------
    dict
        ``{label: share}``, descending by share.
    """
    if not fishers:
        return {}
    labels = list(fishers)
    mats = [np.asarray(fishers[k], dtype=float) for k in labels]
    n_params = mats[0].shape[0]
    for k, m in zip(labels, mats):
        if m.shape != (n_params, n_params):
            raise ValueError(f"Fisher matrix {k!r} has shape {m.shape}, expected square and consistent")

    P = (
        np.zeros((n_params, n_params))
        if prior_precision is None
        else np.asarray(prior_precision, dtype=float)
    )
    total = sum(mats) + P

    sel = (
        np.arange(n_params)
        if parameters is None
        else np.asarray(sorted(set(int(i) for i in parameters)), dtype=int)
    )

    def log_volume(fisher: np.ndarray) -> float:
        sigma = np.linalg.pinv(fisher, rcond=rcond, hermitian=True)
        sub = sigma[np.ix_(sel, sel)]
        sign, logdet = np.linalg.slogdet(sub)
        return np.inf if sign <= 0 else logdet / (2.0 * sel.size)

    base = log_volume(total)
    shares = {}
    for k, m in zip(labels, mats):
        without = log_volume(total - m)
        shares[k] = float(np.exp(without - base) - 1.0) if np.isfinite(without) else np.inf
    return dict(sorted(shares.items(), key=lambda kv: kv[1], reverse=True))
