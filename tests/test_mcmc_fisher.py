"""Tests for linearized error propagation over a posterior sample."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from dendros import (
    fisher_matrix,
    fisher_weights,
    information_shares,
    jacobian_from_samples,
    log_likelihood_bins,
    parameter_covariances,
    rescale_correlation,
    shrink_to_diagonal,
    to_correlation,
    count_variance,
)


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


def test_jacobian_recovers_exact_linear_model():
    rng = np.random.default_rng(0)
    J_true = np.array([[1.0, -2.0], [0.5, 3.0], [0.0, 1.0]])
    m0 = np.array([10.0, 20.0, 30.0])
    state = rng.normal(size=(200, 2))
    prediction = m0 + state @ J_true.T
    fit = jacobian_from_samples(state, prediction)
    np.testing.assert_allclose(fit.jacobian, J_true, atol=1e-10)
    np.testing.assert_allclose(fit.residual_rms, 0.0, atol=1e-10)


def test_quadratic_fit_recovers_derivative_at_centre():
    # A curved model whose linear-fit slope over a wide volume differs from its
    # local derivative; degree 2 must recover the latter.
    rng = np.random.default_rng(1)
    state = rng.uniform(-1.0, 1.0, size=(600, 1))
    prediction = (state**2 + 2.0 * state).reshape(-1, 1)
    quad = jacobian_from_samples(state, prediction, degree=2)
    # Centre is ~0, where d/dx (x^2 + 2x) = 2.
    np.testing.assert_allclose(quad.jacobian[0, 0], 2.0 + 2.0 * quad.center[0], atol=1e-8)
    np.testing.assert_allclose(quad.residual_rms, 0.0, atol=1e-8)
    lin = jacobian_from_samples(state, prediction, degree=1)
    assert lin.residual_rms[0] > 1e-3  # curvature is not captured at degree 1


def test_active_columns_zero_elsewhere():
    rng = np.random.default_rng(2)
    state = rng.normal(size=(100, 4))
    prediction = (state[:, [0]] * 3.0).repeat(2, axis=1)
    fit = jacobian_from_samples(state, prediction, active=[0, 1])
    assert np.all(fit.jacobian[:, 2:] == 0.0)
    np.testing.assert_allclose(fit.jacobian[:, 0], 3.0, atol=1e-10)


def test_weights_bias_the_fit_toward_weighted_samples():
    state = np.array([[0.0], [1.0], [2.0]])
    prediction = np.array([[0.0], [1.0], [10.0]])
    # Down-weighting the outlier pulls the slope toward the first two points.
    heavy = jacobian_from_samples(state, prediction, weights=np.array([1.0, 1.0, 1e-6]))
    assert heavy.jacobian[0, 0] == pytest.approx(1.0, abs=1e-3)


def test_jacobian_rejects_bad_input():
    state = np.zeros((10, 2))
    with pytest.raises(ValueError, match="samples"):
        jacobian_from_samples(state, np.zeros((9, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        jacobian_from_samples(state, np.zeros((10, 3)), weights=-np.ones(10))
    with pytest.raises(ValueError, match="degree"):
        jacobian_from_samples(state, np.zeros((10, 3)), degree=3)
    with pytest.raises(ValueError, match="out-of-range"):
        jacobian_from_samples(state, np.zeros((10, 3)), active=[5])


# ---------------------------------------------------------------------------
# Covariance propagation
# ---------------------------------------------------------------------------


def _setup(n=12, p=2, rho=0.7, seed=3):
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(n, p))
    sigma = np.full(n, 0.5)
    D = np.diag(sigma**2)
    R = np.full((n, n), rho)
    np.fill_diagonal(R, 1.0)
    C = rescale_correlation(R, sigma)
    return J, D, C, sigma


def test_assumed_covariance_matches_direct_inverse():
    J, D, C, sigma = _setup()
    res = parameter_covariances(J, 1.0 / np.diag(D), C)
    expect = np.linalg.inv(J.T @ np.linalg.inv(D) @ J)
    np.testing.assert_allclose(res.assumed, expect, rtol=1e-8)


def test_diagonal_and_full_weights_agree():
    J, D, C, sigma = _setup()
    w = 1.0 / np.diag(D)
    a = parameter_covariances(J, w, C)
    b = parameter_covariances(J, np.diag(w), C)
    np.testing.assert_allclose(a.optimal, b.optimal, rtol=1e-10)
    np.testing.assert_allclose(a.sandwich, b.sandwich, rtol=1e-10)


def test_sandwich_never_tighter_than_optimal():
    # Gauss-Markov: the mis-weighted estimator cannot beat the optimal one.
    for seed in range(5):
        J, D, C, _ = _setup(seed=seed)
        res = parameter_covariances(J, 1.0 / np.diag(D), C)
        eig = np.linalg.eigvalsh(res.sandwich - res.optimal)
        assert eig.min() > -1e-10


def test_correct_weighting_reproduces_itself():
    # Using the true covariance as the assumed one must give no inflation.
    J, D, C, _ = _setup()
    res = parameter_covariances(J, np.linalg.inv(C), C)
    np.testing.assert_allclose(res.factors("optimal"), 1.0, rtol=1e-8)
    np.testing.assert_allclose(res.factors("sandwich"), 1.0, rtol=1e-8)
    assert res.volume_factor("optimal") == pytest.approx(1.0, rel=1e-8)


def test_zero_correlation_gives_no_inflation():
    J, D, C, sigma = _setup()
    res = parameter_covariances(J, 1.0 / np.diag(D), D)
    np.testing.assert_allclose(res.factors("optimal"), 1.0, rtol=1e-8)


def test_prior_precision_tightens_everything():
    J, D, C, _ = _setup()
    P = np.eye(J.shape[1]) * 50.0
    loose = parameter_covariances(J, 1.0 / np.diag(D), C)
    tight = parameter_covariances(J, 1.0 / np.diag(D), C, prior_precision=P)
    assert np.all(tight.sigma("assumed") < loose.sigma("assumed"))
    assert np.all(tight.sigma("optimal") < loose.sigma("optimal"))


def test_correlations_can_tighten_not_only_broaden():
    # A fully correlated mode the model cannot produce is marginalized away,
    # which can leave the remaining information sharper than the diagonal
    # assumption gave.  Inflation is not guaranteed.
    n = 6
    x = np.linspace(-1, 1, n)
    J = np.column_stack([x])          # model produces only a pure gradient
    sigma = np.ones(n)
    R = np.full((n, n), 0.95)         # near-coherent offset, orthogonal to J
    np.fill_diagonal(R, 1.0)
    C = rescale_correlation(R, sigma)
    res = parameter_covariances(J, 1.0 / sigma**2, C)
    assert res.factors("optimal")[0] < 1.0


def test_parameter_covariances_shape_validation():
    J = np.zeros((5, 2))
    with pytest.raises(ValueError, match="covariance_true"):
        parameter_covariances(J, np.ones(5), np.eye(4))
    with pytest.raises(ValueError, match="weights_assumed"):
        parameter_covariances(J, np.ones(4), np.eye(5))


def test_volume_factor_is_geometric_mean_of_factors():
    # For a diagonal-J case the two coincide, which pins the normalization.
    J = np.diag([1.0, 1.0])
    sigma = np.array([1.0, 1.0])
    C = np.diag([4.0, 9.0])
    res = parameter_covariances(J, 1.0 / sigma**2, C)
    f = res.factors("optimal")
    assert res.volume_factor("optimal") == pytest.approx(np.sqrt(f[0] * f[1]))


# ---------------------------------------------------------------------------
# Information shares
# ---------------------------------------------------------------------------


def test_information_shares_rank_by_contribution():
    strong = np.diag([100.0, 100.0])
    weak = np.diag([0.01, 0.01])
    shares = information_shares(
        {"strong": strong, "weak": weak}, prior_precision=np.eye(2)
    )
    assert list(shares) == ["strong", "weak"]
    assert shares["strong"] > shares["weak"]
    assert shares["weak"] < 0.01


def test_information_shares_empty():
    assert information_shares({}) == {}


def test_information_shares_restricted_to_parameters():
    # A constraint on parameter 1 only should score ~0 when asking about 0.
    a = np.diag([10.0, 0.0])
    b = np.diag([0.0, 10.0])
    shares = information_shares(
        {"a": a, "b": b}, prior_precision=np.eye(2), parameters=[0]
    )
    assert shares["a"] > 1.0
    assert shares["b"] == pytest.approx(0.0, abs=1e-9)


def test_fisher_matrix_diagonal_and_full_agree():
    J = np.arange(12, dtype=float).reshape(6, 2)
    w = np.linspace(1.0, 2.0, 6)
    np.testing.assert_allclose(fisher_matrix(J, w), fisher_matrix(J, np.diag(w)))


# ---------------------------------------------------------------------------
# Count statistics
# ---------------------------------------------------------------------------


def test_negative_binomial_reduces_to_poisson_as_dispersion_vanishes():
    count = np.array([0.0, 3.0, 25.0])
    mu = np.array([1.0, 2.5, 22.0])
    pois = log_likelihood_bins(count, mu, 0.0)
    nb = log_likelihood_bins(count, mu, 1e-10)
    np.testing.assert_allclose(nb, pois, atol=1e-4)


def test_negative_binomial_variance_and_weights_are_consistent():
    mu = np.array([10.0, 100.0])
    f = 0.02
    v = count_variance(mu, f)
    np.testing.assert_allclose(v, mu + f * mu**2)
    np.testing.assert_allclose(fisher_weights(mu, f), 1.0 / v)


def test_fisher_weights_zero_where_prediction_is_zero():
    w = fisher_weights(np.array([0.0, 5.0]), 0.01)
    assert w[0] == 0.0
    assert w[1] > 0.0


def test_impossible_bin_when_count_positive_and_prediction_zero():
    out = log_likelihood_bins(np.array([2.0, 0.0]), np.array([0.0, 0.0]), 0.0)
    assert out[0] < -1e29
    assert out[1] == 0.0


def test_dendros_imports_without_scipy():
    """scipy is optional (the `mcmc` extra), so `import dendros` must not need it.

    Read the Docs installs dendros without scipy; a module-level scipy import
    makes every autodoc directive in the API reference render empty.
    """
    code = "import sys; sys.modules['scipy'] = None; import dendros"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_log_likelihood_bins_raises_clear_error_without_scipy(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.special", None)
    with pytest.raises(ImportError, match=r"dendros\[mcmc\]"):
        log_likelihood_bins(np.array([1.0]), np.array([1.0]), 0.0)


def test_observed_information_requires_counts():
    with pytest.raises(ValueError, match="count"):
        fisher_weights(np.array([1.0]), 0.01, observed=True)


def test_observed_information_matches_expected_at_the_mean():
    # With count == mu the observed and expected information coincide.
    mu = np.array([20.0, 50.0])
    f = 0.03
    obs = fisher_weights(mu, f, count=mu, observed=True)
    np.testing.assert_allclose(obs, fisher_weights(mu, f), rtol=1e-12)


# ---------------------------------------------------------------------------
# Covariance utilities
# ---------------------------------------------------------------------------


def test_to_correlation_roundtrip():
    sigma = np.array([2.0, 3.0, 0.5])
    R = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    C = rescale_correlation(R, sigma)
    np.testing.assert_allclose(to_correlation(C), R, rtol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.diag(C)), sigma, rtol=1e-12)


def test_to_correlation_tolerates_empty_bins():
    C = np.array([[4.0, 0.0], [0.0, 0.0]])
    R = to_correlation(C)
    assert np.all(np.isfinite(R))
    assert R[1, 1] == 1.0


def test_shrink_to_diagonal_brackets():
    C = np.array([[1.0, 0.8], [0.8, 1.0]])
    np.testing.assert_allclose(shrink_to_diagonal(C, 0.0), np.eye(2))
    np.testing.assert_allclose(shrink_to_diagonal(C, 1.0), C)
    assert shrink_to_diagonal(C, 0.5)[0, 1] == pytest.approx(0.4)
    assert shrink_to_diagonal(C, 2.0)[0, 1] == pytest.approx(1.6)
    with pytest.raises(ValueError, match="non-negative"):
        shrink_to_diagonal(C, -1.0)


def test_shrink_to_zero_gives_no_inflation():
    J, D, C, _ = _setup()
    res = parameter_covariances(J, 1.0 / np.diag(D), shrink_to_diagonal(C, 0.0))
    np.testing.assert_allclose(res.factors("optimal"), 1.0, rtol=1e-8)
