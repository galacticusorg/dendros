"""Tests for autocorrelation, integrated autocorrelation time, ESS, and acceptance rate."""
from __future__ import annotations

import numpy as np
import pytest

from dendros import (
    acceptance_rate,
    acceptance_rate_trace,
    autocorrelation_function,
    autocorrelation_time,
    effective_sample_size,
    open_mcmc,
)
from dendros._mcmc._autocorr import _acf_1d, _auto_window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ar1(rng, *, n, phi, n_params=1):
    """Generate an AR(1) chain ``x_{t+1} = phi*x_t + eps``.

    Stationary marginal variance is 1 (epsilon variance is 1 - phi^2).
    """
    out = np.zeros((n, n_params))
    sigma_eps = np.sqrt(1.0 - phi**2)
    for j in range(n_params):
        x = 0.0
        for t in range(n):
            x = phi * x + sigma_eps * rng.normal()
            out[t, j] = x
    return out


# ---------------------------------------------------------------------------
# _acf_1d
# ---------------------------------------------------------------------------


def test_acf_1d_zero_lag_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    rho = _acf_1d(x)
    assert rho[0] == pytest.approx(1.0)


def test_acf_1d_constant_returns_delta():
    rho = _acf_1d(np.ones(50))
    assert rho[0] == 1.0
    assert np.all(rho[1:] == 0.0)


def test_acf_1d_white_noise_decays_quickly():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    rho = _acf_1d(x)
    # Beyond lag 0 the ACF should be small (< 0.1) on average.
    assert np.abs(rho[1:50]).mean() < 0.1


def test_acf_1d_ar1_matches_phi():
    rng = np.random.default_rng(2)
    phi = 0.7
    x = _ar1(rng, n=20000, phi=phi)[:, 0]
    rho = _acf_1d(x)
    # Theoretical ACF is phi^k.  Check a few small lags.
    for k in (1, 2, 3, 5):
        assert abs(rho[k] - phi**k) < 0.05, f"k={k}: rho={rho[k]}, expected {phi**k}"


# ---------------------------------------------------------------------------
# Sokal autowindow
# ---------------------------------------------------------------------------


def test_auto_window_returns_first_failing_index():
    # taus = [1, 1, 1, 1, ...]; with c=2 the inequality M < c*taus[M]
    # fails first at M=2 (since 2 >= 2*1).
    taus = np.ones(10)
    assert _auto_window(taus, c=2.0) == 2


def test_auto_window_falls_back_when_never_fails():
    # taus large enough that M < c*taus[M] for all M.
    taus = np.full(5, 1000.0)
    assert _auto_window(taus, c=2.0) == 4


# ---------------------------------------------------------------------------
# autocorrelation_function
# ---------------------------------------------------------------------------


def test_autocorrelation_function_shape(build_chain_set):
    rng = np.random.default_rng(10)
    chains = build_chain_set(("a", "b"), [rng.normal(size=(500, 2)) for _ in range(3)])
    acf = autocorrelation_function(chains, max_lag=20)
    assert acf.shape == (3, 21, 2)
    # Lag-0 entries are 1 (or NaN if a chain was constant — not the case here).
    np.testing.assert_allclose(acf[:, 0, :], 1.0)


def test_autocorrelation_function_max_lag_too_large_raises(build_chain_set):
    rng = np.random.default_rng(11)
    chains = build_chain_set(("a",), [rng.normal(size=(50, 1)), rng.normal(size=(50, 1))])
    with pytest.raises(ValueError, match="max_lag"):
        autocorrelation_function(chains, max_lag=500)


def test_autocorrelation_function_post_burn(build_chain_set):
    rng = np.random.default_rng(12)
    chains = build_chain_set(("a",), [rng.normal(size=(200, 1)), rng.normal(size=(200, 1))])
    acf = autocorrelation_function(chains, post_burn=50, max_lag=10)
    assert acf.shape == (2, 11, 1)


def test_autocorrelation_function_negative_post_burn_raises(build_chain_set):
    rng = np.random.default_rng(13)
    chains = build_chain_set(("a",), [rng.normal(size=(50, 1)), rng.normal(size=(50, 1))])
    with pytest.raises(ValueError, match="post_burn"):
        autocorrelation_function(chains, post_burn=-1)


# ---------------------------------------------------------------------------
# autocorrelation_time / ESS
# ---------------------------------------------------------------------------


def test_autocorrelation_time_white_noise_near_one(build_chain_set):
    rng = np.random.default_rng(20)
    chains = build_chain_set(
        ("a",), [rng.normal(size=(8000, 1)) for _ in range(4)]
    )
    tau = autocorrelation_time(chains, post_burn=0)
    assert tau.shape == (1,)
    assert 0.5 < tau[0] < 2.0, f"tau={tau[0]}"


def test_autocorrelation_time_ar1_matches_theory(build_chain_set):
    rng = np.random.default_rng(21)
    phi = 0.9
    expected = (1 + phi) / (1 - phi)  # = 19
    chains = build_chain_set(
        ("a",), [_ar1(rng, n=20000, phi=phi) for _ in range(4)]
    )
    tau = autocorrelation_time(chains, post_burn=0)
    # Sokal estimator should land in a broad neighbourhood of 19.
    assert 10.0 < tau[0] < 30.0, f"tau={tau[0]}, expected ~{expected}"


def test_effective_sample_size_white_noise_close_to_n(build_chain_set):
    rng = np.random.default_rng(22)
    n_steps, n_chains = 4000, 4
    chains = build_chain_set(
        ("a",), [rng.normal(size=(n_steps, 1)) for _ in range(n_chains)]
    )
    ess = effective_sample_size(chains, post_burn=0)
    n_total = n_steps * n_chains
    # Expect ESS within 30% of N for white noise.
    assert 0.7 * n_total < ess[0] < 1.3 * n_total, f"ESS={ess[0]} vs N={n_total}"


def test_effective_sample_size_ar1_smaller_than_n(build_chain_set):
    rng = np.random.default_rng(23)
    phi = 0.9
    n_steps, n_chains = 8000, 4
    chains = build_chain_set(
        ("a",), [_ar1(rng, n=n_steps, phi=phi) for _ in range(n_chains)]
    )
    ess = effective_sample_size(chains, post_burn=0)
    n_total = n_steps * n_chains
    assert ess[0] < n_total / 5.0, f"ESS={ess[0]} vs N={n_total}"


def test_autocorrelation_time_short_chain_raises(build_chain_set):
    rng = np.random.default_rng(24)
    chains = build_chain_set(("a",), [rng.normal(size=(1, 1)), rng.normal(size=(1, 1))])
    with pytest.raises(ValueError, match="at least two samples"):
        autocorrelation_time(chains, post_burn=0)


# ---------------------------------------------------------------------------
# Auto burn-in (post_burn=None)
# ---------------------------------------------------------------------------


def test_post_burn_auto_warns_when_no_convergence(build_chain_set):
    """A short chain set won't converge on the default grid → fallback warning."""
    rng = np.random.default_rng(30)
    chains = build_chain_set(
        ("a",), [rng.normal(size=(50, 1)) + 5 * j for j in range(3)]
    )
    with pytest.warns(UserWarning, match="convergence"):
        tau = autocorrelation_time(chains, post_burn=None)
    # Falls back to post_burn=0 so we still get a finite τ.
    assert np.isfinite(tau[0])


def test_post_burn_auto_works_for_well_mixed(build_chain_set):
    """Well-mixed chains that converge early — None should pick a small burn."""
    rng = np.random.default_rng(31)
    chains = build_chain_set(
        ("a",), [rng.normal(size=(2000, 1)) for _ in range(4)]
    )
    tau = autocorrelation_time(chains, post_burn=None)
    assert 0.5 < tau[0] < 2.0


# ---------------------------------------------------------------------------
# acceptance_rate
# ---------------------------------------------------------------------------


def test_acceptance_rate_all_changes_is_one(build_chain_set):
    state = np.arange(100, dtype=float).reshape(-1, 1)  # strictly increasing
    chains = build_chain_set(("a",), [state, state])
    rates = acceptance_rate(chains, post_burn=0)
    np.testing.assert_allclose(rates, [1.0, 1.0])


def test_acceptance_rate_no_changes_is_zero(build_chain_set):
    state = np.zeros((100, 2))  # constant
    chains = build_chain_set(("a", "b"), [state, state])
    rates = acceptance_rate(chains, post_burn=0)
    np.testing.assert_allclose(rates, [0.0, 0.0])


def test_acceptance_rate_mixed():
    """Construct a chain whose first n/2 transitions repeat the row, the rest move."""
    state = np.zeros((100, 1))
    state[50:, 0] = np.arange(50, dtype=float)  # changes start at row 50
    from tests.conftest import _build_chain_set

    chains = _build_chain_set(("a",), [state])
    rates = acceptance_rate(chains, post_burn=0)
    # 49 changing transitions out of 99 total.
    assert rates[0] == pytest.approx(49 / 99)


def test_acceptance_rate_too_short_returns_nan(build_chain_set):
    chains = build_chain_set(("a",), [np.zeros((1, 1))])
    rates = acceptance_rate(chains, post_burn=0)
    assert np.isnan(rates[0])


def test_acceptance_rate_post_burn(build_chain_set):
    """Post-burn region all stationary → rate = 0 even though pre-burn moved a lot."""
    state = np.zeros((100, 1))
    state[:30, 0] = np.arange(30, dtype=float)
    chains = build_chain_set(("a",), [state])
    rates = acceptance_rate(chains, post_burn=40)
    assert rates[0] == 0.0


# ---------------------------------------------------------------------------
# acceptance_rate_trace
# ---------------------------------------------------------------------------


def test_acceptance_rate_trace_shape(build_chain_set):
    rng = np.random.default_rng(40)
    chains = build_chain_set(
        ("a",), [rng.normal(size=(200, 1)), rng.normal(size=(200, 1))]
    )
    traces = acceptance_rate_trace(chains, window=30, post_burn=0)
    assert len(traces) == 2
    for t in traces:
        assert t.shape == (200,)
        # First `window` entries are NaN.
        assert np.isnan(t[:30]).all()
        # Beyond that, values are in [0, 1].
        assert np.all((t[30:] >= 0.0) & (t[30:] <= 1.0))


def test_acceptance_rate_trace_constant_chain_is_zero(build_chain_set):
    chains = build_chain_set(("a",), [np.zeros((100, 1))])
    traces = acceptance_rate_trace(chains, window=10, post_burn=0)
    assert (traces[0][10:] == 0.0).all()


def test_acceptance_rate_trace_strict_window_validates(build_chain_set):
    chains = build_chain_set(("a",), [np.zeros((50, 1))])
    with pytest.raises(ValueError, match="window"):
        acceptance_rate_trace(chains, window=0)


# ---------------------------------------------------------------------------
# MCMCRun convenience methods
# ---------------------------------------------------------------------------


def test_run_phase3_methods_smoketest(mcmc_de_run):
    with open_mcmc(mcmc_de_run) as run:
        # Fixture chains have only 5 steps; auto burn-in will warn about
        # never reaching convergence — silence and test the API surface.
        with pytest.warns(UserWarning):
            rates = run.acceptance_rate()
        assert rates.shape == (2,)
        traces = run.acceptance_rate_trace(window=2, post_burn=0)
        assert len(traces) == 2
        with pytest.warns(UserWarning):
            tau = run.autocorrelation_time()
        assert tau.shape == (2,)
        with pytest.warns(UserWarning):
            ess = run.effective_sample_size()
        assert ess.shape == (2,)
