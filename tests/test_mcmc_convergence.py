"""Tests for MCMC convergence diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from dendros import (
    convergence_step,
    gelman_rubin,
    geweke,
    open_mcmc,
    outlier_chains,
)
from dendros._mcmc._grubbs import grubbs_critical_value, iterative_grubbs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _well_mixed_chains(rng, *, n_chains=4, n_steps=2000, n_params=2):
    """Independent draws from N(0, 1) — should yield Rhat ≈ 1."""
    return [rng.normal(size=(n_steps, n_params)) for _ in range(n_chains)]


def _stuck_chains(rng, *, n_chains=4, n_steps=2000, n_params=2, sep=5.0):
    """Chains that stay in widely separated regions — large Rhat."""
    chains = []
    for j in range(n_chains):
        offset = sep * (j - (n_chains - 1) / 2.0)
        chains.append(offset + 0.05 * rng.normal(size=(n_steps, n_params)))
    return chains


# ---------------------------------------------------------------------------
# Grubbs
# ---------------------------------------------------------------------------


def test_grubbs_critical_value_increases_with_n():
    g_small = grubbs_critical_value(5, alpha=0.05)
    g_large = grubbs_critical_value(50, alpha=0.05)
    assert g_large > g_small


def test_grubbs_critical_value_too_few_raises():
    with pytest.raises(ValueError, match="n >= 3"):
        grubbs_critical_value(2, alpha=0.05)


def test_iterative_grubbs_no_outliers_for_clean_data():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(20, 3))
    assert iterative_grubbs(pts) == ()


def test_iterative_grubbs_flags_clear_outlier():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(20, 2))
    pts[7] = [50.0, -50.0]
    flagged = iterative_grubbs(pts)
    assert 7 in flagged


def test_iterative_grubbs_too_few_rows_returns_empty():
    pts = np.array([[0.0], [1.0]])
    assert iterative_grubbs(pts) == ()


def test_iterative_grubbs_respects_max_outliers():
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(20, 1))
    pts[0] = 100.0
    pts[1] = -100.0
    pts[2] = 80.0
    flagged = iterative_grubbs(pts, max_outliers=2)
    assert len(flagged) == 2


def test_grubbs_raises_clear_error_without_scipy(monkeypatch):
    """When scipy is missing, the user should see a clean ImportError pointing
    at the optional `mcmc` extra."""
    from dendros._mcmc import _grubbs

    def _missing():
        raise ImportError(
            "outlier_chains / Grubbs requires the optional `scipy` package. "
            "Install it with: pip install 'dendros[mcmc]'."
        )

    monkeypatch.setattr(_grubbs, "_import_student_t", _missing)
    with pytest.raises(ImportError, match=r"dendros\[mcmc\]"):
        grubbs_critical_value(10, alpha=0.05)


# ---------------------------------------------------------------------------
# Gelman-Rubin
# ---------------------------------------------------------------------------


def test_gelman_rubin_well_mixed_near_one(build_chain_set):
    rng = np.random.default_rng(10)
    chains = build_chain_set(("a", "b"), _well_mixed_chains(rng))
    result = gelman_rubin(chains, n_grid=20)
    final = result.Rhat_c[-1]
    assert np.all(final < 1.1), f"Rhat_c at final step: {final}"


def test_gelman_rubin_stuck_chains_far_from_one(build_chain_set):
    rng = np.random.default_rng(11)
    chains = build_chain_set(("a", "b"), _stuck_chains(rng))
    result = gelman_rubin(chains, n_grid=20)
    final = result.Rhat_c[-1]
    # Chains separated by 5 sigma should give a huge Rhat.
    assert np.all(final > 5.0), f"Rhat_c at final step: {final}"


def test_gelman_rubin_R_interval_well_mixed(build_chain_set):
    rng = np.random.default_rng(12)
    chains = build_chain_set(("a", "b"), _well_mixed_chains(rng))
    result = gelman_rubin(chains, n_grid=20)
    final = result.R_interval[-1]
    assert np.all(np.abs(final - 1.0) < 0.2), f"R_interval at final step: {final}"


def test_gelman_rubin_returns_correct_shapes(build_chain_set):
    rng = np.random.default_rng(13)
    chains = build_chain_set(("a", "b", "c"), _well_mixed_chains(rng, n_params=3))
    result = gelman_rubin(chains, n_grid=15)
    assert result.steps.ndim == 1
    n_eval = result.steps.size
    assert result.Rhat_c.shape == (n_eval, 3)
    assert result.R_interval.shape == (n_eval, 3)
    assert result.parameter_names == ("a", "b", "c")
    assert result.alpha_interval == 0.15
    assert result.chains_used == (0, 1, 2, 3)


def test_gelman_rubin_drop_chains(build_chain_set):
    rng = np.random.default_rng(14)
    states = _well_mixed_chains(rng, n_chains=3)
    states.append(100.0 + 0.01 * rng.normal(size=(2000, 2)))   # huge outlier chain
    chains = build_chain_set(("a", "b"), states)
    bad = gelman_rubin(chains, n_grid=10)
    good = gelman_rubin(chains, n_grid=10, drop_chains=[3])
    assert bad.Rhat_c[-1].max() > good.Rhat_c[-1].max()
    assert good.chains_used == (0, 1, 2)


def test_gelman_rubin_explicit_step_grid(build_chain_set):
    rng = np.random.default_rng(15)
    chains = build_chain_set(("a",), _well_mixed_chains(rng, n_params=1, n_steps=500))
    result = gelman_rubin(chains, step_grid=[100, 250, 500])
    assert result.steps.tolist() == [100, 250, 500]


def test_gelman_rubin_too_few_chains_raises(build_chain_set):
    rng = np.random.default_rng(16)
    chains = build_chain_set(("a",), [rng.normal(size=(200, 1))])
    with pytest.raises(ValueError, match="at least 2 chains"):
        gelman_rubin(chains)


def test_gelman_rubin_short_chain_raises(build_chain_set):
    rng = np.random.default_rng(17)
    chains = build_chain_set(("a",), [rng.normal(size=(5, 1)), rng.normal(size=(5, 1))])
    with pytest.raises(ValueError, match="min_steps"):
        gelman_rubin(chains, min_steps=10)


def test_gelman_rubin_min_steps_too_small_raises(build_chain_set):
    rng = np.random.default_rng(18)
    chains = build_chain_set(("a",), _well_mixed_chains(rng, n_params=1, n_steps=200))
    with pytest.raises(ValueError, match="min_steps"):
        gelman_rubin(chains, min_steps=1)


def test_gelman_rubin_step_grid_out_of_range_raises(build_chain_set):
    rng = np.random.default_rng(19)
    chains = build_chain_set(("a",), _well_mixed_chains(rng, n_params=1, n_steps=200))
    with pytest.raises(ValueError, match="exceeding the shortest chain"):
        gelman_rubin(chains, step_grid=[100, 999])


def test_Rhat_c_max_aggregates_over_params(build_chain_set):
    rng = np.random.default_rng(20)
    chains = build_chain_set(("a", "b"), _well_mixed_chains(rng))
    result = gelman_rubin(chains, n_grid=10)
    np.testing.assert_array_equal(result.Rhat_c_max(), result.Rhat_c.max(axis=1))


# ---------------------------------------------------------------------------
# convergence_step
# ---------------------------------------------------------------------------


def test_convergence_step_simple_crossing():
    arr = np.array([2.0, 1.5, 1.2, 1.05, 1.02])
    assert convergence_step(arr, threshold=1.1) == 3


def test_convergence_step_sustained_for():
    # First crossing at 2 is transient (next value spikes back above), so with
    # sustained_for=2 we should pick up the second crossing instead.
    arr = np.array([1.5, 1.3, 1.05, 1.4, 1.05, 1.05, 1.02])
    assert convergence_step(arr, threshold=1.1) == 2
    assert convergence_step(arr, threshold=1.1, sustained_for=2) == 4


def test_convergence_step_never_below_returns_none():
    arr = np.array([2.0, 2.0, 2.0])
    assert convergence_step(arr, threshold=1.1) is None


def test_convergence_step_empty_returns_none():
    assert convergence_step(np.array([])) is None


def test_convergence_step_bad_sustained_raises():
    with pytest.raises(ValueError, match="sustained_for"):
        convergence_step(np.array([1.0, 1.0]), sustained_for=0)


def test_convergence_step_2d_raises():
    with pytest.raises(ValueError, match="1-D"):
        convergence_step(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Geweke
# ---------------------------------------------------------------------------


def test_geweke_stationary_chain_has_small_z(build_chain_set):
    rng = np.random.default_rng(30)
    chains = build_chain_set(("a",), [rng.normal(size=(5000, 1))])
    z = geweke(chains)
    # |z| < 3 is a generous bound; the chain is genuinely stationary.
    assert np.all(np.abs(z) < 3.0), f"|z|: {np.abs(z)}"


def test_geweke_drifting_chain_has_large_z(build_chain_set):
    rng = np.random.default_rng(31)
    n = 5000
    drift = np.linspace(0.0, 5.0, n).reshape(-1, 1)
    state = drift + 0.3 * rng.normal(size=(n, 1))
    chains = build_chain_set(("a",), [state])
    z = geweke(chains)
    assert np.abs(z[0, 0]) > 5.0


def test_geweke_short_chain_returns_nan(build_chain_set):
    chains = build_chain_set(("a",), [np.zeros((3, 1))])
    z = geweke(chains)
    assert np.isnan(z).all()


def test_geweke_invalid_fractions_raise(build_chain_set):
    rng = np.random.default_rng(32)
    chains = build_chain_set(("a",), [rng.normal(size=(100, 1))])
    with pytest.raises(ValueError, match="first"):
        geweke(chains, first=1.2)
    with pytest.raises(ValueError, match="last"):
        geweke(chains, last=-0.1)
    with pytest.raises(ValueError, match="first \\+ last"):
        geweke(chains, first=0.6, last=0.6)


# ---------------------------------------------------------------------------
# outlier_chains
# ---------------------------------------------------------------------------


def test_outlier_chains_flags_distant_final_state(build_chain_set):
    rng = np.random.default_rng(40)
    states = _well_mixed_chains(rng, n_chains=5, n_steps=500)
    # Make chain 2's final state a huge outlier.
    states[2][-1] = np.array([100.0, -100.0])
    chains = build_chain_set(("a", "b"), states)
    flagged = outlier_chains(chains)
    assert 2 in flagged


def test_outlier_chains_clean_returns_empty(build_chain_set):
    rng = np.random.default_rng(41)
    chains = build_chain_set(("a", "b"), _well_mixed_chains(rng, n_chains=6, n_steps=500))
    assert outlier_chains(chains) == ()


def test_outlier_chains_too_few_returns_empty(build_chain_set):
    rng = np.random.default_rng(42)
    chains = build_chain_set(
        ("a",), [rng.normal(size=(10, 1)), rng.normal(size=(10, 1))]
    )
    assert outlier_chains(chains) == ()


def test_outlier_chains_parameters_subset(build_chain_set):
    rng = np.random.default_rng(43)
    states = _well_mixed_chains(rng, n_chains=5, n_steps=500)
    # Outlier only in parameter "b" — should *not* be flagged when we restrict
    # to "a", but should be flagged when we include "b".
    states[3][-1] = np.array([0.0, 100.0])
    chains = build_chain_set(("a", "b"), states)
    assert outlier_chains(chains, parameters=["a"]) == ()
    assert 3 in outlier_chains(chains, parameters=["b"])


def test_outlier_chains_unknown_parameter_raises(build_chain_set):
    rng = np.random.default_rng(44)
    chains = build_chain_set(("a", "b"), _well_mixed_chains(rng, n_chains=4, n_steps=200))
    with pytest.raises(KeyError, match="Unknown parameter"):
        outlier_chains(chains, parameters=["nope"])


# ---------------------------------------------------------------------------
# MCMCRun convenience methods
# ---------------------------------------------------------------------------


def test_run_convergence_methods_smoketest(mcmc_de_run):
    """Tiny end-to-end check that the methods on MCMCRun are wired up."""
    with open_mcmc(mcmc_de_run) as run:
        # Fixture chains have only 5 steps, so use min_steps=2 to exercise the API.
        result = run.gelman_rubin(min_steps=2)
        assert result.Rhat_c.shape[1] == 2
        # geweke just needs to return the expected shape.
        z = run.geweke()
        assert z.shape == (2, 2)
        # outlier_chains on so few rows returns empty.
        assert run.outlier_chains() == ()
