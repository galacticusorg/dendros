"""Tests for max-posterior/likelihood, posterior sampling, projection pursuit, and MVN fit."""
from __future__ import annotations

from xml.etree import ElementTree as ET

import numpy as np
import pytest

from dendros import (
    MaxResult,
    MVNFit,
    PosteriorSamples,
    ProjectionPursuitResult,
    maximum_likelihood,
    maximum_posterior,
    multivariate_normal_fit,
    open_mcmc,
    parse_mcmc_config,
    posterior_samples,
    projection_pursuit,
)
from dendros._mcmc._chains import Chain, ChainSet
from dendros._mcmc._config import MCMCConfig, ModelParameter, PriorSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain_set(parameters, chain_dicts):
    """Build a ChainSet directly from per-chain dict specs.

    Each entry in *chain_dicts* is a dict with keys: ``state`` (n_steps × n_params),
    optional ``log_posterior``, optional ``log_likelihood``, optional ``step``.
    """
    config = MCMCConfig(
        config_path=None,
        log_file_root=None,
        simulation_kind="differentialEvolution",
        parameters=tuple(parameters),
        likelihood=None,
    )
    chains = []
    for rank, spec in enumerate(chain_dicts):
        state = np.asarray(spec["state"], dtype=float)
        n = state.shape[0]
        chains.append(
            Chain(
                chain_index=rank,
                path=None,
                step=np.asarray(spec.get("step", np.arange(1, n + 1)), dtype=np.int64),
                eval_time=np.zeros(n),
                converged=np.zeros(n, dtype=bool),
                log_posterior=np.asarray(
                    spec.get("log_posterior", np.zeros(n)), dtype=float
                ),
                log_likelihood=np.asarray(
                    spec.get("log_likelihood", np.zeros(n)), dtype=float
                ),
                state=state,
            )
        )
    return ChainSet(config, chains)


def _uniform_param(name, lo, hi, label=None):
    return ModelParameter(
        name=name,
        label=label,
        prior=PriorSpec(kind="uniform", params={"limitLower": lo, "limitUpper": hi}),
        mapper="identity",
    )


# ---------------------------------------------------------------------------
# maximum_posterior / maximum_likelihood
# ---------------------------------------------------------------------------


def test_maximum_posterior_picks_global_max():
    rng = np.random.default_rng(0)
    parameters = [_uniform_param("a", 0, 1), _uniform_param("b", 0, 1)]
    chain0 = {
        "state": rng.normal(size=(50, 2)),
        "log_posterior": rng.normal(size=50),
    }
    chain1 = {
        "state": rng.normal(size=(50, 2)),
        "log_posterior": rng.normal(size=50),
    }
    # Inject a known maximum into chain 1, row 17.
    chain1["log_posterior"][17] = 999.0
    chain1["state"][17] = [0.42, -0.13]
    chains = _make_chain_set(parameters, [chain0, chain1])
    res = maximum_posterior(chains)
    assert isinstance(res, MaxResult)
    assert res.chain_index == 1
    assert res.step == 18                       # 1-based
    assert res.log_posterior == 999.0
    np.testing.assert_allclose(res.state, [0.42, -0.13])
    assert res.parameter_names == ("a", "b")


def test_maximum_likelihood_distinct_from_posterior():
    parameters = [_uniform_param("x", 0, 1)]
    spec = {
        "state": np.zeros((10, 1)),
        "log_posterior": np.full(10, -100.0),
        "log_likelihood": np.full(10, -200.0),
    }
    spec["log_posterior"][3] = -1.0
    spec["log_likelihood"][7] = -2.0
    chains = _make_chain_set(parameters, [spec])
    map_ = maximum_posterior(chains)
    mle = maximum_likelihood(chains)
    assert map_.step == 4
    assert mle.step == 8


def test_maximum_posterior_drop_chains():
    parameters = [_uniform_param("x", 0, 1)]
    rng = np.random.default_rng(1)
    a = {"state": rng.normal(size=(20, 1)), "log_posterior": rng.normal(size=20)}
    b = {"state": rng.normal(size=(20, 1)), "log_posterior": rng.normal(size=20)}
    b["log_posterior"][5] = 1e6   # huge max in chain 1, drop it.
    chains = _make_chain_set(parameters, [a, b])
    res = maximum_posterior(chains, drop_chains=[1])
    assert res.chain_index == 0


def test_maximum_posterior_all_dropped_raises():
    parameters = [_uniform_param("x", 0, 1)]
    chains = _make_chain_set(parameters, [{"state": np.zeros((5, 1))}])
    with pytest.raises(ValueError, match="No chains contributed"):
        maximum_posterior(chains, drop_chains=[0])


# ---------------------------------------------------------------------------
# posterior_samples
# ---------------------------------------------------------------------------


def test_posterior_samples_shape_and_metadata():
    parameters = [_uniform_param("a", 0, 1), _uniform_param("b", 0, 1)]
    rng_chain = np.random.default_rng(10)
    chains = _make_chain_set(
        parameters,
        [
            {"state": rng_chain.normal(size=(100, 2))},
            {"state": rng_chain.normal(size=(100, 2))},
        ],
    )
    samples = posterior_samples(chains, n=25, post_burn=10, rng=np.random.default_rng(42))
    assert isinstance(samples, PosteriorSamples)
    assert samples.state.shape == (25, 2)
    assert samples.log_posterior.shape == (25,)
    assert samples.log_likelihood.shape == (25,)
    assert samples.chain_index.shape == (25,)
    assert samples.step.shape == (25,)
    assert samples.parameter_names == ("a", "b")
    # Without replacement was the default (n=25 < pool=180).
    assert len(set(zip(samples.chain_index.tolist(), samples.step.tolist()))) == 25


def test_posterior_samples_reproducible_with_rng():
    parameters = [_uniform_param("a", 0, 1)]
    rng_chain = np.random.default_rng(20)
    chains = _make_chain_set(
        parameters,
        [
            {"state": rng_chain.normal(size=(100, 1))},
            {"state": rng_chain.normal(size=(100, 1))},
        ],
    )
    s1 = posterior_samples(chains, n=10, post_burn=0, rng=np.random.default_rng(7))
    s2 = posterior_samples(chains, n=10, post_burn=0, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(s1.state, s2.state)
    np.testing.assert_array_equal(s1.chain_index, s2.chain_index)


def test_posterior_samples_replace_required_when_oversampling():
    parameters = [_uniform_param("a", 0, 1)]
    chains = _make_chain_set(parameters, [{"state": np.arange(5).reshape(-1, 1)}])
    # Pool size is 5; n=20 > 5 → must use replacement.
    s = posterior_samples(chains, n=20, post_burn=0, rng=np.random.default_rng(0))
    assert s.state.shape == (20, 1)
    # Explicit replace=False raises.
    with pytest.raises(ValueError, match="without replacement"):
        posterior_samples(chains, n=20, post_burn=0, replace=False)


def test_posterior_samples_n_must_be_positive():
    parameters = [_uniform_param("a", 0, 1)]
    chains = _make_chain_set(parameters, [{"state": np.zeros((10, 1))}])
    with pytest.raises(ValueError, match="n must be positive"):
        posterior_samples(chains, n=0, post_burn=0)


def test_posterior_samples_all_chains_dropped():
    parameters = [_uniform_param("a", 0, 1)]
    chains = _make_chain_set(parameters, [{"state": np.zeros((10, 1))}])
    with pytest.raises(ValueError, match="All chains dropped"):
        posterior_samples(chains, n=5, post_burn=0, drop_chains=[0])


# ---------------------------------------------------------------------------
# projection_pursuit
# ---------------------------------------------------------------------------


def test_projection_pursuit_recovers_constrained_direction():
    """A known-degenerate posterior: y is tightly tied to x via y = x + tiny noise.

    The best-constrained direction should be (1, -1)/sqrt(2) — the
    'difference' axis, with eigenvalue ~ 0.
    """
    rng = np.random.default_rng(30)
    n = 5000
    x = rng.uniform(-1, 1, size=n)
    y = x + 0.01 * rng.normal(size=n)
    state = np.column_stack([x, y])
    parameters = [
        _uniform_param("x", -1, 1, label="x"),
        _uniform_param("y", -1, 1, label="y"),
    ]
    chains = _make_chain_set(parameters, [{"state": state}, {"state": state.copy()}])

    result = projection_pursuit(chains, post_burn=0)
    assert isinstance(result, ProjectionPursuitResult)
    assert result.eigenvalues.shape == (2,)
    # First (smallest) eigenvalue is much smaller than second.
    assert result.eigenvalues[0] < 0.01 * result.eigenvalues[1]
    # First eigenvector is approximately ±(1, -1)/sqrt(2).
    v = result.eigenvectors[:, 0]
    expected = np.array([1.0, -1.0]) / np.sqrt(2)
    assert min(np.abs(v - expected).sum(), np.abs(v + expected).sum()) < 1e-2


def test_projection_pursuit_eigenvalues_sorted_ascending():
    rng = np.random.default_rng(31)
    parameters = [
        _uniform_param("a", -1, 1),
        _uniform_param("b", -1, 1),
        _uniform_param("c", -1, 1),
    ]
    state = rng.uniform(-0.5, 0.5, size=(2000, 3))
    chains = _make_chain_set(parameters, [{"state": state}, {"state": state.copy()}])
    result = projection_pursuit(chains, post_burn=0)
    assert np.all(np.diff(result.eigenvalues) >= 0)


def test_projection_pursuit_direction_summary():
    rng = np.random.default_rng(32)
    n = 2000
    x = rng.uniform(-1, 1, size=n)
    y = x + 0.01 * rng.normal(size=n)
    parameters = [
        _uniform_param("x", -1, 1, label="x"),
        _uniform_param("y", -1, 1, label="y"),
    ]
    chains = _make_chain_set(
        parameters,
        [{"state": np.column_stack([x, y])}, {"state": np.column_stack([x, y])}],
    )
    result = projection_pursuit(chains, post_burn=0)
    components = result.direction(0)
    assert {label for label, _ in components} == {"x", "y"}
    # latex_summary returns a non-empty string.
    assert "x" in result.latex_summary(0)


def test_projection_pursuit_unsupported_mapper_raises():
    parameters = [
        ModelParameter(
            name="x",
            prior=PriorSpec(kind="uniform", params={"limitLower": 0.0, "limitUpper": 1.0}),
            mapper="logarithm",
        ),
    ]
    state = np.linspace(0.1, 0.9, 200).reshape(-1, 1)
    chains = _make_chain_set(parameters, [{"state": state}, {"state": state.copy()}])
    with pytest.raises(NotImplementedError, match="logarithm"):
        projection_pursuit(chains, post_burn=0)


def test_projection_pursuit_unsupported_prior_raises():
    parameters = [
        ModelParameter(
            name="x",
            prior=PriorSpec(kind="cauchy", params={}),
            mapper="identity",
        ),
    ]
    state = np.zeros((200, 1))
    chains = _make_chain_set(parameters, [{"state": state}, {"state": state.copy()}])
    with pytest.raises(NotImplementedError, match="cauchy"):
        projection_pursuit(chains, post_burn=0)


def test_projection_pursuit_truncated_normal_raises():
    parameters = [
        ModelParameter(
            name="x",
            prior=PriorSpec(
                kind="normal",
                params={"variance": 1.0, "limitLower": -1.0, "limitUpper": 1.0},
            ),
            mapper="identity",
        ),
    ]
    state = np.zeros((200, 1))
    chains = _make_chain_set(parameters, [{"state": state}, {"state": state.copy()}])
    with pytest.raises(NotImplementedError, match="Truncated"):
        projection_pursuit(chains, post_burn=0)


# ---------------------------------------------------------------------------
# multivariate_normal_fit
# ---------------------------------------------------------------------------


def test_multivariate_normal_fit_recovers_mean_and_covariance():
    rng = np.random.default_rng(40)
    true_mean = np.array([2.0, -1.0])
    true_cov = np.array([[1.0, 0.4], [0.4, 0.5]])
    samples = rng.multivariate_normal(true_mean, true_cov, size=10000)
    parameters = [_uniform_param("a", -10, 10), _uniform_param("b", -10, 10)]
    chains = _make_chain_set(
        parameters,
        [
            {"state": samples[:5000]},
            {"state": samples[5000:]},
        ],
    )
    fit = multivariate_normal_fit(chains, post_burn=0)
    assert isinstance(fit, MVNFit)
    np.testing.assert_allclose(fit.mean, true_mean, atol=0.05)
    np.testing.assert_allclose(fit.covariance, true_cov, atol=0.05)
    # Cholesky factor satisfies L @ L.T == cov.
    np.testing.assert_allclose(fit.cholesky @ fit.cholesky.T, fit.covariance, atol=1e-10)


def test_multivariate_normal_fit_too_few_samples_raises():
    parameters = [_uniform_param("a", 0, 1), _uniform_param("b", 0, 1)]
    state = np.zeros((2, 2))
    chains = _make_chain_set(parameters, [{"state": state}])
    with pytest.raises(ValueError, match="at least n_params \\+ 1"):
        multivariate_normal_fit(chains, post_burn=0)


def test_write_reparameterization_config_round_trips(tmp_path):
    parameters = [_uniform_param("alpha", -5, 5), _uniform_param("beta", -5, 5)]
    rng = np.random.default_rng(50)
    samples = rng.multivariate_normal([1.0, 2.0], [[1.0, 0.3], [0.3, 0.5]], size=5000)
    chains = _make_chain_set(parameters, [{"state": samples}, {"state": samples.copy()}])
    fit = multivariate_normal_fit(chains, post_burn=0)

    out_path = tmp_path / "reparam.xml"
    written = fit.write_reparameterization_config(out_path)
    assert written.is_file()

    # The output must be a valid <parameters> XML that can be re-parsed.
    root = ET.parse(str(written)).getroot()
    assert root.tag == "parameters"

    active = [el for el in root if el.get("value") == "active"]
    derived = [el for el in root if el.get("value") == "derived"]
    assert len(active) == 2
    assert len(derived) == 2

    # Active parameters are named metaParameter{i} in order, with normal(0,1)
    # priors truncated at +/- n_sigma = 5.
    for i, el in enumerate(active):
        assert el.find("name").get("value") == f"metaParameter{i}"
        prior = el.find("distributionFunction1DPrior")
        assert prior.get("value") == "normal"
        assert float(prior.find("mean").get("value")) == 0.0
        assert float(prior.find("variance").get("value")) == 1.0
        assert float(prior.find("limitLower").get("value")) == -5.0
        assert float(prior.find("limitUpper").get("value")) == 5.0

    # Derived parameter names match the original active parameters.
    derived_names = [el.find("name").get("value") for el in derived]
    assert derived_names == ["alpha", "beta"]

    # The Cholesky factor is lower-triangular, so derived[0] only references
    # metaParameter0, while derived[1] references both metaParameter0 and
    # metaParameter1.
    defs = [el.find("definition").get("value") for el in derived]
    assert "%[metaParameter0]" in defs[0]
    assert "%[metaParameter1]" not in defs[0]
    assert "%[metaParameter0]" in defs[1]
    assert "%[metaParameter1]" in defs[1]


def test_write_reparameterization_config_invalid_n_sigma(tmp_path):
    parameters = [_uniform_param("a", 0, 1), _uniform_param("b", 0, 1)]
    state = np.random.default_rng(0).normal(size=(100, 2))
    chains = _make_chain_set(parameters, [{"state": state}, {"state": state.copy()}])
    fit = multivariate_normal_fit(chains, post_burn=0)
    with pytest.raises(ValueError, match="n_sigma"):
        fit.write_reparameterization_config(tmp_path / "x.xml", n_sigma=-1.0)
    with pytest.raises(ValueError, match="perturber_scale"):
        fit.write_reparameterization_config(tmp_path / "x.xml", perturber_scale=0.0)


# ---------------------------------------------------------------------------
# MCMCRun convenience methods
# ---------------------------------------------------------------------------


def test_run_phase4_methods_smoketest(mcmc_de_run):
    with open_mcmc(mcmc_de_run) as run:
        # Fixture chains have only 5 steps.
        map_ = run.maximum_posterior()
        assert map_.state.shape == (2,)
        mle = run.maximum_likelihood()
        assert mle.state.shape == (2,)
        with pytest.warns(UserWarning):
            samples = run.posterior_samples(n=4)
        assert samples.state.shape == (4, 2)
        with pytest.warns(UserWarning):
            res = run.projection_pursuit()
        assert res.eigenvalues.shape == (2,)
        with pytest.warns(UserWarning):
            fit = run.multivariate_normal_fit()
        assert fit.mean.shape == (2,)
