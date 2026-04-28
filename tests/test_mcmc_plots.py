"""Tests for the corner-plot wrapper.

The plotting backend is gated on the optional `corner` and `matplotlib`
packages — tests are skipped when those aren't available.
"""
from __future__ import annotations

import numpy as np
import pytest

corner = pytest.importorskip("corner")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend for tests
import matplotlib.pyplot as plt  # noqa: E402

from dendros import corner_plot, open_mcmc  # noqa: E402
from dendros._mcmc._chains import Chain, ChainSet  # noqa: E402
from dendros._mcmc._config import MCMCConfig, ModelParameter, PriorSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform(name, lo, hi, label=None):
    return ModelParameter(
        name=name,
        label=label,
        prior=PriorSpec(kind="uniform", params={"limitLower": lo, "limitUpper": hi}),
        mapper="identity",
    )


def _make_chains(parameters, n_steps=2000, n_chains=2, seed=0):
    config = MCMCConfig(
        config_path=None,
        log_file_root=None,
        simulation_kind="differentialEvolution",
        parameters=tuple(parameters),
        likelihood=None,
    )
    rng = np.random.default_rng(seed)
    chains = []
    for rank in range(n_chains):
        state = rng.normal(size=(n_steps, len(parameters)))
        chains.append(
            Chain(
                chain_index=rank,
                path=None,
                step=np.arange(1, n_steps + 1, dtype=np.int64),
                eval_time=np.zeros(n_steps),
                converged=np.zeros(n_steps, dtype=bool),
                log_posterior=np.zeros(n_steps),
                log_likelihood=np.zeros(n_steps),
                state=state,
            )
        )
    return ChainSet(config, chains)


# ---------------------------------------------------------------------------
# corner_plot
# ---------------------------------------------------------------------------


def test_corner_plot_returns_figure():
    chains = _make_chains([_uniform("a", -3, 3), _uniform("b", -3, 3)])
    fig = corner_plot(chains, post_burn=0)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_corner_plot_uses_display_labels_by_default():
    parameters = [
        _uniform("path/a", -3, 3, label=r"\alpha"),
        _uniform("path/b", -3, 3),  # no label → falls back to "b"
    ]
    chains = _make_chains(parameters)
    fig = corner_plot(chains, post_burn=0)
    axes = np.array(fig.axes).reshape(2, 2)
    # The bottom-row axes carry the x-axis labels we set.
    assert axes[1, 0].get_xlabel() == r"$\alpha$"
    assert axes[1, 1].get_xlabel() == "$b$"
    plt.close(fig)


def test_corner_plot_subset_of_parameters():
    parameters = [
        _uniform("a", -3, 3),
        _uniform("b", -3, 3),
        _uniform("c", -3, 3),
    ]
    chains = _make_chains(parameters)
    fig = corner_plot(chains, post_burn=0, parameters=["a", "c"])
    # Two-parameter corner has a 2x2 axes grid.
    assert len(fig.axes) == 4
    plt.close(fig)


def test_corner_plot_unknown_parameter_raises():
    chains = _make_chains([_uniform("a", -3, 3)])
    with pytest.raises(KeyError, match="Unknown parameter"):
        corner_plot(chains, post_burn=0, parameters=["nope"])


def test_corner_plot_explicit_labels():
    chains = _make_chains([_uniform("a", -3, 3), _uniform("b", -3, 3)])
    fig = corner_plot(
        chains, post_burn=0, labels=[r"$x_1$", r"$x_2$"]
    )
    axes = np.array(fig.axes).reshape(2, 2)
    assert axes[1, 0].get_xlabel() == r"$x_1$"
    assert axes[1, 1].get_xlabel() == r"$x_2$"
    plt.close(fig)


def test_corner_plot_label_length_mismatch_raises():
    chains = _make_chains([_uniform("a", -3, 3), _uniform("b", -3, 3)])
    with pytest.raises(ValueError, match="labels"):
        corner_plot(chains, post_burn=0, labels=["only one"])


def test_corner_plot_post_burn_filters_samples(monkeypatch):
    """post_burn must reduce the number of rows passed to corner."""
    captured = []

    real = corner.corner

    def spy(data, *args, **kwargs):
        captured.append(np.asarray(data).shape)
        return real(data, *args, **kwargs)

    monkeypatch.setattr(corner, "corner", spy)

    chains = _make_chains([_uniform("a", -3, 3)], n_steps=400, n_chains=2)
    fig0 = corner_plot(chains, post_burn=0)
    figh = corner_plot(chains, post_burn=200)
    plt.close(fig0)
    plt.close(figh)
    # Two chains × 400 = 800; with burn=200 → 2 × 200 = 400.
    assert captured[0] == (800, 1)
    assert captured[1] == (400, 1)


def test_corner_plot_drop_chains(monkeypatch):
    captured = []
    real = corner.corner

    def spy(data, *args, **kwargs):
        captured.append(np.asarray(data).shape)
        return real(data, *args, **kwargs)

    monkeypatch.setattr(corner, "corner", spy)

    chains = _make_chains([_uniform("a", -3, 3)], n_steps=200, n_chains=3)
    fig_all = corner_plot(chains, post_burn=0)
    fig_dropped = corner_plot(chains, post_burn=0, drop_chains=[1])
    plt.close(fig_all)
    plt.close(fig_dropped)
    assert captured[0] == (600, 1)
    assert captured[1] == (400, 1)


# ---------------------------------------------------------------------------
# MCMCRun.corner_plot
# ---------------------------------------------------------------------------


def test_run_corner_plot_smoketest(mcmc_de_run):
    with open_mcmc(mcmc_de_run) as run:
        # Fixture chains are tiny — silence the auto-burn warning for this test.
        with pytest.warns(UserWarning):
            fig = run.corner_plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
