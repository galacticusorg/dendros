"""Tests for MCMC configuration XML parsing."""
from __future__ import annotations

from pathlib import Path

import pytest

from dendros import (
    Likelihood,
    MCMCConfig,
    ModelParameter,
    parse_mcmc_config,
)


def test_parse_basic_de_config(mcmc_de_run):
    cfg = parse_mcmc_config(mcmc_de_run)
    assert isinstance(cfg, MCMCConfig)
    assert cfg.simulation_kind == "differentialEvolution"
    assert cfg.log_file_root.name == "chains"
    assert cfg.log_file_root.is_absolute()
    assert len(cfg.parameters) == 2
    assert cfg.parameter_names == ("p/a", "p/b")


def test_active_parameter_fields(mcmc_de_run):
    cfg = parse_mcmc_config(mcmc_de_run)
    a, b = cfg.parameters
    assert isinstance(a, ModelParameter)
    assert a.name == "p/a"
    assert a.label == "a"
    assert a.mapper == "identity"
    assert a.prior is not None
    assert a.prior.kind == "uniform"
    assert a.prior.params == {"limitLower": 0.0, "limitUpper": 1.0}
    assert a.perturber is not None
    assert a.perturber.kind == "cauchy"


def test_label_optional_and_falls_back_to_name_tail(mcmc_de_run):
    cfg = parse_mcmc_config(mcmc_de_run)
    _, b = cfg.parameters
    assert b.label is None
    assert b.display_label == "b"  # tail of "p/b"


def test_log_file_root_resolves_relative_to_config(mcmc_de_run):
    cfg = parse_mcmc_config(mcmc_de_run)
    assert cfg.log_file_root.parent == cfg.config_path.parent


def test_likelihood_tree_simple_leaf(mcmc_de_run):
    cfg = parse_mcmc_config(mcmc_de_run)
    assert isinstance(cfg.likelihood, Likelihood)
    assert cfg.likelihood.kind == "galaxyPopulation"
    assert cfg.likelihood.base_parameters_file is not None
    assert cfg.likelihood.base_parameters_file.name == "base.xml"
    assert cfg.likelihood.children == ()


def test_independent_likelihoods_carries_parameter_map(mcmc_independent_config):
    cfg = parse_mcmc_config(mcmc_independent_config)
    assert cfg.likelihood.kind == "independentLikelihoods"
    leaves = cfg.likelihood.leaves()
    assert len(leaves) == 2
    assert leaves[0].parameter_map == ("alpha",)
    assert leaves[1].parameter_map == ("alpha", "beta")
    assert leaves[0].base_parameters_file.name == "baseA.xml"
    assert leaves[1].base_parameters_file.name == "baseB.xml"


def test_state_indices_for_identity_when_no_map(mcmc_de_run):
    cfg = parse_mcmc_config(mcmc_de_run)
    leaf = cfg.likelihood.leaves()[0]
    assert cfg.state_indices_for(leaf) == (0, 1)


def test_state_indices_for_named_subset(mcmc_independent_config):
    cfg = parse_mcmc_config(mcmc_independent_config)
    a_leaf, b_leaf = cfg.likelihood.leaves()
    assert cfg.state_indices_for(a_leaf) == (0,)        # "alpha"
    assert cfg.state_indices_for(b_leaf) == (0, 1)      # "alpha beta"


def test_state_indices_for_unknown_name_raises(mcmc_independent_config):
    cfg = parse_mcmc_config(mcmc_independent_config)
    bad = Likelihood(kind="x", parameter_map=("nope",))
    with pytest.raises(KeyError, match="parameterMap"):
        cfg.state_indices_for(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_mcmc_config(tmp_path / "does_not_exist.xml")


def test_wrong_root_element_raises(tmp_path):
    p = tmp_path / "bad.xml"
    p.write_text("<other/>")
    with pytest.raises(ValueError, match="<parameters>"):
        parse_mcmc_config(p)


def test_missing_simulation_block_raises(tmp_path):
    p = tmp_path / "nosim.xml"
    p.write_text("<parameters/>")
    with pytest.raises(ValueError, match="posteriorSampleSimulation"):
        parse_mcmc_config(p)


def test_missing_log_file_root_raises(tmp_path):
    p = tmp_path / "nolog.xml"
    p.write_text(
        "<parameters>"
        "<posteriorSampleSimulation value=\"differentialEvolution\"/>"
        "</parameters>"
    )
    with pytest.raises(ValueError, match="logFileRoot"):
        parse_mcmc_config(p)
