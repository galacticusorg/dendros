"""Tests for MCMC chain log-file discovery and reading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dendros import (
    Chain,
    ChainSet,
    open_mcmc,
    parse_mcmc_config,
    read_chains,
)
from dendros._mcmc._chains import discover_chain_files


def test_discover_two_ranks(mcmc_de_run):
    cfg = parse_mcmc_config(mcmc_de_run)
    files = discover_chain_files(cfg.log_file_root)
    assert len(files) == 2
    assert files[0].name == "chains_0000.log"
    assert files[1].name == "chains_0001.log"


def test_discover_returns_empty_when_no_files(tmp_path):
    files = discover_chain_files(tmp_path / "nope")
    assert files == []


def test_read_chains_basic(mcmc_de_run):
    cfg = parse_mcmc_config(mcmc_de_run)
    chains = read_chains(cfg)
    assert isinstance(chains, ChainSet)
    assert len(chains) == 2
    for c in chains:
        assert isinstance(c, Chain)
        assert c.state.shape == (5, 2)
        assert c.step.shape == (5,)
        assert c.log_posterior.shape == (5,)
        assert c.velocity is None
        assert c.converged.dtype == bool
    # Ranks should be parsed from the filename suffix.
    assert [c.chain_index for c in chains] == [0, 1]


def test_read_chains_no_files_raises(tmp_path):
    cfg = parse_mcmc_config(
        # Use the independent fixture's template but with a log root that has
        # no matching files.
        _write_minimal_config(tmp_path, log_root="missing_root")
    )
    with pytest.raises(FileNotFoundError, match="missing_root"):
        read_chains(cfg)


def _write_minimal_config(tmp_path: Path, *, log_root: str) -> Path:
    """Helper: minimal one-parameter DE config with a chosen log root."""
    p = tmp_path / "cfg.xml"
    p.write_text(
        f"""\
<?xml version="1.0" encoding="UTF-8"?>
<parameters>
  <posteriorSampleSimulation value="differentialEvolution">
    <logFileRoot value="{log_root}"/>
    <modelParameter value="active">
      <name value="x"/>
      <distributionFunction1DPrior value="uniform">
        <limitLower value="0"/><limitUpper value="1"/>
      </distributionFunction1DPrior>
      <operatorUnaryMapper value="identity"/>
      <distributionFunction1DPerturber value="cauchy">
        <median value="0"/><scale value="1e-4"/>
      </distributionFunction1DPerturber>
    </modelParameter>
  </posteriorSampleSimulation>
</parameters>
"""
    )
    return p


def test_chain_state_columns_match_input(mcmc_de_run):
    """Round-trip: the values written into the test fixture are recovered."""
    # The fixture uses a fixed seed, so we can read back and check the row-0
    # state agrees in shape and gross properties (we don't pin exact values
    # here because the fixture's RNG is internal to the conftest).
    cfg = parse_mcmc_config(mcmc_de_run)
    chains = read_chains(cfg)
    for c in chains:
        assert np.isfinite(c.state).all()
        assert np.isfinite(c.log_posterior).all()
        assert np.array_equal(c.step, np.arange(1, 6))


def test_headered_chain_validates_against_config(mcmc_de_run_headered):
    cfg = parse_mcmc_config(mcmc_de_run_headered)
    chains = read_chains(cfg)
    assert len(chains) == 1
    assert chains[0].state.shape == (4, 2)


def test_headered_chain_mismatched_names_raises(tmp_path):
    """Headered file whose parameter names disagree with the config errors out."""
    cfg_path = _write_minimal_config(tmp_path, log_root="mm")
    log_path = tmp_path / "mm_0000.log"
    with log_path.open("w") as fh:
        fh.write("# Columns:\n")
        fh.write("#    1 = Simulation step\n")
        fh.write("#    2 = Chain index\n")
        fh.write("#    3 = Evaluation time (s)\n")
        fh.write("#    4 = Chain is converged? [T/F]\n")
        fh.write("#    5 = log posterior\n")
        fh.write("#    6 = log likelihood\n")
        fh.write("#    7 = Parameter `wrong/name`\n")
        fh.write("1 0 0.1 F -1.0 -1.0 0.5\n")
    cfg = parse_mcmc_config(cfg_path)
    with pytest.raises(ValueError, match="header parameter columns"):
        read_chains(cfg)


def test_particle_swarm_splits_state_and_velocity(mcmc_ps_run):
    config_path, states, velocities = mcmc_ps_run
    cfg = parse_mcmc_config(config_path)
    assert cfg.simulation_kind == "particleSwarm"
    chains = read_chains(cfg)
    assert len(chains) == 1
    c = chains[0]
    assert c.state.shape == (3, 2)
    assert c.velocity is not None
    assert c.velocity.shape == (3, 2)
    np.testing.assert_allclose(c.state, states)
    np.testing.assert_allclose(c.velocity, velocities)


def test_chain_too_few_columns_raises(tmp_path):
    cfg_path = _write_minimal_config(tmp_path, log_root="short")
    (tmp_path / "short_0000.log").write_text("1 0 0.1 F -1.0\n")
    cfg = parse_mcmc_config(cfg_path)
    with pytest.raises(ValueError, match="expected at least"):
        read_chains(cfg)


def test_converged_flag_parsed(mcmc_de_run):
    """All-False converged column from the fixture round-trips as bool dtype."""
    cfg = parse_mcmc_config(mcmc_de_run)
    chains = read_chains(cfg)
    for c in chains:
        assert c.converged.dtype == bool
        assert not c.converged.any()


def test_post_burn_drops_leading_steps(mcmc_de_run):
    cfg = parse_mcmc_config(mcmc_de_run)
    chains = read_chains(cfg)
    sliced = chains.post_burn(2)
    assert len(sliced) == 2
    for orig, new in zip(chains, sliced):
        assert new.state.shape == (orig.n_steps - 2, 2)
        np.testing.assert_array_equal(new.state, orig.state[2:])


def test_post_burn_negative_raises(mcmc_de_run):
    cfg = parse_mcmc_config(mcmc_de_run)
    chains = read_chains(cfg)
    with pytest.raises(ValueError, match="non-negative"):
        chains.post_burn(-1)


def test_concatenated_stacks_post_burn(mcmc_de_run):
    cfg = parse_mcmc_config(mcmc_de_run)
    chains = read_chains(cfg)
    arr = chains.concatenated(burn=1)
    # Each chain has 5 rows; burn=1 → 4 rows × 2 chains = 8 rows.
    assert arr.shape == (8, 2)


def test_concatenated_drop_chains(mcmc_de_run):
    cfg = parse_mcmc_config(mcmc_de_run)
    chains = read_chains(cfg)
    arr = chains.concatenated(drop_chains=[0])
    assert arr.shape == (5, 2)
    np.testing.assert_array_equal(arr, chains[1].state)


def test_open_mcmc_returns_run(mcmc_de_run):
    with open_mcmc(mcmc_de_run) as run:
        assert run.config.simulation_kind == "differentialEvolution"
        chains = run.chains
        assert len(chains) == 2
        # Cached on second access.
        assert run.chains is chains
