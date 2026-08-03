"""Tests for reading proposed-state logs and using them to widen the pairing."""
from __future__ import annotations

import numpy as np
import pytest

from dendros import (
    ProposalSet,
    discover_proposal_files,
    jacobian_from_samples,
    parse_mcmc_config,
    read_predictions,
    read_proposals,
)


def _write_proposals(path, rows, *, rank=0, parameter_names=("p/a", "p/b"), header=True):
    """Write a proposal log.  Each row is (step, accepted, logp, logl, state...)."""
    with open(path, "w") as fh:
        if header:
            fh.write("# Proposed state log\n")
            fh.write("# One row per proposal evaluated, whether or not it was accepted.\n")
            fh.write("# Columns:\n")
            fh.write("#    1 = Simulation step\n")
            fh.write("#    2 = Chain index\n")
            fh.write("#    3 = Proposal accepted? [T/F]\n")
            fh.write("#    4 = log posterior of proposal\n")
            fh.write("#    5 = log likelihood of proposal\n")
            for i, name in enumerate(parameter_names, start=6):
                fh.write(f"#  {i:3d} = Parameter `{name}`\n")
        for step, accepted, logp, logl, *state in rows:
            fh.write(
                f"   {step}   {rank}   {'T' if accepted else 'F'}   "
                f"{logp!r}   {logl!r}   " + "  ".join(repr(float(x)) for x in state) + "\n"
            )


def _write_samples(path, steps, vectors, *, abscissa=(1.0,), rank=0):
    with open(path, "w") as fh:
        fh.write(f"# Sampled values for chain {rank:04d}\n")
        fh.write(" # Masses:    " + "  ".join(repr(float(a)) for a in abscissa) + "\n")
        for s, v in zip(steps, vectors):
            fh.write(f"   {s}   " + "  ".join(repr(float(x)) for x in v) + "\n")


@pytest.fixture()
def two_param_config(tmp_path, mcmc_de_run):
    return parse_mcmc_config(mcmc_de_run)


def test_discover_proposal_files(tmp_path, two_param_config):
    for rank in (0, 2):
        _write_proposals(tmp_path / f"chainsProposals_{rank:04d}.log",
                         [(1, True, -1.0, -1.5, 0.1, 0.2)], rank=rank)
    files = discover_proposal_files(tmp_path / "chains")
    assert [f.name for f in files] == [
        "chainsProposals_0000.log", "chainsProposals_0002.log"
    ]


def test_discover_ignores_chain_logs(tmp_path, two_param_config):
    (tmp_path / "chains_0000.log").write_text("1 0 0.1 F -1.0 -1.0 0.5 0.5\n")
    assert discover_proposal_files(tmp_path / "chains") == []


def test_read_proposals_parses_all_columns(tmp_path, two_param_config):
    _write_proposals(
        tmp_path / "chainsProposals_0000.log",
        [
            (1, False, -10.0, -11.0, 0.1, 0.2),
            (2, True, -9.0, -9.5, 0.3, 0.4),
            (4, False, -12.0, -12.5, 0.5, 0.6),
        ],
    )
    ps = read_proposals(two_param_config, log_file_root=tmp_path / "chains")
    assert isinstance(ps, ProposalSet)
    assert len(ps) == 1 and ps.n_proposals == 3
    s = ps[0]
    assert s.chain_index == 0
    np.testing.assert_array_equal(s.step, [1, 2, 4])
    np.testing.assert_array_equal(s.accepted, [False, True, False])
    np.testing.assert_allclose(s.log_posterior, [-10.0, -9.0, -12.0])
    np.testing.assert_allclose(s.log_likelihood, [-11.0, -9.5, -12.5])
    np.testing.assert_allclose(s.state, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    assert s.acceptance_rate == pytest.approx(1.0 / 3.0)


def test_read_proposals_missing_raises_with_guidance(tmp_path, two_param_config):
    with pytest.raises(FileNotFoundError, match="logProposals"):
        read_proposals(two_param_config, log_file_root=tmp_path / "chains")


def test_read_proposals_rejects_short_rows(tmp_path, two_param_config):
    p = tmp_path / "chainsProposals_0000.log"
    _write_proposals(p, [(1, True, -1.0, -1.5, 0.1, 0.2)])
    with open(p, "a") as fh:
        fh.write("  2  0  T  -1.0\n")   # truncated
    with pytest.raises(ValueError, match="columns"):
        read_proposals(two_param_config, log_file_root=tmp_path / "chains")


def test_paired_with_proposals_keeps_rejected_records(tmp_path, samples_dir,
                                                      build_chain_set, two_param_config):
    # Chain accepts at steps 2 and 4 only; four proposals were evaluated.
    state = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    chains = build_chain_set(["p/a", "p/b"], [state])
    _write_proposals(
        tmp_path / "chainsProposals_0000.log",
        [
            (2, True, -9.0, -9.5, 1.0, 0.0),     # accepted -> matches chain state
            (3, False, -20.0, -21.0, 7.0, 7.0),  # rejected -> state known only here
            (4, True, -8.0, -8.5, 2.0, 0.0),     # accepted
            (5, False, -30.0, -31.0, 9.0, 9.0),  # rejected
        ],
    )
    proposals = read_proposals(two_param_config, log_file_root=tmp_path / "chains")
    _write_samples(samples_dir / "c_0000.txt", [2, 3, 4, 5],
                   [[20.0], [30.0], [40.0], [50.0]])
    preds = read_predictions(samples_dir, "c")

    without = preds.paired(chains, step_offset=0)
    with_ = preds.paired(chains, step_offset=0, proposals=proposals)

    assert without.n_pairs == 2                       # accepted only
    assert with_.n_pairs == 4                         # every evaluation
    np.testing.assert_array_equal(with_.step, [2, 3, 4, 5])
    # Rejected proposals contribute their own parameters, absent from the chain.
    np.testing.assert_allclose(with_.state[1], [7.0, 7.0])
    np.testing.assert_allclose(with_.prediction.ravel(), [20.0, 30.0, 40.0, 50.0])


def test_paired_with_proposals_zero_weights_rejected(tmp_path, samples_dir,
                                                     build_chain_set, two_param_config):
    state = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    chains = build_chain_set(["p/a", "p/b"], [state])
    _write_proposals(
        tmp_path / "chainsProposals_0000.log",
        [
            (2, True, -9.0, -9.5, 1.0, 0.0),
            (3, False, -20.0, -21.0, 7.0, 7.0),
            (4, True, -8.0, -8.5, 2.0, 0.0),
        ],
    )
    proposals = read_proposals(two_param_config, log_file_root=tmp_path / "chains")
    _write_samples(samples_dir / "c_0000.txt", [2, 3, 4], [[20.0], [30.0], [40.0]])
    pp = read_predictions(samples_dir, "c").paired(
        chains, step_offset=0, proposals=proposals
    )
    # A rejected proposal is a valid model sample but carries no posterior weight.
    np.testing.assert_array_equal(pp.multiplicity, [2, 0, 2])
    # Posterior weight is unchanged by including the rejected records.
    accepted_only = read_predictions(samples_dir, "c").paired(chains, step_offset=0)
    assert pp.multiplicity.sum() == accepted_only.multiplicity.sum()


def test_paired_without_proposals_is_unchanged(tmp_path, samples_dir,
                                               build_chain_set):
    state = np.array([[0.0], [1.0], [1.0], [2.0]])
    chains = build_chain_set(["p/a"], [state])
    _write_samples(samples_dir / "c_0000.txt", [2, 4], [[1.0], [2.0]])
    pp = read_predictions(samples_dir, "c").paired(chains, step_offset=0)
    np.testing.assert_array_equal(pp.step, [2, 4])


def test_jacobian_center_override():
    # Fitting over a wide sample but expanding about a specified point: for a
    # curved model the reported derivative must follow the requested centre.
    rng = np.random.default_rng(0)
    state = rng.uniform(-1.0, 1.0, size=(800, 1))
    prediction = (state**2).reshape(-1, 1)          # d/dx = 2x
    at_half = jacobian_from_samples(
        state, prediction, degree=2, center=np.array([0.5])
    )
    assert at_half.center[0] == pytest.approx(0.5)
    assert at_half.jacobian[0, 0] == pytest.approx(1.0, abs=1e-6)
    at_zero = jacobian_from_samples(
        state, prediction, degree=2, center=np.array([0.0])
    )
    assert at_zero.jacobian[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_jacobian_center_shape_validated():
    with pytest.raises(ValueError, match="center must have shape"):
        jacobian_from_samples(
            np.zeros((10, 2)), np.zeros((10, 3)), center=np.zeros(3)
        )
