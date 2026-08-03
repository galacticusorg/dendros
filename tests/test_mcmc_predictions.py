"""Tests for reading model prediction vectors recorded alongside an MCMC."""
from __future__ import annotations

import numpy as np
import pytest

from dendros import (
    LEGACY_SAMPLE_STEP_OFFSET,
    SAMPLE_STEP_OFFSETS,
    discover_prediction_labels,
    read_predictions,
)


def _write_samples(path, steps, vectors, *, abscissa=(1.0, 2.0, 3.0), rank=0):
    """Write a sample file in Galacticus' `pathSamples` format.

    Note the abscissa header line is written with a leading space, as Galacticus'
    list-directed write produces.
    """
    with open(path, "w") as fh:
        fh.write(f"# Sampled halo mass functions for chain {rank:04d}\n")
        fh.write(" # Masses:    " + "  ".join(repr(float(a)) for a in abscissa) + "\n")
        for s, v in zip(steps, vectors):
            fh.write(f"   {s}   " + "  ".join(repr(float(x)) for x in v) + "\n")


def test_discover_labels_strips_rank_suffix(samples_dir):
    for rank in (0, 1, 7):
        _write_samples(samples_dir / f"thing_z0.000_{rank:04d}.txt", [1], [[1, 2, 3]])
    _write_samples(samples_dir / "other_z1.000_0000.txt", [1], [[1, 2, 3]])
    assert discover_prediction_labels(samples_dir) == ["other_z1.000", "thing_z0.000"]


def test_read_keeps_raw_steps_and_reads_abscissa(samples_dir):
    _write_samples(
        samples_dir / "c_0000.txt",
        [10, 11, 14],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        abscissa=(1e8, 2e8, 3e8),
    )
    ps = read_predictions(samples_dir, "c")
    assert len(ps) == 1
    assert ps.abscissa_name == "Masses"
    np.testing.assert_allclose(ps.abscissa, [1e8, 2e8, 3e8])
    # Steps are kept exactly as recorded; alignment happens in `paired`.
    np.testing.assert_array_equal(ps[0].step, [10, 11, 14])
    np.testing.assert_allclose(ps[0].prediction, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_read_missing_label_raises(samples_dir):
    with pytest.raises(FileNotFoundError):
        read_predictions(samples_dir, "absent")


def test_read_selected_ranks_only(samples_dir):
    for rank in range(4):
        _write_samples(samples_dir / f"c_{rank:04d}.txt", [1], [[1, 2, 3]], rank=rank)
    ps = read_predictions(samples_dir, "c", ranks=[1, 3])
    assert [s.chain_index for s in ps] == [1, 3]


def test_paired_keeps_only_accepted_steps(samples_dir, build_chain_set):
    # A chain whose state changes at steps 2 and 4 only.  Records exist at every
    # step, so the pairing must discard the rejected ones.
    state = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    chains = build_chain_set(["p/a", "p/b"], [state])
    # Chain steps are 1..5, so sample steps must be 0..4 to line up.
    _write_samples(
        samples_dir / "c_0000.txt",
        [0, 1, 2, 3, 4],
        [[i, i, i] for i in range(5)],
        abscissa=(1.0, 2.0, 3.0),
    )
    pp = read_predictions(samples_dir, "c").paired(chains, step_offset=1)

    np.testing.assert_array_equal(pp.step, [2, 4])
    np.testing.assert_allclose(pp.state, [[1.0, 0.0], [2.0, 0.0]])
    # Record labelled step s belongs to chain step s+1, so chain step 2 pairs
    # with the record whose vector is [1,1,1].
    np.testing.assert_allclose(pp.prediction, [[1, 1, 1], [3, 3, 3]])
    assert pp.n_pairs == 2
    assert pp.n_bins == 3
    assert pp.parameter_names == ("p/a", "p/b")


def test_paired_multiplicity_is_posterior_weight(samples_dir, build_chain_set):
    # Accepted at steps 2 and 4 of 5; the first persists for 2 steps, the
    # second to the end of the chain.
    state = np.array([[0.0], [1.0], [1.0], [2.0], [2.0]])
    chains = build_chain_set(["p/a"], [state])
    _write_samples(samples_dir / "c_0000.txt", range(5), [[i] for i in range(5)],
                   abscissa=(1.0,))
    # Records cover every step, so state the convention rather than detecting it.
    pp = read_predictions(samples_dir, "c").paired(chains, step_offset=1)
    np.testing.assert_array_equal(pp.multiplicity, [2, 2])
    assert pp.multiplicity.sum() == 4


def test_paired_pools_across_ranks(samples_dir, build_chain_set):
    s0 = np.array([[0.0], [1.0], [1.0]])
    s1 = np.array([[5.0], [5.0], [6.0]])
    chains = build_chain_set(["p/a"], [s0, s1])
    _write_samples(samples_dir / "c_0000.txt", range(3), [[i] for i in range(3)],
                   abscissa=(1.0,))
    _write_samples(samples_dir / "c_0001.txt", range(3), [[10 + i] for i in range(3)],
                   abscissa=(1.0,))
    pp = read_predictions(samples_dir, "c").paired(chains)
    assert pp.n_pairs == 2
    np.testing.assert_array_equal(sorted(pp.chain_index), [0, 1])
    np.testing.assert_allclose(sorted(pp.state.ravel()), [1.0, 6.0])


def test_paired_handles_gaps_in_recorded_steps(samples_dir, build_chain_set):
    # Proposals rejected on the prior never reach the likelihood, so step
    # indices are not contiguous; the join must not assume alignment.  Accepted
    # at chain steps 2 and 4, with the record for step 3 absent.
    state = np.array([[0.0], [1.0], [1.0], [2.0]])
    chains = build_chain_set(["p/a"], [state])
    _write_samples(samples_dir / "c_0000.txt", [1, 3], [[7.0], [9.0]], abscissa=(1.0,))
    pp = read_predictions(samples_dir, "c").paired(chains)
    np.testing.assert_array_equal(pp.step, [2, 4])
    np.testing.assert_allclose(pp.prediction.ravel(), [7.0, 9.0])


def test_paired_excludes_unclassifiable_first_row(samples_dir, build_chain_set):
    # The first retained row has no predecessor, so it cannot be known to be an
    # acceptance and is dropped even when a record exists for it.
    state = np.array([[0.0], [1.0]])
    chains = build_chain_set(["p/a"], [state])
    _write_samples(samples_dir / "c_0000.txt", [0, 1], [[7.0], [9.0]], abscissa=(1.0,))
    pp = read_predictions(samples_dir, "c").paired(chains)
    np.testing.assert_array_equal(pp.step, [2])


def test_paired_drop_chains(samples_dir, build_chain_set):
    s0 = np.array([[0.0], [1.0]])
    s1 = np.array([[0.0], [2.0]])
    chains = build_chain_set(["p/a"], [s0, s1])
    for rank, off in ((0, 0.0), (1, 10.0)):
        _write_samples(samples_dir / f"c_{rank:04d}.txt", [0, 1],
                       [[off], [off + 1]], abscissa=(1.0,))
    pp = read_predictions(samples_dir, "c").paired(chains, drop_chains=[1])
    assert pp.n_pairs == 1
    assert pp.chain_index.tolist() == [0]


def test_step_offset_detects_legacy_convention(samples_dir, build_chain_set):
    # Records labelled one behind the chain log, as written before the
    # differential_evolution.F90 labelling fix.  Only a subset of steps has a
    # record -- as in a real run, where prior-rejected proposals are never
    # evaluated -- which is what makes the convention decidable.
    state = np.array([[0.0], [1.0], [1.0], [2.0], [2.0]])
    chains = build_chain_set(["p/a"], [state])
    _write_samples(samples_dir / "c_0000.txt", [1, 3], [[1.0], [3.0]], abscissa=(1.0,))
    ps = read_predictions(samples_dir, "c")
    assert ps.step_offset(chains) == LEGACY_SAMPLE_STEP_OFFSET
    np.testing.assert_array_equal(ps.paired(chains).step, [2, 4])


def test_step_offset_detects_current_convention(samples_dir, build_chain_set):
    # Records labelled with the step the chain log records, post-fix: acceptances
    # at chain steps 2 and 4 carry those same indices.
    state = np.array([[0.0], [1.0], [1.0], [2.0], [2.0]])
    chains = build_chain_set(["p/a"], [state])
    _write_samples(samples_dir / "c_0000.txt", [2, 4], [[2.0], [4.0]], abscissa=(1.0,))
    ps = read_predictions(samples_dir, "c")
    assert ps.step_offset(chains) == 0
    pp = ps.paired(chains)
    np.testing.assert_array_equal(pp.step, [2, 4])
    # Chain step 2 must pair with the record labelled 2, whose vector is [2.0].
    np.testing.assert_allclose(pp.prediction.ravel(), [2.0, 4.0])


def test_step_offset_warns_when_indistinguishable(samples_dir, build_chain_set):
    # Records covering every step make the two conventions indistinguishable;
    # guessing silently would mispair every state, so it must warn.
    state = np.array([[0.0], [1.0], [1.0], [2.0], [2.0]])
    chains = build_chain_set(["p/a"], [state])
    _write_samples(samples_dir / "c_0000.txt", range(5), [[i] for i in range(5)],
                   abscissa=(1.0,))
    ps = read_predictions(samples_dir, "c")
    with pytest.warns(UserWarning, match="Cannot determine the sample step"):
        assert ps.step_offset(chains) == 0


def test_paired_step_offset_override(samples_dir, build_chain_set):
    state = np.array([[0.0], [1.0], [1.0], [2.0], [2.0]])
    chains = build_chain_set(["p/a"], [state])
    _write_samples(samples_dir / "c_0000.txt", range(5), [[i] for i in range(5)],
                   abscissa=(1.0,))
    ps = read_predictions(samples_dir, "c")
    # Forcing the wrong convention must change the pairing, not be silently ignored.
    np.testing.assert_allclose(
        ps.paired(chains, step_offset=1).prediction.ravel(), [1.0, 3.0]
    )
    np.testing.assert_allclose(
        ps.paired(chains, step_offset=0).prediction.ravel(), [2.0, 4.0]
    )


def test_sample_step_offsets_constant_covers_both_conventions():
    assert 0 in SAMPLE_STEP_OFFSETS and LEGACY_SAMPLE_STEP_OFFSET in SAMPLE_STEP_OFFSETS


def test_paired_empty_when_nothing_matches(samples_dir, build_chain_set):
    chains = build_chain_set(["p/a"], [np.array([[0.0], [0.0]])])  # never changes
    _write_samples(samples_dir / "c_0000.txt", [0, 1], [[1.0], [2.0]], abscissa=(1.0,))
    pp = read_predictions(samples_dir, "c").paired(chains)
    assert pp.n_pairs == 0
    assert pp.state.shape == (0, 1)


def _write_samples_with_chain(path, rows, *, abscissa=(1.0,), writer=0):
    """Write a sample file in the current format: step, chain index, values."""
    with open(path, "w") as fh:
        fh.write(f"# Sampled halo mass functions written by process {writer:04d}\n")
        fh.write(" # Masses:    " + "  ".join(repr(float(a)) for a in abscissa) + "\n")
        for step, chain, *values in rows:
            fh.write(
                f"   {step}   {chain}   " + "  ".join(repr(float(v)) for v in values) + "\n"
            )


def test_chain_index_column_detected(samples_dir):
    _write_samples_with_chain(
        samples_dir / "c_0000.txt", [(1, 0, 10.0), (2, 3, 20.0)], abscissa=(1.0,)
    )
    ps = read_predictions(samples_dir, "c")
    assert ps[0].has_chain_index
    np.testing.assert_array_equal(ps[0].record_chain, [0, 3])
    np.testing.assert_array_equal(ps[0].step, [1, 2])
    np.testing.assert_allclose(ps[0].prediction.ravel(), [10.0, 20.0])


def test_legacy_files_without_chain_index_still_read(samples_dir):
    _write_samples(samples_dir / "c_0000.txt", [1, 2], [[10.0], [20.0]], abscissa=(1.0,))
    ps = read_predictions(samples_dir, "c")
    assert not ps[0].has_chain_index
    assert ps[0].record_chain is None
    np.testing.assert_allclose(ps[0].prediction.ravel(), [10.0, 20.0])


def test_wrong_column_count_is_rejected(samples_dir):
    # Three values against a two-point abscissa is neither format.
    with open(samples_dir / "c_0000.txt", "w") as fh:
        fh.write("# header\n # Masses: 1.0 2.0\n")
        fh.write("  1  2.0 3.0 4.0 5.0\n")
    with pytest.raises(ValueError, match="columns"):
        read_predictions(samples_dir, "c")


def test_records_attributed_by_chain_index_not_by_file(samples_dir):
    # Load balancing: process 0 evaluated a step belonging to chain 1, and vice
    # versa.  Attribution must follow the chain index, not the file name.
    _write_samples_with_chain(
        samples_dir / "c_0000.txt", [(1, 0, 10.0), (2, 1, 99.0)], writer=0
    )
    _write_samples_with_chain(
        samples_dir / "c_0001.txt", [(1, 1, 20.0), (2, 0, 88.0)], writer=1
    )
    records = read_predictions(samples_dir, "c").records_by_chain()
    np.testing.assert_array_equal(records[0][0], [1, 2])
    np.testing.assert_allclose(records[0][1].ravel(), [10.0, 88.0])
    np.testing.assert_array_equal(records[1][0], [1, 2])
    np.testing.assert_allclose(records[1][1].ravel(), [20.0, 99.0])


def test_paired_uses_chain_index_under_load_balancing(samples_dir, build_chain_set):
    # Chain 0 accepts at step 2, chain 1 at step 3.  Each chain's record was
    # written by the *other* process, as load balancing permits.
    s0 = np.array([[0.0], [1.0], [1.0]])
    s1 = np.array([[5.0], [5.0], [6.0]])
    chains = build_chain_set(["p/a"], [s0, s1])
    _write_samples_with_chain(
        samples_dir / "c_0000.txt", [(2, 1, 0.0), (3, 1, 77.0)], writer=0
    )
    _write_samples_with_chain(
        samples_dir / "c_0001.txt", [(2, 0, 66.0), (3, 0, 0.0)], writer=1
    )
    pp = read_predictions(samples_dir, "c").paired(chains, step_offset=0)
    assert pp.n_pairs == 2
    by_chain = dict(zip(pp.chain_index.tolist(), pp.prediction.ravel().tolist()))
    # Chain 0's acceptance at step 2 must pick up 66.0, written by process 1.
    assert by_chain[0] == pytest.approx(66.0)
    # Chain 1's acceptance at step 3 must pick up 77.0, written by process 0.
    assert by_chain[1] == pytest.approx(77.0)


def test_mixed_formats_rejected(samples_dir):
    _write_samples_with_chain(samples_dir / "c_0000.txt", [(1, 0, 10.0)])
    _write_samples(samples_dir / "c_0001.txt", [1], [[20.0]], abscissa=(1.0,))
    with pytest.raises(ValueError, match="chain index"):
        read_predictions(samples_dir, "c").records_by_chain()
