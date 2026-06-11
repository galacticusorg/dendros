"""Tests for star formation history helpers (sfh_collapse_metallicities / sfh_times)."""
from __future__ import annotations

import numpy as np

from dendros import open_outputs, sfh_collapse_metallicities, sfh_times


# ---------------------------------------------------------------------------
# Shared fixed-time tabulation (the common case)
# ---------------------------------------------------------------------------


def test_sfh_fixed_time_times(sfh_fixed_time_file):
    with open_outputs(sfh_fixed_time_file) as c:
        ds = c["Outputs/Output1/nodeData/diskStarFormationHistoryMass"]
        times = sfh_times(ds)
    np.testing.assert_allclose(times, [1.0, 2.0, 3.0])


def test_sfh_fixed_time_collapse(sfh_fixed_time_file):
    with open_outputs(sfh_fixed_time_file) as c:
        ds = c["Outputs/Output1/nodeData/diskStarFormationHistoryMass"]
        mass = sfh_collapse_metallicities(ds)
    # Non-ragged 2D array: two populated galaxies + one empty (zeros).
    assert mass.shape == (3, 3)
    np.testing.assert_allclose(mass[0], [11.0, 22.0, 33.0])
    np.testing.assert_allclose(mass[1], [44.0, 55.0, 66.0])
    np.testing.assert_allclose(mass[2], [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# fixedAges lightcone tabulation (ragged -> right-aligned 2D)
# ---------------------------------------------------------------------------


def test_sfh_fixedages_collapse_is_non_ragged(sfh_fixedages_file):
    with open_outputs(sfh_fixedages_file, output_root="Lightcone") as c:
        ds = c["Lightcone/Output1/nodeData/diskStarFormationHistoryMass"]
        mass = sfh_collapse_metallicities(ds)
    assert isinstance(mass, np.ndarray)
    assert mass.shape == (3, 3)
    # Right-aligned, front-padded with zeros where bins were dropped.
    np.testing.assert_allclose(mass[0], [11.0, 22.0, 33.0])
    np.testing.assert_allclose(mass[1], [0.0, 44.0, 55.0])
    np.testing.assert_allclose(mass[2], [0.0, 0.0, 0.0])


def test_sfh_fixedages_times_right_aligned(sfh_fixedages_file):
    with open_outputs(sfh_fixedages_file, output_root="Lightcone") as c:
        ds = c["Lightcone/Output1/nodeData/diskStarFormationHistoryMass"]
        times = sfh_times(ds)
    assert times.shape == (3, 3)
    np.testing.assert_allclose(times[0], [1.0, 2.0, 3.0])
    # Dropped bins are NaN-padded at the front; the crossing-time bin is last.
    assert np.isnan(times[1][0])
    np.testing.assert_allclose(times[1][1:], [1.5, 2.5])
    assert np.all(np.isnan(times[2]))


def test_sfh_fixedages_mass_and_times_aligned(sfh_fixedages_file):
    """Mass and times share the same right-alignment, so finite times line up
    with the populated mass columns."""
    with open_outputs(sfh_fixedages_file, output_root="Lightcone") as c:
        ds = c["Lightcone/Output1/nodeData/diskStarFormationHistoryMass"]
        mass = sfh_collapse_metallicities(ds)
        times = sfh_times(ds)
    # For galaxy 1, the padded (NaN) time column is exactly the zero-padded mass.
    pad = np.isnan(times[1])
    np.testing.assert_allclose(mass[1][pad], 0.0)


def test_sfh_fixedages_last_column_is_crossing(sfh_fixedages_file):
    """The final column holds each populated galaxy's latest (crossing) time."""
    with open_outputs(sfh_fixedages_file, output_root="Lightcone") as c:
        ds = c["Lightcone/Output1/nodeData/diskStarFormationHistoryMass"]
        times = sfh_times(ds)
    np.testing.assert_allclose(times[0][-1], 3.0)
    np.testing.assert_allclose(times[1][-1], 2.5)


# ---------------------------------------------------------------------------
# Ragged tabulation without the fixedAges method -> legacy list behaviour
# ---------------------------------------------------------------------------


def test_sfh_variable_time_collapse_is_ragged_list(sfh_variable_time_file):
    with open_outputs(sfh_variable_time_file) as c:
        ds = c["Outputs/Output1/nodeData/diskStarFormationHistoryMass"]
        mass = sfh_collapse_metallicities(ds)
    assert isinstance(mass, list)
    np.testing.assert_allclose(mass[0], [11.0, 22.0, 33.0])
    np.testing.assert_allclose(mass[1], [44.0, 55.0])


def test_sfh_variable_time_times_none(sfh_variable_time_file):
    with open_outputs(sfh_variable_time_file) as c:
        ds = c["Outputs/Output1/nodeData/diskStarFormationHistoryMass"]
        times = sfh_times(ds)
    assert times is None
