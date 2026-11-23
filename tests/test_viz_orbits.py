"""Tests for the subhalo orbit animations in dendros.viz."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from dendros import open_outputs  # noqa: E402
from dendros.viz import SubhaloOrbits, render_orbits, subhalo_orbits  # noqa: E402

TIMES = np.linspace(10.0, 12.0, 21)


@pytest.fixture
def orbit_file(tmp_path):
    """Tree 5: host 1 with subhalo 2, which hosts sub-subhalo 3; subhalo 4 orbits another (unwritten) halo 9 until
    t = 11, after which it is a subhalo of the host. Tree 6 has its own host and a subhalo, to be ignored.
    """
    path = tmp_path / "orbits.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["statusCompletion"] = 0
        for k, t in enumerate(TIMES):
            out = f.create_group(f"Outputs/Output{k + 1}")
            out.attrs["outputTime"] = t
            out.attrs["outputExpansionFactor"] = 1.0 / (1.0 + (12.0 - t) / 10.0)
            phase = 2.0 * np.pi * (t - 10.0) / 2.0
            parent_4 = 1 if t > 11.0 else 9
            nodes = [  # tree, index, parent, isolated, position, bound mass, stellar mass
                (5, 1, -1, 1, (0.0, 0.0, 0.0), 1.0e12, 5.0e10),
                (
                    5,
                    2,
                    1,
                    0,
                    (0.1 * np.cos(phase), 0.1 * np.sin(phase), 0.0),
                    1.0e10 * (1.0 - 0.04 * k),
                    1.0e8,
                ),
                (
                    5,
                    3,
                    2,
                    0,
                    (0.1 * np.cos(phase) + 0.01, 0.1 * np.sin(phase), 0.0),
                    1.0e8,
                    0.0,
                ),
                (5, 4, parent_4, 0, (0.0, -0.15, 0.02), 3.0e9, 0.0),
                (6, 1, -1, 1, (0.0, 0.0, 0.0), 8.0e11, 3.0e10),
                (6, 2, 1, 0, (0.05, 0.0, 0.0), 2.0e9, 0.0),
            ]
            out["mergerTreeIndex"] = np.array([5, 6])
            out["mergerTreeStartIndex"] = np.array([0, 4])
            out["mergerTreeCount"] = np.array([4, 2])
            g = out.create_group("nodeData")
            g["nodeIndex"] = np.array([n[1] for n in nodes])
            g["parentIndex"] = np.array([n[2] for n in nodes])
            g["nodeIsIsolated"] = np.array([n[3] for n in nodes])
            for a, axis in enumerate("XYZ"):
                g[f"positionOrbital{axis}"] = np.array([n[4][a] for n in nodes])
            g["satelliteBoundMass"] = np.array([n[5] for n in nodes])
            g["basicMass"] = np.array([n[5] for n in nodes])
            g["diskMassStellar"] = np.array([n[6] for n in nodes])
            g["spheroidMassStellar"] = np.zeros(len(nodes))
            g["darkMatterOnlyRadiusVirial"] = np.array(
                [0.25 if n[3] else 0.02 for n in nodes]
            )
    return path


def test_subhalo_orbits(orbit_file):
    with open_outputs(str(orbit_file)) as c:
        orbits = subhalo_orbits(c, 5)
    assert np.allclose(orbits.time, TIMES)
    assert list(orbits.index) == [2, 3, 4]
    assert np.allclose(orbits.host_radius, 0.25) and np.allclose(
        orbits.host_mass, 1.0e12
    )
    # The subhalo and its sub-subhalo are present throughout, with their masses.
    assert np.all(np.isfinite(orbits.mass_bound[:2]))
    assert np.allclose(
        orbits.mass_bound[0], 1.0e10 * (1.0 - 0.04 * np.arange(TIMES.size))
    )
    assert np.allclose(
        orbits.position[0, 5], [0.1 * np.cos(np.pi / 2), 0.1 * np.sin(np.pi / 2), 0.0]
    )
    # Subhalo 4 counts only once it orbits the host.
    assert np.array_equal(np.isfinite(orbits.mass_bound[2]), TIMES > 11.0)
    # The peak mass of a subhalo losing mass is its first mass.
    assert np.allclose(orbits.mass_peak[0], 1.0e10)


def _circular(n_times=41, mass_loss=0.9):
    """One subhalo on a circular orbit of radius 0.1 Mpc in the x-y plane, losing mass."""
    t = np.linspace(10.0, 12.0, n_times)
    phase = 2.0 * np.pi * (t - 10.0) / 2.0
    position = np.stack(
        [0.1 * np.cos(phase), 0.1 * np.sin(phase), np.zeros_like(t)], axis=-1
    )[None]
    mass = (1.0e10 * (1.0 - mass_loss * (t - 10.0) / 2.0))[None]
    return SubhaloOrbits(
        t,
        np.zeros_like(t),
        np.array([1]),
        position,
        mass,
        np.full_like(mass, 1.0e7),
        np.full(t.size, 0.25),
        np.full(t.size, 1.0e12),
    )


def test_render_orbits():
    orbits = _circular()
    k = orbits.time.size - 1
    image = render_orbits(
        orbits, k, npix=200, half_width=0.2, elevation=90.0, trail=1.0, bloom=0.0
    )
    assert image.shape == (200, 200, 3)
    assert np.all(np.isfinite(image)) and image.min() >= 0.0 and image.max() < 1.0
    # Seen from above, the subhalo is now at (0.1, 0) (half an orbit after a quarter turn it is at x = +0.1).
    head = orbits.position[0, k]
    px, py = int((head[0] + 0.2) * 500), int((0.2 - head[1]) * 500)
    brightness = image.sum(axis=2)
    assert brightness[py, px] > 0.5

    # The trail, behind the head, is fainter the older it is.
    def at_time(t):
        p = orbits.position[0, np.argmin(np.abs(orbits.time - t))]
        return brightness[int((0.2 - p[1]) * 500), int((p[0] + 0.2) * 500)]

    assert at_time(11.75) > at_time(11.35) > at_time(11.05)
    # Nothing along the part of the orbit older than the trail.
    assert at_time(10.5) < 0.2 * at_time(11.75)


def test_render_orbits_ignores_small_subhalos():
    orbits = _circular()
    k = orbits.time.size - 1
    image = render_orbits(
        orbits, k, npix=100, half_width=0.2, mass_minimum=1.0e11, bloom=0.0
    )
    host_only = render_orbits(
        _circular(mass_loss=0.0),
        k,
        npix=100,
        half_width=0.2,
        mass_minimum=1.0e11,
        bloom=0.0,
    )
    assert np.allclose(image, host_only)


def test_view_matrix():
    from dendros.viz._orbits import _view_matrix

    # From above, the host's x-y plane is the image plane; edge-on, the host's z axis points up the image.
    assert np.allclose(_view_matrix(90.0, 0.0) @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    assert np.allclose(_view_matrix(0.0, 0.0) @ [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    m = _view_matrix(30.0, 70.0)
    assert np.allclose(m @ m.T, np.eye(3))


def test_compose_orbit_animation_frames(tmp_path):
    import os

    from dendros.viz import compose_orbit_animation

    orbits = _circular(n_times=6)
    directory = compose_orbit_animation(
        orbits, str(tmp_path / "orbits.mp4"), size=(320, 180), hold_end=2, encode=False
    )
    assert len(os.listdir(directory)) == 6 + 2


def test_render_orbits_negative_index():
    orbits = _circular()
    last = render_orbits(orbits, orbits.time.size - 1, npix=100, half_width=0.2)
    assert np.array_equal(render_orbits(orbits, -1, npix=100, half_width=0.2), last)
