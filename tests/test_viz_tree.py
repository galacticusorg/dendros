"""Tests for the galaxy merger tree river diagrams in dendros.viz."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from dendros import open_outputs  # noqa: E402
from dendros.viz import galaxy_merger_tree, render_river, river_layout  # noqa: E402

from ._viz_tree_data import TIME_FINAL, _samples  # noqa: E402


def test_reader_builds_the_tree(tree):
    assert tree.root == 100
    assert set(tree.branches) == {100, 101, 102, 103, 104, 200}
    assert tree.branches[102].parent == 101 and tree.branches[
        102
    ].time_merge == pytest.approx(3.0)
    assert sorted(tree.branches[100].children) == [101, 103, 104, 200]
    assert tree.branches[100].merger_times == pytest.approx([2.5, 4.0, 8.0])
    # The surviving satellite hangs off the main branch, and never merges.
    assert tree.branches[200].satellite and tree.branches[200].parent == 100
    assert tree.branches[200].time_merge is None
    assert tree.time_final == pytest.approx(TIME_FINAL)


def test_reader_without_satellites(tree_file):
    with open_outputs(str(tree_file)) as c:
        tree = galaxy_merger_tree(c, "Output1", 7, include_satellites=False)
    assert 200 not in tree.branches


def test_layout_widths_and_root(tree):
    layout = river_layout(tree, exponent=0.5, n_times=400)
    assert np.allclose(layout.center[100], 0.0)
    final = layout.width[100][-1]
    assert final == pytest.approx(1.0, rel=1.0e-6)  # the widest ribbon has width one
    # Widths follow mass**exponent (compare at a sampled time of the satellite).
    k = 300
    m = np.interp(layout.time[k], *(_samples(1.0, TIME_FINAL, 3.0e9)[:2]))
    assert layout.width[200][k] == pytest.approx((m / 5.0e10) ** 0.5, rel=2.0e-2)


def test_layout_ribbons_never_overlap(tree):
    """Except while a branch is merging into its parent, no two ribbons overlap."""
    transition = 0.04
    layout = river_layout(tree, n_times=800, transition=transition)
    ids = list(layout.center)
    merging = {}
    for b in ids:
        branch = tree.branches[b]
        merging[b] = (
            (layout.y > np.log10(branch.time_merge) - 2.0 * transition)
            if branch.time_merge
            else np.zeros(layout.y.size, dtype=bool)
        )
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            both = (
                np.isfinite(layout.center[a])
                & np.isfinite(layout.center[b])
                & ~merging[a]
                & ~merging[b]
            )
            if not np.any(both):
                continue
            gap = (
                np.abs(layout.center[a] - layout.center[b])
                - 0.5 * (layout.width[a] + layout.width[b])
            )[both]
            assert gap.min() > -1.0e-9, (a, b)


def test_merging_branch_ends_inside_its_parent(tree):
    layout = river_layout(tree, n_times=800)
    for b in (101, 102, 103, 104):
        branch = tree.branches[b]
        k = np.nonzero(np.isfinite(layout.center[b]))[0][-1]
        p = branch.parent
        separation = np.abs(layout.center[b][k] - layout.center[p][k])
        if layout.width[b][k] <= layout.width[p][k]:
            assert (
                separation + 0.5 * layout.width[b][k] <= 0.5 * layout.width[p][k] * 1.01
            ), b
        else:
            # A branch wider than its parent ends centered on it.  (The last time on the grid falls just before the
            # merger, hence the tolerance.)
            assert separation < 0.01 * layout.width[p][k], b


def test_render_river(tree):
    layout = river_layout(tree, n_times=300)
    image, extent, mappable = render_river(tree, layout, npix=200)
    assert image.shape == (300, 200, 3)
    assert np.all(np.isfinite(image)) and image.min() >= 0.0 and image.max() <= 1.0
    assert extent[0] < 0.0 < extent[1]
    # The main branch at the final time is drawn, in the color of its (star-forming) sSFR.
    column = int((0.0 - extent[0]) / (extent[1] - extent[0]) * 200)
    expected = mappable.to_rgba(np.log10(0.1 / TIME_FINAL / 1.0e9))[:3]
    assert np.allclose(image[-1, column], expected, atol=0.1)


def test_width_scale(tree):
    """Widths and gaps follow the width scale: here, doubled at the final time relative to the first."""
    plain = river_layout(tree, n_times=400)
    scaled = river_layout(
        tree, n_times=400, width_scale=lambda fraction: 1.0 + fraction
    )
    ok = np.isfinite(plain.width[100])
    factor = 1.0 + (plain.y - plain.y[0]) / (plain.y[-1] - plain.y[0])
    assert np.allclose(scaled.width[100][ok], plain.width[100][ok] * factor[ok])


def test_render_radial(tree):
    from dendros.viz import radial_layout, render_radial

    layout = radial_layout(tree, n_times=300)
    image, mappable = render_radial(tree, layout, npix=200, inner=0.08)
    assert image.shape == (200, 200, 3)
    assert np.all(np.isfinite(image)) and image.min() >= 0.0 and image.max() <= 1.0
    # The center is the final galaxy, in its final color.
    expected = mappable.to_rgba(np.log10(0.1 / TIME_FINAL / 1.0e9))[:3]
    assert np.allclose(image[100, 100], expected, atol=0.1)
    # Beyond the rim lies only background (plus a faint glow).
    assert image[0, 0].max() < 0.1
    # With the tree filling the circle, ribbons reach both the left and right halves of the rim region.
    ring = np.hypot(*np.meshgrid(np.arange(200) - 99.5, np.arange(200) - 99.5)) > 60
    lit = image.max(axis=2) > 0.2
    assert np.any(lit[:, :100] & ring[:, :100]) and np.any(lit[:, 100:] & ring[:, 100:])


@pytest.fixture
def halo_file(tmp_path):
    """A small pre-evolution halo tree.

    Halo 1 (the root, at 13.8 Gyr) has progenitors 2 (main) and 3; halo 2 has progenitors 4 (main) and 6; halo 3 has
    progenitor 5.
    """
    path = tmp_path / "halos.hdf5"
    nodes = {  # index: (parent, time, mass)
        1: (-1, 13.8, 1.0e12),
        2: (1, 10.0, 8.0e11),
        3: (1, 10.0, 1.0e11),
        4: (2, 6.0, 4.0e11),
        5: (3, 6.0, 5.0e10),
        6: (2, 6.0, 1.0e11),
    }
    with h5py.File(path, "w") as f:
        f.attrs["statusCompletion"] = 0
        out = f.create_group("Outputs/Output1")
        out.attrs["outputTime"] = TIME_FINAL
        out.attrs["outputExpansionFactor"] = 1.0
        g = f.create_group("mergerTreeStructures/tree3")
        g["nodeIndex"] = np.array(list(nodes))
        g["parentIndex"] = np.array([v[0] for v in nodes.values()])
        g["time"] = np.array([v[1] for v in nodes.values()])
        g["massBasic"] = np.array([v[2] for v in nodes.values()])
    return path


def test_halo_merger_tree(halo_file):
    from dendros.viz import halo_growth_rate, halo_merger_tree

    with open_outputs(str(halo_file)) as c:
        tree = halo_merger_tree(c, 3)
    assert tree.root == 1
    assert set(tree.branches) == {1, 3, 6}
    main = tree.branches[1]
    assert np.allclose(main.time, [6.0, 10.0, 13.8])
    assert np.allclose(main.properties["MassBasic"], [4.0e11, 8.0e11, 1.0e12])
    assert tree.branches[3].parent == 1 and tree.branches[
        3
    ].time_merge == pytest.approx(13.8)
    assert np.allclose(tree.branches[3].time, [6.0, 10.0])
    assert tree.branches[6].parent == 1 and tree.branches[
        6
    ].time_merge == pytest.approx(10.0)
    assert sorted(main.children) == [3, 6]
    # Growth rate along the main branch: d ln M / dt, positive everywhere here.
    rate = halo_growth_rate(main, window=4.0)
    assert np.all(np.isfinite(rate))
    # At 10 Gyr, over the preceding 4 Gyr the mass doubled; at 13.8 Gyr, it grew by 1.0/(interpolated 0.88) over 4 Gyr.
    assert rate[1] == pytest.approx(np.log10(np.log(2.0) / 4.0), rel=1.0e-6)
    assert rate[2] == pytest.approx(
        np.log10(
            np.log(1.0e12 / np.interp(9.8, main.time, main.properties["MassBasic"]))
            / 4.0
        ),
        rel=1.0e-6,
    )
    layout = river_layout(tree, mass="MassBasic", n_times=200)
    assert set(layout.order) == {1, 3, 6}
