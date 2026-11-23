"""Tests for drawing many histories at once in dendros.viz."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from matplotlib.colors import ListedColormap, Normalize  # noqa: E402

from dendros import open_outputs  # noqa: E402
from dendros.viz import Histories, main_branch_histories, render_tracks  # noqa: E402

WHITE = ListedColormap([[1.0, 1.0, 1.0]])
EXTENT = (0.0, 10.0, 0.0, 10.0)


def _line(n, y=5.0, brightness=0.3, width=1.5, count=1):
    x = [np.linspace(1.0, 9.0, n)] * count
    return render_tracks(
        x,
        [np.full(n, y)] * count,
        [0.0] * count,
        EXTENT,
        size=(200, 200),
        cmap=WHITE,
        norm=Normalize(0.0, 1.0),
        width=width,
        brightness=brightness,
        glow=0.0,
    )


def test_line_brightness():
    """A single line peaks at the given brightness (before the tone curve), whatever its sampling and width."""
    image = _line(2)
    peak = image[:, 100, 0].max()
    assert -np.log(1.0 - peak) == pytest.approx(0.3, rel=0.05)
    assert _line(500)[:, 100, 0].max() == pytest.approx(peak, rel=0.02)
    assert -np.log(1.0 - _line(2, width=3.0)[:, 100, 0].max()) == pytest.approx(
        0.3, rel=0.05
    )


def test_lines_add():
    single = -np.log(1.0 - _line(50)[:, 100, 0].max())
    triple = -np.log(1.0 - _line(50, count=3)[:, 100, 0].max())
    assert triple == pytest.approx(3.0 * single, rel=1.0e-6)


def test_line_position_and_color():
    image = render_tracks(
        [np.array([1.0, 9.0])],
        [np.array([2.5, 2.5])],
        [np.array([0.0, 1.0])],
        EXTENT,
        size=(200, 200),
        norm=Normalize(0.0, 1.0),
        glow=0.0,
    )
    # y = 2.5 in data lies three quarters of the way down the image.
    assert np.argmax(image[:, 100].sum(axis=1)) in (149, 150)
    # The color changes along the line (viridis: purple at 0, yellow at 1).
    left, right = image[150, 30], image[150, 170]
    assert left[2] > left[1] and right[1] > right[2]


def test_time_fraction():
    t = np.array([1.0, 2.0, 4.0, 8.0])
    histories = Histories(
        np.array([1, 2]),
        [t, t],
        {"MassBasic": [np.array([1.0, 2.0, 4.0, 8.0]), np.array([5.0, 6.0, 7.0, 8.0])]},
    )
    formation = histories.time_fraction("MassBasic")
    assert formation[0] == pytest.approx(4.0)  # reaches 4 = 8/2 at t = 4
    assert formation[1] == pytest.approx(1.0)  # above half from the start
    assert np.allclose(histories.final("MassBasic"), [8.0, 8.0])


def test_main_branch_histories(tree_file):
    with open_outputs(str(tree_file)) as c:
        histories = main_branch_histories(c)
    assert len(histories) == 1 and list(histories.tree_index) == [7]
    assert histories.time[0][0] == pytest.approx(0.5) and histories.time[0][
        -1
    ] == pytest.approx(13.8)
    assert histories.final("MassStellarTotal")[0] == pytest.approx(5.0e10)
