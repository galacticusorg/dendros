"""Tests for the galaxy life-story figure and animation in dendros.viz."""

from __future__ import annotations

import os
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
pytest.importorskip("astropy")

import dendros.viz._story as story  # noqa: E402
from dendros.viz import (
    StoryFrame,
    compose_story_animation,
    life_story_figure,
    load_story_frames,
)  # noqa: E402

BANDS = ("SDSS_g", "SDSS_r", "SDSS_i")


def _frames(times):
    """Story frames: a Gaussian blob, brightening with time."""
    xy = np.linspace(-1.0, 1.0, 32)
    blob = np.exp(-np.add.outer(xy**2, xy**2) / 0.1)
    return [
        StoryFrame(t, 1.0 / t, {b: 1.0e3 * t * blob for b in BANDS}, 1.0e9 * t)
        for t in times
    ]


def test_render_main_branch_saves_and_resumes(tmp_path, monkeypatch):
    outputs = [
        SimpleNamespace(name=f"Output{k + 1}", time=t, redshift=z)
        for k, (t, z) in enumerate([(2.0, 3.0), (5.0, 1.0), (13.8, 0.0)])
    ]
    monkeypatch.setattr(
        story, "main_branch_rows", lambda c, tree: [(o, 7) for o in outputs]
    )
    monkeypatch.setattr(
        story,
        "galaxy_scene",
        lambda c, name, row: SimpleNamespace(
            stars=[SimpleNamespace(mass=1.0e10)], name=name
        ),
    )
    calls = []

    def render(scene, azimuth):
        calls.append((scene.name, azimuth))
        return {b: np.full((4, 4), float(len(calls))) for b in BANDS}

    frames = story.render_main_branch(
        None, None, directory=str(tmp_path), render=render, turns=1.0
    )
    assert [f.time for f in frames] == [2.0, 5.0, 13.8]
    assert [a for _, a in calls] == pytest.approx(
        [40.0, 220.0, 400.0]
    )  # one full turn from first to last
    assert len(os.listdir(tmp_path)) == 3
    # A second call reuses the saved renders.
    again = story.render_main_branch(
        None, None, directory=str(tmp_path), render=render, turns=1.0
    )
    assert len(calls) == 3
    assert np.array_equal(again[1].images["SDSS_r"], frames[1].images["SDSS_r"])
    loaded = load_story_frames(str(tmp_path))
    assert [f.time for f in loaded] == [2.0, 5.0, 13.8] and loaded[
        0
    ].mass == pytest.approx(1.0e10)


def test_life_story_figure(tree):
    fig = life_story_figure(tree, _frames([1.0, 3.0, 8.0]))
    assert len(fig.axes) == 1 + 3 + 1  # the river, three renders, and the colorbar


def test_compose_story_animation_frames(tree, tmp_path):
    frames = _frames([1.0, 2.0, 4.0, 8.0])
    directory = compose_story_animation(
        tree,
        frames,
        str(tmp_path / "story.mp4"),
        size=(480, 270),
        hold_start=2,
        hold_end=3,
        encode=False,
        star_formation=lambda t: 1.0,
    )
    written = sorted(os.listdir(directory))
    assert len(written) == 2 + (2 * len(frames) - 1) + 3
    from PIL import Image

    first = np.asarray(Image.open(os.path.join(directory, written[0])), dtype=float)
    last = np.asarray(Image.open(os.path.join(directory, written[-1])), dtype=float)
    assert first.shape == (270, 480, 3)
    # The tree panel is revealed as time passes: brighter at the end than at the start (part of the synthetic tree, and
    # the labels, are already shown at the first frame).
    assert last[:, 270:].sum() > 1.3 * first[:, 270:].sum()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_compose_story_animation_video(tree, tmp_path):
    out = compose_story_animation(
        tree,
        _frames([1.0, 4.0]),
        str(tmp_path / "story.mp4"),
        size=(320, 180),
        hold_start=1,
        hold_end=1,
    )
    assert os.path.getsize(out) > 0
