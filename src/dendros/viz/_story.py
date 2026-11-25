"""The life story of a galaxy: renders of it through time, set beside its merger tree.

:func:`render_main_branch` renders the galaxy on the main branch of a merger tree at
every output of a model; :func:`life_story_figure` sets a few such renders above the
tree, drawn as a horizontal river; and :func:`compose_story_animation` turns renders at
many outputs into a video, with the tree revealed as time passes.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ._galaxy import GalaxyScene, galaxy_scene
from ._render import DustModel, compose_rgb, render_galaxy
from ._texture import SpiralTexture
from ._tree import (
    GalaxyMergerTree,
    RiverLayout,
    main_branch_rows,
    render_river,
    river_layout,
)

_FONT = "DejaVuSans.ttf"


@dataclass
class StoryFrame:
    """One render of the galaxy on a main branch."""

    time: float
    redshift: float
    #: Surface brightness images per band (as returned by :func:`~dendros.viz.render_galaxy`).
    images: Dict[str, np.ndarray]
    #: Stellar mass [M☉].
    mass: float

    def rgb(self, **kwargs) -> np.ndarray:
        """The 8-bit color image (see :func:`~dendros.viz.compose_rgb` for the keywords)."""
        return compose_rgb(self.images, **kwargs)


def presentation_render(
    scene: GalaxyScene,
    half_width: float = 0.025,
    npix: int = 512,
    n_steps: int = 1200,
    inclination: float = 55.0,
    position_angle: float = 20.0,
    azimuth: float = 40.0,
    seed: int = 7,
) -> Dict[str, np.ndarray]:
    """Render ``scene`` with the presentation options (spiral texture, thinner dust and young-star layers).

    The default renderer of :func:`render_main_branch`. ``half_width`` [Mpc] is fixed, so
    that renders at different times share one physical scale.
    """
    texture = (
        SpiralTexture(
            radius_scale=scene.radius_disk,
            arms=2,
            pitch=16.0,
            clumpiness=1.0,
            seed=seed,
        )
        if scene.radius_disk > 0.0
        else None
    )
    return render_galaxy(
        scene,
        inclination=inclination,
        position_angle=position_angle,
        azimuth=azimuth,
        npix=npix,
        half_width=half_width,
        n_steps=n_steps,
        texture=texture,
        dust=DustModel(scale_height_factor=0.4),
        young_scale_height={"disk": 0.3},
    )


def render_main_branch(
    c,
    tree: GalaxyMergerTree,
    directory: Optional[str] = None,
    render: Optional[Callable[..., Dict[str, np.ndarray]]] = None,
    turns: float = 0.0,
    outputs: Optional[Sequence[str]] = None,
    progress: bool = False,
) -> List[StoryFrame]:
    """Render the galaxy on the main branch of ``tree`` at each output in which it is present.

    The galaxy is found in each output by :func:`~dendros.viz.main_branch_rows`.
    ``render(scene, azimuth=...)`` renders one galaxy (default:
    :func:`presentation_render`); the galaxy turns about its own axis by ``turns`` full
    rotations from the first render to the last.  ``outputs`` restricts the outputs used
    (by name).  With ``directory``, each render is saved there (``frame<k>.npz``), and
    renders already saved are reused, so an interrupted run can be resumed and the color
    stretch changed later without rendering again.
    """
    render = render or presentation_render
    rows = main_branch_rows(c, tree)
    if outputs is not None:
        rows = [(o, r) for o, r in rows if o.name in outputs]
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    frames = []
    for k, (output, row) in enumerate(rows):
        target = (
            os.path.join(directory, f"frame{k:04d}.npz")
            if directory is not None
            else None
        )
        if target is not None and os.path.exists(target):
            frames.append(_load_frame(target))
            continue
        scene = galaxy_scene(c, output.name, row)
        azimuth = 40.0 + 360.0 * turns * k / max(len(rows) - 1, 1)
        images = render(scene, azimuth=azimuth)
        mass = float(sum(s.mass for s in scene.stars))
        frame = StoryFrame(float(output.time), float(output.redshift), images, mass)
        if target is not None:
            np.savez_compressed(
                target,
                time=frame.time,
                redshift=frame.redshift,
                mass=frame.mass,
                **{b: v.astype(np.float32) for b, v in images.items()},
            )
        frames.append(frame)
        if progress:
            print(f"rendered {output.name} (z = {output.redshift:.3f})", flush=True)
    return frames


def _load_frame(path: str) -> StoryFrame:
    d = np.load(path)
    bands = [k for k in d.files if k not in ("time", "redshift", "mass")]
    return StoryFrame(
        float(d["time"]),
        float(d["redshift"]),
        {b: d[b] for b in bands},
        float(d["mass"]),
    )


def load_story_frames(directory: str) -> List[StoryFrame]:
    """Load the renders saved by :func:`render_main_branch`, in time order."""
    frames = [
        _load_frame(f) for f in sorted(glob.glob(os.path.join(directory, "frame*.npz")))
    ]
    return sorted(frames, key=lambda f: f.time)


_RGB_DEFAULTS = dict(mu_stretch=23.5, Q=12.0, bloom=0.15, bloom_sigma=0.01)


def life_story_figure(
    tree: GalaxyMergerTree,
    frames: Sequence[StoryFrame],
    layout: Optional[RiverLayout] = None,
    half_width_kpc: float = 25.0,
    rgb: Optional[dict] = None,
    figsize=(24, 11),
    dpi: int = 110,
    **river_kwargs,
):
    """A figure of a galaxy's life: its merger tree as a horizontal river, with ``frames`` above it.

    Each image is joined to the moment it shows on the main branch.  ``half_width_kpc`` is
    the half-width of the renders (for the scale bar), ``rgb`` the keywords of
    :func:`~dendros.viz.compose_rgb`, and other keywords go to
    :func:`~dendros.viz.render_river`.  Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    layout = layout or river_layout(tree, mass_minimum=1.0e6, n_times=1400)
    rgb = {**_RGB_DEFAULTS, **(rgb or {})}
    image, extent, mappable = render_river(tree, layout, npix=700, **river_kwargs)
    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor("#05060a")
        ax_river = fig.add_axes([0.04, 0.07, 0.92, 0.50])
        ax_river.imshow(
            np.transpose(image, (1, 0, 2)),
            origin="lower",
            aspect="auto",
            extent=(extent[2], extent[3], extent[0], extent[1]),
            interpolation="antialiased",
        )
        ax_river.set_facecolor("#05060a")
        ticks = np.array(
            [
                t
                for t in (0.3, 0.5, 1, 2, 3, 5, 8, 13.8)
                if layout.time[0] <= t <= layout.time[-1]
            ]
        )
        ax_river.set_xticks(layout.coordinate(ticks), [f"{t:g}" for t in ticks])
        ax_river.set_xlabel("cosmic time [Gyr]")
        ax_river.set_yticks([])
        for s in ("top", "right", "left"):
            ax_river.spines[s].set_visible(False)
        width = 0.92 / max(len(frames), 1)
        for k, frame in enumerate(frames):
            ax = fig.add_axes([0.04 + k * width + 0.004, 0.63, width - 0.008, 0.33])
            ax.imshow(frame.rgb(**rgb), origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#555a66")
            ax.set_title(
                f"z = {frame.redshift:.1f}   ({frame.time:.1f} Gyr)", fontsize=11
            )
            fig.add_artist(
                ConnectionPatch(
                    xyA=(0.5, 0.0),
                    coordsA=ax.transAxes,
                    xyB=(layout.coordinate(frame.time), 0.0),
                    coordsB=ax_river.transData,
                    color="#9aa3b5",
                    lw=0.6,
                    alpha=0.35,
                    linestyle=(0, (4, 3)),
                )
            )
            ax_river.plot(
                layout.coordinate(frame.time),
                0.0,
                "o",
                ms=5,
                mfc="white",
                mec="#05060a",
                mew=0.8,
                zorder=5,
            )
            if k == 0:
                fraction = 10.0 / (2.0 * half_width_kpc)
                ax.plot(
                    [0.06, 0.06 + fraction],
                    [0.06, 0.06],
                    color="w",
                    lw=2,
                    transform=ax.transAxes,
                )
                ax.text(
                    0.06 + fraction / 2,
                    0.09,
                    "10 kpc",
                    color="w",
                    fontsize=9,
                    ha="center",
                    transform=ax.transAxes,
                )
        fig.colorbar(mappable, ax=ax_river, fraction=0.012, pad=0.005).set_label(
            r"$\log_{10}$ sSFR [yr$^{-1}$]"
        )
    return fig


_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _scientific(value: float) -> str:
    exponent = int(np.floor(np.log10(value))) if value > 0.0 else 0
    return f"{value / 10.0 ** exponent:.1f} × 10{str(exponent).translate(_SUPERSCRIPT)}"


def compose_story_animation(
    tree: GalaxyMergerTree,
    frames: Sequence[StoryFrame],
    output: str,
    star_formation: Optional[Callable[[float], float]] = None,
    layout: Optional[RiverLayout] = None,
    half_width_kpc: float = 25.0,
    rgb: Optional[dict] = None,
    size=(1920, 1080),
    fps: int = 24,
    hold_start: int = 24,
    hold_end: int = 72,
    frame_directory: Optional[str] = None,
    encode: bool = True,
) -> str:
    """Compose an animation of a galaxy's life: the galaxy through time beside its merger tree.

    The galaxy fills a square panel on the left, labeled with its redshift, time, stellar
    mass and (if ``star_formation(time)`` [M☉/yr] is given) star formation rate.  On the
    right the tree is drawn as a vertical river, revealed up to the current time (the
    future appears as a faint ghost) with a marker on the main branch.  Between
    consecutive ``frames`` an in-between frame blends the two.  The first and last frames
    are held for ``hold_start`` and ``hold_end`` frames.

    Frames are written as PNG files to ``frame_directory`` (default: ``output`` with its
    extension replaced by ``_frames``), then, with ``encode``, made into an H.264 video
    with ``ffmpeg`` (which must be installed).  Returns the path of the video (or of the
    frame directory, without ``encode``).
    """
    from PIL import Image, ImageDraw, ImageFont

    if encode and shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is needed to encode the animation (or pass encode=False for the frames alone)"
        )
    width, height = size
    side = height  # the galaxy panel is square
    black = np.zeros(3)
    rgb = {**_RGB_DEFAULTS, **(rgb or {})}
    try:
        fonts = [ImageFont.truetype(_FONT, s) for s in (40, 28, 20)]
    except OSError:
        fonts = [ImageFont.load_default(size=s) for s in (40, 28, 20)]
    font_big, font, font_small = fonts
    layout = layout or river_layout(tree, mass_minimum=1.0e6, n_times=height - 120)
    river, extent, _ = render_river(
        tree, layout, npix=width - side - 140, background=tuple(black)
    )
    river = river[::-1]  # row 0 at the top: the latest time
    time_of_row = layout.time[::-1]
    x0, y0 = side + 110, 60
    rows_of = np.arange(layout.y.size)[::-1]

    def row_at(time):
        return y0 + np.interp(
            np.log10(time) if layout.axis == "log" else time, layout.y, rows_of
        )

    galaxy_images = [
        np.asarray(
            Image.fromarray(f.rgb(**rgb)[::-1]).resize((side, side), Image.LANCZOS),
            dtype=float,
        )
        for f in frames
    ]

    def compose(image, time, redshift, mass):
        frame = np.zeros((height, width, 3))
        frame[:, :side] = image / 255.0
        panel = river.copy()
        future = time_of_row > time
        panel[future] = 0.10 * panel[future]
        frame[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
        img = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        row = row_at(time)
        column = x0 + (0.0 - extent[0]) / (extent[1] - extent[0]) * panel.shape[1]
        draw.line(
            [(x0, row), (x0 + panel.shape[1], row)], fill=(150, 160, 180), width=1
        )
        draw.ellipse(
            [column - 7, row - 7, column + 7, row + 7], outline=(255, 255, 255), width=3
        )
        for t in (0.3, 0.5, 1, 2, 3, 5, 8, 13.8):
            if layout.time[0] <= t <= layout.time[-1]:
                r = row_at(t)
                draw.line([(x0 - 12, r), (x0 - 4, r)], fill=(200, 200, 210), width=2)
                draw.text(
                    (x0 - 16, r),
                    f"{t:g}",
                    font=font_small,
                    fill=(200, 200, 210),
                    anchor="rm",
                )
        draw.text(
            (x0 - 75, y0 + panel.shape[0] / 2),
            "Gyr",
            font=font_small,
            fill=(200, 200, 210),
            anchor="mm",
        )
        draw.text(
            (x0 + panel.shape[1] / 2, 28),
            "galaxy merger tree",
            font=font,
            fill=(220, 220, 230),
            anchor="mm",
        )
        draw.text(
            (x0 + panel.shape[1] / 2, height - 30),
            "color: specific star formation rate (blue: star-forming; red: quenched)",
            font=font_small,
            fill=(170, 175, 190),
            anchor="mm",
        )
        draw.text((40, 36), f"z = {redshift:.2f}", font=font_big, fill=(255, 255, 255))
        draw.text((40, 88), f"t = {time:.2f} Gyr", font=font, fill=(220, 220, 230))
        draw.text(
            (40, height - 120),
            f"stellar mass  {_scientific(mass)} M☉",
            font=font,
            fill=(220, 220, 230),
        )
        if star_formation is not None:
            draw.text(
                (40, height - 76),
                f"star formation  {star_formation(time):.2f} M☉/yr",
                font=font,
                fill=(220, 220, 230),
            )
        bar = 10.0 / (2.0 * half_width_kpc) * side
        draw.line(
            [(side - 40 - bar, height - 50), (side - 40, height - 50)],
            fill=(255, 255, 255),
            width=4,
        )
        draw.text(
            (side - 40 - bar / 2, height - 62),
            "10 kpc",
            font=font_small,
            fill=(255, 255, 255),
            anchor="mb",
        )
        return img

    directory = frame_directory or os.path.splitext(output)[0] + "_frames"
    os.makedirs(directory, exist_ok=True)
    for f in glob.glob(os.path.join(directory, "f*.png")):
        os.remove(f)
    count = 0

    def emit(img):
        nonlocal count
        img.save(os.path.join(directory, f"f{count:05d}.png"))
        count += 1

    first, last = frames[0], frames[-1]
    for _ in range(hold_start):
        emit(compose(galaxy_images[0], first.time, first.redshift, first.mass))
    for k, a in enumerate(frames):
        emit(compose(galaxy_images[k], a.time, a.redshift, a.mass))
        if k + 1 < len(frames):
            b = frames[k + 1]
            emit(
                compose(
                    0.5 * (galaxy_images[k] + galaxy_images[k + 1]),
                    np.sqrt(a.time * b.time),
                    0.5 * (a.redshift + b.redshift),
                    np.sqrt(max(a.mass, 1.0e-30) * max(b.mass, 1.0e-30)),
                )
            )
    for _ in range(hold_end):
        emit(compose(galaxy_images[-1], last.time, last.redshift, last.mass))
    if not encode:
        return directory
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(directory, "f%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "slow",
            output,
        ],
        check=True,
    )
    return output
