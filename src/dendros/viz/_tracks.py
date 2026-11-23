"""Many histories at once: thousands of tracks drawn as lines of light that add.

:func:`main_branch_histories` collects the main-branch history of every central galaxy
in a model (from the ``galaxyMergerTree`` node operator); :func:`render_tracks` draws
any set of tracks additively, so that where many coincide they glow brighter, and
:func:`draw_tracks` shows them on matplotlib axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ._tree import galaxy_merger_tree


@dataclass
class Histories:
    """Main-branch histories of many galaxies.

    ``time`` and each entry of ``properties`` are lists with one array per galaxy (of
    differing lengths), as sampled by the ``galaxyMergerTree`` node operator.
    """

    tree_index: np.ndarray
    time: List[np.ndarray]
    properties: Dict[str, List[np.ndarray]]

    def __len__(self) -> int:
        return len(self.time)

    def final(self, name: str) -> np.ndarray:
        """The value of property ``name`` at the last sample of each history."""
        return np.array([v[-1] if v.size else np.nan for v in self.properties[name]])

    def time_fraction(self, name: str, fraction: float = 0.5) -> np.ndarray:
        """The time [Gyr] at which property ``name`` first reached ``fraction`` of its final value (e.g. the formation
        time of a halo, for ``"MassBasic"`` and 0.5).  NaN where it never did."""
        result = np.full(len(self), np.nan)
        for i, (t, v) in enumerate(zip(self.time, self.properties[name])):
            if v.size < 2 or not v[-1] > 0.0:
                continue
            above = np.nonzero(v >= fraction * v[-1])[0]
            if above.size == 0:
                continue
            j = above[0]
            if j == 0:
                result[i] = t[0]
            else:
                # Interpolate in log mass between the bracketing samples.
                a, b = np.log(max(v[j - 1], 1.0e-300)), np.log(v[j])
                target = np.log(fraction * v[-1])
                w = (target - a) / (b - a) if b > a else 1.0
                result[i] = t[j - 1] + w * (t[j] - t[j - 1])
        return result


def main_branch_histories(
    c, output: Optional[str] = None, properties: Optional[Sequence[str]] = None
) -> Histories:
    """Collect the main-branch history of the central galaxy of every merger tree in ``output``.

    The model must run the ``galaxyMergerTree`` node operator and output its data (see
    :func:`galaxy_merger_tree`).  ``output`` defaults to the last output; ``properties``
    (default: all recorded) names the properties to keep, as in ``Branch.properties``.
    """
    if output is None:
        output = max(c.outputs, key=lambda o: o.time).name
    trees = np.asarray(c[f"Outputs/{output}/mergerTreeIndex"][()])
    indices, times, values = [], [], {}
    for index in trees:
        try:
            tree = galaxy_merger_tree(c, output, int(index), include_satellites=False)
        except (KeyError, ValueError):
            continue  # no central with a galaxy merger tree in this tree
        main = tree.branches[tree.root]
        names = properties if properties is not None else list(main.properties)
        indices.append(int(index))
        times.append(main.time)
        for name in names:
            values.setdefault(name, []).append(main.properties[name])
    return Histories(np.array(indices, dtype=np.int64), times, values)


def render_tracks(
    x: Sequence[np.ndarray],
    y: Sequence[np.ndarray],
    color: Sequence,
    extent,
    size=(1600, 1000),
    cmap=None,
    norm=None,
    width: float = 1.0,
    brightness: float = 0.08,
    glow: float = 0.5,
    glow_sigma: float = 0.006,
) -> np.ndarray:
    """Draw tracks as lines of light that add, returning an RGB float image (row 0 at the top).

    ``x`` and ``y`` hold one array per track, in the coordinates of ``extent`` =
    ``(x0, x1, y0, y1)``.  ``color`` gives, per track, either one value or an array of
    values along it, mapped by ``norm`` and ``cmap`` (by default, the range of the values
    and viridis).  Lines have a Gaussian profile of standard deviation ``width`` pixels,
    and ``brightness`` is the intensity a single line contributes: where many tracks
    coincide they add, and a tone curve lets dense regions saturate gracefully.  ``glow``
    adds a soft halo of width ``glow_sigma`` (in units of the image width).
    """
    import matplotlib
    from matplotlib.colors import Normalize
    from scipy.ndimage import gaussian_filter

    cmap = cmap or matplotlib.colormaps["viridis"]
    if norm is None:
        flat = np.concatenate(
            [np.atleast_1d(np.asarray(v, dtype=float)).ravel() for v in color]
        )
        norm = Normalize(*np.nanpercentile(flat, [2.0, 98.0]))
    w, h = size
    x0, x1, y0, y1 = extent
    buffer = np.zeros((h * w, 3))
    for xs, ys, cs in zip(x, y, color):
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if np.count_nonzero(ok) < 2:
            continue
        cs = np.broadcast_to(np.asarray(cs, dtype=float), xs.shape)
        xs, ys, cs = xs[ok], ys[ok], cs[ok]
        px = (xs - x0) / (x1 - x0) * w
        py = (y1 - ys) / (y1 - y0) * h
        # Resample along the track every half pixel, so the line's intensity does not depend on its sampling.
        step = np.hypot(np.diff(px), np.diff(py))
        arc = np.concatenate([[0.0], np.cumsum(step)])
        if arc[-1] <= 0.0:
            continue
        s = np.arange(0.0, arc[-1], 0.5)
        sx, sy = np.interp(s, arc, px), np.interp(s, arc, py)
        sc = np.interp(s, arc, cs)
        inside = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        if not np.any(inside):
            continue
        rgb = cmap(norm(np.nan_to_num(sc[inside], nan=norm.vmin)))[:, :3]
        flat = sy[inside].astype(int) * w + sx[inside].astype(int)
        for ch in range(3):
            buffer[:, ch] += np.bincount(flat, weights=rgb[:, ch], minlength=h * w)
    image = buffer.reshape(h, w, 3)
    # Each half-pixel sample deposits unit weight: a line of profile width sigma then peaks at 2/(sqrt(2 pi) sigma).
    image = np.stack(
        [gaussian_filter(image[..., ch], width) for ch in range(3)], axis=-1
    )
    image *= brightness * np.sqrt(2.0 * np.pi) * width / 2.0
    if glow > 0.0:
        image = image + glow * np.stack(
            [gaussian_filter(image[..., ch], glow_sigma * w) for ch in range(3)],
            axis=-1,
        )
    return 1.0 - np.exp(-image)


def draw_tracks(ax, x, y, color, extent, size=(1600, 1000), **kwargs):
    """Draw tracks (see :func:`render_tracks`) on the matplotlib Axes ``ax``, in data coordinates ``extent``."""
    image = render_tracks(x, y, color, extent, size=size, **kwargs)
    ax.imshow(
        image, extent=extent, aspect="auto", interpolation="antialiased", origin="upper"
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return image
