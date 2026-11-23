"""Subhalos orbiting their host, drawn with glowing trails that thin as they lose mass.

:func:`subhalo_orbits` collects, from a model with closely spaced outputs, the orbital
positions and masses of every subhalo of a host halo over time; :func:`render_orbits`
draws the subhalos at one moment, each with a fading trail along its recent orbit whose
width follows its bound mass, so that tidal stripping shows as thinning trails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class SubhaloOrbits:
    """Positions and masses of the subhalos of one host over time.

    Arrays of subhalo properties have shape ``(n_subhalos, n_times)``, with NaN where a
    subhalo does not exist (before it falls in, or after it merges or is disrupted).
    """

    #: Output times [Gyr] and redshifts.
    time: np.ndarray
    redshift: np.ndarray
    #: Node index of each subhalo.
    index: np.ndarray
    #: Position relative to the host [Mpc], shape ``(n_subhalos, n_times, 3)``.
    position: np.ndarray
    #: Bound mass [M☉].
    mass_bound: np.ndarray
    #: Stellar mass [M☉].
    mass_stellar: np.ndarray
    #: Host virial radius [Mpc] and mass [M☉], shape ``(n_times,)``.
    host_radius: np.ndarray
    host_mass: np.ndarray

    @property
    def mass_peak(self) -> np.ndarray:
        """The largest bound mass each subhalo has had up to each time [M☉]."""
        return np.fmax.accumulate(np.nan_to_num(self.mass_bound, nan=0.0), axis=1)


def subhalo_orbits(
    c, tree_index: int, outputs: Optional[Sequence[str]] = None
) -> SubhaloOrbits:
    """Collect the orbits of the subhalos of the host of merger tree ``tree_index``.

    The model should write, at closely spaced outputs, the host's main branch and its
    subhalos (e.g. with an ``any`` galactic filter combining ``mainBranch`` and
    ``haloNotIsolated``), with the ``nodeIndices``, ``indicesTree``, ``positionOrbital``
    and ``virialProperties`` node property extractors, and the ``orbiting`` satellite
    component (for ``satelliteBoundMass``).  A subhalo is kept at an output if its chain
    of hosts leads to the host's main branch, so subhalos of other halos are kept only
    once those halos have fallen in.
    """
    metas = sorted(c.outputs, key=lambda o: o.time)
    if outputs is not None:
        metas = [o for o in metas if o.name in outputs]
    records = []
    for k, output in enumerate(metas):
        group = c[output.path]
        trees = np.repeat(
            np.asarray(group["mergerTreeIndex"][()]),
            np.asarray(group["mergerTreeCount"][()]),
        )
        rows = np.nonzero(trees == tree_index)[0]
        nodes = group["nodeData"]

        def read(name, rows=rows, nodes=nodes):
            return np.asarray(nodes[name][()])[rows]

        index = read("nodeIndex")
        parent = read("parentIndex")
        isolated = read("nodeIsIsolated") == 1
        hosts = np.nonzero(isolated)[0]
        if hosts.size != 1:
            continue  # the host is not written (e.g. before the first output with it)
        host = hosts[0]
        position = np.stack([read(f"positionOrbital{a}") for a in "XYZ"], axis=1)
        mass_bound = read("satelliteBoundMass")
        mass_stellar = read("diskMassStellar") + read("spheroidMassStellar")
        row_of = {int(i): j for j, i in enumerate(index)}
        for j in np.nonzero(~isolated)[0]:
            # Follow the chain of hosts to the top.
            p, steps = int(parent[j]), 0
            while p in row_of and not isolated[row_of[p]] and steps < 64:
                p, steps = int(parent[row_of[p]]), steps + 1
            if p in row_of and row_of[p] == host:
                records.append(
                    (int(index[j]), k, position[j], mass_bound[j], mass_stellar[j])
                )
        records.append(
            (
                None,
                k,
                read("darkMatterOnlyRadiusVirial")[host],
                read("basicMass")[host],
                None,
            )
        )
    times = np.array([o.time for o in metas])
    redshifts = np.array([o.redshift for o in metas])
    ids = sorted({r[0] for r in records if r[0] is not None})
    column = {i: n for n, i in enumerate(ids)}
    n_sub, n_t = len(ids), len(metas)
    position = np.full((n_sub, n_t, 3), np.nan)
    mass_bound = np.full((n_sub, n_t), np.nan)
    mass_stellar = np.full((n_sub, n_t), np.nan)
    host_radius = np.full(n_t, np.nan)
    host_mass = np.full(n_t, np.nan)
    for i, k, a, b, s in records:
        if i is None:
            host_radius[k], host_mass[k] = a, b
        else:
            position[column[i], k] = a
            mass_bound[column[i], k] = b
            mass_stellar[column[i], k] = s
    return SubhaloOrbits(
        times,
        redshifts,
        np.array(ids, dtype=np.int64),
        position,
        mass_bound,
        mass_stellar,
        host_radius,
        host_mass,
    )


def _view_matrix(elevation: float, azimuth: float) -> np.ndarray:
    """Rotation taking host-frame coordinates to view coordinates (x right, y up, z towards the viewer).

    ``elevation`` is the angle of the viewer above the host's x-y plane: 90 looks down on the plane, 0 sees it edge-on
    (with the host's z axis pointing up).
    """
    a, e = np.deg2rad(azimuth), np.deg2rad(elevation - 90.0)
    rz = np.array(
        [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(e), -np.sin(e)], [0.0, np.sin(e), np.cos(e)]]
    )
    return rx @ rz


def _stripping_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    # Fraction of peak mass retained: deep red (mostly stripped) through orange to blue-white (intact).
    return LinearSegmentedColormap.from_list(
        "stripping", ["#b3122e", "#f26b38", "#ffd39a", "#e8f1ff", "#9cc8ff"]
    )


def render_orbits(
    orbits: SubhaloOrbits,
    k: int,
    npix: int = 1080,
    half_width: Optional[float] = None,
    trail: float = 1.5,
    elevation: float = 25.0,
    azimuth: float = 0.0,
    mass_minimum: float = 1.0e8,
    width_scale: float = 0.004,
    brightness: float = 1.0,
    galaxy_mass_minimum: float = 1.0e6,
    bloom: float = 0.6,
) -> np.ndarray:
    """Render the subhalos at time index ``k`` as an RGB float image (row 0 at the top).

    Each subhalo with bound mass of at least ``mass_minimum`` is a glowing point, with a
    trail along its orbit over the preceding ``trail`` Gyr that fades with age.  The width
    of point and trail follows the bound mass at each moment (as ``mass**(1/3)``, scaled
    so that 10¹⁰ M☉ has a width of ``width_scale`` image widths), so tidal stripping shows
    as a thinning trail; their color is the fraction of the subhalo's peak mass still
    bound (blue-white: intact; red: mostly stripped).  Subhalos hosting a galaxy of at
    least ``galaxy_mass_minimum`` have a gold core.  The view is from ``elevation`` degrees
    above the host's x-y plane, turned by ``azimuth`` degrees; ``half_width`` [Mpc] defaults
    to 1.3 times the host's final virial radius.  Light adds, and ``bloom`` adds a soft
    glow.
    """
    from scipy.ndimage import gaussian_filter

    k = int(k) % orbits.time.size  # negative indices count from the end
    half_width = half_width or 1.3 * float(np.nanmax(orbits.host_radius))
    view = _view_matrix(elevation, azimuth)
    cmap = _stripping_colormap()
    time = orbits.time
    # The steps of the trail, from the oldest to now, interpolated finely between outputs.
    first = int(np.searchsorted(time, time[k] - trail))
    steps = np.arange(first, k + 1)
    if steps.size >= 2:
        fine = np.linspace(steps[0], steps[-1], (steps.size - 1) * 6 + 1)
    else:
        fine = np.array([float(k)])
    lo = np.floor(fine).astype(int)
    hi = np.minimum(lo + 1, k)
    w = fine - lo
    age = time[k] - (time[lo] * (1.0 - w) + time[hi] * w)
    fade = np.clip(1.0 - age / trail, 0.0, 1.0) ** 1.5

    def at(values):
        """``values`` (per subhalo and time) interpolated to the trail's steps."""
        weight = w[None, :, None] if values.ndim == 3 else w[None, :]
        return values[:, lo] * (1.0 - weight) + values[:, hi] * weight

    position = at(orbits.position) @ view.T
    mass = at(orbits.mass_bound)
    retained = at(
        np.where(
            np.isfinite(orbits.mass_bound),
            orbits.mass_bound / np.maximum(orbits.mass_peak, 1.0),
            np.nan,
        )
    )
    alive_now = np.isfinite(orbits.mass_bound[:, k]) & (
        orbits.mass_bound[:, k] >= mass_minimum
    )
    valid = (
        np.isfinite(mass)
        & np.isfinite(position[..., 0])
        & alive_now[:, None]
        & (mass > 0.0)
    )
    # Pixel coordinates, brightness, color, and width (as sigma in pixels) of every sample.
    scale = npix / (2.0 * half_width)
    px = (position[..., 0] + half_width) * scale
    py = (half_width - position[..., 1]) * scale
    sigma = np.maximum(
        width_scale * npix * (np.nan_to_num(mass, nan=0.0) / 1.0e10) ** (1.0 / 3.0), 0.6
    )
    weight = np.where(valid, fade[None, :], 0.0) * brightness
    color = cmap(np.clip(np.nan_to_num(retained, nan=1.0), 0.0, 1.0))[..., :3]
    image = np.zeros((npix, npix, 3))
    inside = valid & (px >= 0) & (px < npix) & (py >= 0) & (py < npix)
    # Splat samples in bins of width, then blur each bin to its width: light adds.
    edges = np.geomspace(
        0.6, max(float(sigma[inside].max()) if np.any(inside) else 1.0, 0.61) * 1.001, 9
    )
    which = np.digitize(sigma, edges) - 1
    for b in range(edges.size - 1):
        sel = inside & (which == b)
        if not np.any(sel):
            continue
        buffer = np.zeros((npix * npix, 3))
        flat = py[sel].astype(int) * npix + px[sel].astype(int)
        # Weight each sample so that a trail's surface brightness does not depend on its width: the peak brightness of a
        # Gaussian splat falls as 1/sigma^2, a line's as 1/sigma.
        s = np.sqrt(edges[b] * edges[b + 1])
        for ch in range(3):
            buffer[:, ch] = np.bincount(
                flat, weights=(weight * color[..., ch])[sel] * s, minlength=npix * npix
            )
        buffer = buffer.reshape(npix, npix, 3)
        image += np.stack(
            [gaussian_filter(buffer[..., ch], s) for ch in range(3)], axis=-1
        )
    # Gold cores for subhalos hosting galaxies, at their current positions.
    current = alive_now & (
        np.nan_to_num(orbits.mass_stellar[:, k], nan=0.0) >= galaxy_mass_minimum
    )
    if np.any(current):
        p = orbits.position[current, k] @ view.T
        cx, cy = (p[:, 0] + half_width) * scale, (half_width - p[:, 1]) * scale
        ok = (cx >= 0) & (cx < npix) & (cy >= 0) & (cy < npix)
        core = np.zeros((npix, npix))
        stellar = orbits.mass_stellar[current, k][ok]
        np.add.at(
            core,
            (cy[ok].astype(int), cx[ok].astype(int)),
            40.0 * (stellar / 1.0e8) ** 0.25,
        )
        core = gaussian_filter(core, 0.0025 * npix)
        image += core[..., None] * np.array([1.0, 0.8, 0.35])[None, None, :]
    # The host: a soft glow, and a faint ring at its virial radius.
    yy, xx = np.mgrid[0:npix, 0:npix]
    r = np.hypot(xx + 0.5 - npix / 2, yy + 0.5 - npix / 2) / scale
    radius = orbits.host_radius[k]
    if np.isfinite(radius):
        image += (0.25 * np.exp(-((r / (0.15 * radius)) ** 1.0)))[..., None] * np.array(
            [0.55, 0.6, 1.0]
        )
        image += (0.10 * np.exp(-0.5 * ((r - radius) * scale / 1.2) ** 2))[
            ..., None
        ] * np.array([0.6, 0.65, 0.8])
    if bloom > 0.0:
        image = image + bloom * np.stack(
            [gaussian_filter(image[..., ch], 0.012 * npix) for ch in range(3)], axis=-1
        )
    # Tone map, so that dense regions saturate gracefully.
    return 1.0 - np.exp(-image)


def compose_orbit_animation(
    orbits: SubhaloOrbits,
    output: str,
    size=(1920, 1080),
    fps: int = 30,
    zoom: float = 1.6,
    turn: float = 90.0,
    elevation: float = 25.0,
    hold_end: int = 60,
    first: int = 0,
    frame_directory: Optional[str] = None,
    encode: bool = True,
    **render_kwargs,
) -> str:
    """Animate the subhalos of a host along their orbits (see :func:`render_orbits`).

    The view spans ``zoom`` times the host's virial radius (smoothed over time), so it
    follows the host's growth, and turns by ``turn`` degrees over the animation.  Each
    output from index ``first`` on is one frame; the last is held for ``hold_end`` frames.
    Frames are labeled with the time, redshift and number of subhalos shown, with a scale
    bar and a key to the colors.  Frames are written as PNG files to ``frame_directory``
    (default: ``output`` with its extension replaced by ``_frames``), then, with
    ``encode``, made into an H.264 video with ``ffmpeg``.  Other keywords go to
    :func:`render_orbits`.  Returns the path of the video (or of the frame directory).
    """
    import glob
    import os
    import shutil
    import subprocess

    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import gaussian_filter1d

    if encode and shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is needed to encode the animation (or pass encode=False for the frames alone)"
        )
    width, height = size
    side = height
    try:
        font_big, font, font_small = (
            ImageFont.truetype("DejaVuSans.ttf", s) for s in (40, 28, 20)
        )
    except OSError:
        font_big, font, font_small = (
            ImageFont.load_default(size=s) for s in (40, 28, 20)
        )
    radius = np.asarray(orbits.host_radius, dtype=float)
    valid = np.isfinite(radius)
    radius = np.interp(np.arange(radius.size), np.nonzero(valid)[0], radius[valid])
    radius = gaussian_filter1d(radius, 8.0, mode="nearest")
    mass_minimum = render_kwargs.get("mass_minimum", 1.0e8)
    galaxy_minimum = render_kwargs.get("galaxy_mass_minimum", 1.0e6)
    cmap = _stripping_colormap()
    directory = frame_directory or os.path.splitext(output)[0] + "_frames"
    os.makedirs(directory, exist_ok=True)
    for f in glob.glob(os.path.join(directory, "f*.png")):
        os.remove(f)
    count = 0
    steps = list(range(first, orbits.time.size))
    for n, k in enumerate(steps):
        half_width = zoom * radius[k]
        azimuth = turn * n / max(len(steps) - 1, 1)
        image = render_orbits(
            orbits,
            k,
            npix=side,
            half_width=half_width,
            azimuth=azimuth,
            elevation=elevation,
            **render_kwargs,
        )
        frame = np.zeros((height, width, 3))
        x0 = (width - side) // 2
        frame[:, x0 : x0 + side] = image
        img = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        shown = np.isfinite(orbits.mass_bound[:, k]) & (
            orbits.mass_bound[:, k] >= mass_minimum
        )
        galaxies = shown & (np.nan_to_num(orbits.mass_stellar[:, k]) >= galaxy_minimum)
        draw.text(
            (40, 36),
            f"t = {orbits.time[k]:.2f} Gyr",
            font=font_big,
            fill=(255, 255, 255),
        )
        draw.text(
            (40, 90), f"z = {orbits.redshift[k]:.2f}", font=font, fill=(220, 220, 230)
        )
        draw.text(
            (40, height - 120),
            f"{np.count_nonzero(shown)} subhalos",
            font=font,
            fill=(220, 220, 230),
        )
        draw.text(
            (40, height - 76),
            f"{np.count_nonzero(galaxies)} with galaxies",
            font=font,
            fill=(255, 214, 120),
        )
        # Scale bar: a round length near a fifth of the view.
        kpc = 2.0 * half_width * 1.0e3 / 5.0
        bar_kpc = min(
            (1, 2, 5, 10, 20, 50, 100, 200, 500), key=lambda b: abs(np.log(b / kpc))
        )
        bar = bar_kpc / (2.0 * half_width * 1.0e3) * side
        draw.line(
            [(width - 60 - bar, height - 60), (width - 60, height - 60)],
            fill=(255, 255, 255),
            width=4,
        )
        draw.text(
            (width - 60 - bar / 2, height - 72),
            f"{bar_kpc} kpc",
            font=font_small,
            fill=(255, 255, 255),
            anchor="mb",
        )
        # Key: color by the fraction of peak mass retained, and the gold galaxy cores.
        kx, ky, kw = width - 360, 40, 300
        for i in range(kw):
            r, g, b, _ = cmap(i / (kw - 1))
            draw.line(
                [(kx + i, ky), (kx + i, ky + 14)],
                fill=(int(255 * r), int(255 * g), int(255 * b)),
            )
        draw.text((kx, ky + 22), "stripped", font=font_small, fill=(200, 200, 210))
        draw.text(
            (kx + kw, ky + 22),
            "intact",
            font=font_small,
            fill=(200, 200, 210),
            anchor="ra",
        )
        draw.text(
            (kx + kw / 2, ky + 56),
            "bound mass / peak mass",
            font=font_small,
            fill=(200, 200, 210),
            anchor="ma",
        )
        draw.text(
            (kx + kw / 2, ky + 90),
            "trail width: bound mass",
            font=font_small,
            fill=(200, 200, 210),
            anchor="ma",
        )
        draw.ellipse([kx + 40, ky + 128, kx + 52, ky + 140], fill=(255, 214, 120))
        draw.text(
            (kx + 62, ky + 122), "hosts a galaxy", font=font_small, fill=(200, 200, 210)
        )
        repeats = 1 + (hold_end if n == len(steps) - 1 else 0)
        for _ in range(repeats):
            img.save(os.path.join(directory, f"f{count:05d}.png"))
            count += 1
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
