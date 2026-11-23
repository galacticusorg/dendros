"""Ray-marched images of a :class:`GalaxyScene`, with absorption by dust.

The camera looks along its own -z axis.  Each band's image is the integral along
every line of sight of the stellar emissivity, attenuated by the dust between that
point and the observer:

.. math::

   I = \\int j(s) \\, e^{-\\tau(s)} \\, ds , \\qquad \\tau(s) = \\int_s^{\\infty} \\kappa \\rho_\\mathrm{dust} \\, ds' .

The integral is done slab by slab from the front of the volume to the back; each
slab is treated as uniform, so a slab of optical depth :math:`\\Delta\\tau`
contributes :math:`j \\Delta s \\, e^{-\\tau} (1-e^{-\\Delta\\tau})/\\Delta\\tau`.

Dust follows the gas: :math:`\\rho_\\mathrm{dust} = f_\\mathrm{d/m} Z \\rho_\\mathrm{gas}`, with
the gas metallicity :math:`Z` from the model and a dust-to-metals ratio
:math:`f_\\mathrm{d/m}`.  The opacity is
:math:`\\kappa_\\lambda = \\kappa_V (\\lambda/5500\\,\\mathrm{Å})^{-\\beta}`, with default
:math:`\\kappa_V = 2.1\\times10^4\\,\\mathrm{cm^2\\,g^{-1}}` per unit dust mass (Milky Way
extinction per hydrogen atom, :math:`A_V/N_\\mathrm{H}`, for a dust-to-gas ratio of 0.01)
and :math:`\\beta=1`.  This is a screen-free, scattering-free treatment: light is absorbed,
never scattered back in.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from ._galaxy import WAVELENGTH_EFFECTIVE, GalaxyScene

_CM_PER_MPC = 3.0856775814913673e24
_GRAMS_PER_MSUN = 1.98847e33
_PC2_PER_MPC2 = 1.0e12

#: A texture multiplies the emissivity (or dust density) of a component.  It is called
#: as ``texture(x, y, z, component, kind)`` with galaxy-frame coordinates [Mpc], the
#: component name, and ``kind`` one of ``"young"``, ``"old"`` or ``"dust"``, and returns
#: a multiplicative factor (an array broadcastable to ``x``).
Texture = Callable[[np.ndarray, np.ndarray, np.ndarray, str, str], np.ndarray]


@dataclass
class DustModel:
    """Parameters of the dust absorption."""

    dust_to_metals: float = 0.4
    #: Opacity at 5500 Å per unit dust mass [cm²/g].
    kappa_V: float = 2.1e4
    #: Power-law index of the opacity, :math:`\\kappa \\propto \\lambda^{-\\beta}`.
    beta: float = 1.0
    enabled: bool = True
    #: Ratio of the dust's scale height to that of the gas in the model. 1 (the default) is the model: Galacticus gives
    #: disk gas and stars one scale height, so the dust is fully mixed with the starlight. Values below 1 (real disks
    #: have dust layers 2-3 times thinner than their stars) squeeze the dust layer *at fixed column density*, which is
    #: what makes edge-on dust lanes. This is a non-model choice.
    scale_height_factor: float = 1.0

    def kappa(self, band: str, redshift: float = 0.0) -> float:
        """Opacity in ``band`` per unit dust mass [Mpc²/M☉].

        For a galaxy at ``redshift`` seen in an observed-frame band, the dust absorbs at
        the rest-frame wavelength, the band's divided by 1+z.
        """
        wavelength = WAVELENGTH_EFFECTIVE.get(band)
        if wavelength is None:
            raise KeyError(f"no effective wavelength known for band '{band}'")
        kappa = self.kappa_V * (wavelength / (1.0 + redshift) / 5500.0) ** (-self.beta)
        return kappa * _GRAMS_PER_MSUN / _CM_PER_MPC**2


def rotation_matrix(
    inclination: float, position_angle: float = 0.0, azimuth: float = 0.0
) -> np.ndarray:
    """The matrix taking camera coordinates to galaxy-frame coordinates (angles in degrees).

    ``inclination`` 0 is face-on and 90 edge-on (with the disk then lying along the image
    x axis for ``position_angle`` 0); ``position_angle`` rotates the galaxy in the image
    plane; ``azimuth`` rotates the galaxy about its own axis (which matters only for
    non-axisymmetric textures).
    """

    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    i, p, a = np.deg2rad([inclination, position_angle, azimuth])
    # camera = Rz(p) Rx(i) Rz(a) galaxy, so galaxy = Rz(-a) Rx(-i) Rz(-p) camera.
    return rz(-a) @ rx(-i) @ rz(-p)


def render_galaxy(
    scene: GalaxyScene,
    bands: Optional[Sequence[str]] = None,
    inclination: float = 60.0,
    position_angle: float = 0.0,
    azimuth: float = 0.0,
    npix: int = 512,
    half_width: Optional[float] = None,
    n_steps: int = 512,
    dust: Optional[DustModel] = None,
    texture: Optional[Texture] = None,
    components: Optional[Sequence[str]] = None,
    young_scale_height: Optional[Dict[str, float]] = None,
    threads: Optional[int] = None,
    rotation: Optional[np.ndarray] = None,
    redshift: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Render ``scene`` in each of ``bands``.

    Returns a dict mapping band to an ``(npix, npix)`` image of surface brightness in
    Galacticus luminosity units per pc² (so ``21.572 - 2.5 log10(I)`` is the surface
    brightness in AB mag/arcsec²).  ``half_width`` [Mpc] defaults to four stellar
    half-mass radii; the line of sight is sampled over the same depth either side of
    the galaxy center in ``n_steps`` slabs.  ``components`` restricts the stellar
    components drawn (e.g. ``["disk"]``); dust is always included unless disabled.
    ``threads`` sets the number of threads (default: up to 8).

    ``rotation``, if given, is the 3x3 matrix taking camera coordinates to the galaxy's
    frame (z along its axis), and replaces ``inclination``, ``position_angle`` and
    ``azimuth``.  ``redshift`` is the galaxy's redshift when ``bands`` are observed-frame
    bands, so that dust absorbs at the rest-frame wavelength.

    ``young_scale_height`` maps a component name to the ratio of the scale height of
    its young stars to that of the component, e.g. ``{"disk": 0.3}``. Galacticus gives
    all the stars of a component one profile, but young stars in real disks lie in a
    layer a few times thinner than the old ones. As for the dust, the layer is squeezed
    vertically at fixed column density, so the light of every line of sight seen face-on,
    and the total light, are unchanged. This is a non-model choice; it matters mostly
    together with a clumpy texture, whose clumps are otherwise drawn through the full
    thickness of the disk.

    The line of sight is sampled at the centers of ``n_steps`` slabs, so the slab
    thickness ``2 half_width / n_steps`` must be small compared with the thinnest
    structure along the line of sight: face-on, the disk's scale height (0.137 scale
    radii by default), reduced by ``DustModel.scale_height_factor`` for the dust and by
    ``young_scale_height`` for young stars.
    """
    bands = list(bands if bands is not None else scene.bands)
    dust = dust if dust is not None else DustModel()
    half_width = half_width if half_width is not None else scene.extent()
    if not half_width > 0.0:
        raise ValueError("galaxy has no size: cannot choose an image extent")
    stars = [s for s in scene.stars if components is None or s.name in components]
    stars = [s for s in stars if s.mass > 0.0 and not s.density.is_empty]
    gas = [
        g
        for g in scene.gas
        if dust.enabled
        and g.mass > 0.0
        and g.metallicity > 0.0
        and not g.density.is_empty
    ]

    rot = (
        np.asarray(rotation, dtype=float)
        if rotation is not None
        else rotation_matrix(inclination, position_angle, azimuth)
    )
    edges = np.linspace(-half_width, half_width, npix + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    depth = np.linspace(
        half_width, -half_width, n_steps + 1
    )  # front (observer side) to back
    ds = depth[0] - depth[1]
    s_mid = 0.5 * (depth[1:] + depth[:-1])
    kappa = {b: dust.kappa(b, redshift) for b in bands}
    young_scale_height = dict(young_scale_height or {})
    if any(not f > 0.0 for f in young_scale_height.values()):
        raise ValueError("young_scale_height factors must be positive")

    def march(rows: slice) -> Dict[str, np.ndarray]:
        """Integrate the lines of sight through image rows ``rows``, front to back."""
        X, Y = np.meshgrid(centers, centers[rows], indexing="xy")
        image = {b: np.zeros(X.shape) for b in bands}
        tau = {b: np.zeros(X.shape) for b in bands}
        for s in s_mid:
            x = rot[0, 0] * X + rot[0, 1] * Y + rot[0, 2] * s
            y = rot[1, 0] * X + rot[1, 1] * Y + rot[1, 2] * s
            z = rot[2, 0] * X + rot[2, 1] * Y + rot[2, 2] * s
            R2 = x * x + y * y
            squeezed = {}

            def coordinates(f):
                """Spherical (r, theta) of the point (R, z/f): a layer squeezed vertically by a factor f."""
                if f not in squeezed:
                    zf = z / f
                    rf = np.sqrt(R2 + zf * zf)
                    squeezed[f] = (
                        rf,
                        np.arccos(
                            np.clip(np.abs(zf) / np.maximum(rf, 1.0e-300), 0.0, 1.0)
                        ),
                    )
                return squeezed[f]

            r, theta = coordinates(1.0)
            # Dust density in this slab.  With a scale-height factor f, the dust density at (R, z) is that of the gas at
            # (R, z/f), divided by f, conserving column density.
            rho_dust = np.zeros_like(r)
            f = dust.scale_height_factor
            if gas:
                r_dust, theta_dust = coordinates(f)
            for g in gas:
                rho = (
                    g.density(r_dust, theta_dust)
                    * dust.dust_to_metals
                    * g.metallicity
                    / f
                )
                if texture is not None:
                    rho = rho * texture(x, y, z, g.name, "dust")
                rho_dust += rho
            # Emissivity in each band.
            emissivity = {b: np.zeros_like(r) for b in bands}
            for c in stars:
                rho = c.density(r, theta)
                if not c.luminosity_young or (
                    texture is None and c.name not in young_scale_height
                ):
                    factor = (
                        None if texture is None else texture(x, y, z, c.name, "old")
                    )
                    for b in bands:
                        j = rho * c.light_to_mass(b)
                        emissivity[b] += j if factor is None else j * factor
                else:
                    old = 1.0 if texture is None else texture(x, y, z, c.name, "old")
                    young = (
                        1.0 if texture is None else texture(x, y, z, c.name, "young")
                    )
                    f_young = young_scale_height.get(c.name, 1.0)
                    rho_young = (
                        rho
                        if f_young == 1.0
                        else c.density(*coordinates(f_young)) / f_young
                    )
                    for b in bands:
                        emissivity[b] += (
                            rho * c.light_to_mass(b, young=False) * old
                            + rho_young * c.light_to_mass(b, young=True) * young
                        )
            for b in bands:
                dtau = kappa[b] * rho_dust * ds
                # (1-exp(-dtau))/dtau, with its small-dtau limit.
                escape = np.where(
                    dtau > 1.0e-6,
                    -np.expm1(-dtau) / np.maximum(dtau, 1.0e-300),
                    1.0 - 0.5 * dtau,
                )
                image[b] += emissivity[b] * ds * np.exp(-tau[b]) * escape
                tau[b] += dtau
        return image

    # Lines of sight are independent, so blocks of image rows are marched in parallel threads (numpy releases the
    # GIL in the array operations that dominate the cost).
    threads = threads if threads is not None else min(8, os.cpu_count() or 1)
    # One block per thread, but at least 32 rows each: the per-slab Python overhead dominates for thinner blocks.
    block = max(32, -(-npix // threads))
    blocks = [slice(i, min(i + block, npix)) for i in range(0, npix, block)]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(march, blocks))
    return {
        b: np.concatenate([res[b] for res in results], axis=0) / _PC2_PER_MPC2
        for b in bands
    }


def surface_brightness_magnitudes(image: np.ndarray) -> np.ndarray:
    """Convert an image from :func:`render_galaxy` to AB mag/arcsec²."""
    with np.errstate(divide="ignore"):
        return 21.572 - 2.5 * np.log10(image)


def compose_rgb(
    images: Dict[str, np.ndarray],
    red: str = "SDSS_i",
    green: str = "SDSS_r",
    blue: str = "SDSS_g",
    mu_stretch: float = 22.0,
    Q: float = 8.0,
    weights: Optional[Sequence[float]] = None,
    white: Optional[Sequence[float]] = (0.6, 0.3),
    bloom: float = 0.0,
    bloom_sigma: float = 0.02,
) -> np.ndarray:
    """Combine three band images into an 8-bit RGB image with the asinh stretch of Lupton et al. (2004).

    The stretch is set in physical units: ``mu_stretch`` is the surface brightness
    [AB mag/arcsec²] at which the mapping turns from linear to logarithmic, so images
    of different galaxies made with the same settings are directly comparable.
    The channels are white-balanced so that light with colors ``white`` = (g-r, r-i)
    [AB] appears neutral; the default is roughly the integrated color of an Sb galaxy,
    so redder (older) light looks warm and bluer (younger) light cool.  Pass ``white=None``
    to map the bands without balancing, or ``weights`` to give the channel weights
    explicitly.  ``bloom`` > 0 adds a soft glow around the brightest regions (purely
    cosmetic), of width ``bloom_sigma`` in units of the image width.
    """
    from astropy.visualization import make_lupton_rgb

    stretch = 10.0 ** (-0.4 * (mu_stretch - 21.572))
    if weights is None:
        if white is None or (red, green, blue) != ("SDSS_i", "SDSS_r", "SDSS_g"):
            weights = (1.0, 1.0, 1.0)
        else:
            # Flux ratios f_g/f_r = 10^(-0.4 (g-r)) and f_i/f_r = 10^(0.4 (r-i)) for the reference color; weight each
            # channel by the inverse so that color maps to equal channel values.
            g_r, r_i = white
            weights = (10.0 ** (-0.4 * r_i), 1.0, 10.0 ** (0.4 * g_r))
    r, g, b = (images[band] * w for band, w in zip((red, green, blue), weights))
    if bloom > 0.0:
        from scipy.ndimage import gaussian_filter

        sigma = bloom_sigma * r.shape[0]
        r, g, b = (ch + bloom * gaussian_filter(ch, sigma) for ch in (r, g, b))
    with np.errstate(
        invalid="ignore", divide="ignore"
    ):  # empty pixels give 0/0 inside astropy (and are set to zero)
        return make_lupton_rgb(r, g, b, minimum=0.0, stretch=stretch, Q=Q)
